#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 部分改写自 NVIDIA/TensorRT-LLM `.claude/skills/kernel-triton-writing/scripts/verify_kernel.py`
# (本机镜像 50f85bfe3e;Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES,Apache-2.0):
# 沿用它的契约名(kernel_fn / reference_fn / get_inputs)、「先对拍、不过不许测速」、递归比较与逐项误差字段。
# 输出 NaN 填充未初始化内存的做法取自
# fla-org/flash-linear-attention 的 `.agents/skills/fla-optimization-loop/references/TRAPS.md`(MIT)。
# 代码为重写,改了什么:
#   - 一个用例扩成多组用例(get_cases),逐用例对拍,每个输出单独给最大绝对与相对误差
#   - 容差按输出 dtype 取默认值,可整体或逐用例覆盖,容差来源写进 JSON
#   - 输入的复制保留 stride、storage offset 与别名关系,非连续与偏移视图不会被复制成连续张量
#   - RNG 类算子可只比统计量;输出 NaN 填充;wrapper 内同步点检查;覆盖面汇总
#   - 不再生成临时脚本、起子进程;只依赖 torch 与标准库;退出码 0/1/2
"""kernel 对拍:导入约定名字的 kernel、参考实现与用例生成函数,逐用例比较并输出 JSON。

只依赖 torch 与 Python 标准库。复制进任何项目都能用。
"""

from __future__ import annotations

import argparse
import collections
import importlib
import importlib.util
import inspect
import json
import math
import os
import platform
import random
import shlex
import subprocess
import sys
import traceback
import warnings

EXIT_PASS, EXIT_FAIL, EXIT_ENV = 0, 1, 2
SCHEMA = "oss-kernel-dev/verify/v1"
INT32_MAX = 2**31 - 1

CASE_KEYS = {"args", "kwargs", "name", "tags", "atol", "rtol", "compare", "stats_atol", "stats_fn",
             "equal_nan", "compare_args", "seed"}

# (rtol, atol)。torch 预设在运行时优先读所装 torch 的 torch.testing._comparison._DTYPE_PRECISIONS,
# 读不到才用下表(torch.testing.assert_close 文档里的默认值)。其余 dtype 在 torch 里按精确比较。
TORCH_DEFAULTS = {
    "float16": (1e-3, 1e-5), "bfloat16": (1.6e-2, 1e-5), "float32": (1.3e-6, 1e-5), "float64": (1e-7, 1e-7),
    "complex32": (1e-3, 1e-5), "complex64": (1.3e-6, 1e-5), "complex128": (1e-7, 1e-7),
}
# TensorRT-LLM kernel-triton-writing 的 Tolerance guide(float32 取逐元素那一行;matmul 那一行要自己用 --atol/--rtol 给)
TRTLLM_PRESET = {"float16": (1e-3, 1e-3), "bfloat16": (1e-2, 1e-2), "float32": (1e-5, 1e-5)}

KNOWN_TAGS = {"tail", "boundary", "noncontig", "offset", "hard_values", "large_index"}


class EnvError(Exception):
    """用法、契约或环境错误,退出码 2。"""


# ---------------------------------------------------------------- 不依赖 torch 的部分

def load_module(spec: str):
    """'path/to/file.py' 或 'pkg.mod'。"""
    try:
        if spec.endswith(".py"):
            path = os.path.abspath(spec)
            if not os.path.exists(path):
                raise EnvError(f"找不到文件 {path}")
            d = os.path.dirname(path)
            if d not in sys.path:
                sys.path.insert(0, d)
            name = "_verify_user_" + os.path.splitext(os.path.basename(path))[0]
            s = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(s)
            sys.modules[name] = mod
            s.loader.exec_module(mod)
            return mod
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        return importlib.import_module(spec)
    except EnvError:
        raise
    except Exception as e:  # 导入用户代码的任何失败都算用法或环境错误
        raise EnvError(f"导入 {spec!r} 失败:{type(e).__name__}: {e}") from e


def normalize_case(raw, i: int) -> dict:
    if isinstance(raw, (list, tuple)):
        return {"args": list(raw), "kwargs": {}, "name": f"case{i}", "tags": [], "compare": "elementwise"}
    if not isinstance(raw, dict):
        raise EnvError(f"第 {i} 个用例的类型是 {type(raw).__name__};用例要么是参数列表,要么是含 args 或 kwargs 的 dict")
    unknown = set(raw) - CASE_KEYS
    if unknown:
        raise EnvError(f"第 {i} 个用例有未知键 {sorted(unknown)};可用的键:{sorted(CASE_KEYS)}")
    if "args" not in raw and "kwargs" not in raw:
        raise EnvError(f"第 {i} 个用例是 dict 但没有 args 也没有 kwargs;关键字参数放进 kwargs")
    if ("atol" in raw) != ("rtol" in raw):
        raise EnvError(f"第 {i} 个用例的 atol 与 rtol 要么都给,要么都不给")
    c = dict(raw)
    c["args"] = list(c.get("args", []))
    c["kwargs"] = dict(c.get("kwargs", {}))
    c.setdefault("name", f"case{i}")
    c["tags"] = [str(t) for t in c.get("tags", [])]
    c.setdefault("compare", "elementwise")
    if c["compare"] not in ("elementwise", "stats"):
        raise EnvError(f"用例 {c['name']!r} 的 compare 只能是 elementwise 或 stats")
    return c


def pick_tolerance(dtype_name: str, case: dict, cli: tuple | None, preset: str, table: dict, table_src: str):
    """返回 (rtol, atol, 来源)。优先级:用例 > 命令行 > 预设表。"""
    if "atol" in case:
        return float(case["rtol"]), float(case["atol"]), "用例的 atol/rtol"
    if cli is not None:
        return cli[0], cli[1], "命令行 --atol/--rtol"
    if dtype_name in table:
        r, a = table[dtype_name]
        return r, a, f"{table_src}({dtype_name})"
    return 0.0, 0.0, f"{preset} 预设没有 {dtype_name} 的容差,按精确比较;需要时用 --atol/--rtol 或用例的 atol/rtol 给出"


def coverage_warnings(cov: dict, expect_dtypes: list[str] | None = None) -> list[str]:
    w = []
    if cov["cases"] == 0:
        return ["没有跑任何用例"]
    if cov["distinct_input_signatures"] < 2:
        w.append("只有 1 组输入形状;加多组形状,含尾部余数与边界")
    if not ({"tail", "boundary"} & set(cov["tags"])):
        w.append("没有用例标 tail 或 boundary;脚本判断不了尾部余数与边界是否覆盖,覆盖了就给用例加标签")
    if not (cov["noncontiguous_input"] or "noncontig" in cov["tags"]):
        w.append("没有非连续输入(转置、切片、padding 行);kernel 不支持的布局也要测它的拒绝路径")
    if not (cov["storage_offset_input"] or "offset" in cov["tags"]):
        w.append("没有 storage offset 非零的输入视图;新分配的张量都是对齐的,测不到对齐与特化问题")
    if expect_dtypes:
        miss = sorted(set(expect_dtypes) - set(cov["float_dtypes"]))
        if miss:
            w.append(f"--expect-dtypes 声明的 {miss} 没有用例")
    elif len(cov["float_dtypes"]) < 2:
        w.append(f"浮点输入只覆盖了 {cov['float_dtypes'] or '0 种'} dtype;kernel 只支持这一种时用 --expect-dtypes 声明")
    if not (cov["nonfinite_input"] or "hard_values" in cov["tags"]):
        w.append("没有难数值用例(inf、NaN、极大极小值、全零、大量重复值);有就给用例加 hard_values 标签")
    return w


def sanitize(obj):
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def git_info() -> dict | None:
    def run(*a):
        r = subprocess.run(["git", *a], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    try:
        commit = run("rev-parse", "--short", "HEAD")
        if commit is None:
            return None
        return {"commit": commit, "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.TimeoutExpired):
        return None


def _emit(res: dict, out: str | None) -> bool:
    """打印 JSON,给了 --out 再写一份;写文件失败返回 False(调用方按环境错误退出 2)。"""
    text = json.dumps(sanitize(res), ensure_ascii=False, indent=2)
    print(text)
    if out:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError as e:
            print(f"[verify_kernel] 写 --out {out} 失败:{e}", file=sys.stderr)
            return False
    return True


# ---------------------------------------------------------------- 依赖 torch 的部分

def _dtype_name(dt) -> str:
    return str(dt).replace("torch.", "")


def torch_tolerance_table(torch):
    try:
        from torch.testing._comparison import _DTYPE_PRECISIONS  # 私有表,随 torch 版本变
        return {_dtype_name(k): (float(v[0]), float(v[1])) for k, v in _DTYPE_PRECISIONS.items()}, \
            f"torch.testing.assert_close 默认值(torch {torch.__version__})"
    except Exception:
        return dict(TORCH_DEFAULTS), "torch.testing.assert_close 文档默认值"


def _walk(obj, fn, memo):
    import torch
    if isinstance(obj, torch.Tensor):
        return fn(obj, memo)
    if isinstance(obj, list):
        return [_walk(x, fn, memo) for x in obj]
    if isinstance(obj, tuple):
        vals = [_walk(x, fn, memo) for x in obj]
        return type(obj)(*vals) if hasattr(obj, "_fields") else tuple(vals)
    if isinstance(obj, dict):
        return {k: _walk(v, fn, memo) for k, v in obj.items()}
    return obj


def _clone_one(t, memo):
    """复制一个张量:新分配整块 storage,再按原 size、stride、storage offset 取视图;同一 storage 的张量共用一份副本。"""
    import torch
    try:
        with torch.no_grad():
            st = t.untyped_storage()
            es = t.element_size()
            if st.nbytes() % es:
                raise ValueError("storage 字节数不是元素大小的整数倍")
            key = (st.data_ptr(), str(t.device))
            raw = memo.get(key)
            if raw is None:
                whole = t.as_strided((st.nbytes() // es,), (1,), 0)
                raw = whole.view(torch.uint8).clone()
                memo[key] = raw
            out = raw.view(t.dtype).as_strided(t.size(), t.stride(), t.storage_offset())
        if t.requires_grad:
            out.requires_grad_(True)
        return out
    except Exception as e:  # 稀疏、共轭视图等少见布局:退回普通 clone,并记下布局没保留
        memo.setdefault("_warn", []).append(f"有输入按普通 clone 复制,stride 与 offset 未保留({type(e).__name__}: {e})")
        return t.detach().clone().requires_grad_(t.requires_grad)


def clone_preserving(obj):
    memo = {}
    res = _walk(obj, _clone_one, memo)
    return res, memo.get("_warn", [])


class NanPoison:
    """在 kernel 调用期间让 torch.empty 一族返回 NaN 填充的浮点张量,没写满的输出会以 NaN 暴露。"""

    TARGETS = [("torch", "empty"), ("torch", "empty_like"), ("torch", "empty_strided"),
               ("Tensor", "new_empty"), ("Tensor", "new_empty_strided")]

    def __init__(self, enabled: bool):
        self.enabled, self.saved = enabled, []

    def __enter__(self):
        if not self.enabled:
            return self
        import torch
        for owner_name, attr in self.TARGETS:
            owner = torch if owner_name == "torch" else torch.Tensor
            orig = getattr(owner, attr, None)
            if orig is None:
                continue
            self.saved.append((owner, attr, orig, attr in vars(owner)))
            setattr(owner, attr, self._wrap(orig, torch))
        return self

    @staticmethod
    def _wrap(orig, torch):
        def poisoned(*a, **k):
            t = orig(*a, **k)
            if isinstance(t, torch.Tensor) and (t.is_floating_point() or t.is_complex()) and t.numel():
                with torch.no_grad():
                    t.fill_(float("nan"))
            return t
        return poisoned

    def __exit__(self, *exc):
        for owner, attr, orig, own in reversed(self.saved):
            if own:
                setattr(owner, attr, orig)
            else:
                delattr(owner, attr)
        self.saved = []
        return False


def _describe_tensor(t, torch) -> dict:
    d = {"shape": list(t.shape), "dtype": _dtype_name(t.dtype), "device": str(t.device)}
    if t.layout != torch.strided:
        d.update(layout=str(t.layout), aligned16=True, max_offset=0)
        return d
    if not t.is_contiguous():
        d["stride"] = list(t.stride())
    if t.storage_offset():
        d["storage_offset"] = t.storage_offset()
    d["aligned16"] = t.data_ptr() % 16 == 0
    reach = t.storage_offset() + (sum((s - 1) * abs(st) for s, st in zip(t.shape, t.stride())) if t.numel() else 0)
    d["max_offset"] = reach
    if (t.is_floating_point() or t.is_complex()) and t.numel():
        with torch.no_grad():
            d["nonfinite"] = int((~torch.isfinite(t)).sum().item())
    return d


def _flat_inputs(case):
    items = [(f"args[{i}]", v) for i, v in enumerate(case["args"])]
    items += [(f"kwargs[{k!r}]", v) for k, v in case["kwargs"].items()]
    return items


def describe_inputs(case, torch) -> list[dict]:
    out = []
    for path, v in _flat_inputs(case):
        if isinstance(v, torch.Tensor):
            out.append({"arg": path, **_describe_tensor(v, torch)})
        elif isinstance(v, (int, float, bool, str)) or v is None:
            out.append({"arg": path, "value": v})
        else:
            out.append({"arg": path, "type": type(v).__name__})
    return out


def _unravel(flat: int, shape) -> list[int]:
    idx = []
    for s in reversed(list(shape)):
        idx.append(flat % s if s else 0)
        flat = flat // s if s else 0
    return list(reversed(idx))


def compare_tensor(k, r, path, tol, equal_nan, allow_dtype_mismatch, torch) -> dict:
    rec = {"path": path, "kind": "tensor", "shape": list(r.shape), "dtype": _dtype_name(r.dtype)}
    reasons = []
    if tuple(k.shape) != tuple(r.shape):
        reasons.append(f"shape 不一致:kernel {list(k.shape)},参考 {list(r.shape)}")
    if k.dtype != r.dtype and not allow_dtype_mismatch:
        reasons.append(f"dtype 不一致:kernel {_dtype_name(k.dtype)},参考 {_dtype_name(r.dtype)}")
    if k.device != r.device:
        reasons.append(f"device 不一致:kernel {k.device},参考 {r.device}")
    if reasons:
        rec.update(passed=False, reason=";".join(reasons))
        return rec
    rtol, atol, src = tol(_dtype_name(k.dtype))
    rec.update(rtol=rtol, atol=atol, tol_source=src, numel=r.numel())
    if r.numel() == 0:
        rec.update(passed=True, max_abs_err=0.0, max_rel_err=0.0, mismatch=0)
        return rec
    with torch.no_grad():
        if k.dtype == torch.bool or not (k.is_floating_point() or k.is_complex()):
            kk, rr = k.to(torch.int64), r.to(torch.int64)
            bad = kk != rr
            n_bad = int(bad.sum().item())
            rec.update(max_abs_err=float((kk - rr).abs().max().item()), max_rel_err=None, mismatch=n_bad,
                       tol_source="整数与 bool 按精确比较", rtol=0.0, atol=0.0)
            if n_bad:
                fi = int(bad.reshape(-1).nonzero()[0].item())
                rec["worst"] = {"index": _unravel(fi, r.shape), "kernel": kk.reshape(-1)[fi].item(),
                                "reference": rr.reshape(-1)[fi].item()}
            rec["passed"] = n_bad == 0
            return rec
        if k.is_complex():
            k, r = torch.view_as_real(k), torch.view_as_real(r)
        cdt = torch.float64 if torch.float64 in (k.dtype, r.dtype) else torch.float32
        kf, rf = k.to(cdt), r.to(cdt)
        close = torch.isclose(kf, rf, rtol=rtol, atol=atol, equal_nan=equal_nan)
        n_bad = int((~close).sum().item())
        fin = torch.isfinite(kf) & torch.isfinite(rf)
        diff = (kf - rf).abs()
        zero = torch.zeros((), dtype=cdt, device=kf.device)
        diff_f = torch.where(fin, diff, zero)
        denom = torch.where(rf.abs() > 0, rf.abs(), torch.ones((), dtype=cdt, device=kf.device))
        rec.update(max_abs_err=float(diff_f.max().item()),
                   max_rel_err=float(torch.where(fin, diff / denom, zero).max().item()),
                   mismatch=n_bad, mismatch_frac=n_bad / kf.numel(),
                   nonfinite={"kernel_nan": int(torch.isnan(kf).sum().item()),
                              "ref_nan": int(torch.isnan(rf).sum().item()),
                              "kernel_inf": int(torch.isinf(kf).sum().item()),
                              "ref_inf": int(torch.isinf(rf).sum().item())})
        if n_bad:
            inf = torch.full((), float("inf"), dtype=cdt, device=kf.device)
            excess = torch.where(fin, diff - (atol + rtol * rf.abs()), inf)
            excess = torch.where(close, -inf, excess)
            fi = int(excess.reshape(-1).argmax().item())
            rec["worst"] = {"index": _unravel(fi, kf.shape), "kernel": kf.reshape(-1)[fi].item(),
                            "reference": rf.reshape(-1)[fi].item()}
    rec["passed"] = n_bad == 0
    return rec


def _default_stats(t, torch) -> dict:
    x = torch.view_as_real(t).float() if t.is_complex() else t.float()
    n = x.numel()
    fin = x[torch.isfinite(x)]
    return {"mean": fin.mean().item() if fin.numel() else float("nan"),
            "std": fin.std(unbiased=False).item() if fin.numel() else float("nan"),
            "zero_frac": (x == 0).sum().item() / n if n else 0.0,
            "nonfinite_frac": (n - fin.numel()) / n if n else 0.0}


def compare_stats(k, r, path, case, cli_stats_atol, torch) -> dict:
    rec = {"path": path, "kind": "tensor_stats", "shape": list(r.shape), "dtype": _dtype_name(r.dtype)}
    if tuple(k.shape) != tuple(r.shape) or k.dtype != r.dtype:
        rec.update(passed=False, reason=f"shape 或 dtype 不一致:kernel {list(k.shape)} {_dtype_name(k.dtype)},"
                                        f"参考 {list(r.shape)} {_dtype_name(r.dtype)}")
        return rec
    tol = case.get("stats_atol", cli_stats_atol)
    if tol is None:
        raise EnvError(f"用例 {case['name']!r} 用 stats 比较,须在用例里给 stats_atol 或用 --stats-atol")
    fn = case.get("stats_fn") or (lambda t: _default_stats(t, torch))
    with torch.no_grad():
        sk, sr = fn(k), fn(r)
    diffs, ok = {}, True
    for name in sr:
        t = float(tol[name]) if isinstance(tol, dict) else float(tol)
        d = abs(float(sk[name]) - float(sr[name]))
        diffs[name] = {"kernel": float(sk[name]), "reference": float(sr[name]), "abs_diff": d, "atol": t}
        ok &= d <= t
    rec.update(stats=diffs, passed=ok, tol_source="用例的 stats_atol" if "stats_atol" in case else "命令行 --stats-atol")
    return rec


def compare_values(k, r, path, ctx, torch) -> list[dict]:
    if isinstance(r, torch.Tensor) or isinstance(k, torch.Tensor):
        if not (isinstance(r, torch.Tensor) and isinstance(k, torch.Tensor)):
            return [{"path": path, "passed": False,
                     "reason": f"类型不一致:kernel {type(k).__name__},参考 {type(r).__name__}"}]
        if ctx["case"]["compare"] == "stats":
            return [compare_stats(k, r, path, ctx["case"], ctx["stats_atol"], torch)]
        return [compare_tensor(k, r, path, ctx["tol"], ctx["case"].get("equal_nan", True),
                               ctx["allow_dtype_mismatch"], torch)]
    if isinstance(r, (list, tuple)) and isinstance(k, (list, tuple)):
        if len(k) != len(r):
            return [{"path": path, "passed": False, "reason": f"长度不一致:kernel {len(k)},参考 {len(r)}"}]
        out = []
        for i, (a, b) in enumerate(zip(k, r)):
            out += compare_values(a, b, f"{path}[{i}]", ctx, torch)
        return out
    if isinstance(r, dict) and isinstance(k, dict):
        if set(k) != set(r):
            return [{"path": path, "passed": False, "reason": f"键不一致:kernel {sorted(map(str, k))},参考 {sorted(map(str, r))}"}]
        out = []
        for key in r:
            out += compare_values(k[key], r[key], f"{path}[{key!r}]", ctx, torch)
        return out
    if isinstance(r, float) and isinstance(k, (int, float)) and not isinstance(k, bool):
        rtol, atol, src = ctx["tol"]("float64")
        ok = math.isclose(k, r, rel_tol=rtol, abs_tol=atol) or (math.isnan(k) and math.isnan(r))
        return [{"path": path, "kind": "scalar", "kernel": k, "reference": r, "passed": ok, "tol_source": src}]
    try:
        ok = type(k) is type(r) and bool(k == r)
    except Exception:
        ok = False
    return [{"path": path, "kind": "value", "kernel": repr(k)[:200], "reference": repr(r)[:200], "passed": ok}]


def _call(fn, case, inputs):
    args, kwargs = inputs
    return fn(*args, **kwargs)


def _context_alive(torch) -> bool:
    try:
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def run_case(case, kernel_fn, reference_fn, opts, tol_factory, torch) -> tuple[dict, str]:
    """返回 (用例记录, 状态);状态为 pass / fail / ref_error / abort。"""
    rec = {"name": case["name"], "tags": case["tags"], "compare": case["compare"],
           "inputs": describe_inputs(case, torch), "results": [], "warnings": []}
    seed = int(case.get("seed", opts["seed"]))
    orig = (case["args"], case["kwargs"])
    ref_in, w1 = clone_preserving(orig)
    sync_in, w2 = clone_preserving(orig) if opts["sync_check"] != "off" else (None, [])
    rec["warnings"] += sorted(set(w1 + w2))

    try:
        torch.manual_seed(seed)
        ref_out = _call(reference_fn, case, ref_in)
        torch.cuda.synchronize()
    except Exception as e:
        rec.update(passed=False, error={"stage": "reference", "message": f"{type(e).__name__}: {e}",
                                        "traceback_tail": traceback.format_exc().strip().splitlines()[-6:]})
        return rec, "ref_error" if _context_alive(torch) else "abort"

    try:
        torch.manual_seed(seed)
        with NanPoison(opts["nan_poison"]):
            ker_out = _call(kernel_fn, case, orig)
        torch.cuda.synchronize()
    except Exception as e:
        rec.update(passed=False, error={"stage": "kernel", "message": f"{type(e).__name__}: {e}",
                                        "traceback_tail": traceback.format_exc().strip().splitlines()[-6:]})
        return rec, "fail" if _context_alive(torch) else "abort"

    tol = tol_factory(case)
    ctx = {"case": case, "tol": tol, "stats_atol": opts["stats_atol"], "allow_dtype_mismatch": opts["allow_dtype_mismatch"]}
    results = compare_values(ker_out, ref_out, "out", ctx, torch)
    in_place = ker_out is None and ref_out is None
    cmp_args = case.get("compare_args", "auto")
    arg_recs = []
    for (path, kv), (_, rv) in zip(_flat_inputs({"args": orig[0], "kwargs": orig[1]}),
                                   _flat_inputs({"args": ref_in[0], "kwargs": ref_in[1]})):
        # 只比较张量(含张量的列表或 dict 也逐个比);标量参数两边一样,不进结果
        arg_recs += [a for a in compare_values(kv, rv, path, ctx, torch)
                     if str(a.get("kind", "")).startswith("tensor") or not a.get("passed", False)]
    if in_place or cmp_args is True:
        results += arg_recs
    elif cmp_args == "auto":
        for a in arg_recs:
            if not a.get("passed", False):
                a["reason"] = (a.get("reason", "") + ";kernel 与参考调用后这个输入不同:有一方改写了输入").lstrip(";")
                results.append(a)
    rec["results"] = results
    errs = [r for r in results if isinstance(r.get("max_abs_err"), (int, float))]
    rec["max_abs_err"] = max((r["max_abs_err"] for r in errs), default=None)
    rec["max_rel_err"] = max((r["max_rel_err"] for r in errs if isinstance(r.get("max_rel_err"), (int, float))),
                             default=None)

    if opts["sync_check"] != "off":
        rec["sync_check"] = sync_check(kernel_fn, case, sync_in, seed, opts, torch)
    passed = bool(results) and all(r.get("passed", False) for r in results)
    if not results:
        rec["warnings"].append("kernel 与参考都返回 None 且没有张量参数可比:这个用例什么也没验证")
    if opts["sync_check"] == "error" and rec.get("sync_check", {}).get("syncs"):
        passed = False
        rec["warnings"].append("wrapper 里有同步点(--sync-check error)")
    if rec.get("sync_check", {}).get("error"):
        # 第 2 次调用 kernel 出错(非法访存、只能调一次的状态等)同样是 kernel 的失败;上下文坏了就停
        passed = False
        rec["warnings"].append("同步点检查时第 2 次调用 kernel 出错:" + rec["sync_check"]["error"])
        if not _context_alive(torch):
            rec["passed"] = False
            return rec, "abort"
    rec["passed"] = passed
    return rec, "pass" if passed else "fail"


def sync_check(kernel_fn, case, inputs, seed, opts, torch) -> dict:
    """第 2 次调用 kernel(编译与 autotune 已在第 1 次完成),在 sync debug mode 下记录 wrapper 里的同步点。"""
    get = getattr(torch.cuda, "get_sync_debug_mode", None)
    setm = getattr(torch.cuda, "set_sync_debug_mode", None)
    if get is None or setm is None:
        return {"mode": opts["sync_check"], "available": False, "reason": "所装 torch 没有 set_sync_debug_mode"}
    prev = get()
    sites = []
    try:
        torch.manual_seed(seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            setm("warn")
            try:
                with NanPoison(opts["nan_poison"]):
                    _call(kernel_fn, case, inputs)
            finally:
                setm(prev)
        torch.cuda.synchronize()
        for w in caught:
            msg = str(w.message)
            if "synchroniz" in msg.lower():
                sites.append(f"{os.path.basename(w.filename)}:{w.lineno} {msg[:160]}")
    except Exception as e:
        return {"mode": opts["sync_check"], "available": True, "error": f"{type(e).__name__}: {e}"}
    counted = collections.Counter(sites)
    return {"mode": opts["sync_check"], "available": True, "syncs": len(sites),
            "sites": [f"{s} ×{n}" for s, n in counted.most_common(5)]}


def require_cuda(torch):
    if not torch.cuda.is_available():
        raise EnvError(f"torch.cuda.is_available() 为 False(torch {torch.__version__},编译时 CUDA 版本 "
                       f"{torch.version.cuda});kernel 对拍要在 NVIDIA GPU 上跑")


def seed_all(seed: int, torch):
    random.seed(seed)
    torch.manual_seed(seed)  # 同时设所有 CUDA 设备
    try:
        import numpy
        numpy.random.seed(seed % 2**32)
    except Exception:
        pass


def verify(mod, names, opts, command) -> tuple[dict, int]:
    import torch

    require_cuda(torch)
    kernel_fn = getattr(mod, names["kernel"], None)
    reference_fn = getattr(mod, names["reference"], None)
    missing = [n for n, f in ((names["kernel"], kernel_fn), (names["reference"], reference_fn)) if not callable(f)]
    if missing:
        raise EnvError(f"模块里缺少可调用的 {missing}")
    gen = getattr(mod, names["cases"], None)
    single = False
    if gen is None and names["cases"] == "get_cases" and callable(getattr(mod, "get_inputs", None)):
        gen, single = mod.get_inputs, True
    if not callable(gen):
        raise EnvError(f"模块里缺少用例生成函数 {names['cases']}(或单用例的 get_inputs)")

    table, table_src = torch_tolerance_table(torch) if opts["preset"] == "torch" else (TRTLLM_PRESET, "TensorRT-LLM kernel-triton-writing 容差表")
    cli = (opts["rtol"], opts["atol"]) if opts["atol"] is not None else None
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    triton_ver = None
    try:
        import triton
        triton_ver = triton.__version__
    except Exception:
        pass
    res = {"schema": SCHEMA, "command": command, "verdict": None, "reasons": [], "warnings": [],
           "module": getattr(mod, "__file__", None), "names": names, "seed": opts["seed"],
           "tolerance": {"preset": opts["preset"], "table_source": table_src, "table": table,
                         "cli_override": {"rtol": opts["rtol"], "atol": opts["atol"]} if cli else None},
           "nan_poison": opts["nan_poison"], "sync_check": opts["sync_check"], "only": opts["only"] or None,
           "env": {"gpu": props.name, "cc": f"{props.major}.{props.minor}", "torch": torch.__version__,
                   "cuda": torch.version.cuda, "triton": triton_ver, "python": platform.python_version(),
                   "git": git_info()},
           "cases": []}

    def tol_factory(case):
        return lambda dtype_name: pick_tolerance(dtype_name, case, cli, opts["preset"], table, table_src)

    seed_all(opts["seed"], torch)
    try:
        if single:
            raw_iter = iter([list(gen())])
        else:
            try:
                takes_device = "device" in inspect.signature(gen).parameters
            except (TypeError, ValueError):
                takes_device = False
            raw_iter = iter(gen(device="cuda") if takes_device else gen())
    except Exception as e:
        raise EnvError(f"调用 {names['cases']} 失败:{type(e).__name__}: {e}") from e

    cov = {"cases": 0, "tags": collections.Counter(), "float_dtypes": set(), "noncontiguous_input": False,
           "storage_offset_input": False, "misaligned16_input": False, "nonfinite_input": False,
           "large_index_input": False, "signatures": set()}
    status_count = collections.Counter()
    i = -1
    while True:
        seed_all(opts["seed"] + i + 1, torch)  # 每个用例生成前按序号设种子:--only 跳过用例不改变其余用例的数据
        try:
            raw = next(raw_iter)
        except StopIteration:
            break
        except Exception as e:
            raise EnvError(f"生成第 {i + 1} 个用例时出错:{type(e).__name__}: {e}") from e
        i += 1
        case = normalize_case(raw, i)
        if opts["only"] and not any(s in case["name"] for s in opts["only"]):
            continue
        rec, status = run_case(case, kernel_fn, reference_fn, opts, tol_factory, torch)
        res["cases"].append(rec)
        status_count[status] += 1
        cov["cases"] += 1
        cov["tags"].update(case["tags"])
        sig = []
        for d in rec["inputs"]:
            if "shape" not in d:
                continue
            sig.append((tuple(d["shape"]), d["dtype"]))
            if d["dtype"].startswith(("float", "bfloat", "complex")):
                cov["float_dtypes"].add(d["dtype"])
            cov["noncontiguous_input"] |= "stride" in d
            cov["storage_offset_input"] |= "storage_offset" in d
            cov["misaligned16_input"] |= not d["aligned16"]
            cov["nonfinite_input"] |= d.get("nonfinite", 0) > 0
            cov["large_index_input"] |= d["max_offset"] > INT32_MAX
        cov["signatures"].add(tuple(sig))
        print(f"[verify_kernel] {case['name']}: {status}", file=sys.stderr)
        if status == "abort":
            res["reasons"].append(f"用例 {case['name']!r} 出错后 CUDA 上下文已不可用,后面的用例没有跑")
            break

    if i < 0:
        raise EnvError(f"{names['cases']} 没有产生任何用例")
    if cov["cases"] == 0:
        raise EnvError(f"--only {opts['only']} 没有匹配到任何用例")
    cov_out = {"cases": cov["cases"], "tags": dict(cov["tags"]), "float_dtypes": sorted(cov["float_dtypes"]),
               "noncontiguous_input": cov["noncontiguous_input"], "storage_offset_input": cov["storage_offset_input"],
               "misaligned16_input": cov["misaligned16_input"], "nonfinite_input": cov["nonfinite_input"],
               "large_index_input": cov["large_index_input"], "distinct_input_signatures": len(cov["signatures"])}
    unknown_tags = sorted(set(cov["tags"]) - KNOWN_TAGS)
    if unknown_tags:
        res["warnings"].append(f"标签 {unknown_tags} 不在约定词表 {sorted(KNOWN_TAGS)} 里,覆盖汇总不认")
    cov_w = coverage_warnings(cov_out, opts["expect_dtypes"])
    cov_out["gaps"] = cov_w
    res["coverage"] = cov_out
    if opts["only"]:
        res["warnings"].append("用 --only 只跑了部分用例:这是冒烟,不算完整的对拍门")
    if opts["nan_poison"]:
        res["warnings"].append("NaN 填充只覆盖经 torch.empty / empty_like / empty_strided / Tensor.new_empty(_strided) "
                               "分配的张量;C++ 扩展内部的分配不在内")

    n_fail = status_count["fail"] + status_count["abort"]
    res["summary"] = {"cases": cov["cases"], "passed": status_count["pass"], "failed": n_fail,
                      "reference_errors": status_count["ref_error"]}
    if status_count["ref_error"]:
        res["reasons"].append("有用例的 reference_fn 出错:契约本身有问题,先修参考实现或用例")
        res["verdict"] = "error"
        return res, EXIT_ENV
    if n_fail:
        res["reasons"].append(f"{n_fail} 个用例没有通过")
    if cov_w and opts["strict_coverage"]:
        res["reasons"].append("--strict-coverage:覆盖面有缺口")
    fail = bool(n_fail) or (bool(cov_w) and opts["strict_coverage"])
    res["verdict"] = "fail" if fail else "pass"
    return res, EXIT_FAIL if fail else EXIT_PASS


# ---------------------------------------------------------------- 命令行

EPILOG = """\
契约(模块里的约定名字,可用 --kernel/--reference/--cases 改名):
  kernel_fn(*args, **kwargs)      被测实现(wrapper 加 kernel)
  reference_fn(*args, **kwargs)   参考实现,与 kernel_fn 同签名
  get_cases()                     返回或 yield 若干用例;接受 device 参数时传入 "cuda"。
                                  没有 get_cases 而有 get_inputs() 时,把它当作单个用例
  一个用例是参数列表,或含下列键的 dict:
    args / kwargs                 位置参数与关键字参数(二者至少有一个)
    name, tags                    用例名;标签词表:tail boundary noncontig offset hard_values large_index
    atol, rtol                    本用例的容差,覆盖命令行与预设
    compare                       elementwise(默认)或 stats(RNG 类算子只比统计量)
    stats_atol, stats_fn          stats 比较的容差(数,或 {统计量名: 数});自定义统计函数 t -> dict
    equal_nan                     NaN 位置一致时算相等(默认 true)
    compare_args                  auto(默认:kernel 返回 None 时比较调用后的输入,否则只报被改写的输入)、true、false
    seed                          本用例的种子
  反向:在 kernel_fn 与 reference_fn 里各自算出梯度一并返回,脚本按输出比较。

每个用例的流程:
  复制输入(保留 stride、storage offset 与别名)→ 参考实现 → kernel(期间 torch.empty 一族返回 NaN 填充的张量)
  → 逐个输出比较 → 再调用 1 次 kernel,在 torch.cuda sync debug mode 下记录 wrapper 里的同步点(.item()、.cpu() 等);
  这次调用出错也判不通过。
  判定与 torch.isclose 相同:|kernel − ref| ≤ atol + rtol·|ref|;整数与 bool 精确比较。
  max_rel_err 的分母在参考值为 0 处取 1。

容差预设(--tol-preset,按输出 dtype 取,(rtol, atol)):
  torch   torch.testing.assert_close 的默认值:float16 (1e-3, 1e-5)、bfloat16 (1.6e-2, 1e-5)、
          float32 (1.3e-6, 1e-5)、float64 (1e-7, 1e-7);运行时以所装 torch 的表为准
  trtllm  TensorRT-LLM kernel-triton-writing 的表:float16 (1e-3, 1e-3)、bfloat16 (1e-2, 1e-2)、float32 (1e-5, 1e-5)
  预设里没有的浮点 dtype(例如 float8)按精确比较,需要时显式给容差。

退出码:0 全部通过;1 有用例不通过(或 --strict-coverage 下覆盖面有缺口);
        2 用法、契约或环境错误(导入失败、缺名字、没有 CUDA、reference_fn 出错、写 --out 失败)。

例子:
  python verify_kernel.py mykern.py --out .lab/tasks/<任务>/kernel/<短名>/verify/<日期>-v0.json
  python verify_kernel.py mykern.py --only small --sync-check error     # 冒烟
  python verify_kernel.py mykern.py --tol-preset trtllm --strict-coverage
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_kernel.py",
        description="kernel 对拍:逐用例比较 kernel 与参考实现,输出 JSON(每个输出的最大绝对与相对误差、是否通过)。"
                    "对拍通过才准测速。",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("module", help="路径 file.py 或模块名 pkg.mod")
    p.add_argument("--kernel", default="kernel_fn", help="被测实现的名字(默认 kernel_fn)")
    p.add_argument("--reference", default="reference_fn", help="参考实现的名字(默认 reference_fn)")
    p.add_argument("--cases", default="get_cases", help="用例生成函数的名字(默认 get_cases,缺省时退回 get_inputs)")
    p.add_argument("--seed", type=int, default=0, help="生成用例前、每次调用前设的种子(默认 0)")
    p.add_argument("--tol-preset", choices=["torch", "trtllm"], default="torch", help="按 dtype 的默认容差表(默认 torch)")
    p.add_argument("--atol", type=float, help="整体覆盖 atol(与 --rtol 同给)")
    p.add_argument("--rtol", type=float, help="整体覆盖 rtol(与 --atol 同给)")
    p.add_argument("--stats-atol", type=float, help="stats 比较的默认容差")
    p.add_argument("--allow-dtype-mismatch", action="store_true",
                   help="允许 kernel 与参考输出 dtype 不同(参考用更高精度算时),按 kernel 输出 dtype 取容差")
    p.add_argument("--no-nan-poison", action="store_true", help="关掉 kernel 调用期间 torch.empty 一族的 NaN 填充")
    p.add_argument("--sync-check", choices=["off", "warn", "error"], default="warn",
                   help="wrapper 里的同步点:warn 只记录(默认),error 记为不通过,off 不查")
    p.add_argument("--strict-coverage", action="store_true", help="覆盖面有缺口时判不通过")
    p.add_argument("--expect-dtypes", help="kernel 声称支持的浮点 dtype,逗号分隔(例如 float16,bfloat16);覆盖汇总逐个核对")
    p.add_argument("--only", action="append", default=[], metavar="SUBSTR",
                   help="只跑名字含 SUBSTR 的用例,可重复;用于冒烟")
    p.add_argument("--out", help="JSON 另写一份到这个路径")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    command = shlex.join([os.path.basename(sys.argv[0])] + (argv if argv is not None else sys.argv[1:]))
    try:
        if (args.atol is None) != (args.rtol is None):
            raise EnvError("--atol 与 --rtol 要么都给,要么都不给")
        try:
            import torch
        except ImportError as e:
            raise EnvError(f"导入 torch 失败({e});kernel 对拍要在装了 CUDA 版 torch 的环境里跑") from e
        require_cuda(torch)
        mod = load_module(args.module)
        opts = {"seed": args.seed, "preset": args.tol_preset, "atol": args.atol, "rtol": args.rtol,
                "stats_atol": args.stats_atol, "allow_dtype_mismatch": args.allow_dtype_mismatch,
                "nan_poison": not args.no_nan_poison, "sync_check": args.sync_check,
                "strict_coverage": args.strict_coverage, "only": args.only,
                "expect_dtypes": [x.strip() for x in args.expect_dtypes.split(",") if x.strip()] if args.expect_dtypes else None}
        names = {"kernel": args.kernel, "reference": args.reference, "cases": args.cases}
        res, code = verify(mod, names, opts, command)
    except EnvError as e:
        print(f"[verify_kernel] 用法、契约或环境错误:{e}", file=sys.stderr)
        _emit({"schema": SCHEMA, "verdict": "env_error", "reason": str(e), "command": command}, args.out)
        return EXIT_ENV
    except Exception as e:  # 生成用例、比较等处的意外异常:不让裸 traceback 以退出码 1 冒充「不通过」
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        _emit({"schema": SCHEMA, "verdict": "error", "reason": f"{type(e).__name__}: {e}",
               "traceback_tail": tb.strip().splitlines()[-8:], "command": command}, args.out)
        return EXIT_ENV
    if not _emit(res, args.out):
        return EXIT_ENV
    for c in res["cases"]:
        if c.get("passed"):
            continue
        detail = (c.get("error") or {}).get("message")
        bad = [r for r in c.get("results", []) if not r.get("passed", False)]
        if not detail and bad:
            detail = bad[0].get("reason") or f"{bad[0]['path']} max_abs_err={bad[0].get('max_abs_err')} mismatch={bad[0].get('mismatch')}"
        print(f"[verify_kernel] 不通过 {c['name']}:{detail}", file=sys.stderr)
    for g in res.get("coverage", {}).get("gaps", []):
        print(f"[verify_kernel] 覆盖缺口:{g}", file=sys.stderr)
    print(f"[verify_kernel] verdict={res['verdict']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
