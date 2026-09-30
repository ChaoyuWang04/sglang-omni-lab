#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 出处:vllm-project/vllm `.agents/skills/kernel-microbenchmark/benchmarks/graph_replay_benchmark.py`
# (本机镜像 fc2c801ad9,Apache-2.0)。沿用它的做法:多份轮换输入捕获进一个 CUDA graph,
# 用 CUDA event 计 graph 回放,launch 开销不进计时,多轮取中位数;先对拍再计时。
# 改动:
#   - 被测 callable、输入构造、参考实现、工作量公式都从命令行或 bench() 传入,不再写死 matmul
#   - 轮换份数改按 L2 容量算:同一份的 2 次复用之间流过 >= factor×L2 字节、至少 2 份;每次调用
#     流过的字节按张量逻辑大小估(不超过其 storage),不按整块 storage 算。
#     公式取自 flashinfer `flashinfer/testing/utils.py` 的 calculate_rotation_count(Apache-2.0,
#     那里 factor=5);默认 factor=2 与「至少 2 份」取自 cutlass
#     `media/docs/cpp/gemm_performance_measurement_methodology_guidelines.md`(BSD-3-Clause;
#     cutlass 原文是缓冲总量 >= 2×L2,这里按复用间距算,更严)
#   - 对拍:vllm 模板逐份 eager 对拍;这里 eager 对拍第 0 份,graph 回放后再核对第 0 次与最后一次调用
#   - 多个候选同进程按轮交替顺序计时(AB、BA 轮换)
#   - 意外异常(被测函数、参考实现、输入构造抛错)也输出 JSON,退出码 2
#   - 统计量加 p10/p90、样本数;计时期间用 nvidia-smi 采频率、温度、功耗与降频原因
#   - 去掉 pandas,只依赖 torch 与标准库;结果输出 JSON;退出码 0/1/2
"""单个 kernel 或算子的微基准模板:对拍 -> 冷 L2 轮换 -> CUDA graph(或逐次 event)计时 -> JSON。

复制到任意项目里改。只依赖 torch 与 Python 标准库。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback

EXIT_PASS, EXIT_FAIL, EXIT_ENV = 0, 1, 2
SCHEMA = "oss-kernel-microbenchmark/v1"

# NVML nvmlClocksEventReasons 的位定义(nvml.h);nvidia-smi 以十六进制输出
REASON_BITS = {
    0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap",
    0x8: "hw_slowdown", 0x10: "sync_boost", 0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting",
}
THROTTLE_MASK = 0x8 | 0x20 | 0x40 | 0x80  # 热降频与硬件降频;sw_power_cap 单独记,不算降频


class EnvError(Exception):
    """用法或环境错误,退出码 2。"""


# ---------------------------------------------------------------- 不依赖 torch 的部分

def rotation_sets(bytes_per_call: int, l2_bytes: int, factor: float, max_sets: int) -> tuple[int, bool]:
    """返回 (份数, 是否达到冷 L2 要求)。"""
    if bytes_per_call <= 0:
        return 1, False
    need = max(2, math.ceil(factor * l2_bytes / bytes_per_call) + 1)
    if need > max_sets:
        return max_sets, False
    return need, True


def quantile(xs_sorted: list[float], q: float) -> float:
    pos = q * (len(xs_sorted) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs_sorted[lo] + (xs_sorted[hi] - xs_sorted[lo]) * (pos - lo)


def summarize_ms(samples_ms: list[float]) -> dict:
    xs = sorted(samples_ms)
    us = lambda v: round(v * 1e3, 4)  # noqa: E731
    return {
        "median_us": us(statistics.median(xs)), "p10_us": us(quantile(xs, 0.1)),
        "p90_us": us(quantile(xs, 0.9)), "min_us": us(xs[0]), "max_us": us(xs[-1]),
        "mean_us": us(statistics.fmean(xs)),
        "stdev_us": us(statistics.stdev(xs)) if len(xs) > 1 else 0.0, "n_samples": len(xs),
    }


def decode_reasons(mask: int) -> list[str]:
    return [name for bit, name in REASON_BITS.items() if mask & bit]


def _num(s: str):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None  # "[N/A]"、"N/A"、空串


def parse_smi_line(line: str, n_fields: int):
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) != n_fields:
        return None
    sm, temp, power = (_num(p) for p in parts[:3])
    mask = None
    if n_fields > 3 and parts[3].lower().startswith("0x"):
        mask = int(parts[3], 16)
    return sm, temp, power, mask


class ClockSampler:
    """后台跑 `nvidia-smi --query-gpu=... -lms <周期>`,按到达时刻给样本打时间戳。"""

    BASE = ["clocks.sm", "temperature.gpu", "power.draw"]
    REASONS = ["clocks_event_reasons.active", "clocks_throttle_reasons.active"]  # 新旧驱动字段名

    def __init__(self, smi_id, period_ms: int):
        self.exe, self.id, self.period = shutil.which("nvidia-smi"), smi_id, period_ms
        self.samples: list[tuple] = []
        self.fields = self.proc = self.thread = None
        self.error = None if self.exe else "找不到 nvidia-smi"

    def _cmd(self, fields):
        cmd = [self.exe, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
        return cmd + (["-i", str(self.id)] if self.id is not None else [])

    def _once(self, fields):
        try:
            r = subprocess.run(self._cmd(fields), capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as e:
            return None, str(e)
        return (r.stdout.strip().splitlines() or [None])[0] if r.returncode == 0 else None, r.stderr.strip()

    def static_info(self) -> dict:
        if not self.exe or self.error:
            return {}
        line, _ = self._once(["driver_version", "clocks.max.sm", "power.limit"])
        if not line:
            return {}
        p = [x.strip() for x in line.split(",")]
        return {"driver": p[0], "max_sm_mhz": _num(p[1]), "power_limit_w": _num(p[2])}

    def _probe_fields(self):
        err = ""
        for extra in self.REASONS + [None]:
            fields = self.BASE + ([extra] if extra else [])
            line, err = self._once(fields)
            if line and parse_smi_line(line, len(fields)):
                return fields, ""
        return None, err

    def resolve(self):
        """定下查询字段与 -i;static_info() 与 start() 之前调用。"""
        if self.error or self.fields:
            return
        self.fields, err = self._probe_fields()
        if not self.fields and self.id is not None:
            # 按 UUID 或下标找不到卡时,只有一张卡就退回不指定 -i
            saved, self.id = self.id, None
            count, _ = self._once(["count"])
            if count and count.strip() == "1":
                self.fields, err = self._probe_fields()
            else:
                self.id = saved
        if not self.fields:
            self.error = f"nvidia-smi 查询失败:{err}"

    def start(self):
        self.resolve()
        if self.error:
            return
        try:
            self.proc = subprocess.Popen(self._cmd(self.fields) + ["-lms", str(self.period)],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError as e:  # 起不来就降级为「没采到」,不中断计时
            self.error = f"nvidia-smi 循环采样启动失败:{e}"
            return
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            s = parse_smi_line(line, len(self.fields))
            if s:
                self.samples.append((time.perf_counter(),) + s)

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=5)

    def window(self, t0: float, t1: float) -> dict:
        w = [s for s in self.samples if t0 <= s[0] <= t1]
        if not w:
            return {"available": False, "reason": self.error or "窗口内没有样本(计时太短或采样周期太长)"}
        sm = sorted(s[1] for s in w if s[1] is not None)
        temp = [s[2] for s in w if s[2] is not None]
        power = [s[3] for s in w if s[3] is not None]
        mask = 0
        for s in w:
            mask |= s[4] or 0
        got_mask = any(s[4] is not None for s in w)  # 字段不支持或读数全是 N/A 时,降频与否未知,不写成 False
        return {
            "available": True, "source": "nvidia-smi", "samples": len(w),
            "sm_mhz": {"min": sm[0], "median": statistics.median(sm), "max": sm[-1]} if sm else None,
            "temp_c": {"first": temp[0], "max": max(temp), "last": temp[-1]} if temp else None,
            "power_w_median": statistics.median(power) if power else None,
            "reasons_seen": decode_reasons(mask) if got_mask else "未采到(驱动不支持该字段或读数为 N/A)",
            "throttled": bool(mask & THROTTLE_MASK) if got_mask else None,
            "power_capped": bool(mask & 0x4) if got_mask else None,
        }


def load_obj(spec: str):
    """'pkg.mod:attr' 或 'path/to/file.py:attr'。"""
    mod_part, sep, attr = spec.rpartition(":")
    if not sep or not mod_part or not attr:
        raise EnvError(f"无法解析 {spec!r},要写成 模块:属性 或 文件.py:属性")
    try:
        if mod_part.endswith(".py"):
            path = os.path.abspath(mod_part)
            name = "_bench_user_" + os.path.splitext(os.path.basename(path))[0]
            mod = sys.modules.get(name)
            if mod is None:
                spec_ = importlib.util.spec_from_file_location(name, path)
                if spec_ is None:
                    raise EnvError(f"找不到文件 {path}")
                mod = importlib.util.module_from_spec(spec_)
                sys.modules[name] = mod
                spec_.loader.exec_module(mod)
        else:
            if os.getcwd() not in sys.path:
                sys.path.insert(0, os.getcwd())
            mod = importlib.import_module(mod_part)
        obj = mod
        for a in attr.split("."):
            obj = getattr(obj, a)
        return obj
    except EnvError:
        raise
    except Exception as e:  # 导入用户代码的任何失败都算用法或环境错误
        raise EnvError(f"加载 {spec!r} 失败:{type(e).__name__}: {e}") from e


def parse_kv(items: list[str]) -> dict:
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise EnvError(f"参数 {it!r} 要写成 键=值")
        for conv in (int, float):
            try:
                v = conv(v)
                break
            except ValueError:
                pass
        out[k] = v
    return out


def git_info() -> dict | None:
    def run(*a):
        r = subprocess.run(["git", *a], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    try:
        commit = run("rev-parse", "--short", "HEAD")
        if commit is None:
            return None
        top = run("rev-parse", "--show-toplevel") or ""
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
        return {"repo": os.path.basename(top), "commit": commit, "dirty": dirty}
    except (OSError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------- 依赖 torch 的部分

def _tensors(obj, torch):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _tensors(x, torch)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _tensors(x, torch)


def _call(fn, inputs):
    return fn(**inputs) if isinstance(inputs, dict) else fn(*inputs)


def _clone(obj, torch):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone(x, torch) for x in obj)
    if isinstance(obj, dict):
        return {k: _clone(v, torch) for k, v in obj.items()}
    return obj


def _storage_bytes(tensors) -> int:
    seen = {}
    for t in tensors:
        s = t.untyped_storage()
        seen[s.data_ptr()] = s.nbytes()
    return sum(seen.values())


def _touched_bytes(tensors) -> int:
    """每次调用经过 L2 的字节估计:各视图的逻辑大小之和,每块 storage 不超过其大小。
    只按 storage 算会把「大 storage 上的小视图」(偏移视图、切片)高估成整块,轮换份数就少了;
    只按逻辑大小算又会把 expand 出来的广播视图高估。"""
    per_storage, seen = {}, set()
    for t in tensors:
        key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype))
        if key in seen:
            continue
        seen.add(key)
        s = t.untyped_storage()
        cap, acc = per_storage.get(s.data_ptr(), (s.nbytes(), 0))
        per_storage[s.data_ptr()] = (cap, acc + t.numel() * t.element_size())
    return sum(min(cap, acc) for cap, acc in per_storage.values())


def _logical_bytes(tensors) -> int:
    seen = {}
    for t in tensors:
        seen[(t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype))] = t.numel() * t.element_size()
    return sum(seen.values())


def _describe(inputs, torch) -> list[dict]:
    items = inputs.items() if isinstance(inputs, dict) else enumerate(inputs)
    out = []
    for k, v in items:
        if isinstance(v, torch.Tensor):
            d = {"arg": k, "shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
            if not v.is_contiguous():
                d["stride"] = list(v.stride())
            if v.storage_offset():
                d["storage_offset"] = v.storage_offset()
            d["aligned16"] = v.data_ptr() % 16 == 0
            out.append(d)
        elif isinstance(v, (int, float, bool, str)) or v is None:
            out.append({"arg": k, "value": v})
        else:
            out.append({"arg": k, "type": type(v).__name__})
    return out


def _pick(inputs, keys, set0):
    if isinstance(inputs, dict):
        return {k: (set0[k] if k in keys else v) for k, v in inputs.items()}
    return type(inputs)(set0[i] if i in keys else v for i, v in enumerate(inputs))


def _check(torch, out, exp, atol, rtol) -> str | None:
    kw = {} if atol is None else {"atol": atol, "rtol": rtol}
    try:
        torch.testing.assert_close(out, exp, **kw)
        return None
    except (AssertionError, TypeError, ValueError, RuntimeError) as e:
        return f"{type(e).__name__}: {str(e)[:1500]}"


def bench(arms: dict, make_inputs, ref, work=None, *, params=None, hot=(), hot_reason=None,
          mode="graph", atol=None, rtol=None, graph_check=True, seed=0, warmup_calls=50,
          warmup_s=3.0, rounds=30, min_calls=1000, min_graph_calls=10, min_round_us=1000.0,
          l2_bytes=None, l2_factor=2.0, max_sets=2048, peak_gbps=None, peak_tflops=None,
          smi_period_ms=100, label=None, command=None) -> tuple[dict, int]:
    """返回 (结果 dict, 退出码)。环境问题抛 EnvError。"""
    import torch

    if not torch.cuda.is_available():
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        extra = ";检测到 MPS,但本模板不支持" if mps is not None and mps.is_available() else ""
        raise EnvError(f"torch.cuda.is_available() 为 False(torch {torch.__version__},"
                       f"编译时 CUDA 版本 {torch.version.cuda}){extra}。本模板只在 NVIDIA GPU 上计时,"
                       "Mac 上只能跑 --help 与这条检查")
    if (atol is None) != (rtol is None):
        raise EnvError("--atol 与 --rtol 要么都给,要么都不给(不给就用 torch.testing.assert_close 按 dtype 的默认值)")
    if hot and not hot_reason:
        raise EnvError("用了 --hot 就必须用 --hot-reason 写明为什么热 L2 是真实形态")
    params = params or {}
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    uuid = getattr(props, "uuid", None)
    smi_id = (str(uuid) if str(uuid).startswith("GPU-") else f"GPU-{uuid}") if uuid else dev
    sampler = ClockSampler(smi_id, smi_period_ms)
    sampler.resolve()
    l2 = int(l2_bytes) if l2_bytes else int(getattr(props, "L2_cache_size", 0))
    if l2 <= 0:
        raise EnvError("取不到 L2 容量:torch 的设备属性里没有 L2_cache_size,请用 --l2-bytes 给出")
    triton_ver = None
    try:
        import triton  # noqa: F401
        triton_ver = triton.__version__
    except Exception:
        pass
    res = {
        "schema": SCHEMA, "label": label, "command": command, "params": params, "verdict": None,
        "reasons": [], "warnings": [],
        "env": {"gpu": props.name, "cc": f"{props.major}.{props.minor}", "sm_count": props.multi_processor_count,
                "mem_gb": round(props.total_memory / 2**30, 1), "l2_bytes": l2,
                "l2_source": "--l2-bytes" if l2_bytes else "driver (L2_cache_size)",
                "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton_ver,
                "python": platform.python_version(), "git": git_info(), **sampler.static_info()},
    }

    # 1. 造第一份输入并对拍(先验正确再计时)
    torch.manual_seed(seed)
    set0 = make_inputs(**params)
    if not isinstance(set0, (tuple, list, dict)):
        set0 = (set0,)
    res["inputs"] = _describe(set0, torch)
    res["correctness"] = {"atol": atol, "rtol": rtol, "tolerance": "explicit" if atol is not None
                          else "torch.testing.assert_close 按 dtype 的默认值", "eager": {}, "graph": {}}
    exp0 = _clone(_call(ref, set0), torch)
    out_bytes = out_mem = 0  # 新分配输出:每次调用流过的字节;占用的显存
    for name, fn in arms.items():
        try:
            out = _call(fn, set0)
            torch.cuda.synchronize()
        except Exception as e:  # 候选在第 0 份输入上就抛错:算对拍不过
            res["correctness"]["eager"][name] = f"raised {type(e).__name__}: {str(e)[:1500]}"
            res["reasons"].append(f"{name}: eager 调用抛出异常")
            continue
        err = _check(torch, out, exp0, atol, rtol)
        res["correctness"]["eager"][name] = err or "ok"
        if err:
            res["reasons"].append(f"{name}: eager 输出与参考不一致")
        in_ptrs = {t.untyped_storage().data_ptr() for t in _tensors(set0, torch)}
        fresh = [t for t in _tensors(out, torch) if t.untyped_storage().data_ptr() not in in_ptrs]
        out_bytes = max(out_bytes, _touched_bytes(fresh))
        out_mem = max(out_mem, _storage_bytes(fresh))
    if res["reasons"]:
        res["verdict"] = "fail"
        return res, EXIT_FAIL

    # 2. 冷 L2:按每次调用流过的字节数定轮换份数;显存需求按 storage 算
    hot = set(hot)
    rotated = [t for k, v in (set0.items() if isinstance(set0, dict) else enumerate(set0))
               if k not in hot for t in _tensors(v, torch)]
    in_bytes, in_mem = _touched_bytes(rotated), _storage_bytes(rotated)
    per_call = in_bytes + (out_bytes if mode == "graph" else 0)
    n_sets, cold = rotation_sets(per_call, l2, l2_factor, max_sets)
    inner = max(1, math.ceil(min_graph_calls / n_sets)) if mode == "graph" else 1
    free, _ = torch.cuda.mem_get_info()
    need = (n_sets - 1) * in_mem + (n_sets * inner * out_mem * len(arms) if mode == "graph" else 0)
    if need > 0.9 * free:  # 留 10% 给 graph 内存池与临时张量;实现上的余量,不是测量阈值
        if per_call >= 5 * l2:  # flashinfer:单份 >= 5×L2 时不轮换
            want = n_sets
            n_sets, cold, inner = 1, True, max(1, min_graph_calls)
            res["warnings"].append(f"显存放不下 {want} 份输入;单份已 >= 5×L2,足够冷,不再轮换")
        else:
            raise EnvError(f"显存不够放 {n_sets} 份输入(约 {need / 2**30:.1f} GB,空闲 {free / 2**30:.1f} GB);"
                           "缩小形状、调小 --l2-factor、让 INPUTS 别为小视图分配大 storage,"
                           "或把确实热的参数用 --hot 标出")
    if not cold:
        if per_call <= 0:
            res["warnings"].append("没有可轮换的数据(参数全标了 --hot,fn 也不新分配输出),结果是热 L2")
        else:
            res["warnings"].append(f"轮换份数触到上限 {max_sets},同一份的 2 次复用之间流过的数据"
                                   f"不足 {l2_factor}×L2,结果偏热")
    sets = [set0]
    for i in range(1, n_sets):
        torch.manual_seed(seed + i)
        s = make_inputs(**params)
        sets.append(_pick(s if isinstance(s, (tuple, list, dict)) else (s,), hot, set0))
    torch.cuda.synchronize()

    # 3. 每方建计时器:graph 模式捕获 n_sets×inner 次调用;events 模式逐次 event
    runners = {}
    for name, fn in arms.items():
        side = torch.cuda.Stream()  # 旁路 stream 上 eager 预热,触发 JIT、autotune 与库的惰性初始化
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for i in range(warmup_calls):
                _call(fn, sets[i % n_sets])
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        if mode == "graph":
            g, outs = torch.cuda.CUDAGraph(), []
            try:
                with torch.cuda.graph(g):
                    for i in range(n_sets * inner):
                        outs.append(_call(fn, sets[i % n_sets]))
                torch.cuda.synchronize()
            except Exception as e:
                raise EnvError(f"{name} 不能被 CUDA graph 捕获({type(e).__name__}: {e});改用 --mode events") from e
            runners[name] = {"graph": g, "outs": outs}
        else:
            runners[name] = {"fn": fn}

    def launch(name, n):  # 下发一轮,返回 [(start, end, 该段调用次数)]
        r = runners[name]
        if mode == "graph":
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(n):
                r["graph"].replay()
            e.record()
            return [(s, e, n * n_sets * inner)]
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), 1) for _ in range(n)]
        k0 = r.get("k", 0)  # 事件先建好,调用之间只剩 record 与下发
        for j, (s, e, _) in enumerate(evs):
            s.record()
            _call(r["fn"], sets[(k0 + j) % n_sets])
            e.record()
        r["k"] = k0 + n
        return evs

    # 4. graph 回放后的输出再对拍一次:错误结果会被冻进每次回放
    if mode == "graph" and graph_check:
        last = n_sets * inner - 1  # 最后一次调用的输出不会再被覆盖,总能核对
        exps = {}
        for name in arms:
            outs = runners[name]["outs"]
            ptrs = lambda o: {t.untyped_storage().data_ptr() for t in _tensors(o, torch)}  # noqa: E731
            later = set().union(*(ptrs(o) for o in outs[1:])) if len(outs) > 1 else set()
            calls = [last] if (ptrs(outs[0]) & later) else sorted({0, last})  # 第 0 次的输出被后面覆盖就不核对它
            for i in calls:
                if i % n_sets not in exps:
                    exps[i % n_sets] = _clone(_call(ref, sets[i % n_sets]), torch)
            runners[name]["graph"].replay()
            torch.cuda.synchronize()
            errs = [f"call {i}: {err}" for i in calls
                    if (err := _check(torch, outs[i], exps[i % n_sets], atol, rtol))]
            res["correctness"]["graph"][name] = "; ".join(errs) or "ok"
            if errs:
                res["reasons"].append(f"{name}: graph 回放输出与参考不一致")
        if res["reasons"]:
            res["verdict"] = "fail"
            return res, EXIT_FAIL
    elif mode == "graph":
        res["warnings"].append("跳过了 graph 回放输出的对拍(--skip-graph-check)")

    # 5. warmup(按时间,等频率稳定),再定每轮的量
    names = list(arms)
    pending = {n: [] for n in names}
    per_round = {}
    sampler.start()
    try:
        t_w0 = time.perf_counter()
        k = 1  # 每批加倍到至少 10 ms,让 GPU 在 warmup 里持续忙,频率按满载爬升
        while True:
            t_b = time.perf_counter()
            for name in names:
                launch(name, k if mode == "graph" else k * n_sets)
            torch.cuda.synchronize()
            if time.perf_counter() - t_b < 0.01:
                k *= 2
            if time.perf_counter() - t_w0 >= warmup_s:
                break
        t_w1 = time.perf_counter()
        for name in names:
            if mode == "graph":
                (s, e, _), = launch(name, 3)
                torch.cuda.synchronize()
                replay_us = s.elapsed_time(e) * 1e3 / 3
                per_round[name] = max(1, math.ceil(min_calls / (rounds * n_sets * inner)),
                                      math.ceil(min_round_us / max(replay_us, 1e-3)))
            else:
                per_round[name] = max(n_sets, math.ceil(min_calls / rounds))

        # 6. 计时:各方按轮交替顺序(ABAB 与 BABA 轮换),计时区内不同步。
        #    先不计时地排一轮让 GPU 忙着,第一轮的 start event 就不会把 launch 延迟算进去(vllm 模板的做法)
        for name in names:
            launch(name, per_round[name])
        t_m0 = time.perf_counter()
        for r in range(rounds):
            for name in (names if r % 2 == 0 else names[::-1]):
                pending[name].append(launch(name, per_round[name]))
        torch.cuda.synchronize()
        t_m1 = time.perf_counter()
    finally:
        sampler.stop()

    res["method"] = {
        "mode": mode, "timer": "torch.cuda.Event", "interleave": "同进程按轮交替顺序",
        "cold_l2": cold, "hot_args": sorted(map(str, hot)), "hot_reason": hot_reason,
        "sets": n_sets, "bytes_per_call": per_call, "l2_factor": l2_factor,
        "touched_between_reuse_over_l2": round((n_sets - 1) * per_call / l2, 2) if n_sets > 1 else None,
        "fresh_outputs_distinct": (mode == "graph") if out_bytes else "fn 不新分配输出", "calls_per_graph": n_sets * inner if mode == "graph" else None,
        "rounds": rounds, "per_round": per_round, "warmup_calls": warmup_calls, "warmup_s": round(t_w1 - t_w0, 2),
        "sample_unit": "每轮内每次调用的平均" if mode == "graph" else "单次调用",
    }
    res["clocks"] = sampler.window(t_m0, t_m1)
    if not res["clocks"].get("available") and sampler.samples:  # 计时比采样周期短:两边各外扩一个周期
        pad = smi_period_ms / 1e3
        res["clocks"] = sampler.window(t_m0 - pad, t_m1 + pad)
        if res["clocks"].get("available"):
            res["clocks"]["note"] = "计时短于采样周期,窗口两边各外扩一个周期"
    wu = sampler.window(t_w0, t_w1)
    if wu.get("available") and wu.get("sm_mhz"):
        first = [s[1] for s in sampler.samples if t_w0 <= s[0] <= t_w1 and s[1] is not None]
        res["clocks"]["warmup_sm_mhz_first_last"] = [first[0], first[-1]]
    if not res["clocks"].get("available"):
        res["warnings"].append("没采到频率与温度:" + res["clocks"]["reason"])
    elif res["clocks"]["throttled"]:
        res["warnings"].append("计时期间出现热降频或硬件降频:" + ",".join(res["clocks"]["reasons_seen"]))

    # 7. 统计与吞吐
    try:
        w = dict(_call(work, set0)) if work else None
    except Exception as e:
        raise EnvError(f"work 函数出错:{type(e).__name__}: {e}") from e
    if w is None:
        all_in = list(_tensors(set0, torch))
        w = {"bytes": _logical_bytes(all_in) + out_bytes, "flops": None,
             "source": "默认:每个输入与新分配输出的元素各读写一次;请用 --work 按数学定义给"}
    else:
        w.setdefault("source", "--work")
    res["work"] = w
    res["arms"] = {}
    for name in names:
        samples = [s.elapsed_time(e) / n for batch in pending[name] for (s, e, n) in batch]
        st = summarize_ms(samples)
        st["n_calls"] = sum(n for batch in pending[name] for (_, _, n) in batch)
        sec = st["median_us"] * 1e-6
        if w.get("bytes"):
            st["gbps"] = round(w["bytes"] / sec / 1e9, 2)
            if peak_gbps:
                st["frac_peak_bw"] = round(st["gbps"] / peak_gbps, 3)
        if w.get("flops"):
            st["tflops"] = round(w["flops"] / sec / 1e12, 3)
            if peak_tflops:
                st["frac_peak_flops"] = round(st["tflops"] / peak_tflops, 3)
        for key in ("frac_peak_bw", "frac_peak_flops"):
            if st.get(key, 0) > 1.0:
                res["reasons"].append(f"{name}: {key}={st[key]} 超过实测峰值,先查单位、公式、被跳过的工作与缓存")
        res["arms"][name] = st
    base = names[0]
    res["speedup_vs_" + base] = {n: round(res["arms"][base]["median_us"] / res["arms"][n]["median_us"], 4)
                                 for n in names[1:]}
    res["verdict"] = "suspect" if res["reasons"] else "pass"
    return res, (EXIT_FAIL if res["reasons"] else EXIT_PASS)


# ---------------------------------------------------------------- 内置示例(--demo)

def demo_inputs(n=1 << 24, dtype="bfloat16"):
    import torch
    x = torch.rand(n, device="cuda").mul_(2).sub_(1).to(getattr(torch, dtype))  # 均匀 [-1, 1](cutlass 的填充)
    return (x, torch.empty_like(x))


def demo_mul(x, out):
    import torch
    return torch.mul(x, 2, out=out)


def demo_add(x, out):
    import torch
    return torch.add(x, x, out=out)


def demo_ref(x, out):
    return (x.float() * 2).to(x.dtype)


def demo_work(x, out):
    return {"bytes": x.numel() * x.element_size() + out.numel() * out.element_size(), "flops": x.numel()}


# ---------------------------------------------------------------- 命令行

EPILOG = """\
契约:
  INPUTS(**params) 返回 tuple(按位置传)或 dict(按关键字传)的一份输入;每次调用都要能造出新张量
  FN(*inputs) 与 REF(*inputs) 返回要比对的张量(或张量的 tuple);原地写的 kernel 返回被写的张量
  WORK(*inputs) 返回 {"flops": ..., "bytes": ...},按操作的数学定义算理论最小工作量
  REF 不许写输入;原地累加的 FN 多次调用后结果会变,用 --skip-graph-check 并自己核对

例子:
  python bench_template.py --demo --param n=16777216 --out /tmp/demo.json
  python bench_template.py --fn old=mykern.py:run_v1 --fn new=mykern.py:run_v2 \\
      --inputs mykern.py:make_inputs --ref mykern.py:reference --work mykern.py:workload \\
      --param tokens=4096 --param hidden=7168 --peak-gbps <klab probe 的 measured_copy_gbps>
  A/A 噪声底:同一个实现写 2 次 --fn a1=m.py:f --fn a2=m.py:f

退出码:0 通过;1 对拍不过(含候选在第 0 份输入上抛错)或结果可疑(超过实测峰值);
  2 用法或环境错误(含没有 CUDA),以及参考实现、输入构造、计时途中的意外异常(JSON 的 verdict 为 error)
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_template.py", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG,
        description="单个 kernel 或算子的微基准:先对拍,再在冷 L2 下用 CUDA graph + CUDA event 计时,输出 JSON。")
    a = p.add_argument
    a("--fn", action="append", default=[], metavar="[NAME=]MOD:ATTR", help="被测 callable,可重复;第一个是比较基准")
    a("--inputs", metavar="MOD:ATTR", help="输入构造函数;模块写 pkg.mod 或 path/file.py")
    a("--ref", metavar="MOD:ATTR", help="参考实现(必填,先验正确再计时)")
    a("--work", metavar="MOD:ATTR", help="理论工作量函数;不给就按输入加输出各读写一次估字节,并在 JSON 里标明")
    a("--param", action="append", default=[], metavar="K=V", help="传给 INPUTS 的关键字参数,可重复")
    a("--demo", action="store_true", help="跑内置示例(bf16 逐元素 x*2 对 x+x)")
    a("--mode", choices=["graph", "events"], default="graph", help="graph:捕获后计回放(默认);events:逐次 event,给不能捕获的 fn")
    a("--atol", type=float)
    a("--rtol", type=float)
    a("--skip-graph-check", action="store_true", help="不核对 graph 回放后的输出(原地累加类 fn 用)")
    a("--hot", action="append", default=[], metavar="IDX|KEY", help="不轮换的参数(下标或关键字),必须配 --hot-reason")
    a("--hot-reason", help="为什么这些参数在真实形态里就是 L2 热的")
    a("--seed", type=int, default=0)
    a("--warmup-calls", type=int, default=50, help="eager 预热次数(默认 50)")
    a("--warmup-s", type=float, default=3.0, help="计时前回放预热秒数(默认 3:固定功耗下频率约 3 s 后稳定)")
    a("--rounds", type=int, default=30, help="每方样本轮数(默认 30)")
    a("--min-calls", type=int, default=1000, help="每方至少计时的调用次数(默认 1000)")
    a("--min-graph-calls", type=int, default=10, help="每个 graph 至少捕获的调用数(默认 10)")
    a("--min-round-us", type=float, default=1000.0, help="每轮至少多长,摊薄约 0.5 µs 的 event 分辨率(默认 1000)")
    a("--l2-bytes", type=int, help="L2 容量;不给就用驱动报告的 L2_cache_size")
    a("--l2-factor", type=float, default=2.0, help="同一份输入的 2 次复用之间流过的数据 >= 几倍 L2"
                                                   "(默认 2;5 为保守值)")
    a("--max-sets", type=int, default=2048, help="轮换份数上限,防止 graph 过大;触顶时 JSON 标 cold_l2=false")
    a("--peak-gbps", type=float, help="实测带宽峰值(klab probe 的 measured_copy_gbps)")
    a("--peak-tflops", type=float, help="同 dtype 的实测算力峰值")
    a("--smi-period-ms", type=int, default=100, help="nvidia-smi 采样周期")
    a("--label", help="写进 JSON 的短名")
    a("--out", help="JSON 另存到这个路径;不论是否给,JSON 都打印到 stdout")
    return p


def _emit(res: dict, out_path: str | None):
    text = json.dumps(res, ensure_ascii=False, indent=2)
    print(text)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    command = shlex.join([os.path.basename(sys.argv[0])] + (argv if argv is not None else sys.argv[1:]))
    try:
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise EnvError(f"导入 torch 失败({e});本模板需要装了 CUDA 版 torch 的环境") from e
        if args.demo:
            arms = {"mul": demo_mul, "add": demo_add}
            make_inputs, ref, work = demo_inputs, demo_ref, demo_work
        else:
            if not (args.fn and args.inputs and args.ref):
                raise EnvError("需要 --fn、--inputs、--ref(先验正确再计时);或用 --demo")
            arms = {}
            for spec in args.fn:
                name, sep, target = spec.partition("=")
                if not sep:
                    name, target = spec.rpartition(":")[2], spec
                if name in arms:
                    raise EnvError(f"候选名 {name!r} 重复;用 NAME=MOD:ATTR 起不同的名字")
                arms[name] = load_obj(target)
            make_inputs, ref = load_obj(args.inputs), load_obj(args.ref)
            work = load_obj(args.work) if args.work else None
        hot = [int(h) if h.isdigit() else h for h in args.hot]
        res, code = bench(
            arms, make_inputs, ref, work, params=parse_kv(args.param), hot=hot, hot_reason=args.hot_reason,
            mode=args.mode, atol=args.atol, rtol=args.rtol, graph_check=not args.skip_graph_check,
            seed=args.seed, warmup_calls=args.warmup_calls, warmup_s=args.warmup_s, rounds=args.rounds,
            min_calls=args.min_calls, min_graph_calls=args.min_graph_calls, min_round_us=args.min_round_us,
            l2_bytes=args.l2_bytes, l2_factor=args.l2_factor, max_sets=args.max_sets,
            peak_gbps=args.peak_gbps, peak_tflops=args.peak_tflops, smi_period_ms=args.smi_period_ms,
            label=args.label, command=command)
    except EnvError as e:
        print(f"[bench_template] 环境或用法错误:{e}", file=sys.stderr)
        _emit({"schema": SCHEMA, "verdict": "env_error", "reason": str(e), "command": command}, args.out)
        return EXIT_ENV
    except Exception as e:  # 参考实现、输入构造、捕获后回放等处的意外异常:不让裸 traceback 以退出码 1 冒充「不通过」
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        _emit({"schema": SCHEMA, "verdict": "error", "reason": f"{type(e).__name__}: {e}",
               "traceback_tail": tb.strip().splitlines()[-8:], "command": command}, args.out)
        return EXIT_ENV
    _emit(res, args.out)
    for name, st in res.get("arms", {}).items():
        print(f"[bench_template] {name}: median {st['median_us']} µs (p10 {st['p10_us']}, p90 {st['p90_us']}, "
              f"n={st['n_samples']}) {st.get('gbps', '-')} GB/s {st.get('tflops', '-')} TFLOPS", file=sys.stderr)
    for wmsg in res.get("warnings", []):
        print(f"[bench_template] 注意:{wmsg}", file=sys.stderr)
    print(f"[bench_template] verdict={res['verdict']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
