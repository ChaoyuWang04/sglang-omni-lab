#!/usr/bin/env python3
"""按级别落盘函数或 custom op 每次调用的输入,找出崩溃、非法访存发生在哪一次调用。

库用法(级别等配置在包装时读取,环境变量要在 import 被测代码之前设好):

    import sys; sys.path.insert(0, "<本目录>")
    from dump_inputs import dump_inputs, wrap, patch, unpatch, load_dump

    @dump_inputs()                        # 包自己能改的函数
    def fused_op(x, w, eps=1e-6): ...

    op = wrap(torch.ops.mylib.fused_op)   # 包 custom op 或任意可调用对象
    orig = patch(some_module, "kernel")   # 不改源码,替换模块或类上的属性
    unpatch(some_module, "kernel", orig)

    args, kwargs = load_dump("<dir>/tensors/rank0.pid123/000042_fused_op")  # level 10 的输入读回

命令行只做离线的事,见 --help。
"""

import argparse
import dataclasses
import fnmatch
import functools
import glob
import inspect
import json
import math
import os
import re
import sys
import tempfile
import threading
import time

FORMAT_VERSION = 1

ENV_LEVEL = "OSS_DUMP_LEVEL"
ENV_DIR = "OSS_DUMP_DIR"
ENV_SYNC = "OSS_DUMP_SYNC"
ENV_INCLUDE = "OSS_DUMP_INCLUDE"
ENV_EXCLUDE = "OSS_DUMP_EXCLUDE"
ENV_MAX_MB = "OSS_DUMP_MAX_MB"

DEFAULT_DIR = "oss_dump"
DEFAULT_MAX_MB = 4096
MAX_ITEMS = 16
MAX_DEPTH = 3
MAX_STR = 200
ENV_SNAPSHOT_KEYS = (
    "CUDA_LAUNCH_BLOCKING",
    "PYTORCH_NO_CUDA_MEMORY_CACHING",
    "CUDA_VISIBLE_DEVICES",
    "TORCH_COMPILE_DISABLE",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
)
REPR_KEY = "__oss_dump_repr__"

HELP_EPILOG = """\
级别(OSS_DUMP_LEVEL,取不超过所给值的最高一档):
  0    关闭(默认)。包装器直接返回原函数,没有开销
  1    只记函数名、序号、时间
  3    加每个参数与返回值的形状、dtype、设备、stride、连续性、storage_offset、指针对齐
  5    加数值统计:min、max、mean、nan_count、inf_count(整数只记 min、max)。会同步 GPU
  10   执行前把输入张量拷到 CPU 并 torch.save,返回后再存输出。会同步 GPU,写盘量大

其他环境变量:
  OSS_DUMP_DIR       输出目录,默认 ./oss_dump
  OSS_DUMP_SYNC=1    每次调用返回前 torch.cuda.synchronize(),让异步错误落在真正出错的那次调用上;
                     与 CUDA_LAUNCH_BLOCKING=1 一样会改变时序
  OSS_DUMP_INCLUDE   只对名字匹配的函数做 level 10 落盘,逗号分隔的 shell 通配,例如 'mylib.*,*attn*'
  OSS_DUMP_EXCLUDE   不做 level 10 落盘的函数名通配
  OSS_DUMP_MAX_MB    每个进程 level 10 张量落盘的总上限,默认 4096

输出布局:
  <dir>/calls.rank<r>.pid<p>.jsonl          每进程一个文件;事件 header、call、return、exception
  <dir>/tensors/rank<r>.pid<p>/<序号>_<函数名>/inputs.pt、outputs.pt
  rank 取 torch.distributed(已初始化时),否则取环境变量 RANK、LOCAL_RANK,都没有记 na。
  call 记录在执行前写出并 flush:进程崩了,最后一条「有 call 没有 return」就是出事的边界。

自动跳过:
  CUDA Graph 捕获期间不做统计与落盘(记录里 capturing=true);torch.compile 追踪期间不记录,
  要记录编译区域里的调用,用 TORCH_COMPILE_DISABLE=1 在 eager 下复现。

退出码:
  summarize  0 = 所有调用都有返回;1 = 找到有输入无输出或抛异常的调用;2 = 目录不存在或没有记录文件
  selftest   0 = 全部检查通过;1 = 有检查失败;2 = 环境错误(例如没有 torch、要求 cuda 但不可用)
"""


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def _parse_level(raw):
    try:
        value = int(str(raw).strip() or "0")
    except ValueError:
        raise ValueError(f"{ENV_LEVEL} 必须是整数(0、1、3、5、10),收到 {raw!r}") from None
    return value


def _split_patterns(raw):
    return tuple(p.strip() for p in (raw or "").split(",") if p.strip())


@dataclasses.dataclass(frozen=True)
class Config:
    level: int = 0
    out_dir: str = DEFAULT_DIR
    sync: bool = False
    include: tuple = ()
    exclude: tuple = ()
    max_mb: float = DEFAULT_MAX_MB

    @classmethod
    def from_env(cls):
        env = os.environ
        return cls(
            level=_parse_level(env.get(ENV_LEVEL, "0")),
            out_dir=env.get(ENV_DIR, DEFAULT_DIR) or DEFAULT_DIR,
            sync=env.get(ENV_SYNC, "0").strip() not in ("", "0", "false", "False"),
            include=_split_patterns(env.get(ENV_INCLUDE)),
            exclude=_split_patterns(env.get(ENV_EXCLUDE)),
            max_mb=float(env.get(ENV_MAX_MB, DEFAULT_MAX_MB)),
        )

    def dump_selected(self, name):
        if self.include and not any(fnmatch.fnmatch(name, p) for p in self.include):
            return False
        return not any(fnmatch.fnmatch(name, p) for p in self.exclude)


# ---------------------------------------------------------------------------
# 与 torch 相关的探测;torch 只在已被导入时使用,不主动导入
# ---------------------------------------------------------------------------


def _torch():
    return sys.modules.get("torch")


def _is_tensor(torch, value):
    return torch is not None and isinstance(value, torch.Tensor)


def _is_capturing(torch):
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _compiling_probe():
    """包装时取 torch.compiler.is_compiling;Dynamo 把它当常量处理,追踪期间直通不会引出 graph break。"""
    torch = _torch()
    if torch is None:
        return None
    return getattr(getattr(torch, "compiler", None), "is_compiling", None)


def _cuda_ready(torch):
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    except Exception:
        return False


def _rank():
    torch = _torch()
    if torch is not None:
        try:
            dist = torch.distributed
            if dist.is_available() and dist.is_initialized():
                return str(dist.get_rank())
        except Exception:
            pass
    for key in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(key)
        if value not in (None, ""):
            return re.sub(r"[^0-9A-Za-z_-]", "_", value)
    return "na"


def _is_device_error(exc):
    """CUDA 粘滞错误要继续往外抛;统计本身不支持(比如某些 dtype)只记下来。"""
    torch = _torch()
    oom = getattr(torch, "OutOfMemoryError", None) if torch is not None else None
    if oom is not None and isinstance(exc, oom):
        return False
    if type(exc).__name__ == "AcceleratorError":
        return True
    text = str(exc)
    return any(s in text for s in ("CUDA error", "cudaError", "HIP error", "illegal memory access"))


def _num(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _qualname(fn):
    for attr in ("_qualname", "_qualified_op_name"):
        value = getattr(fn, attr, None)
        if isinstance(value, str) and value:
            return value
    qual = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    if qual:
        module = getattr(fn, "__module__", None)
        return f"{module}.{qual}" if module else qual
    return str(fn)


# ---------------------------------------------------------------------------
# 描述参数(level 3 与 5)
# ---------------------------------------------------------------------------


def _ptr_align(t):
    try:
        if t.device.type == "meta":
            return None
        ptr = t.data_ptr()
    except Exception:
        return None
    if not ptr:
        return None
    return min(ptr & -ptr, 256)


def _stats(torch, t):
    if t.numel() == 0:
        return {"numel": 0}
    if t.device.type == "meta":
        return {"skipped": "meta tensor"}
    if t.is_complex():
        return {"skipped": "complex dtype"}
    x = t.detach()
    out = {"numel": int(t.numel())}
    if t.dtype == torch.bool:
        out["true_count"] = int(x.sum().item())
        return out
    if not t.is_floating_point():
        out["min"] = int(x.min().item())
        out["max"] = int(x.max().item())
        return out
    try:
        out["nan_count"] = int(torch.isnan(x).sum().item())
        out["inf_count"] = int(torch.isinf(x).sum().item())
        out["min"] = _num(float(x.min().item()))
        out["max"] = _num(float(x.max().item()))
        out["mean"] = _num(float(x.mean(dtype=torch.float32).item()))
    except Exception as exc:
        if _is_device_error(exc):
            raise
        y = x.float()  # 例如 float8 上部分归约不支持,转 float32 再算
        out["nan_count"] = int(torch.isnan(y).sum().item())
        out["inf_count"] = int(torch.isinf(y).sum().item())
        out["min"] = _num(float(y.min().item()))
        out["max"] = _num(float(y.max().item()))
        out["mean"] = _num(float(y.mean().item()))
    return out


def _describe_tensor(torch, t, level, capturing):
    info = {
        "type": "tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "requires_grad": bool(t.requires_grad),
    }
    if type(t) is not torch.Tensor:
        info["class"] = type(t).__name__
    for key, getter in (
        ("stride", lambda: list(t.stride())),
        ("contiguous", lambda: bool(t.is_contiguous())),
        ("storage_offset", lambda: int(t.storage_offset())),
    ):
        try:
            info[key] = getter()
        except Exception as exc:
            info[key] = f"<{type(exc).__name__}>"
    info["ptr_align"] = _ptr_align(t)
    if level >= 5:
        if capturing:
            info["stats"] = {"skipped": "cuda graph capture"}
        else:
            try:
                info["stats"] = _stats(torch, t)
            except Exception as exc:
                if _is_device_error(exc):
                    raise
                info["stats"] = {"error": f"{type(exc).__name__}: {str(exc)[:MAX_STR]}"}
    return info


def describe(value, level=3, capturing=False, depth=0):
    """把一个参数或返回值变成可写进 JSON 的描述。level >= 5 时张量带统计。"""
    torch = _torch()
    if _is_tensor(torch, value):
        return _describe_tensor(torch, value, level, capturing)
    if value is None or isinstance(value, (bool, int)):
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, float):
        return {"type": "float", "value": _num(value)}
    if isinstance(value, str):
        return {"type": "str", "value": value[:MAX_STR]}
    if torch is not None and isinstance(value, (torch.dtype, torch.device, torch.Size)):
        return {"type": type(value).__name__, "value": str(value)}
    if depth >= MAX_DEPTH:
        return {"type": type(value).__name__, "truncated": "depth"}
    if isinstance(value, (list, tuple)):
        items = [describe(v, level, capturing, depth + 1) for v in list(value)[:MAX_ITEMS]]
        return {"type": type(value).__name__, "len": len(value), "items": items}
    if isinstance(value, dict):
        keys = list(value.keys())[:MAX_ITEMS]
        items = {str(k): describe(value[k], level, capturing, depth + 1) for k in keys}
        return {"type": "dict", "len": len(value), "items": items}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            f.name: describe(getattr(value, f.name, None), level, capturing, depth + 1)
            for f in dataclasses.fields(value)[:MAX_ITEMS]
        }
        return {"type": type(value).__qualname__, "fields": fields}
    info = {"type": type(value).__qualname__}
    if not capturing:  # 自定义 __repr__ 可能打印张量数值,触发同步;捕获期间只记类名
        try:
            info["repr"] = repr(value)[:MAX_STR]
        except Exception as exc:
            info["repr"] = f"<repr failed: {type(exc).__name__}>"
    return info


# ---------------------------------------------------------------------------
# 张量落盘(level 10)
# ---------------------------------------------------------------------------


def _to_saveable(torch, value, path, meta, depth=0):
    """转成 torch.load(weights_only=True) 能读回的结构:张量、基本类型、容器、dtype、device、Size。"""
    if _is_tensor(torch, value):
        try:
            meta[path] = {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "storage_offset": int(value.storage_offset()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "requires_grad": bool(value.requires_grad),
            }
            t = value.detach().to("cpu")
            if type(t) is not torch.Tensor:
                t = t.as_subclass(torch.Tensor)
            if t.untyped_storage().nbytes() > t.numel() * t.element_size():
                t = t.clone(memory_format=torch.preserve_format)  # 视图只存自己那部分,不存整块 storage
            return t
        except Exception as exc:
            if _is_device_error(exc):
                raise
            meta.pop(path, None)
            return {REPR_KEY: f"<tensor not saved: {type(exc).__name__}: {str(exc)[:MAX_STR]}>",
                    "type": type(value).__qualname__}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (torch.dtype, torch.device, torch.Size)):
        return value
    if depth < MAX_DEPTH + 1:
        if isinstance(value, (list, tuple)):
            items = [_to_saveable(torch, v, f"{path}[{i}]", meta, depth + 1) for i, v in enumerate(value)]
            return tuple(items) if isinstance(value, tuple) else items
        if isinstance(value, dict) and all(isinstance(k, (str, int)) for k in value):
            return {k: _to_saveable(torch, v, f"{path}.{k}", meta, depth + 1) for k, v in value.items()}
    try:
        text = repr(value)[:MAX_STR]
    except Exception:
        text = type(value).__qualname__
    return {REPR_KEY: text, "type": type(value).__qualname__}


def _nbytes(torch, value):
    if _is_tensor(torch, value):
        try:
            return int(value.numel()) * int(value.element_size())
        except Exception:
            return 0
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(torch, v) for v in value)
    if isinstance(value, dict):
        return sum(_nbytes(torch, v) for v in value.values())
    return 0


def _safe_name(name):
    return re.sub(r"[^0-9A-Za-z_.-]", "_", name)[-80:]


# ---------------------------------------------------------------------------
# 写记录
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self, out_dir):
        self.out_dir = os.path.abspath(out_dir)
        self._reinit()

    def _reinit(self):
        self.lock = threading.Lock()
        self.files = {}
        self.seq = 0
        self.dumped_bytes = 0

    def next_seq(self):
        with self.lock:
            self.seq += 1
            return self.seq

    def _stem(self):
        return f"rank{_rank()}.pid{os.getpid()}"

    def _header(self, stem):
        torch = _torch()
        rec = {
            "event": "header",
            "format": FORMAT_VERSION,
            "pid": os.getpid(),
            "rank": _rank(),
            "t": time.time(),
            "argv": sys.argv[:20],
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", None) if torch is not None else None,
            "env": {k: os.environ.get(k) for k in ENV_SNAPSHOT_KEYS if k in os.environ},
        }
        if torch is not None and _cuda_ready(torch):
            try:
                dev = torch.cuda.current_device()
                rec["cuda_device"] = torch.cuda.get_device_name(dev)
                rec["cuda_capability"] = list(torch.cuda.get_device_capability(dev))
                rec["cuda_runtime"] = torch.version.cuda
            except Exception:
                pass
        return rec

    def write(self, rec):
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        with self.lock:
            stem = self._stem()
            key = (os.getpid(), stem)
            handle = self.files.get(key)
            if handle is None:
                os.makedirs(self.out_dir, exist_ok=True)
                handle = open(os.path.join(self.out_dir, f"calls.{stem}.jsonl"), "a", encoding="utf-8")
                self.files[key] = handle
                handle.write(json.dumps(self._header(stem), ensure_ascii=False, default=str) + "\n")
            handle.write(line)
            handle.flush()  # 进程被 abort 时,已 flush 的内容仍留在内核缓冲里

    def save_tensors(self, cfg, seq, name, kind, payload, capturing):
        torch = _torch()
        if torch is None:
            return {"skipped": "torch not imported"}
        if capturing:
            return {"skipped": "cuda graph capture"}
        if not cfg.dump_selected(name):
            return {"skipped": "filtered by OSS_DUMP_INCLUDE/EXCLUDE"}
        size = _nbytes(torch, payload)
        with self.lock:
            if self.dumped_bytes + size > cfg.max_mb * 1024 * 1024:
                return {"skipped": f"OSS_DUMP_MAX_MB={cfg.max_mb:g} reached"}
            self.dumped_bytes += size
        rel = os.path.join("tensors", self._stem(), f"{seq:06d}_{_safe_name(name)}")
        target = os.path.join(self.out_dir, rel)
        try:
            os.makedirs(target, exist_ok=True)
            meta = {}
            tree = _to_saveable(torch, payload, kind, meta)
            blob = {"format": FORMAT_VERSION, "fn": name, "seq": seq, "kind": kind, "tree": tree,
                    "tensor_meta": meta}
            torch.save(blob, os.path.join(target, f"{kind}.pt"))
        except Exception as exc:
            if _is_device_error(exc):
                raise
            return {"error": f"{type(exc).__name__}: {str(exc)[:MAX_STR]}"}  # 落盘失败不改变被测程序的行为
        return {"path": os.path.join(rel, f"{kind}.pt"), "bytes": size}


_RECORDERS = {}
_RECORDERS_LOCK = threading.Lock()


def _recorder(out_dir):
    key = os.path.abspath(out_dir)
    with _RECORDERS_LOCK:
        rec = _RECORDERS.get(key)
        if rec is None:
            rec = _RECORDERS[key] = _Recorder(key)
        return rec


def _after_fork_in_child():
    global _RECORDERS_LOCK
    _RECORDERS_LOCK = threading.Lock()
    for rec in _RECORDERS.values():
        rec._reinit()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


# ---------------------------------------------------------------------------
# 包装
# ---------------------------------------------------------------------------


def wrap(fn, name=None, config=None):
    """返回记录调用的包装函数;级别为 0 时原样返回 fn。"""
    cfg = config if config is not None else Config.from_env()
    if cfg.level <= 0 or getattr(fn, "__oss_dump_wrapped__", False):
        return fn
    label = name or _qualname(fn)
    rec = _recorder(cfg.out_dir)
    compiling = _compiling_probe()

    def run(*args, **kwargs):
        if compiling is not None and compiling():
            return fn(*args, **kwargs)
        torch = _torch()
        seq = rec.next_seq()
        capturing = _is_capturing(torch)
        call = {"event": "call", "seq": seq, "fn": label, "t": time.time(), "level": cfg.level,
                "capturing": capturing}
        try:
            if cfg.level >= 3:
                call["args"] = [describe(a, cfg.level, capturing) for a in args]
                call["kwargs"] = {k: describe(v, cfg.level, capturing) for k, v in kwargs.items()}
            if cfg.level >= 10:
                call["dump"] = rec.save_tensors(cfg, seq, label, "inputs", {"args": args, "kwargs": kwargs}, capturing)
        except BaseException as exc:
            rec.write(call)
            rec.write({"event": "exception", "seq": seq, "fn": label, "t": time.time(), "phase": "inputs",
                       "error_type": type(exc).__name__, "error": str(exc)[:2000]})
            raise
        rec.write(call)
        try:
            out = fn(*args, **kwargs)
            if cfg.sync and torch is not None and not capturing and _cuda_ready(torch):
                torch.cuda.synchronize()
            ret = {"event": "return", "seq": seq, "fn": label, "t": time.time()}
            if cfg.level >= 3:
                ret["outputs"] = describe(out, cfg.level, capturing)
            if cfg.level >= 10:
                ret["dump"] = rec.save_tensors(cfg, seq, label, "outputs", out, capturing)
        except BaseException as exc:
            rec.write({"event": "exception", "seq": seq, "fn": label, "t": time.time(), "phase": "call",
                       "error_type": type(exc).__name__, "error": str(exc)[:2000]})
            raise
        rec.write(ret)
        return out

    try:
        functools.update_wrapper(run, fn)
    except Exception:
        pass
    run.__oss_dump_wrapped__ = True
    return run


def dump_inputs(name=None, config=None):
    """装饰器形式的 wrap。"""

    def deco(fn):
        return wrap(fn, name=name, config=config)

    return deco


def patch(owner, attr, name=None, config=None):
    """把 owner.attr 换成包装版,返回原对象,供 unpatch 还原。

    只影响经由 owner.attr 的查找;别处早先 `from m import f` 拿到的引用不受影响。
    """
    raw = inspect.getattr_static(owner, attr)
    label = name or f"{getattr(owner, '__name__', type(owner).__name__)}.{attr}"
    if isinstance(raw, staticmethod):
        new = staticmethod(wrap(raw.__func__, label, config))
    elif isinstance(raw, classmethod):
        new = classmethod(wrap(raw.__func__, label, config))
    else:
        new = wrap(getattr(owner, attr), label, config)
    setattr(owner, attr, new)
    return raw


def unpatch(owner, attr, original):
    setattr(owner, attr, original)


def _restore_tree(torch, tree, meta, path, device, restore_strides):
    if _is_tensor(torch, tree):
        m = meta.get(path, {})
        dev = device or m.get("device", "cpu")
        t = tree
        want = m.get("stride")
        if restore_strides and want is not None and list(t.stride()) != want:
            try:
                r = torch.empty_strided(tuple(m["shape"]), tuple(want), dtype=t.dtype, device=dev)
                r.copy_(t)
                t = r
            except Exception:
                t = t.to(dev)
        else:
            t = t.to(dev)
        if m.get("requires_grad") and (t.is_floating_point() or t.is_complex()):
            t.requires_grad_(True)
        return t
    if isinstance(tree, list):
        return [_restore_tree(torch, v, meta, f"{path}[{i}]", device, restore_strides) for i, v in enumerate(tree)]
    if isinstance(tree, tuple):
        return tuple(_restore_tree(torch, v, meta, f"{path}[{i}]", device, restore_strides) for i, v in enumerate(tree))
    if isinstance(tree, dict) and REPR_KEY not in tree:
        return {k: _restore_tree(torch, v, meta, f"{path}.{k}", device, restore_strides) for k, v in tree.items()}
    return tree


def load_dump(path, device=None, restore_strides=True, kind="inputs"):
    """读回 level 10 落盘的输入,返回 (args, kwargs);kind="outputs" 时返回输出。

    张量放回原设备(或 device 指定的设备),按记录恢复 stride;storage_offset 与指针对齐不恢复,
    对齐敏感的问题(misaligned address)按记录里的 storage_offset 用 as_strided 自己重建。
    无法序列化的参数读回来是 {"__oss_dump_repr__": ...} 占位,要自己换成真对象。
    """
    import torch

    file = path if os.path.isfile(path) else os.path.join(path, f"{kind}.pt")
    blob = torch.load(file, map_location="cpu", weights_only=True)
    tree = _restore_tree(torch, blob["tree"], blob.get("tensor_meta", {}), blob.get("kind", kind), device,
                         restore_strides)
    if blob.get("kind", kind) == "inputs":
        return list(tree["args"]), dict(tree["kwargs"])
    return tree


# ---------------------------------------------------------------------------
# 离线汇总(不需要 torch)
# ---------------------------------------------------------------------------


def summarize(out_dir, tail=5):
    files = sorted(glob.glob(os.path.join(out_dir, "calls.*.jsonl")))
    report = {"dir": os.path.abspath(out_dir), "files": [], "first_pending": None, "verdict": None}
    pending_all = []
    for path in files:
        header, calls, finished, exceptions, bad = None, {}, set(), [], 0
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1  # 进程在写一行时死掉,最后一行可能不完整
                    continue
                ev = rec.get("event")
                if ev == "header":
                    header = header or rec
                elif ev == "call":
                    calls[rec.get("seq")] = rec
                elif ev == "return":
                    finished.add(rec.get("seq"))
                elif ev == "exception":
                    finished.add(rec.get("seq"))
                    exceptions.append({k: rec.get(k) for k in ("seq", "fn", "phase", "error_type", "error")})
        seqs = sorted(s for s in calls if isinstance(s, int))
        pending = [calls[s] for s in seqs if s not in finished]
        for p in pending:
            pending_all.append(dict(p, file=os.path.basename(path)))
        last = calls[seqs[-1]] if seqs else None
        report["files"].append({
            "file": os.path.basename(path),
            "rank": (header or {}).get("rank"),
            "pid": (header or {}).get("pid"),
            "torch": (header or {}).get("torch"),
            "cuda_device": (header or {}).get("cuda_device"),
            "calls": len(calls),
            "completed": len(calls) - len(pending),
            "pending": [{k: p.get(k) for k in ("seq", "fn", "t", "capturing")} for p in pending[-tail:]],
            "last_pending_call": pending[-1] if pending else None,
            "exceptions": exceptions[-tail:],
            "last_call": {k: last.get(k) for k in ("seq", "fn", "t")} if last else None,
            "bad_lines": bad,
        })
    if pending_all:
        first = min(pending_all, key=lambda p: p.get("t") or 0)
        report["first_pending"] = {k: first.get(k) for k in ("file", "seq", "fn", "t")}
    has_problem = bool(pending_all) or any(f["exceptions"] for f in report["files"])
    if not files:
        report["verdict"] = "no_records"
    else:
        report["verdict"] = "incomplete_calls_found" if has_problem else "all_calls_completed"
    return report


# ---------------------------------------------------------------------------
# 自检(需要 torch)
# ---------------------------------------------------------------------------


def _read_events(out_dir):
    events = []
    for path in sorted(glob.glob(os.path.join(out_dir, "calls.*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            events.extend(json.loads(line) for line in fh if line.strip())
    return events


def selftest(device="cpu", keep=None):
    torch = sys.modules.get("torch")
    if torch is None:
        import torch  # noqa: F401  调用方已确认可导入
    results = []

    def check(name, fn):
        try:
            detail = fn()
            results.append({"name": name, "ok": True, "detail": detail})
        except Exception as exc:
            results.append({"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})

    base = keep or tempfile.mkdtemp(prefix="oss_dump_selftest_")
    os.makedirs(base, exist_ok=True)

    def cfg(level, sub, **kw):
        return Config(level=level, out_dir=os.path.join(base, sub), **kw)

    def scale_add(x, y, scale=2.0):
        return x * scale + y

    def t_level1():
        f = wrap(scale_add, "selftest.scale_add", cfg(1, "l1"))
        f(torch.ones(2, device=device), torch.ones(2, device=device))
        ev = _read_events(os.path.join(base, "l1"))
        kinds = [e["event"] for e in ev]
        assert kinds == ["header", "call", "return"], kinds
        assert "args" not in ev[1]
        return kinds

    def t_level3():
        f = wrap(scale_add, "selftest.scale_add", cfg(3, "l3"))
        x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4).t()
        f(x, torch.zeros(4, 3, device=device), scale=3.0)
        call = [e for e in _read_events(os.path.join(base, "l3")) if e["event"] == "call"][0]
        a0 = call["args"][0]
        assert a0["shape"] == [4, 3] and a0["stride"] == [1, 4] and a0["contiguous"] is False, a0
        assert a0["dtype"] == "torch.float32" and call["kwargs"]["scale"]["value"] == 3.0, call
        return {"arg0": a0}

    def t_level5():
        f = wrap(scale_add, "selftest.scale_add", cfg(5, "l5"))
        x = torch.tensor([1.0, float("nan"), float("inf"), -2.0], device=device)
        f(x, torch.zeros(4, device=device))
        call = [e for e in _read_events(os.path.join(base, "l5")) if e["event"] == "call"][0]
        st = call["args"][0]["stats"]
        assert st["nan_count"] == 1 and st["inf_count"] == 1, st
        g = wrap(lambda i: i + 1, "selftest.int", cfg(5, "l5i"))
        g(torch.tensor([3, -7, 5], device=device))
        st2 = [e for e in _read_events(os.path.join(base, "l5i")) if e["event"] == "call"][0]["args"][0]["stats"]
        assert st2["min"] == -7 and st2["max"] == 5, st2
        return {"float": st, "int": st2}

    def t_level10():
        c = cfg(10, "l10")
        f = wrap(scale_add, "selftest.scale_add", c)
        x = torch.randn(5, 3, device=device).t()
        y = torch.randn(3, 5, device=device)
        out = f(x, y, scale=0.5)
        call = [e for e in _read_events(c.out_dir) if e["event"] == "call"][0]
        dump_dir = os.path.join(c.out_dir, os.path.dirname(call["dump"]["path"]))
        args, kwargs = load_dump(dump_dir)
        assert torch.equal(args[0].cpu(), x.cpu()) and torch.equal(args[1].cpu(), y.cpu())
        assert list(args[0].stride()) == list(x.stride()), (args[0].stride(), x.stride())
        assert kwargs == {"scale": 0.5}, kwargs
        saved_out = load_dump(dump_dir, kind="outputs")
        assert torch.allclose(saved_out.cpu(), out.cpu())
        return {"dump": call["dump"], "restored_stride": list(args[0].stride())}

    def t_exception():
        c = cfg(3, "exc")

        def boom(x):
            raise ValueError("selftest boom")

        f = wrap(boom, "selftest.boom", c)
        try:
            f(torch.ones(1, device=device))
        except ValueError:
            pass
        rep = summarize(c.out_dir)
        assert rep["verdict"] == "incomplete_calls_found", rep
        assert rep["files"][0]["exceptions"][0]["error_type"] == "ValueError", rep
        return rep["files"][0]["exceptions"]

    def t_crash_boundary():
        import subprocess

        out_dir = os.path.join(base, "crash")
        code = (
            "import sys, os; sys.path.insert(0, %r); import torch\n"
            "from dump_inputs import wrap, Config\n"
            "c = Config(level=3, out_dir=%r)\n"
            "ok = wrap(lambda x: x + 1, 'selftest.ok', c)\n"
            "die = wrap(lambda x: os._exit(3), 'selftest.die', c)\n"
            "ok(torch.ones(2)); die(torch.ones(2))\n"
        ) % (os.path.dirname(os.path.abspath(__file__)), out_dir)
        env = dict(os.environ, RANK="7")
        proc = subprocess.run([sys.executable, "-c", code], env=env, timeout=120)
        assert proc.returncode == 3, proc.returncode
        rep = summarize(out_dir)
        fp = rep["first_pending"]
        assert fp and fp["fn"] == "selftest.die" and fp["file"].startswith("calls.rank7."), rep
        return fp

    def t_capture_skip():
        c = cfg(10, "cap")
        f = wrap(lambda x: x * 2, "selftest.cap", c)
        if device == "cuda":
            x = torch.ones(8, device="cuda")
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    x * 2  # 预热不经过包装,避免把预热调用混进记录
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                f(x)
            g.replay()
            torch.cuda.synchronize()
            mode = "real cuda graph capture"
        else:
            global _is_capturing
            saved = _is_capturing
            _is_capturing = lambda torch_mod: True  # noqa: E731
            try:
                f(torch.ones(8))
            finally:
                _is_capturing = saved
            mode = "simulated capture on cpu"
        call = [e for e in _read_events(c.out_dir) if e["event"] == "call"][0]
        assert call["capturing"] is True and "skipped" in call["dump"], call
        assert call["args"][0]["stats"] == {"skipped": "cuda graph capture"}, call
        return {"mode": mode, "dump": call["dump"]}

    def t_custom_op():
        lib = getattr(torch, "library", None)
        if lib is None or not hasattr(lib, "custom_op"):
            return "skipped: torch.library.custom_op not available"

        @lib.custom_op("ossdump_selftest::scale2", mutates_args=())
        def scale2(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        c = cfg(3, "op")
        f = wrap(scale2, config=c)
        f(torch.ones(3, device=device))
        ev = [e for e in _read_events(c.out_dir) if e["event"] in ("call", "return")]
        assert [e["event"] for e in ev] == ["call", "return"], ev
        return {"fn": ev[0]["fn"]}

    def t_sync():
        if device != "cuda":
            return "skipped: needs cuda"
        f = wrap(scale_add, "selftest.sync", cfg(3, "sync", sync=True))
        f(torch.ones(4, device="cuda"), torch.ones(4, device="cuda"))
        return "ok"

    def t_level0_passthrough():
        f = wrap(scale_add, config=Config(level=0))
        assert f is scale_add
        return "returns original function"

    for name, fn in (
        ("level0_passthrough", t_level0_passthrough),
        ("level1_names_only", t_level1),
        ("level3_metadata", t_level3),
        ("level5_stats", t_level5),
        ("level10_roundtrip", t_level10),
        ("exception_recorded", t_exception),
        ("crash_boundary_subprocess", t_crash_boundary),
        ("capture_skips_stats_and_dump", t_capture_skip),
        ("custom_op", t_custom_op),
        ("sync_after_call", t_sync),
    ):
        check(name, fn)
    info = {"torch": torch.__version__, "device": device, "workdir": base, "checks": results,
            "passed": all(r["ok"] for r in results)}
    if device == "cuda":
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    return info


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def _emit(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="dump_inputs.py",
        description="按级别落盘每次调用的输入(库用法见文件开头);命令行汇总记录、做自检。结果输出 JSON。",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")
    p_sum = sub.add_parser("summarize", help="汇总记录目录,找出有输入无输出或抛异常的调用(不需要 torch)")
    p_sum.add_argument("dir", help="OSS_DUMP_DIR 指向的目录")
    p_sum.add_argument("--tail", type=int, default=5, help="每个文件列出最近几条 pending 与异常,默认 5")
    p_self = sub.add_parser("selftest", help="在本机 torch 上跑一遍包装、统计、落盘、读回、捕获跳过的自检")
    p_self.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p_self.add_argument("--keep", metavar="DIR", help="自检产物写到这里并保留;默认写临时目录")
    args = parser.parse_args(argv)

    if args.cmd == "summarize":
        if not os.path.isdir(args.dir):
            _emit({"error": f"目录不存在: {args.dir}"})
            return 2
        report = summarize(args.dir, tail=args.tail)
        if not report["files"]:
            _emit(dict(report, error="目录里没有 calls.*.jsonl"))
            return 2
        _emit(report)
        return 1 if report["verdict"] == "incomplete_calls_found" else 0

    if args.cmd == "selftest":
        try:
            import torch
        except ImportError as exc:
            _emit({"error": f"torch 不可导入: {exc}", "hint": "在装了 torch 的环境里运行;--help 与 summarize 不需要 torch"})
            return 2
        if args.device == "cuda" and not torch.cuda.is_available():
            _emit({"error": "要求 --device cuda,但 torch.cuda.is_available() 为 False"})
            return 2
        info = selftest(device=args.device, keep=args.keep)
        _emit(info)
        return 0 if info["passed"] else 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
