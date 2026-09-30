#!/usr/bin/env python3
"""在 torch.cuda.set_sync_debug_mode 下反复运行一个可调用对象,收集 host-device 同步点的去重调用栈,输出 JSON。

目标写成 `模块:属性` 或 `文件.py:属性`(属性可带点,如 `harness.py:Runner.step`)。
- 默认:目标本身是无参可调用对象,每轮调用一次。
- --factory:目标是无参工厂,先调用一次(不开调试模式)拿到真正每轮要跑的无参可调用对象,
  用来把建模型、造输入这类一次性工作挪出统计区间。

两种模式:
- warn(默认):sync debug mode 设为 warn,拦截 "called a synchronizing CUDA operation" 警告,
  按调用栈去重计数。用来定位。
- error:sync debug mode 设为 error,第一次同步就抛 RuntimeError,记下它的调用栈。用来复验;
  目标代码自己过滤或改写 warnings 时,它仍然有效。

覆盖范围与 torch 的 sync debug mode 相同:只有经过 c10 `memcpy_and_sync` 或 `stream_synchronize`
的同步会被报告(`.item()`、`.cpu()`、`nonzero`、`Stream.synchronize()` 等)。`torch.cuda.synchronize()`、
`Event.synchronize()`、pageable 内存上的 non_blocking 拷贝、第三方扩展直接调用的 CUDA 同步、
torch.distributed 与 torch.sparse 里的同步都不在其内,要用 nsys 核对。

退出码:0 统计区间内没有被检测到的同步;1 检测到同步;2 用法或环境错误
(没有 torch、没有 CUDA、加载目标失败、目标抛出与同步无关的异常、写不了 -o 指定的文件)。只依赖 torch 与标准库。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import re
import shlex
import sys
import threading
import traceback
import warnings

EXIT_CLEAN, EXIT_FOUND, EXIT_ENV = 0, 1, 2
SCHEMA = "oss-sync-free-cudagraph/find_syncs/v1"
SYNC_MSG = "called a synchronizing CUDA operation"  # c10/cuda/CUDAFunctions.cpp warn_or_error_on_sync
THIS_FILE = os.path.abspath(__file__)
RUNNER_FUNC = "_call_target"  # 调用栈里这一帧及其外层都属于本脚本,截掉
WARNINGS_FILES = {"warnings.py", "_py_warnings.py"}

COVERAGE_NOTE = (
    "sync debug mode 不报告:torch.cuda.synchronize()、Event.synchronize()、pageable 内存上的 "
    "non_blocking 拷贝(驱动层变成同步)、第三方扩展直接调用的 CUDA 同步、torch.distributed 与 "
    "torch.sparse 里的同步;用 nsys analyze -r cuda_api_sync,cuda_memcpy_sync,cuda_memcpy_async 核对"
)


class EnvError(Exception):
    """用法或环境错误,退出码 2。"""


# ---------------------------------------------------------------- 加载目标

def load_obj(spec: str):
    """'pkg.mod:attr' 或 'path/to/file.py:attr';attr 可带点。"""
    mod_part, sep, attr = spec.rpartition(":")
    if not sep or not mod_part or not attr:
        raise EnvError(f"无法解析 {spec!r},要写成 模块:属性 或 文件.py:属性")
    try:
        if mod_part.endswith(".py"):
            path = os.path.abspath(mod_part)
            if not os.path.isfile(path):
                raise EnvError(f"找不到文件 {path}")
            name = "_find_syncs_user_" + re.sub(r"\W", "_", os.path.splitext(os.path.basename(path))[0])
            mod = sys.modules.get(name)
            if mod is None:
                sys.path.insert(0, os.path.dirname(path))
                spec_ = importlib.util.spec_from_file_location(name, path)
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
    except EnvError:
        raise
    except Exception as e:  # 导入用户代码的任何失败都算用法或环境错误
        raise EnvError(f"加载 {spec!r} 失败:{type(e).__name__}: {e}") from e
    if not callable(obj):
        raise EnvError(f"{spec!r} 不是可调用对象")
    return obj


# ---------------------------------------------------------------- 调用栈处理

def short_path(filename: str) -> str:
    """site-packages 下的路径缩成 <site-packages>/...,工作目录下的写相对路径,便于跨机器去重。"""
    norm = filename.replace(os.sep, "/")
    for marker in ("/site-packages/", "/dist-packages/"):
        i = norm.rfind(marker)
        if i >= 0:
            return "<site-packages>/" + norm[i + len(marker):]
    try:
        rel = os.path.relpath(filename)
    except ValueError:
        return filename
    return filename if rel.startswith("..") else rel


def trim_stack(frames, torch_dir: str | None, keep_torch: bool, depth: int) -> dict:
    """frames:traceback.FrameSummary 列表,外层在前。返回保留的帧、site 与 torch 入口。"""
    frames = list(frames)
    # 截掉本脚本的外层:取最后一个 RUNNER_FUNC 帧之后的部分
    cut = 0
    for i, fs in enumerate(frames):
        if os.path.abspath(fs.filename) == THIS_FILE and fs.name == RUNNER_FUNC:
            cut = i + 1
    frames = frames[cut:]
    # 截掉内层的 warnings 模块帧与本脚本的 hook 帧
    while frames and (os.path.basename(frames[-1].filename) in WARNINGS_FILES
                      or os.path.abspath(frames[-1].filename) == THIS_FILE):
        frames.pop()

    def in_torch(fs) -> bool:
        return bool(torch_dir) and os.path.abspath(fs.filename).startswith(torch_dir + os.sep)

    torch_entry = None
    if frames and in_torch(frames[-1]):
        j = len(frames) - 1
        while j > 0 and in_torch(frames[j - 1]):
            j -= 1
        torch_entry = frames[j]  # 用户代码进入 torch 的那一帧,例如 torch/_tensor.py 的 __format__
    kept = frames if keep_torch else [fs for fs in frames if not in_torch(fs)]
    site = next((fs for fs in reversed(frames) if not in_torch(fs)), frames[-1] if frames else None)
    kept = kept[-depth:] if depth > 0 else kept
    return {"kept": kept, "site": site, "torch_entry": torch_entry}


def fmt_frame(fs) -> str:
    s = f"{short_path(fs.filename)}:{fs.lineno} in {fs.name}"
    line = (fs.line or "").strip()
    return f"{s} | {line}" if line else s


class Collector:
    def __init__(self, torch_dir: str | None, keep_torch: bool, depth: int, iters: int):
        self.torch_dir, self.keep_torch, self.depth, self.iters = torch_dir, keep_torch, depth, iters
        self.iter = -1
        self.stacks: dict[tuple, dict] = {}
        self.per_iter = [0] * iters
        self.lock = threading.Lock()

    def record(self, frames, thread_name: str) -> None:
        t = trim_stack(frames, self.torch_dir, self.keep_torch, self.depth)
        key = tuple((fs.filename, fs.lineno, fs.name) for fs in t["kept"])
        with self.lock:
            ent = self.stacks.get(key)
            if ent is None:
                ent = self.stacks[key] = {
                    "count": 0, "per_iter": [0] * self.iters, "first_iter": self.iter, "threads": [],
                    "site": fmt_frame(t["site"]) if t["site"] else None,
                    "torch_entry": fmt_frame(t["torch_entry"]) if t["torch_entry"] else None,
                    "frames": [fmt_frame(fs) for fs in t["kept"]],
                    "_order": len(self.stacks),
                }
            ent["count"] += 1
            if 0 <= self.iter < self.iters:
                ent["per_iter"][self.iter] += 1
                self.per_iter[self.iter] += 1
            if thread_name not in ent["threads"]:
                ent["threads"].append(thread_name)

    def summary(self) -> dict:
        stacks = sorted(self.stacks.values(), key=lambda e: (-e["count"], e["_order"]))
        by_site: dict[str, int] = {}
        for e in stacks:
            by_site[e["site"] or "?"] = by_site.get(e["site"] or "?", 0) + e["count"]
            e.pop("_order", None)
        return {
            "total_sync_events": sum(e["count"] for e in stacks),
            "per_iter_events": self.per_iter,
            "unique_stacks": stacks,
            "by_site": [{"site": s, "count": c} for s, c in sorted(by_site.items(), key=lambda kv: -kv[1])],
        }


# ---------------------------------------------------------------- 运行

def _call_target(fn):
    return fn()


def env_info(torch) -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": getattr(torch, "__version__", None),
            "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None)}
    try:
        info["device"] = torch.cuda.get_device_name(0)
        info["capability"] = "%d.%d" % tuple(torch.cuda.get_device_capability(0))
    except Exception as e:  # 设备信息拿不到不影响主流程
        info["device_error"] = f"{type(e).__name__}: {e}"
    return info


def run(target, *, factory: bool, mode: str, iters: int, warmup: int, depth: int, keep_torch: bool) -> tuple[dict, int]:
    try:
        import torch
    except ImportError as e:
        raise EnvError(f"导入 torch 失败({e});需要装了 CUDA 版 torch 的环境") from e
    if not hasattr(torch.cuda, "set_sync_debug_mode"):
        raise EnvError(f"torch {getattr(torch, '__version__', '?')} 没有 torch.cuda.set_sync_debug_mode")
    if not torch.cuda.is_available():
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        extra = ";检测到 MPS,但 sync debug mode 只作用于 CUDA" if mps is not None and mps.is_available() else ""
        raise EnvError(f"torch.cuda.is_available() 为 False(torch {torch.__version__},编译时 CUDA 版本 "
                       f"{torch.version.cuda}){extra}。定位同步点需要 NVIDIA GPU 与 CUDA 版 torch")
    torch_dir = os.path.dirname(os.path.abspath(torch.__file__)) if getattr(torch, "__file__", None) else None
    res: dict = {"env": env_info(torch), "notes": [COVERAGE_NOTE]}

    fn = target
    try:
        if factory:
            fn = target()
            if not callable(fn):
                raise EnvError("--factory 的返回值不是可调用对象")
        for _ in range(warmup):
            _call_target(fn)
        torch.cuda.synchronize()  # 不经过 sync debug 检查;把预热的错误在这里暴露出来
    except EnvError:
        raise
    except Exception as e:
        raise EnvError(f"预热或工厂调用失败:{type(e).__name__}: {e}\n{traceback.format_exc()}") from e

    col = Collector(torch_dir, keep_torch, depth, iters)
    prev_mode = torch.cuda.get_sync_debug_mode()
    failure = None
    with warnings.catch_warnings():
        orig_show = warnings.showwarning

        def hook(message, category, filename, lineno, file=None, line=None):
            if SYNC_MSG in str(message):
                col.record(traceback.extract_stack(), threading.current_thread().name)
            else:
                orig_show(message, category, filename, lineno, file, line)

        warnings.filterwarnings("always", message=".*" + re.escape(SYNC_MSG))
        warnings.showwarning = hook
        torch.cuda.set_sync_debug_mode(mode)
        try:
            for i in range(iters):
                col.iter = i
                try:
                    _call_target(fn)
                except RuntimeError as e:
                    if mode == "error" and SYNC_MSG in str(e):
                        col.record(traceback.extract_tb(e.__traceback__), threading.current_thread().name)
                        break
                    failure = e
                    break
                except Exception as e:
                    failure = e
                    break
        finally:
            torch.cuda.set_sync_debug_mode(prev_mode)
    if failure is not None:
        tb = "".join(traceback.format_exception(type(failure), failure, failure.__traceback__))
        raise EnvError(f"目标在第 {col.iter} 轮抛出与同步无关的异常:{type(failure).__name__}: {failure}\n{tb}")
    try:
        torch.cuda.synchronize()
    except Exception as e:
        raise EnvError(f"收尾 torch.cuda.synchronize() 失败(可能是目标里的异步 CUDA 错误):{e}") from e

    res.update(col.summary())
    found = res["total_sync_events"] > 0
    res["verdict"] = "syncs_found" if found else "clean"
    if mode == "error":
        res["notes"].append("error 模式在第一次同步处停下,只报告这一处;修掉后重跑,直到 clean")
        res["notes"].append("目标代码若用宽泛的 except 吞掉了 RuntimeError,error 模式会漏报")
    else:
        res["notes"].append("目标代码若自己改了 warnings 过滤或 showwarning,warn 模式会漏报;复验用 --mode error")
    pi = res["per_iter_events"]
    if len(pi) >= 2 and pi[0] > max(pi[1:]):
        res["notes"].append("第 0 轮的同步比后续轮次多,可能是首次调用的惰性初始化;加大 --warmup 再看稳态")
    if any("backward(" in (e["site"] or "") for e in res["unique_stacks"]):
        res["notes"].append("反向里的同步只能定位到调用 backward() 的那一行;要细到算子,用 nsys 的 CUDA 回溯")
    return res, (EXIT_FOUND if found else EXIT_CLEAN)


# ---------------------------------------------------------------- 命令行

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="find_syncs.py",
        description="在 torch.cuda.set_sync_debug_mode 下反复运行目标,输出同步点的去重调用栈与次数(JSON)。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python3 find_syncs.py harness.py:step --iters 5\n"
            "  python3 find_syncs.py harness.py:make_step --factory --warmup 3 --iters 5 -o syncs.json\n"
            "  python3 find_syncs.py harness.py:make_step --factory --mode error   # 复验\n\n"
            "退出码:0 没有检测到同步;1 检测到同步;2 用法或环境错误。\n"
            "覆盖范围:" + COVERAGE_NOTE + "。"
        ),
    )
    p.add_argument("target", help="模块:属性 或 文件.py:属性;默认是每轮调用一次的无参可调用对象")
    p.add_argument("--factory", action="store_true",
                   help="目标是无参工厂:先调用一次(不开调试模式),返回值才是每轮要跑的无参可调用对象")
    p.add_argument("--mode", choices=("warn", "error"), default="warn",
                   help="warn:收集全部同步并去重计数(默认);error:第一次同步即停,用于复验")
    p.add_argument("--iters", type=int, default=3, help="统计的轮数(默认 3)")
    p.add_argument("--warmup", type=int, default=1,
                   help="统计前不开调试模式先跑的轮数,排除首次调用的惰性初始化(默认 1;设 0 则一并统计)")
    p.add_argument("--depth", type=int, default=12, help="每个调用栈保留的最内层帧数(默认 12;0 为不限)")
    p.add_argument("--keep-torch-frames", action="store_true",
                   help="保留 torch 包内的 Python 帧(默认去掉,只在 torch_entry 字段记用户代码进入 torch 的那一帧)")
    p.add_argument("--label", default=None, help="写进 JSON 的标签,例如 before、after")
    p.add_argument("-o", "--out", default=None, help="JSON 写到这个文件;不给则打印到标准输出")
    return p


def _emit(res: dict, out: str | None) -> bool:
    """写出 JSON;写不了 -o 指定的文件时改打到标准输出,返回 False(调用方按退出码 2 处理)。"""
    text = json.dumps(res, ensure_ascii=False, indent=2)
    if out:
        try:
            with open(out, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            return True
        except OSError as e:
            print(f"[find_syncs] 写不了 {out}({e}),结果改打到标准输出", file=sys.stderr)
    print(text)
    return not out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    command = shlex.join([os.path.basename(sys.argv[0])] + (list(argv) if argv is not None else sys.argv[1:]))
    base = {"schema": SCHEMA, "command": command, "label": args.label, "target": args.target,
            "mode": args.mode, "iters": args.iters, "warmup": args.warmup}
    try:
        if args.iters < 1 or args.warmup < 0 or args.depth < 0:
            raise EnvError("--iters 至少为 1,--warmup 与 --depth 不能为负")
        target = load_obj(args.target)
        res, code = run(target, factory=args.factory, mode=args.mode, iters=args.iters,
                        warmup=args.warmup, depth=args.depth, keep_torch=args.keep_torch_frames)
    except EnvError as e:
        print(f"[find_syncs] 环境或用法错误:{str(e).splitlines()[0]}", file=sys.stderr)
        _emit({**base, "verdict": "env_error", "reason": str(e)}, args.out)
        return EXIT_ENV
    res = {**base, **res}
    if not _emit(res, args.out):
        return EXIT_ENV
    print(f"[find_syncs] verdict={res['verdict']} total={res['total_sync_events']} "
          f"unique_stacks={len(res['unique_stacks'])} per_iter={res['per_iter_events']}", file=sys.stderr)
    for s in res["by_site"][:10]:
        print(f"[find_syncs]   {s['count']:>6}  {s['site']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
