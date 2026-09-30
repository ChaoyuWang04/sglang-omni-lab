#!/usr/bin/env python3
"""在 GPU 机器上跑一次构建加负载:子进程组后台运行,输出落日志,旁路监控挂起标志、CUDA 报错、
日志静默与总时限,命中就停,按统一状态值写 status.json。

用法见 --help。参数可以来自 job spec(--spec,推荐;下游只读它、不重算),或全部写在命令行上。
状态值:PASSED、FAILED、TIMEOUT、HANG_DETECTED、OOM、BUILD_FAILED、SKIPPED_HW。
构建或负载命令起不来(可执行文件不存在等 OSError)算这次运行的失败,照样写 status.json:
构建的记 BUILD_FAILED,负载的记 FAILED,原因里带报错原文。
退出码:0 = PASSED 或 SKIPPED_HW;1 = 其余状态;2 = 用法或环境错误(还没开始做任何事:参数或 spec 不合法、
status.json 已存在、查不到 compute capability、clean 越界等;不写 status.json)。
只依赖 Python 标准库(>= 3.8)。
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

SCHEMA = "oss-remote-gpu-run/v1"
EXIT_PASS, EXIT_FAIL, EXIT_ENV = 0, 1, 2
STATUSES = ("PASSED", "FAILED", "TIMEOUT", "HANG_DETECTED", "OOM", "BUILD_FAILED", "SKIPPED_HW")

# 默认模式。hang:命中立即停;error:命中后给进程 error_grace_s 自己退出,再停;oom:只用于分类。
DEFAULT_HANG = [
    r"(?i)(?<!no )hang detected",                      # TensorRT-LLM 的挂起标志
    r"Watchdog caught collective operation timeout",   # PyTorch ProcessGroupNCCL 的超时
]
DEFAULT_ERROR = [
    r"CUDA error: ",                                   # c10 CUDA_CHECK 的报错前缀
    r"device-side assert triggered",
    r"an illegal memory access was encountered",
]
DEFAULT_OOM = [
    r"CUDA out of memory",                             # c10 CUDACachingAllocator
    r"OutOfMemoryError",
    r"cudaErrorMemoryAllocation",
    r"CUBLAS_STATUS_ALLOC_FAILED",
]
# 只写进 hints,不改状态
HINTS = [
    (r"no kernel image is available for execution on the device",
     "no_kernel_image:产物里没有当前 GPU 架构的代码,查构建的架构开关(5090 是 sm_120)"),
]

SPEC_KEYS_USED = {
    "name", "run", "cwd", "env", "clean", "build", "build_timeout_s", "build_stall_s",
    "timeout_s", "stall_s", "requires_cc", "hang_patterns", "error_patterns", "oom_patterns",
    "pass_patterns", "fail_patterns", "default_patterns", "error_grace_s", "kill_grace_s",
    "on_hang", "on_hang_timeout_s", "out_dir", "fingerprint", "python", "tail_lines", "head_lines",
    "repo_commit", "dirty",
}
# 只供记账、原样带进 status.json 的字段
SPEC_KEYS_CARRIED = {"task", "backend", "series", "artifacts", "pull_to", "notes", "hlab", "version"}
RUNTIME_FLAGS_WITH_SPEC = {"out_dir", "device_cc", "quiet", "poll_s"}

DEFAULTS = {
    "cwd": ".", "env": {}, "clean": [], "build": None, "build_timeout_s": None, "build_stall_s": 0,
    "requires_cc": [], "hang_patterns": [], "error_patterns": [], "oom_patterns": [],
    "pass_patterns": [], "fail_patterns": [], "default_patterns": True, "error_grace_s": 30,
    "kill_grace_s": 10, "on_hang": None, "on_hang_timeout_s": 60, "out_dir": None,
    "fingerprint": False, "python": None, "tail_lines": 100, "head_lines": 50,
    "repo_commit": None, "dirty": None,
}


class UsageError(Exception):
    """用法或环境错误,退出码 2。"""


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ 配置


def _num(cfg, key, allow_none=False, allow_zero=True):
    v = cfg.get(key)
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise UsageError(f"{key} 必须是数字,拿到 {v!r}")
    if v < 0 or (v == 0 and not allow_zero):
        raise UsageError(f"{key} 必须{'大于' if not allow_zero else '不小于'} 0,拿到 {v!r}")
    return float(v)


def _str_list(cfg, key):
    v = cfg.get(key) or []
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise UsageError(f"{key} 必须是字符串列表")
    return v


def _command(v, key, required):
    if v is None:
        if required:
            raise UsageError(f"缺少 {key}")
        return None
    if isinstance(v, str) and v.strip():
        return v
    if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
        return v
    raise UsageError(f"{key} 必须是非空字符串(交给 /bin/sh -c)或非空字符串列表(argv)")


def validate(cfg: dict) -> dict:
    c = dict(DEFAULTS)
    c.update({k: v for k, v in cfg.items() if v is not None or k in ("build", "on_hang")})
    if not isinstance(c.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9._-]+", c["name"]):
        raise UsageError("name 必填,只能含字母、数字、. _ -")
    c["run"] = _command(c.get("run"), "run", True)
    c["build"] = _command(c.get("build"), "build", False)
    c["timeout_s"] = _num(c, "timeout_s", allow_zero=False)
    c["stall_s"] = _num(c, "stall_s")           # 必填;0 表示不监控静默
    c["build_timeout_s"] = _num(c, "build_timeout_s", allow_none=True, allow_zero=False)
    c["build_stall_s"] = _num(c, "build_stall_s")
    for k in ("error_grace_s", "kill_grace_s", "on_hang_timeout_s"):
        c[k] = _num(c, k)
    for k in ("tail_lines", "head_lines"):
        if isinstance(c[k], bool) or not isinstance(c[k], int) or c[k] < 0:
            raise UsageError(f"{k} 必须是非负整数")
    for k in ("hang_patterns", "error_patterns", "oom_patterns", "pass_patterns", "fail_patterns"):
        for p in _str_list(c, k):
            try:
                re.compile(p)
            except re.error as e:
                raise UsageError(f"{k} 里的正则 {p!r} 不合法:{e}")
    for cc in _str_list(c, "requires_cc"):
        if not re.fullmatch(r"\d+\.\d+", cc):
            raise UsageError(f"requires_cc 写成 12.0 这种形式,拿到 {cc!r}")
    for p in _str_list(c, "clean"):
        parts = p.replace("\\", "/").split("/")
        if os.path.isabs(p) or ".." in parts or p.strip() in ("", ".", "./", "*"):
            raise UsageError(f"clean 只收工作目录内的相对路径或通配,拿到 {p!r}")
    env = c.get("env") or {}
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise UsageError("env 必须是 {字符串: 字符串}")
    if c["on_hang"] is not None and (not isinstance(c["on_hang"], list)
                                      or not all(isinstance(x, str) for x in c["on_hang"]) or not c["on_hang"]):
        raise UsageError("on_hang 必须是非空 argv 列表,可含 {pid}")
    if not isinstance(c["default_patterns"], bool) or not isinstance(c["fingerprint"], bool):
        raise UsageError("default_patterns、fingerprint 必须是 true/false")
    if c["dirty"] is not None and not isinstance(c["dirty"], bool):
        raise UsageError("dirty 必须是 true/false")
    if not os.path.isdir(c["cwd"]):
        raise UsageError(f"cwd 不存在:{c['cwd']}")
    return c


def load_spec(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
    except (OSError, ValueError) as e:
        raise UsageError(f"读不了 spec {path}:{e}")
    if not isinstance(spec, dict):
        raise UsageError("spec 顶层必须是 JSON 对象")
    unknown = set(spec) - SPEC_KEYS_USED - SPEC_KEYS_CARRIED
    if unknown:
        raise UsageError(f"spec 有未知字段(拼错了?):{sorted(unknown)}")
    return spec


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="后台跑构建加负载,监控挂起、CUDA 报错、日志静默与超时,按统一状态值写 status.json。",
        epilog=("例:run_guarded.py --spec .lab/tasks/<任务>/jobs/<短名>.json\n"
                "    run_guarded.py --name smoke --timeout-s 600 --stall-s 120 -- python -m pytest tests/x.py -q\n"
                "状态值:" + "、".join(STATUSES) + "。构建命令起不来记 BUILD_FAILED,负载命令起不来记 FAILED,"
                "都写 status.json。\n"
                "退出码:0 PASSED 或 SKIPPED_HW,1 其余状态,2 用法或环境错误(还没开始做任何事,不写 status.json)。"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", help="job spec JSON;给了就不能再用下面的参数型选项")
    p.add_argument("--name")
    p.add_argument("--timeout-s", type=float, help="负载总时限(秒),应小于 hlab 等外层时限")
    p.add_argument("--stall-s", type=float, help="日志多少秒不增长算挂起;0 表示不监控")
    p.add_argument("--build", help="构建命令(交给 /bin/sh -c);失败记 BUILD_FAILED,不跑负载")
    p.add_argument("--build-timeout-s", type=float)
    p.add_argument("--build-stall-s", type=float, help="构建的静默时限;默认 0(不监控,编译常长时间无输出)")
    p.add_argument("--clean", action="append", help="构建前删掉的产物路径(相对 cwd,可用通配;git 跟踪的文件拒删),可多次")
    p.add_argument("--cwd")
    p.add_argument("--env", action="append", metavar="K=V", help="传给子进程的环境变量,可多次")
    p.add_argument("--requires-cc", action="append", help="允许的 compute capability(如 10.0),可多次;不满足记 SKIPPED_HW")
    p.add_argument("--hang-pattern", action="append", help="额外的挂起模式(正则),命中立即停")
    p.add_argument("--error-pattern", action="append", help="额外的报错模式(正则),命中后宽限期内不退出就停")
    p.add_argument("--oom-pattern", action="append")
    p.add_argument("--pass-pattern", action="append", help="退出码 0 时每个都必须出现,否则记 FAILED")
    p.add_argument("--fail-pattern", action="append", help="出现任一个就记 FAILED")
    p.add_argument("--no-default-patterns", action="store_true", help="不用内置的挂起、CUDA 报错、OOM 模式")
    p.add_argument("--error-grace-s", type=float, help="出现 CUDA 报错后等进程自己退出的秒数,默认 30")
    p.add_argument("--kill-grace-s", type=float, help="SIGTERM 之后等多久再 SIGKILL,默认 10")
    p.add_argument("--fingerprint", action="store_true", help="开跑前用同目录的 env_fingerprint.py 采环境指纹")
    p.add_argument("--python", help="采指纹时查 torch 用的解释器(默认当前解释器)")
    p.add_argument("--repo-commit", help="被测仓 commit(远端没有 .git 时由 Mac 侧给出)")
    p.add_argument("--out-dir", help="产物目录;默认 $HLAB_OUTPUT_ROOT/<name>,否则 ./guarded-runs/<name>-<时间>")
    p.add_argument("--device-cc", help="不查 nvidia-smi,直接用这个 compute capability 判断 requires")
    p.add_argument("--poll-s", type=float, default=1.0, help="监控轮询间隔(秒)")
    p.add_argument("--quiet", action="store_true", help="不把子进程输出回显到本进程 stdout")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- 之后是负载 argv")
    a = p.parse_args(argv)
    if a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    return a


def config_from_args(a) -> tuple:
    runtime = {"out_dir": a.out_dir, "device_cc": a.device_cc, "quiet": a.quiet, "poll_s": a.poll_s}
    param_flags = {k: v for k, v in vars(a).items()
                   if k not in RUNTIME_FLAGS_WITH_SPEC | {"spec", "cmd"} and v not in (None, False, [])}
    if a.spec:
        if param_flags or a.cmd:
            raise UsageError("给了 --spec 就不要再在命令行写参数(只允许 --out-dir --device-cc --quiet --poll-s):"
                             + ", ".join(sorted(param_flags) + (["-- cmd"] if a.cmd else [])))
        spec = load_spec(a.spec)
        cfg = {k: v for k, v in spec.items() if k in SPEC_KEYS_USED}
        carried = {k: v for k, v in spec.items()}
    else:
        env = {}
        for kv in a.env or []:
            if "=" not in kv:
                raise UsageError(f"--env 要写成 K=V,拿到 {kv!r}")
            k, v = kv.split("=", 1)
            env[k] = v
        cfg = {
            "name": a.name, "run": a.cmd or None, "cwd": a.cwd, "env": env, "clean": a.clean,
            "build": a.build, "build_timeout_s": a.build_timeout_s, "build_stall_s": a.build_stall_s,
            "timeout_s": a.timeout_s, "stall_s": a.stall_s, "requires_cc": a.requires_cc,
            "hang_patterns": a.hang_pattern, "error_patterns": a.error_pattern,
            "oom_patterns": a.oom_pattern, "pass_patterns": a.pass_pattern,
            "fail_patterns": a.fail_pattern, "default_patterns": not a.no_default_patterns,
            "error_grace_s": a.error_grace_s, "kill_grace_s": a.kill_grace_s,
            "fingerprint": a.fingerprint, "python": a.python, "repo_commit": a.repo_commit,
        }
        carried = None
    if runtime["out_dir"] is not None:
        cfg["out_dir"] = runtime["out_dir"]
    if runtime["poll_s"] is None or runtime["poll_s"] <= 0:
        raise UsageError("--poll-s 必须大于 0")
    return validate(cfg), carried, runtime


# ------------------------------------------------------------------ 运行前


def query_device_cc(override):
    if override:
        if not re.fullmatch(r"\d+\.\d+", override):
            raise UsageError(f"--device-cc 写成 12.0 这种形式,拿到 {override!r}")
        return override
    exe = shutil.which("nvidia-smi")
    if not exe:
        raise UsageError("判断 requires_cc 需要 nvidia-smi,找不到;可用 --device-cc 直接给出")
    try:
        r = subprocess.run([exe, "--query-gpu=compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise UsageError(f"nvidia-smi 查 compute_cap 失败:{e}")
    ccs = sorted({x.strip() for x in r.stdout.splitlines() if x.strip()})
    if r.returncode != 0 or not ccs or not all(re.fullmatch(r"\d+\.\d+", x) for x in ccs):
        raise UsageError(f"nvidia-smi 查 compute_cap 失败:{(r.stderr or r.stdout).strip()[:200]}")
    if len(ccs) > 1:
        raise UsageError(f"机器上有多种 compute capability {ccs},用 --device-cc 指明要用的卡")
    return ccs[0]


def _git_tracked(cwd, path):
    if not shutil.which("git"):
        return None
    try:
        inside = subprocess.run(["git", "-C", cwd, "rev-parse", "--is-inside-work-tree"],
                                capture_output=True, text=True, timeout=30)
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return None
        r = subprocess.run(["git", "-C", cwd, "ls-files", "--", os.path.relpath(path, cwd)],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return [x for x in r.stdout.splitlines() if x.strip()]


def resolve_clean(patterns, cwd):
    """把 clean 里的路径与通配展开成要删的实际路径;越出 cwd 或含 git 跟踪文件就拒绝。"""
    real_cwd = os.path.realpath(cwd)
    targets = []
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(cwd, pat))):
            parent = os.path.realpath(os.path.dirname(os.path.abspath(path)))
            if not (parent == real_cwd or parent.startswith(real_cwd + os.sep)):
                raise UsageError(f"clean 的 {pat!r} 展开到工作目录之外:{path}")
            if os.path.realpath(path) == real_cwd:
                raise UsageError(f"clean 的 {pat!r} 会删掉整个工作目录")
            tracked = _git_tracked(cwd, path)
            if tracked:
                raise UsageError(f"clean 的 {pat!r} 含 git 跟踪的文件(如 {tracked[0]}),只能列编译产物")
            targets.append(path)
    return targets


def remove_paths(paths):
    done = []
    for path in paths:
        if os.path.islink(path) or os.path.isfile(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        else:
            continue
        done.append(path)
    return done


# ------------------------------------------------------------------ 监控一个阶段


class Stage:
    def __init__(self, label, cmd, *, cwd, env, log_path, timeout_s, stall_s, hang, error, oom,
                 passp, failp, error_grace_s, kill_grace_s, poll_s, echo, on_hang, on_hang_timeout_s,
                 dump_path, stop_flag):
        self.label, self.cmd, self.cwd, self.env = label, cmd, cwd, env
        self.log_path, self.timeout_s, self.stall_s = log_path, timeout_s, stall_s
        self.pats = {"hang": hang, "error": error, "oom": oom, "pass": passp, "fail": failp,
                     "hint": [re.compile(h) for h, _ in HINTS]}
        self.hint_text = {h: t for h, t in HINTS}
        self.error_grace_s, self.kill_grace_s, self.poll_s = error_grace_s, kill_grace_s, poll_s
        self.echo, self.on_hang, self.on_hang_timeout_s = echo, on_hang, on_hang_timeout_s
        self.dump_path, self.stop_flag = dump_path, stop_flag
        self.lock = threading.Lock()
        self.matches = []            # 前若干条命中
        self.seen = {k: set() for k in self.pats}
        self.first_hang = None
        self.first_error_t = None
        self.last_output_t = None
        self.lines = 0
        self.echo_ends_newline = True

    def _scan_line(self, text):
        self.lines += 1
        for kind, regs in self.pats.items():
            for rx in regs:
                if rx.search(text):
                    with self.lock:
                        if rx.pattern not in self.seen[kind] and len(self.matches) < 50:
                            self.matches.append({"kind": kind, "pattern": rx.pattern,
                                                 "line_no": self.lines, "line": text[:500]})
                        self.seen[kind].add(rx.pattern)
                        if kind == "hang" and self.first_hang is None:
                            self.first_hang = (rx.pattern, self.lines)
                        if kind == "error" and self.first_error_t is None:
                            self.first_error_t = time.monotonic()

    def _reader(self, fd, log):
        buf = b""
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            self.last_output_t = time.monotonic()
            log.write(chunk)
            log.flush()
            if self.echo:
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    self.echo_ends_newline = chunk.endswith(b"\n")
                except (OSError, ValueError):
                    pass
            buf += chunk
            *full, buf = re.split(rb"\r\n|\n|\r", buf)
            for raw in full:
                self._scan_line(raw.decode("utf-8", "replace"))
            if len(buf) > 1 << 20:     # 超长无换行:按 1 MiB 截断扫描
                self._scan_line(buf.decode("utf-8", "replace"))
                buf = b""
        if buf:
            self._scan_line(buf.decode("utf-8", "replace"))

    def _signal_group(self, proc, sig):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _stop(self, proc):
        self._signal_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=self.kill_grace_s)
        except subprocess.TimeoutExpired:
            self._signal_group(proc, signal.SIGKILL)
            proc.wait()

    def _dump(self, pid):
        if not self.on_hang:
            return None
        argv = [x.replace("{pid}", str(pid)) for x in self.on_hang]
        try:
            with open(self.dump_path, "wb") as f:
                r = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=self.on_hang_timeout_s)
            return {"argv": argv, "exit_code": r.returncode, "file": os.path.basename(self.dump_path)}
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"argv": argv, "error": str(e)}

    def run(self):
        args = ["/bin/sh", "-c", self.cmd] if isinstance(self.cmd, str) else list(self.cmd)
        t0 = time.monotonic()
        started = now_iso()
        with open(self.log_path, "wb") as log:
            try:
                proc = subprocess.Popen(args, cwd=self.cwd, env=self.env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
            except OSError as e:  # 起不来是这次运行的结果,由 classify 记状态,不当用法错误
                return {
                    "argv": args, "log": os.path.basename(self.log_path), "started_at": started,
                    "duration_s": round(time.monotonic() - t0, 3), "exit_code": None, "signal": None,
                    "killed_for": None, "detail": None, "start_error": f"{type(e).__name__}: {e}",
                    "stragglers_killed": False, "lines": 0, "matches": [], "seen": {}, "hang_dump": None,
                }
            self.last_output_t = t0
            reader = threading.Thread(target=self._reader, args=(proc.stdout.fileno(), log), daemon=True)
            reader.start()
            killed_for, detail, dump = None, None, None
            while proc.poll() is None:
                now = time.monotonic()
                if self.stop_flag.get("signal"):
                    killed_for, detail = "cancelled", f"包装器收到信号 {self.stop_flag['signal']}"
                elif now - t0 >= self.timeout_s:
                    killed_for, detail = "timeout", f"超过总时限 {self.timeout_s:g} s"
                elif self.first_hang is not None:
                    killed_for = "hang_pattern"
                    detail = f"第 {self.first_hang[1]} 行命中挂起模式 {self.first_hang[0]!r}"
                    dump = self._dump(proc.pid)
                elif self.stall_s and now - self.last_output_t >= self.stall_s:
                    killed_for, detail = "stall", f"日志 {self.stall_s:g} s 没有增长"
                    dump = self._dump(proc.pid)
                elif self.first_error_t is not None and now - self.first_error_t >= self.error_grace_s:
                    killed_for = "error"
                    detail = f"日志出现 CUDA 报错,{self.error_grace_s:g} s 内进程没有自己退出"
                if killed_for:
                    self._stop(proc)
                    break
                time.sleep(self.poll_s)
            rc = proc.wait()
            # 主进程退出后,同一进程组里残留的子进程(服务的 worker 等)一并清掉
            stragglers = False
            try:
                os.killpg(proc.pid, 0)
                stragglers = True
                self._signal_group(proc, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            reader.join(timeout=10)
            proc.stdout.close()
        return {
            "argv": args, "log": os.path.basename(self.log_path), "started_at": started,
            "duration_s": round(time.monotonic() - t0, 3), "exit_code": rc if rc >= 0 else None,
            "signal": -rc if rc < 0 else None, "killed_for": killed_for, "detail": detail,
            "start_error": None, "stragglers_killed": stragglers, "lines": self.lines, "matches": self.matches,
            "seen": {k: sorted(v) for k, v in self.seen.items() if v}, "hang_dump": dump,
        }


# ------------------------------------------------------------------ 分类与记录


def classify(res, passp, label="负载"):
    seen = res["seen"]
    kf = res["killed_for"]
    if res.get("start_error"):
        return "FAILED", f"{label}无法启动:{res['start_error']}"
    if kf == "cancelled":
        return "FAILED", "被取消:" + res["detail"]
    if kf == "timeout":
        return "TIMEOUT", res["detail"]
    if kf in ("hang_pattern", "stall"):
        return "HANG_DETECTED", res["detail"]
    if "oom" in seen and (kf == "error" or res["exit_code"] != 0):
        return "OOM", "日志命中 OOM 模式:" + seen["oom"][0]
    if kf == "error" or "error" in seen:
        return "FAILED", (res["detail"] or "日志出现 CUDA 报错:" + seen["error"][0])
    if res["exit_code"] != 0:
        how = f"退出码 {res['exit_code']}" if res["exit_code"] is not None else f"被信号 {res['signal']} 结束"
        return "FAILED", how
    if "fail" in seen:
        return "FAILED", "退出码 0,但命中失败模式:" + seen["fail"][0]
    missing = [p.pattern for p in passp if p.pattern not in set(seen.get("pass", []))]
    if missing:
        return "FAILED", "退出码 0,但没出现成功模式:" + missing[0]
    return "PASSED", "退出码 0"


def read_tail(path, n):
    if n <= 0 or not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size, block, data = f.tell(), 65536, b""
        while size > 0 and data.count(b"\n") <= n:
            step = min(block, size)
            size -= step
            f.seek(size)
            data = f.read(step) + data
    return [x.decode("utf-8", "replace") for x in data.splitlines()[-n:]]


def read_head(path, n):
    out = []
    if n <= 0 or not os.path.exists(path):
        return out
    with open(path, "rb") as f:
        for raw in f:
            out.append(raw.rstrip(b"\r\n").decode("utf-8", "replace")[:500])
            if len(out) >= n:
                break
    return out


def hints_of(res):
    seen = set(res["seen"].get("hint", []))
    return [t for h, t in HINTS if h in seen]


def default_out_dir(name):
    root = os.environ.get("HLAB_OUTPUT_ROOT")
    if root:
        return os.path.join(root, name)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join("guarded-runs", f"{name}-{stamp}")


def main(argv=None) -> int:
    stop_flag = {}
    try:
        a = parse_args(argv)
        cfg, carried, runtime = config_from_args(a)
        out_dir = cfg["out_dir"] or default_out_dir(cfg["name"])
        status_path = os.path.join(out_dir, "status.json")
        if os.path.exists(status_path):
            raise UsageError(f"{status_path} 已存在;原始记录不覆盖,换 --out-dir 或 name")
        os.makedirs(out_dir, exist_ok=True)
        cwd = cfg["cwd"]
        env = dict(os.environ)
        env.update(cfg["env"])
        dflt = cfg["default_patterns"]
        comp = lambda xs: [re.compile(x) for x in xs]  # noqa: E731
        hang = comp((DEFAULT_HANG if dflt else []) + cfg["hang_patterns"])
        error = comp((DEFAULT_ERROR if dflt else []) + cfg["error_patterns"])
        oom = comp((DEFAULT_OOM if dflt else []) + cfg["oom_patterns"])
        passp, failp = comp(cfg["pass_patterns"]), comp(cfg["fail_patterns"])
        device_cc = query_device_cc(runtime["device_cc"]) if cfg["requires_cc"] else runtime["device_cc"]
        clean_targets = resolve_clean(cfg["clean"], cwd) if cfg["clean"] else []
    except UsageError as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return EXIT_ENV

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda s, _f: stop_flag.__setitem__("signal", signal.Signals(s).name))

    record = {"schema": SCHEMA, "name": cfg["name"], "status": None, "reason": None,
              "started_at": now_iso(), "out_dir": os.path.abspath(out_dir), "cwd": os.path.abspath(cwd),
              "device_cc": device_cc, "repo_commit": cfg["repo_commit"], "dirty": cfg["dirty"],
              "env_overrides": cfg["env"], "spec": carried,
              "cleaned": [], "fingerprint": None, "build": None, "run": None, "hints": [],
              "tail": [], "head": []}
    echo = not runtime["quiet"]

    def finish(status, reason, stage_log=None):
        record["status"], record["reason"], record["ended_at"] = status, reason, now_iso()
        if stage_log:
            record["tail"] = read_tail(stage_log, cfg["tail_lines"])
            if status not in ("PASSED", "SKIPPED_HW"):
                record["head"] = read_head(stage_log, cfg["head_lines"])
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        summary = {"status": status, "reason": reason, "status_file": os.path.abspath(status_path)}
        if echo and not getattr(finish, "newline_ok", True):
            sys.stdout.write("\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return EXIT_PASS if status in ("PASSED", "SKIPPED_HW") else EXIT_FAIL

    if cfg["requires_cc"] and device_cc not in cfg["requires_cc"]:
        return finish("SKIPPED_HW", f"本机 compute capability {device_cc} 不在 requires_cc {cfg['requires_cc']} 里,没有运行")

    if cfg["fingerprint"]:
        fp_path = os.path.join(out_dir, "fingerprint.json")
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import env_fingerprint  # noqa: WPS433
            fp = env_fingerprint.collect(python=cfg["python"], commit=cfg["repo_commit"],
                                         dirty=cfg["dirty"], repo=cwd)
            with open(fp_path, "w", encoding="utf-8") as f:
                json.dump(fp, f, ensure_ascii=False, indent=2)
            record["fingerprint"] = os.path.basename(fp_path)
        except Exception as e:  # 指纹采不到不挡运行,记下原因
            record["fingerprint"] = {"error": f"{type(e).__name__}: {e}"}

    record["cleaned"] = [os.path.relpath(p, cwd) for p in remove_paths(clean_targets)]

    common = dict(cwd=cwd, env=env, error_grace_s=cfg["error_grace_s"], kill_grace_s=cfg["kill_grace_s"],
                  poll_s=runtime["poll_s"], echo=echo, on_hang=cfg["on_hang"],
                  on_hang_timeout_s=cfg["on_hang_timeout_s"], stop_flag=stop_flag)
    # 从这里起已经开始做事(可能已采指纹、删过产物):结果一律写进 status.json,不再返回 2
    if cfg["build"] is not None:
        # 构建只看退出码、总时限与静默:编译输出会原样引用源码里的 "CUDA error: " 之类字符串
        b_log = os.path.join(out_dir, "build.log")
        st = Stage("build", cfg["build"], log_path=b_log,
                   timeout_s=cfg["build_timeout_s"] or cfg["timeout_s"], stall_s=cfg["build_stall_s"],
                   hang=[], error=[], oom=[], passp=[], failp=[],
                   dump_path=os.path.join(out_dir, "build-hang-dump.txt"), **common)
        res = st.run()
        finish.newline_ok = st.echo_ends_newline
        record["build"] = res
        record["hints"] = hints_of(res)
        b_status, b_reason = classify(res, [], label="命令")
        if b_status != "PASSED":
            return finish("BUILD_FAILED", f"构建 {b_status}:{b_reason};负载没有运行", b_log)
    r_log = os.path.join(out_dir, "run.log")
    st = Stage("run", cfg["run"], log_path=r_log, timeout_s=cfg["timeout_s"], stall_s=cfg["stall_s"],
               hang=hang, error=error, oom=oom, passp=passp, failp=failp,
               dump_path=os.path.join(out_dir, "hang-dump.txt"), **common)
    res = st.run()
    finish.newline_ok = st.echo_ends_newline
    record["run"] = res
    record["hints"] = sorted(set(record["hints"] + hints_of(res)))
    status, reason = classify(res, passp)
    return finish(status, reason, r_log)


if __name__ == "__main__":
    sys.exit(main())
