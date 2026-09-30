#!/usr/bin/env python3
"""采集一次 GPU 运行的环境指纹,输出 JSON:GPU 型号与驱动、SM 当前与最高频率、降频原因、卡上的计算进程、
CUDA 工具链、nsys 与 ncu 版本、torch 版本与编译时 CUDA、torch 编进去的架构、L2 容量、被测仓 commit 与是否 dirty。

torch 在子进程里查(--python 指定解释器),本进程不建 CUDA 上下文、不占显存。
退出码:0 = 采到了 GPU 信息(其余缺项写在 missing 里);1 = nvidia-smi 不可用或查询失败;2 = 用法错误。
只依赖 Python 标准库(>= 3.8)。本工具包新写,没有复制上游代码。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys

SCHEMA = "oss-remote-gpu-run/fingerprint-v1"
EXIT_PASS, EXIT_FAIL, EXIT_ENV = 0, 1, 2

GPU_FIELDS = ["index", "name", "driver_version", "compute_cap", "pci.bus_id", "clocks.sm", "clocks.max.sm",
              "temperature.gpu", "power.draw", "power.limit", "pstate", "memory.used", "memory.total"]
# 新驱动叫 clocks_event_reasons,旧驱动叫 clocks_throttle_reasons;以 nvidia-smi --help-query-gpu 为准
REASON_FIELDS = ["clocks_event_reasons.active", "clocks_throttle_reasons.active"]
# NVML nvmlClocksEventReasons 的位定义(nvml.h),与 oss-kernel-microbenchmark 的 bench_template.py 一致
REASON_BITS = {
    0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap",
    0x8: "hw_slowdown", 0x10: "sync_boost", 0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting",
}
THROTTLE_MASK = 0x8 | 0x20 | 0x40 | 0x80   # 热降频与硬件降频;sw_power_cap 单独记
# HLAB_RUN_ID、HLAB_COMMIT 由 hlab 的沙箱注入(工作区里的 .git 被遮住,commit 只能从这里或 Mac 侧拿)
ENV_KEYS = ["CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "TORCH_CUDA_ARCH_LIST", "HLAB_RUN_ID", "HLAB_COMMIT"]

TORCH_PROBE = r"""
import json, sys
out = {"python": sys.version.split()[0], "executable": sys.executable}
try:
    import torch
except Exception as e:
    out["error"] = "import torch failed: %s: %s" % (type(e).__name__, e)
    print(json.dumps(out)); sys.exit(0)
out["version"] = torch.__version__
out["cuda"] = torch.version.cuda
try:
    out["arch_list"] = torch.cuda.get_arch_list()
except Exception as e:
    out["arch_list_error"] = str(e)
try:
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["device"] = {"name": p.name, "cc": "%d.%d" % (p.major, p.minor),
                         "sm_count": p.multi_processor_count, "total_memory_bytes": p.total_memory,
                         "l2_cache_bytes": getattr(p, "L2_cache_size", None)}
    else:
        out["device_error"] = "torch.cuda.is_available() is False"
except Exception as e:
    out["device_error"] = "%s: %s" % (type(e).__name__, e)
try:
    import triton
    out["triton"] = triton.__version__
except Exception:
    pass
print(json.dumps(out))
"""


def _run(argv, timeout=60):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, str(e)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout).strip()[:300]
    return r.stdout, None


def _num(s):
    s = s.strip()
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return None if s in ("[N/A]", "N/A", "[Not Supported]", "") else s


def find_tool(name):
    """PATH 优先,其次 $CUDA_HOME/bin 与 /usr/local/cuda/bin(非交互 shell 的 PATH 常不含它们)。"""
    hit = shutil.which(name)
    if hit:
        return hit
    for base in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if base:
            cand = os.path.join(base, "bin", name)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    return None


def decode_reasons(raw):
    try:
        mask = int(str(raw).strip(), 16)
    except (TypeError, ValueError):
        return None
    return {"mask": hex(mask), "active": [n for b, n in REASON_BITS.items() if mask & b],
            "throttled": bool(mask & THROTTLE_MASK)}


def query_gpus(smi, missing):
    last_err = None
    for extra in REASON_FIELDS + [None]:
        fields = GPU_FIELDS + ([extra] if extra else [])
        out, err = _run([smi, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
        if out is None:
            last_err = err
            continue
        rows = []
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != len(fields):
                rows = None
                break
            row = {f: _num(v) if f not in ("name", "driver_version", "compute_cap", "pci.bus_id", "pstate") else v
                   for f, v in zip(fields, parts)}
            if extra:
                row["clock_reasons"] = decode_reasons(row.pop(extra))
                row["clock_reasons_field"] = extra
            rows.append(row)
        if rows:
            if extra is None:
                missing.append({"item": "clock_reasons", "why": f"两种字段名都查不到:{last_err}"})
            return rows, None
    return None, last_err


def query_apps(smi, missing):
    out, err = _run([smi, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    if out is None:
        missing.append({"item": "compute_apps", "why": err})
        return None
    apps = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3:
            apps.append({"pid": _num(parts[0]), "process_name": parts[1], "used_memory_mib": _num(parts[2])})
    return apps


def tool_version(name, missing):
    path = find_tool(name)
    if not path:
        missing.append({"item": name, "why": "PATH、$CUDA_HOME/bin、/usr/local/cuda/bin 都找不到"})
        return None
    out, err = _run([path, "--version"])
    if out is None:
        missing.append({"item": name, "why": err})
        return {"path": path, "version": None}
    lines = [x.strip() for x in out.splitlines() if x.strip()]
    pick = None
    for pat in (r"release \d+\.\d+", r"(?i)^version\b", r"\d+\.\d+"):
        pick = next((x for x in lines if re.search(pat, x)), None)
        if pick:
            break
    return {"path": path, "version": pick or (lines[-1] if lines else None)}


def torch_info(python, missing):
    exe = python or sys.executable
    out, err = _run([exe, "-c", TORCH_PROBE], timeout=180)
    if out is None:
        missing.append({"item": "torch", "why": err})
        return None
    try:
        info = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        missing.append({"item": "torch", "why": "探测输出不是 JSON"})
        return None
    for k in ("error", "device_error", "arch_list_error"):
        if k in info:
            missing.append({"item": "torch." + k, "why": info[k]})
    dev = info.get("device") or {}
    if dev.get("cc") and isinstance(info.get("arch_list"), list):
        sm = "sm_" + dev["cc"].replace(".", "")
        info["arch_list_covers_device"] = any(a == sm or a.startswith(sm) for a in info["arch_list"])
    return info


def repo_info(repo, commit, dirty, missing):
    info = {"commit": commit, "dirty": dirty, "source": "argument" if commit else None}
    git = shutil.which("git")
    if git and repo and os.path.isdir(repo):
        head, _ = _run([git, "-C", repo, "rev-parse", "HEAD"])
        if head:
            status, _ = _run([git, "-C", repo, "status", "--porcelain"])
            changed = [x for x in (status or "").splitlines() if x.strip()]
            info.update({"git_commit": head.strip(), "git_dirty": bool(changed), "git_changed_paths": len(changed)})
            if commit and not head.strip().startswith(commit) and not commit.startswith(head.strip()):
                info["warning"] = f"git HEAD {head.strip()[:12]} 与给出的 commit {commit[:12]} 不一致"
            if not commit:
                info.update({"commit": head.strip(), "dirty": bool(changed), "source": "git"})
    if not info["commit"] and os.environ.get("HLAB_COMMIT"):
        # 工作台的导出 commit;push-edits 过的工作区与它不一致,dirty 只能由 Mac 侧给出
        info.update({"commit": os.environ["HLAB_COMMIT"], "source": "HLAB_COMMIT"})
    if not info["commit"]:
        info["source"] = "unavailable"
        missing.append({"item": "repo.commit", "why": "没有 .git,也没给 --commit;远端工作区要由 Mac 侧给出同步的 commit"})
    if info["dirty"] is None and "git_dirty" not in info:
        missing.append({"item": "repo.dirty", "why": "没有 .git,也没给 --dirty;用过 push-edits 的工作区算 dirty"})
    return info


def collect(python=None, commit=None, dirty=None, repo="."):
    missing = []
    fp = {"schema": SCHEMA, "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
          "host": platform.node(), "gpus": None, "compute_apps": None, "env": {k: os.environ.get(k) for k in ENV_KEYS},
          "nvcc": None, "nsys": None, "ncu": None, "torch": None, "repo": None, "missing": missing}
    smi = shutil.which("nvidia-smi")
    if smi:
        fp["gpus"], err = query_gpus(smi, missing)
        if fp["gpus"] is None:
            missing.append({"item": "gpus", "why": f"nvidia-smi 查询失败:{err}"})
        fp["compute_apps"] = query_apps(smi, missing)
    else:
        missing.append({"item": "gpus", "why": "找不到 nvidia-smi"})
    for name in ("nvcc", "nsys", "ncu"):
        fp[name] = tool_version(name, missing)
    fp["torch"] = torch_info(python, missing)
    fp["repo"] = repo_info(repo, commit, dirty, missing)
    return fp


def _bool(s):
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("写 true 或 false")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                epilog="退出码:0 采到 GPU 信息;1 nvidia-smi 不可用或失败;2 用法错误。")
    p.add_argument("--python", help="查 torch 用的解释器(远端环境的 python);默认当前解释器")
    p.add_argument("--commit", help="被测仓 commit;远端没有 .git 时由 Mac 侧给出")
    p.add_argument("--dirty", type=_bool, help="被测仓是否有未提交改动(true/false)")
    p.add_argument("--repo", default=".", help="被测仓目录(有 .git 时从 git 读 commit 与 dirty)")
    p.add_argument("-o", "--out", help="同时写到这个文件")
    a = p.parse_args(argv)
    fp = collect(python=a.python, commit=a.commit, dirty=a.dirty, repo=a.repo)
    text = json.dumps(fp, ensure_ascii=False, indent=2)
    if a.out:
        try:
            with open(a.out, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError as e:
            print(json.dumps({"error": f"写不了 {a.out}:{e}"}, ensure_ascii=False))
            return EXIT_ENV
    print(text)
    return EXIT_PASS if fp["gpus"] else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
