#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the SGLang-Omni project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 出处:
#   1. vllm-project/vllm-omni .claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py
#      (本机镜像 commit 4af28f33b)。原文件没有许可证头;上面的许可取自该仓库 LICENSE,
#      版权行取自该仓库源文件通用的 SPDX 头。沿用了它的骨架:读 .json/.json.gz、顶层是
#      traceEvents 或裸事件数组、GPU 事件(kernel、gpu_memcpy、gpu_memset)取区间并集、
#      按阈值找空闲间隙、查间隙中点处的 CPU 事件。
#   2. sgl-project/sglang-omni .claude/skills/omni-gpu-deep-dive/scripts/omni_trace_pair.py
#      (本机镜像 commit af0c9cd4,SPDX Apache-2.0):稳态门的三组标记与判法。
# 本文件是改写版(2026-10-01),改了什么:
#   1. 输出从打印文本改为 JSON;退出码 0/1/2;子命令 analyze、pair、scrub
#   2. 新增稳态门:检出编译或图捕获标记即拒收(退出码 2);Lazy Function Loading 只在无栈 trace 上致命
#   3. 新增归因:kernel -> cuda_runtime(args.correlation)-> cpu_op(args["External id"])
#      -> 同线程在发射时刻的 python_function 栈,跳过 torch 运行时帧取第一处业务代码行
#   4. 间隙:中点处的 CPU 事件改为按线程列 runtime 调用、cpu_op、user_annotation、Python 栈,
#      并算下一个 kernel 的 issue lag;标出 profiler 自身开销(Command Buffer Full、overhead 类别)
#   5. 新增重叠候选(跨流遮盖率与同流紧邻风险)与融合候选(同流连续短 kernel)
#   6. 新增 mapping/formal 双 trace 按 kernel 名合并、--expect-frame/--expect-kernel、分享前检查与 scrub
#   7. 去掉 NCCL 汇总;多卡 trace 默认只取 GPU 时间最多的那张卡
"""torch.profiler chrome trace 的离线归因:稳态门、kernel 占比、空闲间隙、融合与重叠候选。

子命令:
  analyze TRACE                     单份 trace:稳态门 + kernel 表 + 间隙 + 候选;有栈时附归因
  pair --mapping M --formal F       双 trace:位置取 mapping(关图、开栈),时长取 formal(真实配置)
  scrub TRACE -o OUT                写一份分享用的副本:抹掉像密钥的键值、主机名与家目录路径

只依赖 Python 标准库。输入是 torch.profiler 导出的 chrome trace(.json 或 .json.gz),
或只含一份这种文件的目录。
退出码:0 = 完成且预期都满足(pair 另要求至少一行可下结论);
       1 = 完成但有预期没满足,或 pair 没有一行可下结论;
       2 = 用法、文件或格式错误,或 trace 没过稳态门。
"""

import argparse
import bisect
import collections
import gzip
import json
import re
import sys
from pathlib import Path

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "memcpy", "memset"}
RUNTIME_CATS = {"cuda_runtime", "cuda_driver", "runtime"}
CPU_OP_CATS = {"cpu_op", "operator"}
ANNOTATION_CATS = {"user_annotation"}
PYTHON_CAT = "python_function"

# 稳态门标记,取自 sglang-omni omni_trace_pair.py。按子串匹配每个事件的 name。
COMPILE_MARKERS = (
    "(dynamo_timed)",
    "entire_frame_compile",
    "backend_compile",
    "torch/_dynamo/convert_frame",
    "torch/_inductor/compile_fx",
    "torch/_inductor/async_compile",
    "torch/_inductor/codecache",
    "cudaModuleLoad",
    "cuModuleLoad",
)
# 普通 kernel 第一次调用也会出现;有栈时编译必然同时命中路径标记,所以只在无栈 trace 上致命。
FIRST_CALL_MARKERS = ("Lazy Function Loading",)
# 捕获而非重放:cudaGraphLaunch 是 formal trace 的正常内容,不在这里。
CAPTURE_MARKERS = ("cudaStreamBeginCapture", "cudaStreamEndCapture", "cudaGraphInstantiate")

DEFAULT_RUNTIME_REGEX = (
    r"(^|/)torch/|(^|/)triton/|^<built-in|^nn\.Module:|^<frozen |"
    r"(^|/)(contextlib|functools|threading|runpy)\.py\("
)
FRAME_RE = re.compile(r"^(?P<file>.*)\((?P<line>\d+)\): (?P<func>.*)$")
REDACTED = "<redacted>"
HOME_RE = re.compile(r"/(?:home|Users)/[^/\s\"']+/|/root/")
# host_name:kineto 用 gethostname() 写进 trace 顶层元数据,外发前一并抹掉。
SECRET_KEY_RE = re.compile(r"(?i)(^|_)(token|secret|password|passwd|api_?key|access_?key|credentials?|host_?name)$")
SECRET_VALUE_RE = re.compile(
    r"\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY)[A-Z0-9_]*)=(?!<redacted>)(\S+)|\bhf_[A-Za-z0-9]{30,}"
)

CATEGORY_RULES = (
    ("communication", re.compile(r"(?i)nccl|all_?reduce|all_?gather|reduce_?scatter|all_?to_?all|alltoall")),
    ("memory", re.compile(r"(?i)memcpy|memset|copy|fill")),
    ("compute", re.compile(r"(?i)gemm|matmul|cutlass|cublas|xmma|flash|fmha|attention|attn|wgmma|mma")),
    ("elementwise", re.compile(r"(?i)elementwise|vectorized|reduce|norm|rope|rotary|softmax|silu|gelu|"
                               r"act_and_mul|cast|topk|sigmoid|triton_poi|triton_per|triton_red")),
)


class TraceError(Exception):
    """用法、文件或格式错误,退出码 2。"""


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_path(raw):
    path = Path(raw)
    if path.is_dir():
        found = sorted(p for p in path.iterdir()
                       if p.is_file() and (p.name.endswith(".json") or p.name.endswith(".json.gz")))
        if len(found) != 1:
            raise TraceError(f"{raw} 是目录,需要恰好一份 .json 或 .json.gz,实际 {len(found)} 份:"
                             f"{[p.name for p in found][:10]}")
        return found[0]
    if not path.is_file():
        raise TraceError(f"文件不存在: {raw}")
    return path


def load_json(path):
    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        with open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, EOFError, UnicodeDecodeError, ValueError) as exc:
        raise TraceError(f"读不了 {path}: {exc}") from exc


def split_trace(data, path):
    if isinstance(data, list):
        events, meta = data, {}
    elif isinstance(data, dict) and isinstance(data.get("traceEvents"), list):
        events = data["traceEvents"]
        meta = {k: v for k, v in data.items() if k != "traceEvents"}
    else:
        raise TraceError(f"{path} 不是 chrome trace:顶层既不是含 traceEvents 列表的对象,也不是事件数组")
    events = [e for e in events if isinstance(e, dict)]
    if not events:
        raise TraceError(f"{path} 里没有事件:profiler 什么也没录到,查是否挂在了执行前向的进程上")
    return events, meta


class Trace:
    """把事件按用途分桶。时间单位 us(chrome trace 的 ts、dur)。"""

    def __init__(self, raw_path, device=None):
        self.path = resolve_path(raw_path)
        self.data = load_json(self.path)
        self.events, self.meta = split_trace(self.data, self.path)
        self.categories = collections.Counter()
        self.runtime, self.cpu_ops, self.annotations, self.overhead = [], [], [], []
        self.python = collections.defaultdict(list)
        gpu_all = []
        for event in self.events:
            cat = str(event.get("cat", ""))
            self.categories[cat or "<none>"] += 1
            if event.get("ph") != "X":
                continue
            ts, dur = num(event.get("ts")), num(event.get("dur"))
            if ts is None or dur is None or dur < 0:
                continue
            args = event.get("args") if isinstance(event.get("args"), dict) else {}
            row = {"name": str(event.get("name", "")), "cat": cat, "ts": ts, "end": ts + dur, "dur": dur,
                   "pid": str(event.get("pid")), "tid": str(event.get("tid")),
                   "corr": as_int(args.get("correlation")), "ext": as_int(args.get("External id"))}
            low = cat.lower()
            if low in GPU_CATS:
                if dur <= 0:
                    continue
                row["device"] = as_int(args.get("device", event.get("pid")))
                row["stream"] = str(args.get("stream", event.get("tid")))
                gpu_all.append(row)
            elif low in RUNTIME_CATS:
                self.runtime.append(row)
            elif low in CPU_OP_CATS:
                self.cpu_ops.append(row)
            elif low in ANNOTATION_CATS:
                self.annotations.append(row)
            elif low == PYTHON_CAT:
                callfrom = args.get("CallFrom")
                self.python[(row["pid"], row["tid"])].append(
                    (ts, ts + dur, row["name"], str(callfrom) if callfrom else None))
            if low == "overhead" or "Command Buffer Full" in row["name"]:
                self.overhead.append(row)
        for frames in self.python.values():
            frames.sort(key=lambda f: (f[0], -f[1]))
        self.device_time = collections.Counter()
        for row in gpu_all:
            self.device_time[row["device"]] += row["dur"]
        if device is not None:
            self.device = device
        elif self.device_time:
            self.device = max(self.device_time.items(), key=lambda kv: kv[1])[0]
        else:
            self.device = None
        self.gpu = sorted((r for r in gpu_all if r["device"] == self.device), key=lambda r: r["ts"])
        self.rt_by_corr = {}
        for row in self.runtime:
            if row["corr"] is not None:
                self.rt_by_corr.setdefault(row["corr"], row)
        self.cpu_by_ext = {}
        for row in self.cpu_ops:
            if row["ext"] is not None:
                self.cpu_by_ext.setdefault(row["ext"], row["name"])
        top_stack = self.meta.get("with_stack")
        self.with_stack = bool(self.python) or str(top_stack) == "1"

    def inventory(self):
        graph_corrs = {r["corr"] for r in self.runtime if r["name"].startswith("cudaGraphLaunch")}
        return {
            "path": str(self.path),
            "categories": dict(self.categories.most_common()),
            "devices_gpu_time_us": {str(k): round(v, 3) for k, v in self.device_time.items()},
            "device_used": self.device,
            "python_function_events": sum(len(v) for v in self.python.values()),
            "with_stack": self.with_stack,
            "graph_launches": sum(1 for r in self.runtime if r["name"].startswith("cudaGraphLaunch")),
            "kernels_from_graph_launch": sum(1 for r in self.gpu if r["corr"] in graph_corrs),
            "profiler_steps": sum(1 for a in self.annotations if a["name"].startswith("ProfilerStep#")),
        }


# ---------------------------------------------------------------- 稳态门

def steady_state_gate(trace, allow_capture=False, extra_markers=(), samples=3):
    markers = list(COMPILE_MARKERS) + list(FIRST_CALL_MARKERS) + list(extra_markers)
    if not allow_capture:
        markers += list(CAPTURE_MARKERS)
    starts = [num(e.get("ts")) for e in trace.events if num(e.get("ts")) is not None]
    lo, hi = (min(starts), max(starts)) if starts else (0.0, 0.0)
    span = max(hi - lo, 1e-9)
    hits = {}
    for event in trace.events:
        name = str(event.get("name", ""))
        for marker in markers:
            if marker not in name:
                continue
            ts = num(event.get("ts"))
            slot = hits.setdefault(marker, {"count": 0, "samples": [], "rel_pos": []})
            slot["count"] += 1
            if ts is not None:
                slot["rel_pos"].append((ts - lo) / span)
            if len(slot["samples"]) < samples:
                slot["samples"].append({"name": name[:200], "cat": event.get("cat"), "ts": event.get("ts")})
    demoted = set(FIRST_CALL_MARKERS) if trace.with_stack else set()
    fatal, notes = {}, {}
    for marker, slot in hits.items():
        pos = slot.pop("rel_pos")
        slot["first_rel_pos"] = round(min(pos), 3) if pos else None
        slot["last_rel_pos"] = round(max(pos), 3) if pos else None
        (notes if marker in demoted else fatal)[marker] = slot
    gate = {"status": "REJECTED" if fatal else "PASS", "with_stack": trace.with_stack,
            "fatal": fatal, "notes": notes}
    if fatal:
        gate["hint"] = ("加长预热并把每个 shape bucket 都预热到,再重采;不要事后扣除。rel_pos 是命中在"
                        "trace 时间窗里的位置(0 为开头,1 为结尾):集中在开头说明有 bucket 没预热,"
                        "散布全程说明每次调用都在重编译或重捕获。")
    elif notes:
        gate["hint"] = "有栈 trace 里只有 Lazy Function Loading:普通 kernel 的首次加载,加预热可去掉,不影响结论。"
    return gate


# ---------------------------------------------------------------- 归因

def stacks_at(frames, queries):
    """frames 为 (start, end, 帧名, CallFrom),按 (start, -end) 排好;queries 为 [(t, key)]。
    返回 {key: [帧, 外层 → 内层]}。"""
    out, stack, i = {}, [], 0
    for t, key in sorted(queries, key=lambda q: q[0]):
        while i < len(frames) and frames[i][0] <= t:
            frame = frames[i]
            while stack and stack[-1][1] < frame[0]:
                stack.pop()
            stack.append(frame)
            i += 1
        while stack and stack[-1][1] < t:
            stack.pop()
        out[key] = [f for f in stack if f[0] <= t <= f[1]]
    return out


class SiteRules:
    def __init__(self, user_regex=None, runtime_regex=DEFAULT_RUNTIME_REGEX, strip_prefixes=()):
        self.user = re.compile(user_regex) if user_regex else None
        self.runtime = re.compile(runtime_regex)
        self.strip = tuple(strip_prefixes)

    def short(self, path):
        for marker in ("site-packages/", "dist-packages/"):
            if marker in path:
                return path.rsplit(marker, 1)[1]
        for prefix in self.strip:
            if path.startswith(prefix):
                return path[len(prefix):].lstrip("/")
        return HOME_RE.sub("~/", path)

    def fmt(self, name, call_line=None):
        """帧名形如 file(行): func,行号是函数入口行;有子帧的 CallFrom 时换成实际调用行。"""
        match = FRAME_RE.match(name)
        if not match:
            return name[:160]
        if call_line is not None:
            return f"{self.short(match['file'])}:{call_line} {match['func'][:80]}"
        return f"{self.short(match['file'])}:{match['line']} {match['func'][:80]} [def]"

    @staticmethod
    def call_line(frame, child):
        """child 的 CallFrom 是 frame 里发出这次调用的那一行(torch 2.14 起导出)。"""
        if child is None or not child[3]:
            return None
        match = FRAME_RE.match(frame[2])
        path, _, line = child[3].rpartition(":")
        if match and path == match["file"] and line.isdigit():
            return int(line)
        return None

    def choose(self, stack):
        """stack 外层 → 内层。返回 (site, rule)。"""
        if not stack:
            return None, "no_stack"
        n = len(stack)
        order = list(range(n - 1, -1, -1))
        picks = []
        if self.user:
            picks.append(("user", [i for i in order if (m := FRAME_RE.match(stack[i][2]))
                                   and self.user.search(m["file"])]))
        picks.append(("non_runtime", [i for i in order if FRAME_RE.match(stack[i][2])
                                      and not self.runtime.search(stack[i][2])]))
        for rule, hits in picks:
            if hits:
                i = hits[0]
                child = stack[i + 1] if i + 1 < n else None
                return self.fmt(stack[i][2], self.call_line(stack[i], child)), rule
        return self.fmt(stack[-1][2]), "runtime_only"

    def show(self, stack, depth=8):
        out = []
        for i in range(len(stack) - 1, -1, -1):
            child = stack[i + 1] if i + 1 < len(stack) else None
            out.append(self.fmt(stack[i][2], self.call_line(stack[i], child)))
            if len(out) >= depth:
                break
        return out


def attribute_kernels(trace, rules):
    """逐个 GPU 事件找发射它的 runtime 调用与当时的 Python 栈,按 kernel 名汇总(按时长加权)。"""
    per_thread = collections.defaultdict(list)
    launch = {}
    for idx, gpu in enumerate(trace.gpu):
        runtime = trace.rt_by_corr.get(gpu["corr"]) if gpu["corr"] is not None else None
        if runtime is None:
            continue
        launch[idx] = runtime
        per_thread[(runtime["pid"], runtime["tid"])].append(((runtime["ts"] + runtime["end"]) / 2, idx))
    stacks = {}
    for key, queries in per_thread.items():
        frames = trace.python.get(key)
        if frames:
            stacks.update(stacks_at(frames, queries))
    by_name = {}
    for idx, gpu in enumerate(trace.gpu):
        slot = by_name.setdefault(gpu["name"], {"sites": collections.Counter(), "runtime": collections.Counter(),
                                                "rules": collections.Counter(), "cpu_ops": collections.Counter(),
                                                "stacks": {}, "total": 0.0})
        slot["total"] += gpu["dur"]
        runtime = launch.get(idx)
        if runtime is None:
            slot["rules"]["no_launch"] += gpu["dur"]
            continue
        op = trace.cpu_by_ext.get(runtime["ext"]) or trace.cpu_by_ext.get(gpu["ext"])
        if op:
            slot["cpu_ops"][op] += gpu["dur"]
        if runtime["name"].startswith("cudaGraphLaunch"):
            slot["rules"]["graph_replay"] += gpu["dur"]
            continue
        stack = stacks.get(idx, [])
        site, rule = rules.choose(stack)
        slot["rules"][rule] += gpu["dur"]
        if rule in ("user", "non_runtime"):
            slot["sites"][site] += gpu["dur"]
            slot["stacks"].setdefault(site, rules.show(stack))
        elif site:
            slot["runtime"][site] += gpu["dur"]
    result = {}
    for name, slot in by_name.items():
        total = slot["total"] or 1e-9
        attributed = slot["rules"]["user"] + slot["rules"]["non_runtime"]
        # 没过半时,状态取归不到的那几类里时长最多的一类(runtime_only、graph_replay、no_stack、no_launch)
        rest = collections.Counter({k: v for k, v in slot["rules"].items() if k not in ("user", "non_runtime")})
        dominant = rest.most_common(1)[0][0] if rest else "no_launch"
        status = "attributed" if attributed >= 0.5 * total else dominant
        sites = [{"site": s, "share": round(d / total, 4)} for s, d in slot["sites"].most_common(3)]
        top = sites[0]["site"] if sites else None
        result[name] = {
            "status": status,
            "attributed_fraction": round(attributed / total, 4),
            "sites": sites,
            "cpu_ops": [op for op, _ in slot["cpu_ops"].most_common(3)],
            "top_stack": slot["stacks"].get(top, []) if top else [],
        }
        if slot["runtime"]:
            result[name]["runtime_frames"] = [s for s, _ in slot["runtime"].most_common(3)]
    return result


# ---------------------------------------------------------------- kernel 表

def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def overlap_len(start, end, merged, starts):
    i = max(bisect.bisect_right(starts, start) - 1, 0)
    total = 0.0
    while i < len(merged) and merged[i][0] < end:
        lo, hi = max(start, merged[i][0]), min(end, merged[i][1])
        if hi > lo:
            total += hi - lo
        i += 1
    return total


def category(name, cat):
    if cat.lower() in {"gpu_memcpy", "gpu_memset", "memcpy", "memset"}:
        return "memory"
    for label, pattern in CATEGORY_RULES:
        if pattern.search(name):
            return label
    return "other"


def kernel_table(trace, top, attribution=None):
    stats = {}
    for gpu in trace.gpu:
        slot = stats.setdefault(gpu["name"], {"cat": gpu["cat"], "count": 0, "total": 0.0, "max": 0.0,
                                              "streams": set()})
        slot["count"] += 1
        slot["total"] += gpu["dur"]
        slot["max"] = max(slot["max"], gpu["dur"])
        slot["streams"].add(gpu["stream"])
    gpu_time = sum(s["total"] for s in stats.values())
    merged = union((g["ts"], g["end"]) for g in trace.gpu)
    busy = sum(e - s for s, e in merged)
    span = (merged[-1][1] - merged[0][0]) if merged else 0.0
    rows = []
    for name, slot in sorted(stats.items(), key=lambda kv: kv[1]["total"], reverse=True):
        row = {"name": name[:300], "cat": slot["cat"], "category": category(name, slot["cat"]),
               "count": slot["count"], "total_us": round(slot["total"], 3),
               "avg_us": round(slot["total"] / slot["count"], 3), "max_us": round(slot["max"], 3),
               "share": round(slot["total"] / gpu_time, 4) if gpu_time else 0.0,
               "streams": sorted(slot["streams"])}
        if attribution is not None:
            row.update({k: v for k, v in attribution.get(name, {}).items() if k != "top_stack"})
        rows.append(row)
    summary = {"gpu_time_us": round(gpu_time, 3), "busy_union_us": round(busy, 3), "span_us": round(span, 3),
               "idle_ratio": round(1 - busy / span, 4) if span else None, "distinct_kernels": len(rows)}
    return summary, rows[:top], rows


# ---------------------------------------------------------------- 间隙

def innermost(rows, t):
    best = None
    for row in rows:
        if row["ts"] <= t <= row["end"] and (best is None or row["dur"] < best["dur"]):
            best = row
    return best


def gap_table(trace, rules, min_gap_us, top_gaps):
    groups = []
    for gpu in trace.gpu:
        if groups and gpu["ts"] <= groups[-1]["end"]:
            group = groups[-1]
            if gpu["end"] >= group["end"]:
                group["end"], group["last"] = gpu["end"], gpu
        else:
            groups.append({"start": gpu["ts"], "end": gpu["end"], "first": gpu, "last": gpu})
    gaps = []
    for prev, nxt in zip(groups, groups[1:]):
        dur = nxt["start"] - prev["end"]
        if dur >= min_gap_us:
            gaps.append((dur, prev, nxt))
    gaps.sort(key=lambda g: g[0], reverse=True)
    span = (groups[-1]["end"] - groups[0]["start"]) if groups else 0.0
    out = {"min_gap_us": min_gap_us, "count": len(gaps), "sum_us": round(sum(g[0] for g in gaps), 3),
           "sum_share_of_span": round(sum(g[0] for g in gaps) / span, 4) if span else None, "top": []}
    threads = collections.defaultdict(list)
    for row in trace.cpu_ops:
        threads[(row["pid"], row["tid"])].append(row)
    for dur, prev, nxt in gaps[:top_gaps]:
        start, end = prev["end"], nxt["start"]
        mid = (start + end) / 2
        entry = {"start_us": round(start, 3), "dur_us": round(dur, 3),
                 "prev_kernel": prev["last"]["name"][:160], "next_kernel": nxt["first"]["name"][:160]}
        runtime = trace.rt_by_corr.get(nxt["first"]["corr"]) if nxt["first"]["corr"] is not None else None
        if runtime is not None:
            entry["next_launch"] = runtime["name"]
            entry["issue_lag_us"] = round(runtime["ts"] - start, 3)
            frames = trace.python.get((runtime["pid"], runtime["tid"]))
            if frames:
                stack = stacks_at(frames, [((runtime["ts"] + runtime["end"]) / 2, 0)])[0]
                entry["next_launch_site"] = rules.choose(stack)[0]
        busy_calls = []
        for row in trace.runtime:
            lap = min(row["end"], end) - max(row["ts"], start)
            if lap > 0:
                busy_calls.append((lap, row))
        busy_calls.sort(key=lambda x: x[0], reverse=True)
        entry["runtime_in_gap"] = [{"name": r["name"], "overlap_us": round(lap, 3), "thread": r["tid"]}
                                   for lap, r in busy_calls[:5]]
        entry["cpu_ops_at_mid"] = []
        for (pid, tid), rows in threads.items():
            hit = innermost(rows, mid)
            if hit:
                entry["cpu_ops_at_mid"].append({"thread": tid, "op": hit["name"][:160]})
        entry["annotations_at_mid"] = sorted({a["name"][:120] for a in trace.annotations
                                              if a["ts"] <= mid <= a["end"]
                                              and not a["name"].startswith("ProfilerStep#")})[:5]
        entry["python_at_mid"] = []
        for (pid, tid), frames in trace.python.items():
            stack = stacks_at(frames, [(mid, 0)])[0]
            if stack:
                entry["python_at_mid"].append({"thread": tid, "site": rules.choose(stack)[0],
                                               "stack": rules.show(stack, 6)})
        entry["profiler_overhead"] = any(r["ts"] < end and r["end"] > start for r in trace.overhead)
        out["top"].append(entry)
    return out


# ---------------------------------------------------------------- 候选

def overlap_candidates(trace, all_rows, min_share, adjacent_us, top):
    by_stream = collections.defaultdict(list)
    for gpu in trace.gpu:
        by_stream[gpu["stream"]].append(gpu)
    others = {}
    for stream in by_stream:
        merged = union((g["ts"], g["end"]) for s, rows in by_stream.items() if s != stream for g in rows)
        others[stream] = (merged, [m[0] for m in merged])
    stats = collections.defaultdict(lambda: {"total": 0.0, "hidden": 0.0, "n": 0, "prev": 0, "next": 0})
    for stream, rows in by_stream.items():
        merged, starts = others[stream]
        for i, gpu in enumerate(rows):
            slot = stats[gpu["name"]]
            slot["total"] += gpu["dur"]
            slot["n"] += 1
            if merged:
                slot["hidden"] += overlap_len(gpu["ts"], gpu["end"], merged, starts)
            if i > 0 and 0 <= gpu["ts"] - rows[i - 1]["end"] < adjacent_us:
                slot["prev"] += 1
            if i + 1 < len(rows) and 0 <= rows[i + 1]["ts"] - gpu["end"] < adjacent_us:
                slot["next"] += 1
    share = {r["name"]: r for r in all_rows}
    out = []
    for name, slot in stats.items():
        row = share.get(name[:300])
        if row is None or row["share"] < min_share:
            continue
        exposed = slot["total"] - slot["hidden"]
        hidden_ratio = slot["hidden"] / slot["total"] if slot["total"] else 0.0
        both = min(slot["prev"], slot["next"]) / slot["n"]
        one = max(slot["prev"], slot["next"]) / slot["n"]
        risk = "high" if both >= 0.5 else ("medium" if one >= 0.5 else "low")
        if hidden_ratio >= 0.8:
            label = "low-roi-hidden"
        elif row["category"] == "memory":
            label = "memory-exposed"
        elif row["category"] == "communication":
            label = "comm-exposed"
        elif risk == "high":
            label = "check-deps"
        else:
            label = "exposed"
        out.append({"name": name[:300], "category": row["category"], "share": row["share"],
                    "exposed_us": round(exposed, 3), "hidden_ratio": round(hidden_ratio, 4),
                    "dep_risk": risk, "label": label})
    out.sort(key=lambda r: r["exposed_us"], reverse=True)
    return out[:top]


def fusion_candidates(trace, short_us, min_run, top, site_of=None):
    by_stream = collections.defaultdict(list)
    for gpu in trace.gpu:
        if gpu["cat"].lower() == "kernel":
            by_stream[gpu["stream"]].append(gpu)
    runs = []
    for rows in by_stream.values():
        current = []
        for gpu in rows:
            if gpu["dur"] < short_us:
                current.append(gpu)
                continue
            if len(current) >= min_run:
                runs.append(current)
            current = []
        if len(current) >= min_run:
            runs.append(current)
    patterns = {}
    for run in runs:
        key = tuple(g["name"] for g in run)
        slot = patterns.setdefault(key, {"occurrences": 0, "kernel_us": 0.0, "wall_us": 0.0})
        slot["occurrences"] += 1
        slot["kernel_us"] += sum(g["dur"] for g in run)
        slot["wall_us"] += run[-1]["end"] - run[0]["ts"]
    out = []
    for key, slot in sorted(patterns.items(), key=lambda kv: kv[1]["wall_us"], reverse=True)[:top]:
        row = {"length": len(key), "names": [n[:120] for n in key[:8]], "occurrences": slot["occurrences"],
               "kernel_us": round(slot["kernel_us"], 3), "wall_us": round(slot["wall_us"], 3),
               "gap_us": round(slot["wall_us"] - slot["kernel_us"], 3), "label": "fuse-candidate"}
        if site_of is not None:
            sites = {site_of.get(n) for n in key}
            row["sites"] = sorted(s for s in sites if s)[:5]
            row["dep_risk"] = "low" if len(sites) == 1 and None not in sites else "unclear"
        out.append(row)
    return out


# ---------------------------------------------------------------- 预期与分享检查

def expectations(trace, frame_regexes, kernel_regexes):
    out = []
    names_py = [f[2] for frames in trace.python.values() for f in frames]
    names_cpu = [r["name"] for r in trace.cpu_ops + trace.annotations]
    for regex in frame_regexes:
        pattern = re.compile(regex)
        hits = [n for n in names_py if pattern.search(n)] + [n for n in names_cpu if pattern.search(n)]
        out.append({"kind": "frame", "regex": regex, "trace": str(trace.path), "matches": len(hits),
                    "sample": hits[0][:200] if hits else None, "met": bool(hits)})
    for regex in kernel_regexes:
        pattern = re.compile(regex)
        hits = [g["name"] for g in trace.gpu if pattern.search(g["name"])]
        out.append({"kind": "kernel", "regex": regex, "trace": str(trace.path), "matches": len(hits),
                    "sample": hits[0][:200] if hits else None, "met": bool(hits)})
    return out


def walk_strings(obj, path, found, limit=20):
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}"
            if SECRET_KEY_RE.search(str(key)) and value != REDACTED and len(found["secret_like"]) < limit:
                found["secret_like"].append(here)
            walk_strings(value, here, found, limit)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            walk_strings(value, f"{path}[{i}]", found, limit)
    elif isinstance(obj, str):
        if SECRET_VALUE_RE.search(obj) and len(found["secret_like"]) < limit:
            found["secret_like"].append(path)
        if HOME_RE.search(obj):
            found["home_path_strings"] += 1


def share_check(trace):
    found = {"secret_like": [], "home_path_strings": 0}
    walk_strings(trace.data, "$", found)
    found["top_level_keys"] = sorted(trace.meta.keys())
    found["scrub_before_sharing"] = bool(found["secret_like"] or found["home_path_strings"])
    return found


def scrub(obj, stats):
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if SECRET_KEY_RE.search(str(key)) and isinstance(value, (str, int, float)):
                out[key] = REDACTED
                stats["redacted_keys"] += 1
            else:
                out[key] = scrub(value, stats)
        return out
    if isinstance(obj, list):
        return [scrub(v, stats) for v in obj]
    if isinstance(obj, str):
        new, n = SECRET_VALUE_RE.subn(lambda m: f"{m.group(1)}={REDACTED}" if m.group(1) else REDACTED, obj)
        stats["redacted_values"] += n
        new, n = HOME_RE.subn("~/", new)
        stats["home_paths_rewritten"] += n
        return new
    return obj


# ---------------------------------------------------------------- 子命令

def load(raw, device, allow_capture, extra_markers):
    trace = Trace(raw, device)
    if not trace.gpu:
        where = f"device {device} 上" if device is not None else ""
        raise TraceError(f"{trace.path} 里{where}没有 GPU 事件(kernel/gpu_memcpy/gpu_memset):"
                         "采集时要开 ProfilerActivity.CUDA;给了 --device 的核对卡号")
    return trace, steady_state_gate(trace, allow_capture, extra_markers)


def top_sites(attribution):
    return {name: (a["sites"][0]["site"] if a["sites"] and a["status"] == "attributed" else None)
            for name, a in attribution.items()}


def run_analyze(args, rules):
    trace, gate = load(args.trace, args.device, args.allow_capture, args.extra_marker)
    out = {"tool": "trace_attribution", "mode": "analyze", "label": args.label,
           "inventory": trace.inventory(), "gate": gate}
    if gate["status"] != "PASS":
        out["status"] = "REJECTED"
        return out, 2
    attribution = attribute_kernels(trace, rules) if trace.with_stack else None
    summary, rows, all_rows = kernel_table(trace, args.top, attribution)
    site_of = top_sites(attribution) if attribution is not None else None
    out.update({
        "kernel_table": {**summary, "rows": rows},
        "gaps": gap_table(trace, rules, args.min_gap_us, args.top_gaps),
        "candidates": {"overlap": overlap_candidates(trace, all_rows, args.min_share, args.adjacent_us, args.top),
                       "fusion": fusion_candidates(trace, args.short_kernel_us, args.min_run, args.top, site_of)},
        "expectations": expectations(trace, args.expect_frame, args.expect_kernel),
        "share_check": share_check(trace),
        "notes": [],
    })
    if attribution is not None:
        out["top_stacks"] = {r["name"]: attribution[r["name"]]["top_stack"] for r in rows[:5]
                             if r["name"] in attribution}
    else:
        out["notes"].append("trace 没有 python_function 事件,只出时长,不出源码行;要位置就另采开栈的 mapping trace。")
    if out["inventory"]["graph_launches"] and trace.with_stack:
        out["notes"].append("有栈 trace 里有 cudaGraphLaunch:图重放的 kernel 只能归到重放调用处,归不到源码行。")
    code = 0 if all(e["met"] for e in out["expectations"]) else 1
    out["status"] = "OK" if code == 0 else "EXPECTATION_UNMET"
    return out, code


def run_pair(args, rules):
    mapping, mgate = load(args.mapping, args.device, args.allow_capture, args.extra_marker)
    formal, fgate = load(args.formal, args.device, args.allow_capture, args.extra_marker)
    out = {"tool": "trace_attribution", "mode": "pair", "label": args.label,
           "mapping": {"inventory": mapping.inventory(), "gate": mgate},
           "formal": {"inventory": formal.inventory(), "gate": fgate}, "notes": []}
    if mgate["status"] != "PASS" or fgate["status"] != "PASS":
        out["status"] = "REJECTED"
        return out, 2
    if not mapping.python:
        raise TraceError(f"mapping trace {mapping.path} 没有 python_function 事件:采集时要开 with_stack=True")
    if out["mapping"]["inventory"]["graph_launches"]:
        out["notes"].append("mapping trace 里还有 cudaGraphLaunch:图没关干净,这部分 kernel 归不到源码行。")
    if formal.python:
        out["notes"].append("formal trace 开了栈:它的时长含栈采集开销,formal 应按真实配置关栈重采。")
    attribution = attribute_kernels(mapping, rules)
    _, _, mrows = kernel_table(mapping, args.top, attribution)
    summary, _, frows = kernel_table(formal, args.top)
    mstats = {r["name"]: r for r in mrows}
    merged, conclusions, ratios = [], [], []
    for row in frows:
        m = mstats.get(row["name"])
        att = attribution.get(row["name"]) if m else None
        status = att["status"] if att else "not_in_mapping"
        entry = {"name": row["name"], "category": row["category"], "formal_share": row["share"],
                 "formal_total_us": row["total_us"], "formal_count": row["count"],
                 "mapping_count": m["count"] if m else 0, "status": status,
                 "sites": att["sites"] if att else [], "cpu_ops": att["cpu_ops"] if att else []}
        if m and m["count"]:
            entry["count_ratio"] = round(row["count"] / m["count"], 3)
            ratios.append(entry["count_ratio"])
        entry["conclusive"] = row["share"] >= args.min_share and status == "attributed"
        if entry["conclusive"]:
            conclusions.append({"name": entry["name"], "formal_share": entry["formal_share"],
                                "site": entry["sites"][0]["site"], "top_stack": att["top_stack"]})
        merged.append(entry)
    formal_names = {r["name"] for r in frows}
    mapping_only = [{"name": r["name"], "mapping_share": r["share"]} for r in mrows
                    if r["name"] not in formal_names and r["share"] >= args.min_share][:5]
    if mapping_only:
        out["notes"].append("mapping 里占比不小的 kernel 在 formal 里没有:多半被图或编译替换,两边工作可能不同。")
    ratios.sort()
    site_of = top_sites(attribution)
    out.update({
        "merged": {"formal_summary": summary, "rows": merged[:args.top], "conclusions": conclusions,
                   "mapping_only": mapping_only,
                   "count_ratio_median": ratios[len(ratios) // 2] if ratios else None},
        "gaps": gap_table(formal, rules, args.min_gap_us, args.top_gaps),
        "candidates": {"overlap": overlap_candidates(formal, frows, args.min_share, args.adjacent_us, args.top),
                       "fusion": fusion_candidates(formal, args.short_kernel_us, args.min_run, args.top, site_of)},
        "expectations": expectations(mapping, args.expect_frame, [])
        + expectations(formal, [], args.expect_kernel),
        "share_check": {"mapping": share_check(mapping), "formal": share_check(formal)},
    })
    met = all(e["met"] for e in out["expectations"])
    code = 0 if met and conclusions else 1
    out["status"] = "OK" if code == 0 else ("NO_CONCLUSION" if met else "EXPECTATION_UNMET")
    return out, code


def run_scrub(args):
    src = resolve_path(args.trace)
    dst = Path(args.output)
    if dst.resolve() == src.resolve():
        raise TraceError("scrub 不覆盖原文件,-o 换一个路径")
    data = load_json(src)
    split_trace(data, src)
    stats = {"redacted_keys": 0, "redacted_values": 0, "home_paths_rewritten": 0}
    cleaned = scrub(data, stats)
    opener = gzip.open if dst.name.endswith(".gz") else open
    with opener(dst, "wt", encoding="utf-8") as handle:
        json.dump(cleaned, handle)
    return {"tool": "trace_attribution", "mode": "scrub", "input": str(src), "output": str(dst),
            **stats, "status": "OK"}, 0


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", type=int, help="只看这个 GPU(默认取 GPU 时间最多的一张)")
    common.add_argument("--top", type=int, default=20, help="各表最多列几行(默认 20)")
    common.add_argument("--min-share", type=float, default=0.01,
                        help="占 GPU 时间的比例低于它不进候选、不下结论(默认 0.01)")
    common.add_argument("--min-gap-us", type=float, default=100.0, help="只列不短于它的 GPU 空闲间隙(默认 100)")
    common.add_argument("--top-gaps", type=int, default=10, help="列出最长的几个间隙(默认 10)")
    common.add_argument("--short-kernel-us", type=float, default=10.0,
                        help="短于它的 kernel 算融合候选里的短 kernel(默认 10)")
    common.add_argument("--min-run", type=int, default=3, help="同流连续几个短 kernel 才算一串(默认 3)")
    common.add_argument("--adjacent-us", type=float, default=5.0,
                        help="同流前后 kernel 间隔短于它算紧邻,用于重叠候选的依赖风险(默认 5)")
    common.add_argument("--user-regex", help="业务代码路径的正则(如 'python/sglang/|vllm/'),命中的帧优先作归因位置")
    common.add_argument("--runtime-regex", default=DEFAULT_RUNTIME_REGEX,
                        help="归因时跳过的运行时帧(默认跳过 torch/、triton/、内建函数与 nn.Module 帧)")
    common.add_argument("--strip-prefix", action="append", default=[], help="从文件路径去掉的前缀,可重复")
    common.add_argument("--allow-capture", action="store_true",
                        help="不把图捕获当作拒收理由(只在捕获本身就是研究对象时用)")
    common.add_argument("--extra-marker", action="append", default=[], help="额外的拒收标记子串,可重复")
    common.add_argument("--expect-frame", action="append", default=[],
                        help="要求 trace 里出现匹配的 Python 帧或 cpu_op 名(pair 查 mapping),可重复;缺了退出码 1")
    common.add_argument("--expect-kernel", action="append", default=[],
                        help="要求 trace 里出现匹配的 kernel 名(pair 查 formal),可重复;缺了退出码 1")
    common.add_argument("--label", help="写进输出的标签,如 decode-bs8")
    common.add_argument("-o", "--output", help="JSON 写到文件(默认 stdout)")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_an = sub.add_parser("analyze", parents=[common], help="单份 trace")
    p_an.add_argument("trace", help="trace 文件(.json/.json.gz)或只含一份 trace 的目录")
    p_pair = sub.add_parser("pair", parents=[common], help="mapping + formal 双 trace")
    p_pair.add_argument("--mapping", required=True, help="关 CUDA Graph、关编译、开 with_stack 的 trace")
    p_pair.add_argument("--formal", required=True, help="真实配置(开图、关栈)的 trace")
    p_scrub = sub.add_parser("scrub", help="写一份分享用的副本")
    p_scrub.add_argument("trace", help="trace 文件")
    p_scrub.add_argument("-o", "--output", required=True, help="副本路径;以 .gz 结尾则压缩")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "scrub":
            out, code = run_scrub(args)
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return code
        rules = SiteRules(args.user_regex, args.runtime_regex, args.strip_prefix)
        out, code = run_analyze(args, rules) if args.command == "analyze" else run_pair(args, rules)
    except TraceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except re.error as exc:
        print(f"error: 正则写错了: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: 写不了输出文件: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        try:
            Path(args.output).write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: 写不了输出文件: {exc}", file=sys.stderr)
            return 2
    else:
        print(text)
    if out.get("status") == "REJECTED":
        print("error: trace 没过稳态门(编译或图捕获进了采集窗口),见输出里的 gate", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
