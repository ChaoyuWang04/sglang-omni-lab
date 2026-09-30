#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
# 出处:NVIDIA/TensorRT-LLM .claude/skills/perf-host-analysis/scripts/detect_host_overhead.py
#   (本机镜像 commit 50f85bfe3e)。原文件没有独立许可证头,上面的版权与许可取自该仓库 LICENSE
#   与同目录 analyze_host_overhead.py 的 SPDX 头。
# 本文件是改写版(2026-10-01),改了什么:
#   1. 分析窗口:从「第一个到最后一个 kernel」改为按 NVTX 名正则切出的稳态 step;去掉 TRT-LLM
#      专属的 "[Executor] _forward_step N: X ctx reqs, Y gen reqs" 文本,改用命名分组 prefill/decode
#   2. 投票:GPU 空闲率与利用率互为补数,只算一票;host prep 三个子指标只算一票(3b 与 3c 同时越线)。
#      可用票少于 3(没给 --host-regex、它在 step 内没匹配到 range,或缺 runtime 表)且恰好一票越线时
#      判 INCONCLUSIVE,JSON 的 missing_votes 写明缺哪类证据;整体结论:任一阶段「是」即「是」
#   3. GPU 忙时间:kernel 之外并入 CUDA Graph 整图记录与 memcpy/memset,取区间并集;
#      有图重放而并集里没有图的执行记录时判 INCONCLUSIVE
#   4. launch 耗时:API 名先去掉 _vNNNN 版本后缀再匹配;除 cudaLaunchKernel 外也算 cuLaunchKernel、
#      cudaGraphLaunch 等 launch API;launch 区间取并集
#   5. NVTX 文本同时兼容 text 列与 textId -> StringIds
#   6. host prep 的 NVTX 名改由 --host-regex 指定(上游写死 _prepare_tp_inputs 等 TRT-LLM 名字)
#   7. 新增 kernel 间隙分桶与最长间隙定位;只输出 JSON;退出码 0/1/2
"""判断稳态 step 里 host 开销是否为瓶颈,并给 GPU 空闲间隙分桶。

输入 nsys 导出的 SQLite(nsys export --type sqlite)。只依赖 Python 标准库。
退出码:0 = 判定「否」;1 = 判定「是」(host 是瓶颈);2 = 用法或环境错误,或采集、证据不全(INCONCLUSIVE)。
"""

import argparse
import bisect
import json
import os
import pathlib
import re
import sqlite3
import sys

DEFAULT_STEP_REGEX = r"step=(?P<step>\d+)\s+prefill=(?P<prefill>\d+)\s+decode=(?P<decode>\d+)"

# 起点值:上游 references/thresholds.md,在 Llama 3.2 1B、TP=2、B200 上标定(2025-02);本机重标后用 --*-threshold 覆盖
THRESHOLD_SOURCE = (
    "starting values from TensorRT-LLM perf-host-analysis references/thresholds.md "
    "(calibrated on Llama 3.2 1B, TP=2, B200, 2025-02); override with --*-threshold after local recalibration"
)
IDLE_THRESHOLD = {"all": 0.30, "prefill": 0.30, "decode": 0.15}
LAUNCH_THRESHOLD = 0.10
EXPOSED_WALL_THRESHOLD = 0.05
EXPOSED_IDLE_THRESHOLD = 0.50
NCCL_CAVEAT_THRESHOLD = 0.20

LAUNCH_APIS = {
    "cudaLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchCooperativeKernel",
    "cuLaunchKernel", "cuLaunchKernelEx", "cuLaunchCooperativeKernel",
    "cudaGraphLaunch", "cuGraphLaunch",
}
GRAPH_LAUNCH_APIS = {"cudaGraphLaunch", "cuGraphLaunch"}

# 上游 references/kernel-level-analysis.md 的分桶与典型来源(经验归因,不是判据)。
GAP_BUCKETS = [
    (0, 1, "<1us", "normal kernel dispatch pipeline"),
    (1, 5, "1-5us", "CUDA graph segment replay overhead"),
    (5, 10, "5-10us", "cudaGraphLaunch dispatch"),
    (10, 50, "10-50us", "light Python dispatch between graph segments"),
    (50, 100, "50-100us", "end-to-end gap between graph segment replays (incl. Python dispatch loop)"),
    (100, 500, "100-500us", "Python interpreter overhead between kernel launches"),
    (500, 1000, "500us-1ms", "heavy Python processing (tensor view chains, metadata prep)"),
    (1000, 5000, "1-5ms", "Python interpreter overhead in eager code paths"),
    (5000, float("inf"), ">5ms", "host-device sync (.item()) or runtime object creation"),
]

_VERSION_SUFFIX = re.compile(r"_v\d+$")


class UsageError(Exception):
    pass


# ---------------------------------------------------------------- schema helpers
def list_tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def resolve_table(tables, base):
    """精确名优先,其次取带 _V<n> 后缀的最新版本(不同 nsys 版本的导出表名可能带后缀)。"""
    if base in tables:
        return base
    cands = sorted((t for t in tables if re.fullmatch(re.escape(base) + r"_V\d+", t)),
                   key=lambda t: int(t.rsplit("_V", 1)[1]))
    return cands[-1] if cands else None


def table_columns(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


# ---------------------------------------------------------------- interval helpers
def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def intersect(a, b):
    """两组已合并区间的交集(双指针)。"""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append([s, e])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


class Cover:
    """已合并区间上的 [lo, hi) 覆盖长度查询。"""

    def __init__(self, merged):
        self.m = merged
        self.ends = [e for _, e in merged]

    def covered(self, lo, hi):
        i = bisect.bisect_right(self.ends, lo)
        total = 0
        while i < len(self.m) and self.m[i][0] < hi:
            s, e = self.m[i]
            total += min(e, hi) - max(s, lo)
            i += 1
        return total


def count_in(starts, lo, hi):
    return bisect.bisect_left(starts, hi) - bisect.bisect_left(starts, lo)


# ---------------------------------------------------------------- loading
def load_gpu(conn, tables, device, kernels_only, used):
    """返回 [(start, end, name, kind, correlationId)],按 start 排序。"""
    acts = []
    specs = [("CUPTI_ACTIVITY_KIND_KERNEL", "kernel")]
    if not kernels_only:
        specs += [("CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "graph"),
                  ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                  ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")]
    for base, kind in specs:
        t = resolve_table(tables, base)
        if t is None:
            continue
        cols = table_columns(conn, t)
        if not {"start", "end"} <= cols:
            if kind == "kernel":
                raise UsageError(f"{t} lacks start/end columns")
            continue
        name_col = "shortName" if "shortName" in cols else ("demangledName" if "demangledName" in cols else None)
        sel = ["a.start", "a.end"]
        sel.append("s.value" if (kind == "kernel" and name_col) else "NULL")
        sel.append("a.graphNodeId" if "graphNodeId" in cols else "NULL")
        sel.append("a.correlationId" if "correlationId" in cols else "NULL")
        sql = f"SELECT {', '.join(sel)} FROM {t} a"
        if kind == "kernel" and name_col:
            sql += f" LEFT JOIN StringIds s ON a.{name_col} = s.id"
        params = []
        if device is not None and "deviceId" in cols:
            sql += " WHERE a.deviceId = ?"
            params.append(device)
        n = 0
        for s, e, name, gnode, corr in conn.execute(sql, params):
            if s is None or e is None:
                continue
            k = "graph_node" if (kind == "kernel" and gnode is not None) else kind
            acts.append((s, e, name or f"[{kind}]", k, corr))
            n += 1
        used[kind] = {"table": t, "rows": n}
    acts.sort()
    return acts


def load_nvtx(conn, tables):
    t = resolve_table(tables, "NVTX_EVENTS")
    if t is None:
        raise UsageError("table NVTX_EVENTS not found (profile with -t cuda,nvtx)")
    cols = table_columns(conn, t)
    if "textId" in cols:
        sql = (f"SELECT n.start, n.end, COALESCE(n.text, s.value) FROM {t} n "
               "LEFT JOIN StringIds s ON n.textId = s.id")
    else:
        sql = f"SELECT n.start, n.end, n.text FROM {t} n"
    sql += " WHERE n.end IS NOT NULL AND n.end > n.start"
    return [(s, e, txt) for s, e, txt in conn.execute(sql) if txt]


def load_runtime(conn, tables):
    t = resolve_table(tables, "CUPTI_ACTIVITY_KIND_RUNTIME")
    if t is None:
        return None, None
    rows = []
    for s, e, name in conn.execute(
            f"SELECT r.start, r.end, s.value FROM {t} r LEFT JOIN StringIds s ON r.nameId = s.id"):
        if s is not None and e is not None and name:
            rows.append((s, e, _VERSION_SUFFIX.sub("", name)))
    return t, rows


# ---------------------------------------------------------------- analysis
def select_steps(nvtx, step_re, skip_first, skip_last, steady_max_batch):
    steps = []
    for s, e, txt in sorted(nvtx):
        m = step_re.search(txt)
        if not m:
            continue
        g = m.groupdict()
        pre, dec = g.get("prefill"), g.get("decode")
        if pre is None or dec is None:
            phase, dec_n = "unclassified", None
        else:
            pre_n, dec_n = int(pre), int(dec)
            if pre_n == 0 and dec_n == 0:
                continue  # 空迭代不是稳态 step
            phase = "prefill" if pre_n > 0 else "decode"
        steps.append({"start": s, "end": e, "text": txt, "phase": phase, "decode": dec_n})
    matched = len(steps)
    for i, st in enumerate(steps):  # 相邻两个 step range 之间、落在 NVTX 区间外的时间
        st["gap_after"] = max(steps[i + 1]["start"] - st["end"], 0) if i + 1 < len(steps) else None
    steps = steps[skip_first:len(steps) - skip_last if skip_last else None]
    if steady_max_batch:
        dec = [x["decode"] for x in steps if x["phase"] == "decode"]
        if dec:
            top = max(dec)
            steps = [x for x in steps if x["phase"] != "decode" or x["decode"] == top]
    return matched, steps


def bucket_of(gap_us):
    for lo, hi, label, _ in GAP_BUCKETS:
        if lo <= gap_us < hi:
            return label
    return GAP_BUCKETS[-1][2]


def phase_report(name, steps, busy, host, host_busy, launch, nccl, acts_starts, kinds_starts,
                 graph_launch_starts, host_given, runtime_ok, th):
    wall = sum(s["end"] - s["start"] for s in steps)
    b = sum(busy.covered(s["start"], s["end"]) for s in steps)
    idle = wall - b
    n = len(steps)
    rep = {"steps": n}
    if n == 0 or wall <= 0:
        rep["verdict"] = "NO_DATA"
        return rep
    per = {
        "wall_us": wall / n / 1e3, "gpu_busy_us": b / n / 1e3, "gpu_idle_us": idle / n / 1e3,
        "gpu_activities": sum(count_in(acts_starts, s["start"], s["end"]) for s in steps) / n,
    }
    gaps_after = [s["gap_after"] for s in steps if s["gap_after"] is not None]
    if gaps_after:
        per["between_steps_us"] = sum(gaps_after) / len(gaps_after) / 1e3
    idle_ratio = idle / wall
    metrics = {"gpu_idle_ratio": idle_ratio, "gpu_utilization": 1 - idle_ratio}
    votes = {"gpu_idle": {"value": idle_ratio, "op": ">", "threshold": th["idle"][name],
                          "crossed": idle_ratio > th["idle"][name]}}
    if runtime_ok:
        lt = sum(launch.covered(s["start"], s["end"]) for s in steps)
        per["launch_api_us"] = lt / n / 1e3
        metrics["launch_ratio"] = lt / wall
        votes["launch"] = {"value": lt / wall, "op": ">", "threshold": th["launch"],
                           "crossed": lt / wall > th["launch"]}
    if host_given:
        ht = sum(host.covered(s["start"], s["end"]) for s in steps)
        exp = ht - sum(host_busy.covered(s["start"], s["end"]) for s in steps)
        if ht > 0:
            per["host_total_us"], per["host_exposed_us"] = ht / n / 1e3, exp / n / 1e3
            ew, ei = exp / wall, (exp / idle if idle > 0 else 0.0)
            metrics.update({"host_exposed_of_host": exp / ht, "host_exposed_wall_ratio": ew,
                            "host_exposed_idle_ratio": ei})
            votes["host_exposed"] = {
                "value": {"wall": ew, "idle": ei}, "op": ">",
                "threshold": {"wall": th["exposed_wall"], "idle": th["exposed_idle"]},
                "crossed": ew > th["exposed_wall"] and ei > th["exposed_idle"]}
    if b > 0:
        metrics["nccl_share_of_busy"] = sum(nccl.covered(s["start"], s["end"]) for s in steps) / b
    crossed = sum(v["crossed"] for v in votes.values())
    missing = []  # 缺的票:缺哪类证据、怎么补
    if "launch" not in votes:
        missing.append({"vote": "launch", "evidence": "CUDA runtime API trace",
                        "fix": "CUPTI_ACTIVITY_KIND_RUNTIME missing: re-profile with -t cuda and re-export"})
    if "host_exposed" not in votes:
        if not host_given:
            missing.append({"vote": "host_exposed", "evidence": "host-work NVTX ranges",
                            "fix": "wrap schedule/prepare/process in fixed-name NVTX ranges, "
                                   "then rerun with --host-regex"})
        else:
            missing.append({"vote": "host_exposed", "evidence": "host-work NVTX ranges",
                            "fix": "--host-regex matched no range inside these steps: check names with "
                                   "`nsys stats -r nvtx_sum`, fix --host-regex and rerun"})
    rep.update({"per_step": per, "metrics": metrics, "votes": votes,
                "crossed_count": crossed, "applicable_count": len(votes)})
    if missing:
        rep["missing_votes"] = missing
    notes = []
    graph_launches = sum(count_in(graph_launch_starts, s["start"], s["end"]) for s in steps)
    graph_acts = sum(count_in(kinds_starts[k], s["start"], s["end"]) for s in steps
                     for k in ("graph", "graph_node"))
    if graph_launches and not graph_acts:
        rep["verdict"] = "INCONCLUSIVE"
        notes.append(f"{graph_launches} graph launches in these steps but no graph execution rows: "
                     "GPU busy time is under-counted; re-export or re-profile with --cuda-graph-trace=node")
    elif len(votes) < 2:
        rep["verdict"] = "INCONCLUSIVE"
        notes.append("fewer than 2 applicable votes; supply the evidence listed in missing_votes and rerun")
    elif len(votes) < 3 and crossed == 1:
        # 只剩 2 票时,一票越线既不够判「是」,也不能判「否」:缺的那一票可能正好越线
        rep["verdict"] = "INCONCLUSIVE"
        which = next(k for k, v in votes.items() if v["crossed"])
        lacking = ", ".join(f"{m['vote']} ({m['evidence']})" for m in missing)
        notes.append(f"only {len(votes)} of 3 votes applicable and exactly one crossed ({which}); "
                     f"missing: {lacking}. Supply it (usually host NVTX ranges + --host-regex) and rerun")
    else:
        rep["verdict"] = "YES" if crossed >= 2 else "NO"
        if rep["verdict"] == "NO" and votes["gpu_idle"]["crossed"]:
            notes.append("GPU idle crossed but no other vote did: the idle is not explained by launch "
                         "or exposed host work; read gap_buckets/top_gaps (sync, communication)")
    if metrics.get("nccl_share_of_busy", 0) > th["nccl"]:
        notes.append("NCCL share of busy time above caveat threshold: idle may be communication")
    if notes:
        rep["notes"] = notes
    return rep


def analyze(args):
    if not os.path.isfile(args.trace):
        raise UsageError(f"trace not found: {args.trace}")
    try:
        step_re = re.compile(args.step_regex)
        host_re = re.compile(args.host_regex) if args.host_regex else None
    except re.error as exc:
        raise UsageError(f"bad regex: {exc}")
    uri = pathlib.Path(os.path.abspath(args.trace)).as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise UsageError(f"cannot open {args.trace}: {exc}")
    try:
        tables = list_tables(conn)
        for req in ("StringIds", "CUPTI_ACTIVITY_KIND_KERNEL"):
            if resolve_table(tables, req) is None:
                raise UsageError(f"table {req} not found (profile with -t cuda and export sqlite)")
        used, warnings = {}, []
        acts = load_gpu(conn, tables, args.device, args.kernels_only, used)
        nvtx = load_nvtx(conn, tables)
        rt_table, runtime = load_runtime(conn, tables)
        if rt_table is None:
            warnings.append("CUPTI_ACTIVITY_KIND_RUNTIME missing: launch vote not applicable")
        else:
            used["runtime"] = {"table": rt_table, "rows": len(runtime)}
        if not acts:
            raise UsageError("no GPU activity rows (kernel/graph/memcpy/memset) in the export")

        matched, steps = select_steps(nvtx, step_re, args.skip_first, args.skip_last,
                                      args.steady_max_batch)
        if not steps:
            raise UsageError(f"no steady-state step matched --step-regex ({matched} matched "
                             "before skipping); check NVTX text with `nsys stats -r nvtx_sum`")

        if any(b["start"] < a["end"] for a, b in zip(steps, steps[1:])):
            warnings.append("step ranges overlap (several processes or ranks in one report?): "
                            "analyze one process per report")

        busy = merge([(s, e) for s, e, *_ in acts])
        busy_c = Cover(busy)
        host_m = merge([(s, e) for s, e, t in nvtx
                        if host_re and host_re.search(t) and not step_re.search(t)])
        host_c, host_busy_c = Cover(host_m), Cover(intersect(host_m, busy))
        launch_c = Cover(merge([(s, e) for s, e, nm in (runtime or []) if nm in LAUNCH_APIS]))
        graph_launch_starts = sorted(s for s, _, nm in (runtime or []) if nm in GRAPH_LAUNCH_APIS)
        nccl_c = Cover(merge([(s, e) for s, e, nm, k, _ in acts if k in ("kernel", "graph_node")
                              and "nccl" in nm.lower()]))
        acts_starts = [a[0] for a in acts]
        kinds_starts = {k: [a[0] for a in acts if a[3] == k] for k in ("graph", "graph_node")}
        th = {"idle": dict(IDLE_THRESHOLD), "launch": args.launch_threshold,
              "exposed_wall": args.exposed_wall_threshold, "exposed_idle": args.exposed_idle_threshold,
              "nccl": NCCL_CAVEAT_THRESHOLD}
        if args.idle_threshold is not None:
            th["idle"]["all"] = th["idle"]["prefill"] = args.idle_threshold
        if args.decode_idle_threshold is not None:
            th["idle"]["decode"] = args.decode_idle_threshold

        groups = {"all": steps}
        for ph in ("prefill", "decode"):
            sel = [s for s in steps if s["phase"] == ph]
            if sel:
                groups[ph] = sel
        phases = {ph: phase_report(ph, sel, busy_c, host_c, host_busy_c, launch_c, nccl_c,
                                   acts_starts, kinds_starts, graph_launch_starts,
                                   host_re is not None, rt_table is not None, th)
                  for ph, sel in groups.items()}
        if host_re is not None and not host_m:
            warnings.append("--host-regex matched no NVTX range: host_exposed vote not applicable")

        # 间隙:相邻两段 GPU 忙区间之间的空闲,按中点落在哪个 step 归阶段
        wstarts = [s["start"] for s in steps]
        buckets = {ph: {b[2]: {"count": 0, "total_us": 0.0} for b in GAP_BUCKETS} for ph in groups}
        in_steps = []
        for i in range(len(busy) - 1):
            g0, g1 = busy[i][1], busy[i + 1][0]
            mid = (g0 + g1) / 2
            k = bisect.bisect_right(wstarts, mid) - 1
            if k < 0 or mid >= steps[k]["end"]:
                continue
            us = (g1 - g0) / 1e3
            label = bucket_of(us)
            for ph in ("all", steps[k]["phase"]):
                if ph in buckets:
                    buckets[ph][label]["count"] += 1
                    buckets[ph][label]["total_us"] += us
            in_steps.append((g1 - g0, g0, g1, k))
        for ph, bk in buckets.items():
            idle_total = sum(v["total_us"] for v in bk.values())
            for v in bk.values():
                v["share_of_gap_time"] = v["total_us"] / idle_total if idle_total else 0.0
        top = sorted(in_steps, reverse=True)[:args.top_gaps]
        top_gaps = [describe_gap(conn, rt_table, acts, acts_starts, nvtx, step_re, g0, g1, steps[k])
                    for _, g0, g1, k in top]
    except sqlite3.Error as exc:  # 不是 SQLite 文件,或导出 schema 与预期不符
        raise UsageError(f"SQLite error: {exc}")
    finally:
        conn.close()

    # 任一阶段「是」则整体「是」(上游:分阶段只会抬高、不会压低结论);判「是」的阶段自身已过了
    # 图执行记录与票数检查,别的阶段证据不全不推翻它。没有「是」时,有 INCONCLUSIVE 就 INCONCLUSIVE
    verdicts = [p["verdict"] for p in phases.values()]
    if "YES" in verdicts:
        verdict = "YES"
    elif "INCONCLUSIVE" in verdicts:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NO"
    crossed = [f"{ph}.{v}" for ph, p in phases.items() for v, d in p.get("votes", {}).items()
               if d["crossed"]]
    return {
        "tool": "host_overhead.py", "trace": os.path.abspath(args.trace),
        "verdict": verdict, "crossed": crossed,
        "rule": ("per phase: INCONCLUSIVE if graph launches lack execution rows, if fewer than 2 votes "
                 "are applicable, or if fewer than 3 are applicable and exactly 1 crossed; otherwise YES "
                 "if >=2 crossed, else NO. Overall: YES if any phase YES, else INCONCLUSIVE if any phase "
                 "INCONCLUSIVE, else NO"),
        "thresholds": {"gpu_idle_ratio": th["idle"], "launch_ratio": th["launch"],
                       "host_exposed_wall_ratio": th["exposed_wall"],
                       "host_exposed_idle_ratio": th["exposed_idle"],
                       "nccl_caveat": th["nccl"], "source": THRESHOLD_SOURCE},
        "steps": {"matched": matched, "used": len(steps), "skip_first": args.skip_first,
                  "skip_last": args.skip_last, "steady_max_batch": args.steady_max_batch,
                  "by_phase": {ph: len(sel) for ph, sel in groups.items()}},
        "tables_used": used, "warnings": warnings, "phases": phases,
        "gap_buckets": {"scope": "idle gaps between merged GPU-busy intervals, midpoint inside a step",
                        "source": "TensorRT-LLM perf-host-analysis references/kernel-level-analysis.md",
                        "typical_source": {b[2]: b[3] for b in GAP_BUCKETS}, "by_phase": buckets},
        "top_gaps": top_gaps,
    }


def describe_gap(conn, rt_table, acts, acts_starts, nvtx, step_re, g0, g1, step):
    d = {"start_ns": g0, "len_us": (g1 - g0) / 1e3, "step": step["text"]}
    j = bisect.bisect_left(acts_starts, g1)
    if j < len(acts):
        _, _, name, kind, corr = acts[j]
        d["next_gpu"] = name
        if rt_table and corr is not None:
            row = conn.execute(
                f"SELECT r.start, s.value FROM {rt_table} r LEFT JOIN StringIds s ON r.nameId = s.id "
                "WHERE r.correlationId = ? LIMIT 1", (corr,)).fetchone()
            if row and row[0] is not None:
                d["issued_by"] = _VERSION_SUFFIX.sub("", row[1] or "?")
                # >0:发这段 GPU 工作的 API 在 GPU 空下来之后才被调用(host 晚了)
                d["issue_lag_us"] = (row[0] - g0) / 1e3
    if rt_table:
        rows = conn.execute(
            f"SELECT s.value, MIN(r.end, ?) - MAX(r.start, ?) FROM {rt_table} r "
            "LEFT JOIN StringIds s ON r.nameId = s.id WHERE r.start < ? AND r.end > ?",
            (g1, g0, g1, g0)).fetchall()
        agg = {}
        for nm, ov in rows:
            nm = _VERSION_SUFFIX.sub("", nm or "?")
            agg[nm] = agg.get(nm, 0) + ov
        d["host_apis_in_gap"] = [{"api": k, "overlap_us": v / 1e3}
                                 for k, v in sorted(agg.items(), key=lambda x: -x[1])[:3]]
    mid = (g0 + g1) / 2
    covering = sorted((e - s, t) for s, e, t in nvtx if s <= mid < e and not step_re.search(t))
    d["nvtx_at_gap"] = [t for _, t in covering[:3]]
    return d


def rounded(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    return x


def main(argv=None):
    p = argparse.ArgumentParser(
        description="判断 nsys 稳态 step 里 host 开销是否为瓶颈(读 nsys 导出的 SQLite,输出 JSON)。",
        epilog="退出码:0 = 否;1 = 是;2 = 用法/环境错误或 INCONCLUSIVE。"
               "阈值是起点值,本机重标后用 --*-threshold 覆盖。")
    p.add_argument("--trace", required=True, help="nsys export --type sqlite 得到的 .sqlite")
    p.add_argument("--step-regex", default=DEFAULT_STEP_REGEX,
                   help="匹配 step NVTX 文本的正则;命名分组 prefill、decode 可选(默认:%(default)s)")
    p.add_argument("--host-regex", help="host 工作 NVTX 区间名的正则(如 'schedule|prepare|process'),"
                                        "用于暴露 host 时间一票;step 区间自身会被排除")
    p.add_argument("--skip-first", type=int, default=1, help="丢掉开头几个 step(默认 1)")
    p.add_argument("--skip-last", type=int, default=1, help="丢掉结尾几个 step(默认 1)")
    p.add_argument("--steady-max-batch", action="store_true",
                   help="decode 只留 decode 请求数等于最大值的 step(两版对比时用)")
    p.add_argument("--device", type=int, help="只看这个 deviceId 的 GPU 活动")
    p.add_argument("--kernels-only", action="store_true",
                   help="忙时间只并 kernel 表(只作对照;默认还并图记录与 memcpy/memset)")
    p.add_argument("--idle-threshold", type=float, help="all/prefill 的 GPU 空闲率阈值(默认 0.30)")
    p.add_argument("--decode-idle-threshold", type=float, help="decode 的 GPU 空闲率阈值(默认 0.15)")
    p.add_argument("--launch-threshold", type=float, default=LAUNCH_THRESHOLD,
                   help="launch API 并集占墙钟的阈值(默认 %(default)s)")
    p.add_argument("--exposed-wall-threshold", type=float, default=EXPOSED_WALL_THRESHOLD,
                   help="暴露 host 时间占墙钟的阈值(默认 %(default)s)")
    p.add_argument("--exposed-idle-threshold", type=float, default=EXPOSED_IDLE_THRESHOLD,
                   help="暴露 host 时间占 GPU 空闲的阈值(默认 %(default)s)")
    p.add_argument("--top-gaps", type=int, default=10, help="列出最长的几个间隙(默认 10)")
    p.add_argument("-o", "--output", help="JSON 写到文件(默认 stdout)")
    args = p.parse_args(argv)
    if args.skip_first < 0 or args.skip_last < 0 or args.top_gaps < 0:
        p.error("--skip-first/--skip-last/--top-gaps must be >= 0")
    try:
        result = rounded(analyze(args))
    except UsageError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return {"NO": 0, "YES": 1}.get(result["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
