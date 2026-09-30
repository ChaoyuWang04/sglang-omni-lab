#!/usr/bin/env python3
"""按 SOL% 给 ncu 导出 CSV 里的每次 kernel 发射分类,列出支撑分类的指标,输出 JSON。

输入二选一(都来自 ncu --import <rep> --csv):
  --page raw      一行一次发射,表头是指标全名,第二行是单位(推荐,指标最全;stall 与 pipe 明细只有它有)
  --page details  一行一个指标,按 Section Name / Metric Name 取值;另带 ncu 规则结果
导出时加 --print-units base 最省事;不加时 us、Tbyte/s、Ghz 之类缩放单位与千分位也能识别。

分类(默认阈值 = oss-perf-investigation 判据表「kernel 分类」行,重标后用 --hi/--lo 覆盖):
  compute   Compute(SM) > hi 且 Memory < lo
  memory    Memory > hi 且 Compute < lo
  latency   两者都 < lo
  balanced  两者都在 [lo, hi]
  outside_table  以上都不是(例如 70/50),给出 leaning,两组 section 都补
  unclassified   缺 SpeedOfLight 指标
另给 ncu 自带 SpeedOfLight 规则的看法(ncu_rule_view:两者都 < 60 为 latency_issue,Waves Per SM < 1 为
small_grid;任一 >= 80 为 high_throughput;相差 >= 10 个百分点为 compute_heavier / memory_heavier;否则 balanced)。

退出码:0 = 每次发射都落进四类之一;1 = 有发射为 outside_table 或 unclassified(要人看);
        2 = 文件不存在、不是 ncu CSV、没有数据行、--kernel 写错或没匹配到、没有任何一行带 SOL 指标,
            或写不了输出文件。
只依赖 Python 标准库。
"""

import argparse
import csv
import io
import json
import os
import re
import statistics
import sys
from collections import Counter, OrderedDict

DEFAULT_HI = 60.0
DEFAULT_LO = 40.0
DEFAULT_SHORT_US = 10.0          # 单次短于它且大量发射 -> 回 nsys 看 launch
DEFAULT_OCC_HINT = 50.0          # 延迟型且 achieved occupancy 高于它 -> 看 InstructionStats
NCU_RULE = {"latency": 60.0, "high": 80.0, "balanced_gap": 10.0, "waves": 1.0}

# 逻辑指标 -> raw 页的候选全名(按顺序试)与 details 页的 (Section Name 子串, Metric Name)。
# 指标名随 ncu 版本变:例如 Issue Slots Busy 有的版本是 _active,有的是 _elapsed。
METRICS = OrderedDict([
    ("compute_sol_pct", (["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
                         [("Speed Of Light", "Compute (SM) Throughput")])),
    ("memory_sol_pct", (["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"],
                        [("Speed Of Light", "Memory Throughput")])),
    ("duration", (["gpu__time_duration.sum"], [("Speed Of Light", "Duration")])),
    ("sm_frequency", (["gpc__cycles_elapsed.avg.per_second"], [("Speed Of Light", "SM Frequency")])),
    ("dram_throughput_pct", (["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                              "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
                             [("Speed Of Light", "DRAM Throughput")])),
    ("l1tex_throughput_pct", (["l1tex__throughput.avg.pct_of_peak_sustained_active"],
                              [("Speed Of Light", "L1/TEX Cache Throughput")])),
    ("l2_throughput_pct", (["lts__throughput.avg.pct_of_peak_sustained_elapsed"],
                           [("Speed Of Light", "L2 Cache Throughput")])),
    ("l1_hit_rate_pct", (["l1tex__t_sector_hit_rate.pct"], [("Memory Workload", "L1/TEX Hit Rate")])),
    ("l2_hit_rate_pct", (["lts__t_sector_hit_rate.pct"], [("Memory Workload", "L2 Hit Rate")])),
    ("mem_busy_pct", (["gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed"],
                      [("Memory Workload", "Mem Busy")])),
    ("max_bandwidth_pct", (["gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed"],
                           [("Memory Workload", "Max Bandwidth")])),
    ("mem_pipes_busy_pct", (["sm__memory_throughput.avg.pct_of_peak_sustained_elapsed"],
                            [("Memory Workload", "Mem Pipes Busy")])),
    ("local_spilling_requests", (["derived__local_spilling_requests"],
                                 [("Memory Workload", "Local Memory Spilling Requests")])),
    ("issue_slots_busy_pct", (["sm__inst_issued.avg.pct_of_peak_sustained_active",
                               "sm__inst_issued.avg.pct_of_peak_sustained_elapsed"],
                              [("Compute Workload", "Issue Slots Busy")])),
    ("sm_busy_pct", (["sm__instruction_throughput.avg.pct_of_peak_sustained_active",
                      "sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed"],
                     [("Compute Workload", "SM Busy")])),
    ("executed_ipc_active", (["sm__inst_executed.avg.per_cycle_active"],
                             [("Compute Workload", "Executed Ipc Active")])),
    ("tensor_pipe_pct", (["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
                          "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"], [])),
    ("achieved_occupancy_pct", (["sm__warps_active.avg.pct_of_peak_sustained_active"],
                                [("Occupancy", "Achieved Occupancy")])),
    ("theoretical_occupancy_pct", (["sm__maximum_warps_per_active_cycle_pct"],
                                   [("Occupancy", "Theoretical Occupancy")])),
    ("block_limit_registers", (["launch__occupancy_limit_registers"], [("Occupancy", "Block Limit Registers")])),
    ("block_limit_shared_mem", (["launch__occupancy_limit_shared_mem"], [("Occupancy", "Block Limit Shared Mem")])),
    ("block_limit_warps", (["launch__occupancy_limit_warps"], [("Occupancy", "Block Limit Warps")])),
    ("block_limit_sm", (["launch__occupancy_limit_blocks"], [("Occupancy", "Block Limit SM")])),
    ("registers_per_thread", (["launch__registers_per_thread"], [("Launch Statistics", "Registers Per Thread")])),
    ("waves_per_sm", (["launch__waves_per_multiprocessor"], [("Launch Statistics", "Waves Per SM")])),
    ("eligible_warps_per_scheduler", (["smsp__warps_eligible.avg.per_cycle_active"],
                                      [("Scheduler Statistics", "Eligible Warps Per Scheduler")])),
    ("active_warps_per_scheduler", (["smsp__warps_active.avg.per_cycle_active"],
                                    [("Scheduler Statistics", "Active Warps Per Scheduler")])),
    ("no_eligible_pct", (["smsp__issue_inst0.avg.pct_of_peak_sustained_active"],
                         [("Scheduler Statistics", "No Eligible")])),
    ("warp_cycles_per_issued_inst", (["smsp__average_warp_latency_per_inst_issued.ratio"],
                                     [("Warp State", "Warp Cycles Per Issued Instruction")])),
])

# 每类要看的证据;缺哪项就说明还没补哪个 section。
EVIDENCE = {
    "memory": ["dram_throughput_pct", "l2_throughput_pct", "l1tex_throughput_pct", "l1_hit_rate_pct",
               "l2_hit_rate_pct", "mem_busy_pct", "max_bandwidth_pct", "mem_pipes_busy_pct",
               "local_spilling_requests"],
    "compute": ["issue_slots_busy_pct", "sm_busy_pct", "executed_ipc_active", "tensor_pipe_pct"],
    "latency": ["achieved_occupancy_pct", "theoretical_occupancy_pct", "block_limit_registers",
                "block_limit_shared_mem", "block_limit_warps", "block_limit_sm", "registers_per_thread",
                "waves_per_sm", "eligible_warps_per_scheduler", "no_eligible_pct",
                "warp_cycles_per_issued_inst"],
}
NEXT_SECTIONS = {
    "memory": ["MemoryWorkloadAnalysis"],
    "compute": ["ComputeWorkloadAnalysis", "InstructionStats"],
    "latency": ["Occupancy", "LaunchStats", "WarpStateStats", "SchedulerStats"],
}
STALL_RE = [
    re.compile(r"^smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$"),
    re.compile(r"^smsp__warp_issue_stalled_(\w+?)_per_warp_active\.pct$"),
]
PIPE_RE = [
    re.compile(r"^sm__inst_executed_pipe_(\w+)\.avg\.pct_of_peak_sustained_active$"),
    re.compile(r"^sm__pipe_(\w+)_cycles_active\.avg\.pct_of_peak_sustained_active$"),
]
TIME_TO_US = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3,
              "s": 1e6, "second": 1e6}
FREQ_TO_GHZ = {"ghz": 1.0, "mhz": 1e-3, "khz": 1e-6, "hz": 1e-9, "cycle/second": 1e-9,
               "cycle/nsecond": 1.0, "cycle/usecond": 1e-3, "cycle/msecond": 1e-6}
BASE_COLS = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream",
             "Block Size", "Grid Size", "Device", "CC"]


class InputError(Exception):
    pass


# ---------- 读取 ----------

def read_table(path):
    if not os.path.isfile(path):
        raise InputError(f"输入不存在: {path}")
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.read().splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.lstrip("﻿").startswith('"ID"')), None)
    if start is None:
        raise InputError("没找到以 \"ID\" 开头的表头行,不像 ncu --csv 的输出")
    rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    if not rows or "Kernel Name" not in rows[0]:
        raise InputError("表头里没有 Kernel Name 列,不像 ncu --csv 的输出")
    return rows, lines[:start]


def detect_number_format(values):
    """ncu 按系统 locale 输出数字:en 是 1,234.5,de 是 1.234,5。扫一遍数据定一种。"""
    comma_decimal = dot_decimal = False
    for v in values:
        if not re.fullmatch(r"-?[\d.,]+", v):
            continue
        if re.search(r"\d\.\d{3}\.\d{3}", v) or re.search(r",\d{1,2}$", v) or re.search(r",\d{4,}", v):
            comma_decimal = True
        if re.search(r"\d,\d{3},\d{3}", v) or re.search(r"\.\d{1,2}$", v) or re.search(r"\.\d{4,}", v):
            dot_decimal = True
    if comma_decimal and not dot_decimal:
        return "comma"
    return "dot"


def parse_number(text, fmt):
    if text is None:
        return None
    s = text.strip()
    if not s or not re.fullmatch(r"-?[\d.,]+(e[-+]?\d+)?", s, flags=re.I):
        return None
    if fmt == "comma":
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def load_launches(rows, fmt_opt):
    hdr = rows[0]
    if "Metric Name" in hdr and "Metric Value" in hdr:
        page = "details"
    elif any("__" in h for h in hdr):
        page = "raw"
    else:
        raise InputError("既不是 --page raw(表头是指标名)也不是 --page details(有 Metric Name/Metric Value 列)")
    idx = {h: i for i, h in enumerate(hdr)}

    def cell(r, name):
        i = idx.get(name)
        return r[i] if i is not None and i < len(r) else ""

    launches = OrderedDict()
    values_for_fmt = []
    if page == "raw":
        data = rows[1:]
        units = [""] * len(hdr)
        if data and cell(data[0], "ID") == "" and cell(data[0], "Kernel Name") == "":
            units, data = data[0], data[1:]
        for r in data:
            if not any(x.strip() for x in r):
                continue
            key = cell(r, "ID") or str(len(launches))
            rec = {"base": {c: cell(r, c) for c in BASE_COLS if c in idx}, "raw": {}, "details": {}, "rules": []}
            for i, h in enumerate(hdr):
                if "__" in h and i < len(r):
                    rec["raw"][h] = (units[i] if i < len(units) else "", r[i])
                    values_for_fmt.append(r[i])
            launches[key] = rec
    else:
        for r in rows[1:]:
            if not any(x.strip() for x in r):
                continue
            key = cell(r, "ID")
            rec = launches.setdefault(key, {"base": {c: cell(r, c) for c in BASE_COLS if c in idx},
                                            "raw": {}, "details": {}, "rules": []})
            metric = cell(r, "Metric Name")
            if metric:
                rec["details"][(cell(r, "Section Name"), metric)] = (cell(r, "Metric Unit"), cell(r, "Metric Value"))
                values_for_fmt.append(cell(r, "Metric Value"))
            elif cell(r, "Rule Name"):
                rec["rules"].append({
                    "section": cell(r, "Section Name"), "rule": cell(r, "Rule Name"),
                    "type": cell(r, "Rule Type"), "description": cell(r, "Rule Description")[:300],
                    "est_speedup_type": cell(r, "Estimated Speedup Type"),
                    "est_speedup_raw": cell(r, "Estimated Speedup"),
                })
    if not launches:
        raise InputError("CSV 里没有数据行")
    fmt = fmt_opt if fmt_opt != "auto" else detect_number_format(values_for_fmt)
    return page, launches, fmt


# ---------- 取值 ----------

def lookup(rec, logical, fmt):
    raw_names, detail_keys = METRICS[logical]
    for name in raw_names:
        if name in rec["raw"]:
            unit, text = rec["raw"][name]
            v = parse_number(text, fmt)
            if v is not None:
                return {"value": v, "unit": unit, "metric": name}
    for (sec, lab), (unit, text) in rec["details"].items():
        if lab in raw_names:  # --metrics 采的指标在 details 页以全名出现(Section 为 Command line profiler metrics)
            v = parse_number(text, fmt)
            if v is not None:
                return {"value": v, "unit": unit, "metric": lab}
    for sec_sub, label in detail_keys:
        for (sec, lab), (unit, text) in rec["details"].items():
            if lab == label and sec_sub.lower() in sec.lower():
                if logical.endswith("_pct") and unit and unit != "%":
                    continue  # Memory Workload Analysis 也有一行 Memory Throughput(byte/s)
                v = parse_number(text, fmt)
                if v is not None:
                    return {"value": v, "unit": unit, "metric": f"{sec} / {lab}"}
    return None


def normalize(logical, m):
    if m is None:
        return None
    out = dict(m)
    u = (m.get("unit") or "").strip()
    if logical == "duration":
        factor = TIME_TO_US.get(u, None if u else 1e-3)  # 空单位按 ncu 基本单位 ns
        out["value_us"] = None if factor is None else m["value"] * factor
    elif logical == "sm_frequency":
        factor = FREQ_TO_GHZ.get(u.lower(), None if u else 1e-9)
        out["value_ghz"] = None if factor is None else m["value"] * factor
    return out


def top_named(rec, patterns, fmt, n, exclude=()):
    found = {}
    for name, (unit, text) in rec["raw"].items():
        for pat in patterns:
            mt = pat.match(name)
            if mt and mt.group(1) not in exclude:
                v = parse_number(text, fmt)
                if v is not None and (mt.group(1) not in found or v > found[mt.group(1)]["value"]):
                    found[mt.group(1)] = {"name": mt.group(1), "value": v, "unit": unit, "metric": name}
                break
    return sorted(found.values(), key=lambda d: -d["value"])[:n]


# ---------- 分类 ----------

def classify(c, m, hi, lo):
    if c is None or m is None:
        return "unclassified", None
    if c > hi and m < lo:
        return "compute", None
    if m > hi and c < lo:
        return "memory", None
    if c < lo and m < lo:
        return "latency", None
    if lo <= c <= hi and lo <= m <= hi:
        return "balanced", None
    return "outside_table", ("compute" if c > m else "memory" if m > c else None)


def ncu_rule_view(c, m, waves):
    if c is None or m is None:
        return None
    if c < NCU_RULE["high"] and m < NCU_RULE["high"]:
        if c < NCU_RULE["latency"] and m < NCU_RULE["latency"]:
            if waves is not None and waves < NCU_RULE["waves"]:
                return "small_grid"
            return "latency_issue"
        if abs(c - m) >= NCU_RULE["balanced_gap"]:
            return "compute_heavier" if c > m else "memory_heavier"
        return "balanced"
    return "high_throughput"


def analyze(page, launches, fmt, args):
    kre = re.compile(args.kernel) if args.kernel else None
    results = []
    for key, rec in launches.items():
        name = rec["base"].get("Kernel Name", "")
        if kre and not kre.search(name):
            continue
        vals = {k: normalize(k, lookup(rec, k, fmt)) for k in METRICS}
        c = vals["compute_sol_pct"]["value"] if vals["compute_sol_pct"] else None
        m = vals["memory_sol_pct"]["value"] if vals["memory_sol_pct"] else None
        waves = vals["waves_per_sm"]["value"] if vals["waves_per_sm"] else None
        cls, leaning = classify(c, m, args.hi, args.lo)
        if cls in EVIDENCE:
            groups = [cls]
        elif cls == "outside_table":
            groups = ([leaning] if leaning else []) + [g for g in ("compute", "memory") if g != leaning]
        elif cls == "balanced":
            groups = ["compute", "memory"]
        else:
            groups = []
        evidence, missing = [], []
        for g in groups:
            for k in EVIDENCE[g]:
                if vals.get(k):
                    evidence.append({"key": k, **{f: vals[k][f] for f in ("value", "unit", "metric")}})
                else:
                    missing.append(k)
        next_sections = []
        for g in groups:
            next_sections += [s for s in NEXT_SECTIONS[g] if s not in next_sections]
        stalls = top_named(rec, STALL_RE, fmt, args.top, exclude=("selected",))
        pipes = top_named(rec, PIPE_RE, fmt, args.top)
        if cls == "latency" and not stalls:
            missing.append("stall_reasons(raw 页的 smsp__average_warps_issue_stalled_*)")
        if cls in ("compute", "balanced", "outside_table") and not pipes and page == "raw":
            missing.append("pipe_utilization(raw 页的 sm__inst_executed_pipe_* / sm__pipe_*_cycles_active)")
        flags = []
        dur_us = vals["duration"].get("value_us") if vals["duration"] else None
        if dur_us is not None and dur_us < args.short_us:
            flags.append({"flag": "short_kernel",
                          "note": f"单次 {dur_us:.2f} us < {args.short_us:g} us;若大量发射,是 launch 问题,回 nsys"})
        occ = vals["achieved_occupancy_pct"]["value"] if vals["achieved_occupancy_pct"] else None
        if cls == "latency" and occ is not None and occ > args.occ_hint:
            flags.append({"flag": "instruction_bound_hint",
                          "note": f"延迟型但 achieved occupancy {occ:.1f}% > {args.occ_hint:g}%:补 InstructionStats"})
            if "InstructionStats" not in next_sections:
                next_sections.append("InstructionStats")
        if waves is not None and waves < 1:
            flags.append({"flag": "small_grid", "note": f"Waves Per SM {waves:g} < 1:网格填不满所有 SM"})
        spill = vals["local_spilling_requests"]["value"] if vals["local_spilling_requests"] else None
        if spill:
            flags.append({"flag": "local_spilling", "note": f"Local Memory Spilling Requests = {spill:g};"
                          "与 cuobjdump -res-usage 的 LOCAL/STACK 对照"})
        rules = []
        for r in rec["rules"]:
            r = dict(r)
            r["est_speedup_pct"] = parse_number(r.pop("est_speedup_raw"), fmt)
            rules.append(r)
        rules.sort(key=lambda r: -(r["est_speedup_pct"] or 0))
        results.append({
            "id": key, "kernel": name,
            "block_size": rec["base"].get("Block Size"), "grid_size": rec["base"].get("Grid Size"),
            "cc": rec["base"].get("CC"),
            "class": cls, "leaning": leaning,
            "compute_sol_pct": c, "memory_sol_pct": m,
            "duration_us": dur_us,
            "sm_frequency_ghz": vals["sm_frequency"].get("value_ghz") if vals["sm_frequency"] else None,
            "ncu_rule_view": ncu_rule_view(c, m, waves),
            "evidence": evidence, "missing": missing, "next_sections": next_sections,
            "top_stalls": stalls, "top_pipes": pipes, "flags": flags, "ncu_rules": rules[:args.top],
        })
    return results


def summarize(results):
    by = OrderedDict()
    for r in results:
        by.setdefault(r["kernel"], []).append(r)
    out = []
    for name, rs in by.items():
        classes = Counter(r["class"] for r in rs)

        def med(k):
            xs = [r[k] for r in rs if r[k] is not None]
            return statistics.median(xs) if xs else None
        out.append({"kernel": name, "launches": len(rs), "classes": dict(classes),
                    "consistent": len(classes) == 1,
                    "median_compute_sol_pct": med("compute_sol_pct"),
                    "median_memory_sol_pct": med("memory_sol_pct"),
                    "median_duration_us": med("duration_us")})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description="按 SOL% 给 ncu CSV(--page raw 或 --page details)里的每次 kernel 发射分类,"
                    "列出支撑指标与还该补的 section,输出 JSON。"
                    "退出码 0 = 全部落进四类;1 = 有表外或缺 SOL 的发射;2 = 输入错误。",
        epilog="导出:ncu --import <rep>.ncu-rep --page raw --csv --print-units base > raw.csv")
    p.add_argument("csv", help="ncu --import <rep> --csv --page raw|details 的输出文件")
    p.add_argument("--kernel", help="只看 Kernel Name 匹配这个正则的发射")
    p.add_argument("--hi", type=float, default=DEFAULT_HI, help="高阈值,%%(默认 60)")
    p.add_argument("--lo", type=float, default=DEFAULT_LO, help="低阈值,%%(默认 40)")
    p.add_argument("--short-us", type=float, default=DEFAULT_SHORT_US,
                   help="单次时长低于它标 short_kernel(默认 10 us)")
    p.add_argument("--occ-hint", type=float, default=DEFAULT_OCC_HINT,
                   help="延迟型且 achieved occupancy 高于它时提示指令受限(默认 50%%)")
    p.add_argument("--decimal", choices=["auto", "dot", "comma"], default="auto",
                   help="数字格式:dot = 1,234.5;comma = 1.234,5;默认自动识别")
    p.add_argument("--top", type=int, default=3, help="stall、pipe、ncu 规则各列前几项(默认 3)")
    p.add_argument("-o", "--output", help="JSON 写到文件(默认 stdout)")
    args = p.parse_args(argv)
    if not args.lo < args.hi:
        p.error("--lo 必须小于 --hi")

    try:
        rows, preamble = read_table(args.csv)
        page, launches, fmt = load_launches(rows, args.decimal)
        results = analyze(page, launches, fmt, args)
        if not results:
            raise InputError(f"--kernel {args.kernel!r} 没有匹配到任何发射")
        if all(r["compute_sol_pct"] is None or r["memory_sol_pct"] is None for r in results):
            raise InputError("没有任何发射带 SpeedOfLight 指标(sm__throughput / gpu__compute_memory_throughput);"
                             "采集时加 --section SpeedOfLight")
    except (InputError, csv.Error, UnicodeError, OSError) as e:
        print(json.dumps({"error": str(e), "input": args.csv}, ensure_ascii=False), file=sys.stderr)
        return 2
    except re.error as e:
        print(json.dumps({"error": f"--kernel 正则写错了: {e}", "input": args.csv}, ensure_ascii=False),
              file=sys.stderr)
        return 2

    bad = [r for r in results if r["class"] in ("outside_table", "unclassified")]
    report = {
        "input": os.path.abspath(args.csv), "page": page, "number_format": fmt,
        "preamble": [ln for ln in preamble if ln.strip()][:10],
        "thresholds": {"hi": args.hi, "lo": args.lo, "short_us": args.short_us, "occ_hint": args.occ_hint,
                       "ncu_rule": NCU_RULE},
        "kernels": summarize(results), "launches": results,
        "verdict": "needs_review" if bad else "classified",
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError as e:
            print(json.dumps({"error": f"写不了输出文件: {e}"}, ensure_ascii=False), file=sys.stderr)
            return 2
    else:
        print(text)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
