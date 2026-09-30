#!/usr/bin/env python3
"""比较两次运行的同名指标:逐步带符号相对差、最大偏离的 step、按阈值判定,并生成可贴进 PR 的 Markdown 表。

只依赖 Python 标准库。

输入是两份指标日志(先基线、后候选),每条记录一个 step、若干同名指标:
  - JSON lines:每行一个 JSON 对象,含 step 键(默认 "step",用 --step-key 改)
  - CSV:首行表头,含 step 列
  - golden JSON:{"<指标>": {"values": {"<step>": <值>, ...}}, ...}
  - JSON 数组:[{...}, {...}],每个元素同 JSON lines 的一行

相对差 = (cand - base) / |base|,正值表示候选更大。基线绝对值小于 --zero-eps 的 step、
任一侧为 NaN 或 Inf 的 step 不进统计;前者另计「基线为零而候选非零」的步数,后者另计「只有一侧非有限」
的步数,有复审线的指标出现后者直接判 review。

判定按指标的阈值(相对差,小数;1e-4 即 0.01%),对 --stat 选的统计量取绝对值后比较:
  noise        <= 噪声线
  above-noise  高于噪声线、不超过复审线
  within       没设噪声线、不超过复审线
  review       超过复审线,或只有一侧出现 NaN/Inf
  report-only  这个指标没有复审线,只报告
  no-data      没有可比的 step
退出码:0 = 有被判定的指标且都没越线;1 = 至少一个指标 review;
       2 = 用法、输入或格式错误,写不出结果,或没有任何指标被判定(全是 report-only 或 no-data)。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys

SIGN_CONVENTION = "(cand - base) / |base|; positive means the candidate is higher"

# (噪声线, 复审线),相对差的小数。复审线为 None 表示只报告。出处见 skill 目录的 SOURCES.md。
DEFAULT_THRESHOLDS = {
    "lm loss": (1e-4, 1e-3),
    "num-zeros": (None, 1e-3),
    "mem-allocated-bytes": (None, 1e-4),
    "mem-max-allocated-bytes": (None, 1e-4),
    "iteration-time": (None, None),
}

STAT_FIELDS = {"mean": "mean_rel", "meanabs": "mean_abs_rel", "maxabs": "max_abs_rel"}
STAT_TITLES = {
    "mean": "the signed mean over compared steps",
    "meanabs": "the mean of |rel. diff| over compared steps",
    "maxabs": "the largest |rel. diff| over compared steps",
}


class InputError(Exception):
    pass


# ---------------------------------------------------------------- 解析

def _to_float(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _to_step(value):
    f = _to_float(value)
    if f is None or not math.isfinite(f):
        return None
    return int(f) if f.is_integer() else f


def _looks_golden(obj):
    return (isinstance(obj, dict) and bool(obj)
            and all(isinstance(v, dict) and isinstance(v.get("values"), dict) for v in obj.values()))


def detect_format(path, text):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return "csv"
    if ext in (".jsonl", ".ndjson"):
        return "jsonl"
    stripped = text.lstrip()
    if stripped[:1] in ("{", "["):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return "jsonl"
        if _looks_golden(obj):
            return "golden"
        if isinstance(obj, list):
            return "jsonarray"
        return "jsonl"
    return "csv"


def _parse_jsonl(text, where):
    rows = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{where}: line {lineno} is not valid JSON ({exc.msg})") from None
        if not isinstance(obj, dict):
            raise InputError(f"{where}: line {lineno} is not a JSON object")
        rows.append(obj)
    return rows


def _parse_json_whole(text, where):
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(f"{where}: not valid JSON ({exc.msg} at line {exc.lineno})") from None


def _parse_csv(text, where, step_key):
    try:
        reader = csv.DictReader(io.StringIO(text))
        fields = reader.fieldnames
        if not fields:
            raise InputError(f"{where}: CSV has no header row")
        if step_key not in fields:
            raise InputError(f"{where}: CSV header has no {step_key!r} column (columns: {', '.join(fields)})")
        return list(reader)
    except csv.Error as exc:
        raise InputError(f"{where}: malformed CSV ({exc})") from None


def _rows_to_series(rows, step_key, where):
    series = {}
    seen = set()
    info = {"records": 0, "records_without_step": 0, "duplicate_steps": 0}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise InputError(f"{where}: record {index} is not an object")
        info["records"] += 1
        step = _to_step(row.get(step_key))
        if step is None:
            info["records_without_step"] += 1
            continue
        if step in seen:
            info["duplicate_steps"] += 1
        seen.add(step)
        for key, value in row.items():
            if key == step_key or not isinstance(key, str):
                continue
            f = _to_float(value)
            if f is not None:
                series.setdefault(key, {})[step] = f
    if not seen:
        raise InputError(f"{where}: no record carries a usable {step_key!r} value")
    info["steps"] = len(seen)
    return series, info


def _golden_to_series(obj, where):
    if not _looks_golden(obj):
        raise InputError(f'{where}: not a golden-value JSON ({{"<metric>": {{"values": {{...}}}}}})')
    series = {}
    steps = set()
    for metric, block in obj.items():
        for key, value in block["values"].items():
            step, f = _to_step(key), _to_float(value)
            if step is None or f is None:
                continue
            series.setdefault(metric, {})[step] = f
            steps.add(step)
    if not steps:
        raise InputError(f"{where}: golden-value JSON has no numeric values")
    return series, {"records": None, "records_without_step": 0, "duplicate_steps": 0, "steps": len(steps)}


def load(path, fmt, step_key):
    if not os.path.isfile(path):
        raise InputError(f"input not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"{path}: cannot read as UTF-8 text ({exc})") from None
    if not text.strip():
        raise InputError(f"{path}: file is empty")
    used = detect_format(path, text) if fmt == "auto" else fmt
    if used == "csv":
        series, info = _rows_to_series(_parse_csv(text, path, step_key), step_key, path)
    elif used == "jsonl":
        series, info = _rows_to_series(_parse_jsonl(text, path), step_key, path)
    elif used == "jsonarray":
        obj = _parse_json_whole(text, path)
        if not isinstance(obj, list):
            raise InputError(f"{path}: expected a JSON array of records")
        series, info = _rows_to_series(obj, step_key, path)
    else:  # golden
        series, info = _golden_to_series(_parse_json_whole(text, path), path)
    info["format"] = used
    return series, info


# ---------------------------------------------------------------- 阈值

def _parse_levels(spec, what):
    spec = spec.strip()
    if spec.lower() == "off":
        return None, None
    noise_text, sep, review_text = spec.rpartition(":")
    if not sep:
        noise_text, review_text = "", spec
    try:
        noise = float(noise_text) if noise_text.strip() else None
        review = float(review_text)
    except ValueError:
        raise InputError(f"{what}: expected [NOISE:]REVIEW or off, got {spec!r}") from None
    for v in (noise, review):
        if v is not None and (not math.isfinite(v) or v < 0):
            raise InputError(f"{what}: thresholds must be finite and >= 0, got {spec!r}")
    if noise is not None and noise > review:
        raise InputError(f"{what}: noise threshold {noise} exceeds review threshold {review}")
    return noise, review


def build_thresholds(args):
    table = {} if args.no_default_thresholds else {k: v + ("default",) for k, v in DEFAULT_THRESHOLDS.items()}
    for spec in args.threshold or []:
        name, sep, levels = spec.rpartition("=")
        if not sep or not name.strip():
            raise InputError(f"--threshold: expected NAME=[NOISE:]REVIEW or NAME=off, got {spec!r}")
        table[name.strip()] = _parse_levels(levels, f"--threshold {name.strip()}") + ("cli",)
    fallback = None
    if args.default_threshold:
        fallback = _parse_levels(args.default_threshold, "--default-threshold") + ("cli-default",)
    return table, fallback


# ---------------------------------------------------------------- 比较

def compare_metric(base, cand, zero_eps):
    per_step = []
    counts = {"skipped_zero_base": 0, "zero_base_nonzero_cand": 0,
              "skipped_nonfinite_both": 0, "nonfinite_one_side": 0}
    for step in sorted(set(base) & set(cand)):
        b, c = base[step], cand[step]
        fin_b, fin_c = math.isfinite(b), math.isfinite(c)
        if not (fin_b and fin_c):
            counts["nonfinite_one_side" if fin_b != fin_c else "skipped_nonfinite_both"] += 1
            continue
        if abs(b) < zero_eps:
            counts["skipped_zero_base"] += 1
            if abs(c) >= zero_eps:
                counts["zero_base_nonzero_cand"] += 1
            continue
        per_step.append((step, b, c, (c - b) / abs(b)))
    out = {"n_steps": len(per_step), **counts,
           "steps_only_in_base": len(set(base) - set(cand)),
           "steps_only_in_cand": len(set(cand) - set(base))}
    if per_step:
        rels = [r for _, _, _, r in per_step]
        worst = max(per_step, key=lambda t: abs(t[3]))
        out.update({
            "step_range": [per_step[0][0], per_step[-1][0]],
            "mean_rel": sum(rels) / len(rels),
            "mean_abs_rel": sum(abs(r) for r in rels) / len(rels),
            "max_abs_rel": abs(worst[3]),
            "max_abs_step": worst[0],
            "max_abs_base": worst[1],
            "max_abs_cand": worst[2],
        })
    else:
        out.update({"step_range": None, "mean_rel": None, "mean_abs_rel": None, "max_abs_rel": None,
                    "max_abs_step": None, "max_abs_base": None, "max_abs_cand": None})
    return out, per_step


def judge(stats, noise, review, stat):
    if review is not None and stats["nonfinite_one_side"] > 0:
        return "review"
    if stats["n_steps"] == 0:
        return "no-data"
    if review is None:
        return "report-only"
    value = abs(stats[STAT_FIELDS[stat]])
    if value > review:
        return "review"
    if noise is None:
        return "within"
    return "noise" if value <= noise else "above-noise"


# ---------------------------------------------------------------- 输出

def _pct(x, signed):
    if x is None:
        return "—"
    if x == 0:
        return "`0%`"
    return f"`{100 * x:+.6f}%`" if signed else f"`{100 * x:.6f}%`"


def _pct_plain(x):
    return "—" if x is None else f"{100 * x:g}%"


def render_markdown(result):
    base, cand = result["base"]["label"], result["cand"]["label"]
    lines = [
        f"Signed relative difference `(cand - base) / |base|` per metric, **{cand}** vs **{base}** "
        f"(positive = {cand} is higher). Verdicts use {STAT_TITLES[result['stat']]}.",
        "",
        "| Metric | Steps | Mean rel. diff | Max \\|rel. diff\\| (step) | Threshold (noise / review) | Verdict |",
        "| --- | --: | --: | --: | --- | --- |",
    ]
    notes = []
    for name in sorted(result["metrics"]):
        m = result["metrics"][name]
        th = m["threshold"]
        th_cell = "report only" if th["review"] is None else f"{_pct_plain(th['noise'])} / {_pct_plain(th['review'])}"
        if m["max_abs_rel"] is None or m["max_abs_rel"] == 0:
            worst = _pct(m["max_abs_rel"], False)
        else:
            worst = f"{_pct(m['max_abs_rel'], False)} (step {m['max_abs_step']})"
        lines.append(f"| `{name}` | {m['n_steps']} | {_pct(m['mean_rel'], True)} | {worst} | {th_cell} | {m['verdict']} |")
        skipped = []
        if m["skipped_zero_base"]:
            skipped.append(f"{m['skipped_zero_base']} with |base| < {result['zero_eps']:g}"
                           + (f" ({m['zero_base_nonzero_cand']} of them non-zero in {cand})"
                              if m["zero_base_nonzero_cand"] else ""))
        if m["skipped_nonfinite_both"]:
            skipped.append(f"{m['skipped_nonfinite_both']} non-finite in both runs")
        if m["nonfinite_one_side"]:
            skipped.append(f"{m['nonfinite_one_side']} non-finite in only one run")
        if skipped:
            notes.append(f"- `{name}`: skipped steps: " + "; ".join(skipped) + ".")
    for side, key in ((base, "only_in_base"), (cand, "only_in_cand")):
        if result[key]:
            notes.append(f"- Metrics logged only by {side}: " + ", ".join(f"`{n}`" for n in result[key]) + ".")
    if notes:
        lines += [""] + notes
    return "\n".join(lines) + "\n"


def run(args):
    if args.zero_eps < 0 or not math.isfinite(args.zero_eps):
        raise InputError("--zero-eps must be finite and >= 0")
    thresholds, fallback = build_thresholds(args)
    base_series, base_info = load(args.base, args.format, args.step_key)
    cand_series, cand_info = load(args.cand, args.format, args.step_key)

    names_base, names_cand = set(base_series), set(cand_series)
    if args.metric:
        wanted = list(dict.fromkeys(args.metric))
        missing = [n for n in wanted if n not in names_base or n not in names_cand]
        if missing:
            raise InputError("requested metric(s) missing from at least one run: " + ", ".join(missing))
        shared = wanted
        only_base, only_cand = [], []
    else:
        shared = sorted(names_base & names_cand)
        only_base, only_cand = sorted(names_base - names_cand), sorted(names_cand - names_base)
    if not shared:
        raise InputError("the two runs share no metric names")

    metrics, notes = {}, []
    for name in shared:
        stats, per_step = compare_metric(base_series[name], cand_series[name], args.zero_eps)
        noise, review, source = thresholds.get(name) or fallback or (None, None, "none")
        stats["threshold"] = {"noise": noise, "review": review, "source": source}
        stats["verdict"] = judge(stats, noise, review, args.stat)
        if not args.no_steps:
            stats["steps"] = [list(t) for t in per_step]
        metrics[name] = stats
        if stats["zero_base_nonzero_cand"]:
            notes.append(f"{name}: {stats['zero_base_nonzero_cand']} step(s) have a zero base and a non-zero "
                         "candidate; they are excluded from the relative difference, inspect them directly")
        if stats["nonfinite_one_side"]:
            notes.append(f"{name}: {stats['nonfinite_one_side']} step(s) are NaN/Inf in only one run")

    verdicts = [m["verdict"] for m in metrics.values()]
    if "review" in verdicts:
        overall = "review"
    elif any(v in ("noise", "above-noise", "within") for v in verdicts):
        overall = "pass"
    else:
        overall = "inconclusive"
        notes.append("no metric was judged: give --threshold NAME=[NOISE:]REVIEW or --default-threshold")

    result = {
        "tool": "metric_diff",
        "base": {"path": args.base, "label": args.label_base, **base_info},
        "cand": {"path": args.cand, "label": args.label_cand, **cand_info},
        "sign_convention": SIGN_CONVENTION,
        "stat": args.stat,
        "zero_eps": args.zero_eps,
        "verdict": overall,
        "metrics": metrics,
        "only_in_base": only_base,
        "only_in_cand": only_cand,
        "notes": notes,
    }
    result["markdown"] = render_markdown(result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(
        description="比较两次运行的同名指标:逐步带符号相对差 (cand - base)/|base|、最大偏离的 step、"
                    "按阈值判定,输出 JSON 与 Markdown 摘要表。",
        epilog="默认阈值(相对差小数):lm loss 噪声 1e-4、复审 1e-3;num-zeros 复审 1e-3;"
               "mem-allocated-bytes 与 mem-max-allocated-bytes 复审 1e-4;iteration-time 只报告;"
               "其余指标只报告,除非给 --threshold 或 --default-threshold。"
               "退出码:0 = 通过;1 = 有指标 review;2 = 用法或输入错误、写不出结果,或没有指标被判定。")
    p.add_argument("base", help="基线运行的指标日志")
    p.add_argument("cand", help="候选运行的指标日志")
    p.add_argument("--format", choices=["auto", "jsonl", "csv", "golden", "jsonarray"], default="auto",
                   help="两份输入的格式(默认按扩展名与内容判断)")
    p.add_argument("--step-key", default="step", help="step 的键名或列名(默认 %(default)s)")
    p.add_argument("--metric", action="append", help="只比较这个指标;可重复。缺在任一侧即报错")
    p.add_argument("--threshold", action="append", metavar="NAME=[NOISE:]REVIEW",
                   help="给某指标设阈值(相对差小数,1e-3 即 0.1%%),NAME=off 表示只报告;可重复,覆盖默认值")
    p.add_argument("--default-threshold", metavar="[NOISE:]REVIEW",
                   help="没有专门阈值的指标用这个阈值")
    p.add_argument("--no-default-thresholds", action="store_true", help="不用内置的默认阈值表")
    p.add_argument("--stat", choices=sorted(STAT_FIELDS), default="mean",
                   help="拿哪个统计量比阈值:mean 为带符号均值(默认),meanabs 为绝对值均值,maxabs 为最大绝对值")
    p.add_argument("--zero-eps", type=float, default=1e-12,
                   help="|基线| 小于它的 step 不算相对差(默认 %(default)g)")
    p.add_argument("--label-base", default="base", help="表里基线的名字(默认 %(default)s)")
    p.add_argument("--label-cand", default="cand", help="表里候选的名字(默认 %(default)s)")
    p.add_argument("--md", help="把 Markdown 摘要表写到这个文件")
    p.add_argument("-o", "--output", help="JSON 写到文件(默认 stdout)")
    p.add_argument("--no-steps", action="store_true", help="JSON 里不放逐步明细")
    args = p.parse_args(argv)
    try:
        result = run(args)
    except InputError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2, ensure_ascii=False)
    try:
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        else:
            print(text)
        if args.md:
            with open(args.md, "w", encoding="utf-8") as f:
                f.write(result["markdown"])
    except OSError as exc:  # 写不出结果是环境错误,不能以 1 冒充 review
        print(json.dumps({"error": f"cannot write output: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2
    return {"pass": 0, "review": 1}.get(result["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
