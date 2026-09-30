#!/usr/bin/env python3
"""按层序比较两个目录里的张量 dump,报告每层的误差与余弦相似度,指出第一个越过阈值的层。

依赖 torch(加载 .pt);没有 torch 时退出码 2。

输入:参照目录与被测目录,各放若干 .pt,文件名(去掉 .pt 的相对路径)即层名,两侧同名的才比较。
.pt 里可以是一个张量,也可以是 dict / list / tuple(逐级展开成 <层名>/<键>)。
层序:给 --order 时按该文件逐行列出的名字;否则按名字的自然序(数字按数值比)。
自然序只在名字自带执行顺序时才对,例如 dump 时加零填充的序号前缀 000_embed、001_layer0.attn_out。

逐项计算(float64):
  max_abs  = max|test - ref|
  rel      = max_abs / max(max|ref|, 1e-12)
  cos      = <test, ref> / (|test| |ref|)
  identical:两侧形状、dtype 相同且逐位相等
  equal:同 dtype 时即 identical;dtype 不同时为数值完全相等(max_abs 为 0 且非有限值位置相同)
越线条件(任一):形状不同;整数张量不相等;NaN/+Inf/-Inf 位置不同;cos < --cos-min;rel > --rel-max。
加 --bitwise 时,任何数值差异都算越线。
另报:第一个不相等(equal 为假)的项、第一个 cos 比上一项掉超过 --drop-max 的项;越线项注明上一项是否相等
(上一项相等而本项不同,即「输入相同、输出不同」)。

退出码:0 = 没有越线;1 = 找到越线的层;2 = 用法或环境错误(目录不存在、没有 torch、
没有同名项、文件无法加载、写不出结果)。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

DEFAULT_COS_MIN = 0.99
DEFAULT_REL_MAX = 0.10
DEFAULT_DROP_MAX = 0.001


class InputError(Exception):
    pass


def natural_key(name):
    return [(0, int(part), "") if part.isdigit() else (1, 0, part) for part in re.split(r"(\d+)", name) if part]


def list_files(root, pattern_suffix):
    found = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(pattern_suffix):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root)[: -len(pattern_suffix)]
                found[rel.replace(os.sep, "/")] = full
    return found


def flatten(obj, prefix, torch, out, skipped):
    if torch.is_tensor(obj):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            flatten(value, f"{prefix}/{key}", torch, out, skipped)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            flatten(value, f"{prefix}/{i}", torch, out, skipped)
    else:
        skipped.append(prefix)


def load_dir(root, torch):
    files = list_files(root, ".pt")
    entries, skipped = {}, []
    for name, path in files.items():
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 — 任何加载失败都是输入错误
            raise InputError(f"cannot load {path} with weights_only=True ({type(exc).__name__}: {exc}); "
                             "save plain tensors or dicts of tensors") from None
        flatten(obj, name, torch, entries, skipped)
    return entries, skipped, len(files)


def order_entries(names, order_file):
    if not order_file:
        return sorted(names, key=natural_key), [], [], "natural"
    if not os.path.isfile(order_file):
        raise InputError(f"order file not found: {order_file}")
    with open(order_file, encoding="utf-8") as f:
        listed = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    listed = list(dict.fromkeys(listed))
    present = set(names)
    ordered = [n for n in listed if n in present]
    unlisted = sorted(present - set(ordered), key=natural_key)
    unknown = [n for n in listed if n not in present]
    return ordered + unlisted, unlisted, unknown, "file"


def bit_equal(ref, test, torch):
    """同 dtype、同形状且逐字节相等(NaN 与 NaN 按位比较,-0.0 与 0.0 视为不同)。"""
    if ref.dtype != test.dtype or ref.shape != test.shape:
        return False
    try:
        return bool(torch.equal(ref.contiguous().reshape(-1).view(torch.uint8),
                                test.contiguous().reshape(-1).view(torch.uint8)))
    except (RuntimeError, TypeError):
        return bool(torch.equal(ref, test))


def compare_pair(ref, test, torch):
    item = {
        "shape_ref": list(ref.shape), "shape_test": list(test.shape),
        "dtype_ref": str(ref.dtype).replace("torch.", ""), "dtype_test": str(test.dtype).replace("torch.", ""),
        "identical": False, "equal": False, "max_abs": None, "mean_abs": None, "rel": None, "cos": None,
        "nonfinite_ref": 0, "nonfinite_test": 0, "status": "ok",
    }
    if ref.shape != test.shape:
        item["status"] = "shape_mismatch"
        return item
    item["identical"] = bit_equal(ref, test, torch)
    item["equal"] = item["identical"]
    if not (ref.is_floating_point() or ref.is_complex() or test.is_floating_point() or test.is_complex()):
        item["status"] = "ok" if torch.equal(ref.to(torch.int64), test.to(torch.int64)) else "int_mismatch"
        item["max_abs"] = float((ref.to(torch.int64) - test.to(torch.int64)).abs().max().item()) if ref.numel() else 0.0
        item["equal"] = item["identical"] or (ref.dtype != test.dtype and item["status"] == "ok")
        return item
    if ref.is_complex() or test.is_complex():
        a, b = torch.view_as_real(ref.to(torch.complex128)), torch.view_as_real(test.to(torch.complex128))
    else:
        a, b = ref.to(torch.float64), test.to(torch.float64)
    a, b = a.flatten(), b.flatten()
    fin_a, fin_b = torch.isfinite(a), torch.isfinite(b)
    item["nonfinite_ref"] = int((~fin_a).sum().item())
    item["nonfinite_test"] = int((~fin_b).sum().item())
    same_masks = (torch.equal(torch.isnan(a), torch.isnan(b))
                  and torch.equal(a == math.inf, b == math.inf)
                  and torch.equal(a == -math.inf, b == -math.inf))
    if not same_masks:
        item["status"] = "nonfinite_mismatch"
    both = fin_a & fin_b
    a, b = a[both], b[both]
    if a.numel() == 0:
        item.update({"max_abs": 0.0, "mean_abs": 0.0, "rel": 0.0, "cos": None})
        item["equal"] = item["identical"] or (ref.dtype != test.dtype and item["status"] == "ok")
        return item
    diff = (b - a).abs()
    max_abs = float(diff.max().item())
    ref_scale = float(a.abs().max().item())
    na, nb = float(a.norm().item()), float(b.norm().item())
    if na == 0.0 and nb == 0.0:
        cos = 1.0
    elif na == 0.0 or nb == 0.0:
        cos = None
    else:
        cos = float((a * b).sum().item()) / (na * nb)
    item.update({"max_abs": max_abs, "mean_abs": float(diff.mean().item()),
                 "rel": max_abs / max(ref_scale, 1e-12), "cos": cos})
    if ref.dtype != test.dtype:
        item["equal"] = item["status"] == "ok" and max_abs == 0.0
    return item


def crossed(item, args):
    if item["status"] != "ok":
        return True
    if args.bitwise:
        return not item["equal"]
    if item["cos"] is None:
        return item["max_abs"] not in (None, 0.0)
    return item["cos"] < args.cos_min or (item["rel"] is not None and item["rel"] > args.rel_max)


def analyze(args):
    for d in (args.ref_dir, args.test_dir):
        if not os.path.isdir(d):
            raise InputError(f"directory not found: {d}")
    try:
        import torch  # noqa: PLC0415 — 延迟导入,让 --help 与参数检查不依赖 torch
    except ImportError:
        raise InputError("torch is not importable in this Python; run where torch is installed") from None

    ref, ref_skipped, ref_files = load_dir(args.ref_dir, torch)
    test, test_skipped, test_files = load_dir(args.test_dir, torch)
    if not ref_files or not test_files:
        raise InputError(f"no .pt files under {args.ref_dir if not ref_files else args.test_dir}")
    common = set(ref) & set(test)
    if not common:
        raise InputError("the two directories share no entry names")
    names, unlisted, unknown, order_source = order_entries(common, args.order)

    entries, first_crossing, first_unequal, first_drop = [], None, None, None
    prev = None
    for name in names:
        item = {"name": name, **compare_pair(ref[name], test[name], torch)}
        item["crossed"] = crossed(item, args)
        item["prev_equal"] = None if prev is None else prev["equal"]
        if first_unequal is None and not item["equal"]:
            first_unequal = name
        if (first_drop is None and prev is not None and prev["cos"] is not None and item["cos"] is not None
                and prev["cos"] - item["cos"] > args.drop_max):
            first_drop = {"name": name, "prev": prev["name"], "drop": prev["cos"] - item["cos"]}
        if first_crossing is None and item["crossed"]:
            first_crossing = item
        entries.append(item)
        prev = item

    notes = []
    if order_source == "natural":
        notes.append("layer order is the natural sort of names; pass --order if names do not encode execution order")
    if unlisted:
        notes.append(f"{len(unlisted)} common entries are not in the order file and were appended at the end")
    if unknown:
        notes.append(f"{len(unknown)} names in the order file match no common entry, e.g. {unknown[:3]}")
    if first_crossing is not None and first_crossing["prev_equal"] is False:
        notes.append("the previous entry already differs: the first crossing inherits upstream drift; "
                     "look at first_unequal and first_cos_drop too")
    return {
        "tool": "first_divergence",
        "ref_dir": args.ref_dir,
        "test_dir": args.test_dir,
        "criteria": {"bitwise": args.bitwise, "cos_min": args.cos_min, "rel_max": args.rel_max,
                     "drop_max": args.drop_max, "rel_definition": "max|test-ref| / max|ref|",
                     "precision": "float64"},
        "order_source": order_source,
        "verdict": "diverged" if first_crossing else "within",
        "first_crossing": first_crossing and {k: first_crossing[k] for k in
                                              ("name", "status", "cos", "rel", "max_abs", "prev_equal")},
        "first_unequal": first_unequal,
        "first_cos_drop": first_drop,
        "only_in_ref": sorted(set(ref) - common, key=natural_key),
        "only_in_test": sorted(set(test) - common, key=natural_key),
        "non_tensor_entries": {"ref": ref_skipped, "test": test_skipped},
        "entries": entries,
        "notes": notes,
    }


def clean(obj):
    """把非有限浮点换成 None,保证输出是标准 JSON。"""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def main(argv=None):
    p = argparse.ArgumentParser(
        description="按层序比较两个目录的 .pt 张量 dump(float64 算 max_abs、rel、cos),"
                    "指出第一个越过阈值的层。需要 torch。",
        epilog="退出码:0 = 没有越线;1 = 找到越线的层;2 = 用法或环境错误。"
               "默认阈值 cos < 0.99 或 rel > 0.10 越线,出处见 SOURCES.md。")
    p.add_argument("ref_dir", help="参照实现的 dump 目录")
    p.add_argument("test_dir", help="被测实现的 dump 目录")
    p.add_argument("--order", help="层序文件:每行一个名字(去掉 .pt 的相对路径,dict 展开后为 名字/键),# 开头为注释")
    p.add_argument("--cos-min", type=float, default=DEFAULT_COS_MIN, help="cos 低于它越线(默认 %(default)s)")
    p.add_argument("--rel-max", type=float, default=DEFAULT_REL_MAX,
                   help="rel = max_abs / max|ref| 高于它越线(默认 %(default)s)")
    p.add_argument("--bitwise", action="store_true", help="任何数值差异都算越线(批不变、同 kernel 对比时用)")
    p.add_argument("--drop-max", type=float, default=DEFAULT_DROP_MAX,
                   help="相邻两项 cos 掉幅超过它时报 first_cos_drop(默认 %(default)s)")
    p.add_argument("-o", "--output", help="JSON 写到文件(默认 stdout)")
    p.add_argument("--summary", action="store_true", help="JSON 里不放逐项明细")
    args = p.parse_args(argv)
    if not (0.0 <= args.rel_max) or not (-1.0 <= args.cos_min <= 1.0) or args.drop_max < 0:
        p.error("--rel-max and --drop-max must be >= 0; --cos-min must be in [-1, 1]")
    try:
        result = analyze(args)
    except InputError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    if args.summary:
        result.pop("entries")
    text = json.dumps(clean(result), indent=2, ensure_ascii=False)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError as exc:  # 写不出结果是环境错误,不能以 1 冒充越线
            print(json.dumps({"error": f"cannot write output: {exc}"}, ensure_ascii=False), file=sys.stderr)
            return 2
    else:
        print(text)
    return 1 if result["verdict"] == "diverged" else 0


if __name__ == "__main__":
    sys.exit(main())
