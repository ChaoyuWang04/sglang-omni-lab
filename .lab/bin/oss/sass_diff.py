#!/usr/bin/env python3
"""规范化对比两版 SASS:按函数给出 opcode 计数差、新增与消失的指令类型、资源用量变化,输出 JSON。

输入:两份反汇编文本,cuobjdump -sass(按 "Function : <名>" 分函数)或 nvdisasm(按 ".text.<名>:" 分函数)
的输出都认;同一文件里若还有 cuobjdump -res-usage 或 nvcc -Xptxas -v 的输出,资源用量一并读出,
也可以用 --res-a/--res-b 另给。

规范化:去掉地址与编码注释、寄存器编号(R12 -> R,UR4 -> UR,P0 -> P,B1 -> B)、.reuse、跳转标签与
符号名;立即数默认保留(--ignore-immediates 一并抹掉)。规范化后序列相同记 noise_only(原文也相同记
identical),否则 changed,并给出前几个差异区段。

退出码:0 = 符合预期(默认 --expect changed:至少一个配对函数的规范化序列变了;--expect same:全部没变;
        另加的 --expect-up/--expect-down 也都成立);1 = 不符合;
        2 = 文件不存在、解析不出函数、两边目标架构不一致、--arch/--function 没匹配到或没有可配对的函数,
            或写不了输出文件。
只依赖 Python 标准库。
"""

import argparse
import difflib
import json
import os
import re
import sys
from collections import Counter, OrderedDict

INSTR_RE = re.compile(r"^\s*/\*([0-9a-fA-F]{4,})\*/\s*(.*)$")
ENCODING_RE = re.compile(r"/\*\s*0x[0-9a-fA-F]+\s*\*/")
CUOBJ_FUNC_RE = re.compile(r"^\s*Function\s*:\s*(\S+)")
CODE_FOR_RE = re.compile(r"^\s*code for (sm_\w+)")
ARCH_EQ_RE = re.compile(r"^\s*arch\s*=\s*(sm_\w+)")
HEADERFLAG_RE = re.compile(r"EF_CUDA_SM(\d+[A-Za-z]?)\b")
NV_SECTION_RE = re.compile(r"^\s*\.section\s+\.text\.([^,\s]+)")
NV_LABEL_RE = re.compile(r"^\s*\.text\.(\S+?):\s*$")
NV_REGS_RE = re.compile(r"SHI_REGISTERS=(\d+)")
RES_FUNC_RE = re.compile(r"^\s*Function\s+([^\s:]+)\s*:\s*$")
RES_KV_RE = re.compile(r"([A-Z_]+(?:\[\d+\])?):(\d+)")
PTXAS_ENTRY_RE = re.compile(r"Compiling entry function '([^']+)' for '(sm_\w+)'")
PTXAS_PROPS_RE = re.compile(r"Function properties for (\S+)")
PTXAS_STACK_RE = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
PTXAS_USED_RE = re.compile(r"Used (\d+) registers")
PTXAS_SMEM_RE = re.compile(r"(\d+) bytes smem")

CONTROL_OPS = {"BRA", "BRX", "JMP", "JMX", "CALL", "CAL", "JCAL", "SSY", "PBK", "PCNT", "BSSY", "BREAK",
               "RET", "WARPSYNC", "BMOV"}
CATEGORIES = OrderedDict([
    ("local_mem", {"LDL", "STL"}),
    ("shared_mem", {"LDS", "STS", "LDSM", "STSM", "ATOMS"}),
    ("global_mem", {"LDG", "STG", "LD", "ST", "ATOM", "ATOMG", "RED", "REDG", "LDGSTS"}),
    ("tensor", {"HMMA", "IMMA", "DMMA", "BMMA", "QMMA", "OMMA", "HGMMA", "IGMMA", "QGMMA", "BGMMA",
                "UTCHMMA", "UTCIMMA", "UTCQMMA", "UTCOMMA"}),
    ("async_copy", {"LDGSTS", "LDGDEPBAR", "UTMALDG", "UTMASTG", "UTMAPF", "UBLKCP", "UTMACCTL", "SYNCS"}),
    ("barrier_sync", {"BAR", "BSYNC", "WARPSYNC", "MEMBAR", "DEPBAR", "ERRBAR", "SYNCS"}),
    ("control", {"BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT", "BSSY", "BREAK", "YIELD"}),
])


class InputError(Exception):
    pass


# ---------- 解析 ----------

def arch_from_flag(text):
    m = HEADERFLAG_RE.search(text)
    return f"sm_{m.group(1).lower()}" if m else None


def parse_instruction(addr, body):
    body = ENCODING_RE.sub("", body)
    body = body.replace("{", " ").replace("}", " ")
    if ";" in body:
        body = body[:body.rfind(";")]
    body = body.strip()
    if not body:
        return None
    pred = None
    m = re.match(r"^@(!?U?P(?:\d+|T))\s+", body)
    if m:
        pred, body = m.group(1), body[m.end():]
    parts = body.split(None, 1)
    return {"addr": addr, "pred": pred, "opcode": parts[0], "operands": parts[1] if len(parts) > 1 else "",
            "text": ("@" + pred + " " if pred else "") + body}


def parse_res_usage(lines, default_arch=None):
    """cuobjdump -res-usage 与 nvcc -Xptxas -v 的输出 -> {(arch, 函数名): {...}}。"""
    res = {}
    arch = default_arch
    cur = None
    props = None
    for ln in lines:
        m = ARCH_EQ_RE.match(ln) or CODE_FOR_RE.match(ln)
        if m:
            arch = m.group(1)
        m = RES_FUNC_RE.match(ln)
        if m:
            cur = m.group(1)
            continue
        if cur and re.match(r"^\s*REG:\d+", ln):
            d = res.setdefault((arch, cur), {})
            for k, v in RES_KV_RE.findall(ln):
                d[k.lower()] = int(v)
            cur = None
            continue
        m = PTXAS_ENTRY_RE.search(ln)
        if m:
            cur_ptxas = m.group(1)
            arch = m.group(2)
            props = cur_ptxas
            res.setdefault((arch, cur_ptxas), {})
            continue
        m = PTXAS_PROPS_RE.search(ln)
        if m:
            props = m.group(1)
            continue
        m = PTXAS_STACK_RE.search(ln)
        if m and props and (arch, props) in res:
            d = res[(arch, props)]
            d["stack_frame_bytes"], d["spill_store_bytes"], d["spill_load_bytes"] = map(int, m.groups())
            continue
        m = PTXAS_USED_RE.search(ln)
        if m and props and (arch, props) in res:
            d = res[(arch, props)]
            d["reg"] = int(m.group(1))
            sm = PTXAS_SMEM_RE.search(ln)
            if sm:
                d["shared"] = int(sm.group(1))
    return res


def parse_sass(path):
    if not os.path.isfile(path):
        raise InputError(f"输入不存在: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    funcs = OrderedDict()
    fmt = None
    arch = None
    cur = None
    nv_section = None
    nv_regs = {}
    for ln in lines:
        m = CODE_FOR_RE.match(ln) or ARCH_EQ_RE.match(ln)
        if m:
            arch = m.group(1)
            continue
        if ".headerflags" in ln and arch is None:
            arch = arch_from_flag(ln)
        m = CUOBJ_FUNC_RE.match(ln)
        if m:
            fmt = fmt or "cuobjdump"
            cur = (arch, m.group(1))
            funcs.setdefault(cur, {"instructions": [], "registers": None})
            continue
        m = NV_SECTION_RE.match(ln)
        if m:
            nv_section = m.group(1)
            continue
        m = NV_REGS_RE.search(ln)
        if m and nv_section:
            nv_regs[nv_section] = int(m.group(1))
            continue
        m = NV_LABEL_RE.match(ln)
        if m:
            fmt = fmt or "nvdisasm"
            cur = (arch, m.group(1))
            funcs.setdefault(cur, {"instructions": [], "registers": nv_regs.get(m.group(1))})
            continue
        m = INSTR_RE.match(ln)
        if m:
            ins = parse_instruction(m.group(1).lower(), m.group(2))
            if ins is None:
                continue
            if cur is None:
                fmt = fmt or "bare"
                cur = (arch, "<anonymous>")
                funcs.setdefault(cur, {"instructions": [], "registers": None})
            funcs[cur]["instructions"].append(ins)
    funcs = OrderedDict((k, v) for k, v in funcs.items() if v["instructions"])
    if not funcs:
        raise InputError(f"{path}: 没解析出任何带指令的函数;要的是 cuobjdump -sass 或 nvdisasm 的文本输出")
    return {"path": os.path.abspath(path), "format": fmt, "functions": funcs,
            "res": parse_res_usage(lines, default_arch=None)}


# ---------- 规范化与比较 ----------

def normalize(ins, ignore_imm):
    ops = ins["operands"]
    ops = re.sub(r"`\(([^)]*)\)", "LABEL", ops)
    ops = re.sub(r"\.L_\w+", "LABEL", ops)
    ops = re.sub(r"\b_Z\w+", "SYM", ops)
    ops = ops.replace(".reuse", "")
    ops = re.sub(r"\bUR\d+\b", "UR", ops)
    ops = re.sub(r"\bR\d+\b", "R", ops)
    ops = re.sub(r"\bUP\d\b", "UP", ops)
    ops = re.sub(r"\bP\d\b", "P", ops)
    ops = re.sub(r"\bB\d+\b", "B", ops)
    base = ins["opcode"].split(".")[0]
    if base in CONTROL_OPS:
        ops = re.sub(r"\b0x[0-9a-fA-F]+\b", "LABEL", ops)
    if ignore_imm:
        ops = re.sub(r"\b0x[0-9a-fA-F]+\b", "IMM", ops)
        ops = re.sub(r"(?<![\w.])-?\d+(\.\d+)?(e[-+]?\d+)?\b", "IMM", ops)
    ops = re.sub(r"\s+", " ", ops).strip()
    pred = ""
    if ins["pred"]:
        neg, reg = ins["pred"].startswith("!"), ins["pred"].lstrip("!")
        if reg in ("PT", "UPT"):
            pred = "@!PT " if neg else ""
        else:
            pred = ("@!" if neg else "@") + ("UP " if reg.startswith("UP") else "P ")
    return f"{pred}{ins['opcode']} {ops}".strip()


def counts(instrs):
    full = Counter(i["opcode"] for i in instrs)
    base = Counter(i["opcode"].split(".")[0] for i in instrs)
    return full, base


def delta(ca, cb):
    out = {}
    for k in sorted(set(ca) | set(cb)):
        if ca.get(k, 0) != cb.get(k, 0):
            out[k] = {"a": ca.get(k, 0), "b": cb.get(k, 0), "delta": cb.get(k, 0) - ca.get(k, 0)}
    return out


def categories(base):
    return {name: sum(base.get(op, 0) for op in ops) for name, ops in CATEGORIES.items()}


def regions(na, nb, ia, ib, top, max_lines):
    sm = difflib.SequenceMatcher(a=na, b=nb, autojunk=False)
    blocks = [op for op in sm.get_opcodes() if op[0] != "equal"]
    blocks.sort(key=lambda op: -max(op[2] - op[1], op[4] - op[3]))
    out = []
    for tag, i1, i2, j1, j2 in blocks[:top]:
        out.append({
            "kind": tag,
            "a_index": [i1, i2], "b_index": [j1, j2],
            "a_addr": ia[i1]["addr"] if i1 < len(ia) else None,
            "b_addr": ib[j1]["addr"] if j1 < len(ib) else None,
            "a": [ia[k]["text"] for k in range(i1, min(i2, i1 + max_lines))],
            "b": [ib[k]["text"] for k in range(j1, min(j2, j1 + max_lines))],
        })
    return out, round(sm.ratio(), 4)


def lookup_res(side, key):
    arch, name = key
    for k in ((arch, name), (None, name)):
        if k in side["res"]:
            return side["res"][k]
    for (a, n), v in side["res"].items():
        if n == name:
            return v
    return None


def compare_pair(ka, fa, kb, fb, A, B, args):
    ia, ib = fa["instructions"], fb["instructions"]
    na = [normalize(i, args.ignore_immediates) for i in ia]
    nb = [normalize(i, args.ignore_immediates) for i in ib]
    raw_same = [i["text"] for i in ia] == [i["text"] for i in ib]
    if raw_same:
        status = "identical"
    elif na == nb:
        status = "noise_only"
    else:
        status = "changed"
    fa_full, fa_base = counts(ia)
    fb_full, fb_base = counts(ib)
    ra, rb = lookup_res(A, ka), lookup_res(B, kb)
    reg_a = fa["registers"] if fa["registers"] is not None else (ra or {}).get("reg")
    reg_b = fb["registers"] if fb["registers"] is not None else (rb or {}).get("reg")
    out = {
        "a_name": ka[1], "b_name": kb[1], "arch": ka[0] or kb[0], "status": status,
        "instructions": {"a": len(ia), "b": len(ib), "delta": len(ib) - len(ia)},
        "registers": {"a": reg_a, "b": reg_b},
        "resources": {"a": ra, "b": rb},
        "categories": {"a": categories(fa_base), "b": categories(fb_base)},
        "opcode_delta": delta(fa_full, fb_full),
        "base_opcode_delta": delta(fa_base, fb_base),
        "new_opcodes": sorted(set(fb_full) - set(fa_full)),
        "gone_opcodes": sorted(set(fa_full) - set(fb_full)),
        "new_base_opcodes": sorted(set(fb_base) - set(fa_base)),
        "gone_base_opcodes": sorted(set(fa_base) - set(fb_base)),
    }
    if status == "changed":
        out["diff_regions"], out["similarity"] = regions(na, nb, ia, ib, args.top, args.max_lines)
    return out, fa_full, fa_base, fb_full, fb_base


def select(side, args, label):
    funcs = side["functions"]
    if args.arch:
        funcs = OrderedDict((k, v) for k, v in funcs.items() if k[0] == args.arch)
        if not funcs:
            archs = sorted({str(k[0]) for k in side["functions"]})
            raise InputError(f"{label}: 没有 --arch {args.arch} 的函数(有 {archs})")
    if args.function:
        rx = re.compile(args.function)
        funcs = OrderedDict((k, v) for k, v in funcs.items() if rx.search(k[1]))
        if not funcs:
            raise InputError(f"{label}: --function {args.function!r} 没匹配到函数")
    return funcs


def main(argv=None):
    p = argparse.ArgumentParser(
        description="规范化对比两份 SASS(cuobjdump -sass 或 nvdisasm 输出),按函数给出 opcode 计数差、"
                    "新增与消失的指令类型、寄存器与溢出变化,输出 JSON。"
                    "退出码 0 = 符合 --expect;1 = 不符合;2 = 输入或配对错误。")
    p.add_argument("a", help="基线的反汇编文本")
    p.add_argument("b", help="改动后的反汇编文本")
    p.add_argument("--res-a", help="基线的 cuobjdump -res-usage 或 nvcc -Xptxas -v 输出(可选)")
    p.add_argument("--res-b", help="改动后的 cuobjdump -res-usage 或 nvcc -Xptxas -v 输出(可选)")
    p.add_argument("--arch", help="只比这个架构的函数,如 sm_120(fatbin 里有多个架构时用)")
    p.add_argument("--function", help="只比函数名(mangled)匹配这个正则的函数;两边各剩 1 个时不看名字直接配对")
    p.add_argument("--expect", choices=["changed", "same"], default="changed",
                   help="changed(默认):确认改动改变了生成代码;same:确认没改变(纯重构)")
    p.add_argument("--expect-up", action="append", default=[], metavar="OPCODE",
                   help="断言该 opcode 在 B 里比 A 多;含点号按完整 opcode 比,否则按基础 opcode;可重复")
    p.add_argument("--expect-down", action="append", default=[], metavar="OPCODE",
                   help="断言该 opcode 在 B 里比 A 少;规则同上;可重复")
    p.add_argument("--ignore-immediates", action="store_true", help="规范化时连立即数一起抹掉")
    p.add_argument("--top", type=int, default=5, help="每个函数列出最大的几个差异区段(默认 5)")
    p.add_argument("--max-lines", type=int, default=12, help="每个差异区段最多列几条指令(默认 12)")
    p.add_argument("-o", "--output", help="JSON 写到文件(默认 stdout)")
    args = p.parse_args(argv)

    try:
        A, B = parse_sass(args.a), parse_sass(args.b)
        for side, extra in ((A, args.res_a), (B, args.res_b)):
            if extra:
                if not os.path.isfile(extra):
                    raise InputError(f"输入不存在: {extra}")
                with open(extra, "r", encoding="utf-8", errors="replace") as f:
                    side["res"].update(parse_res_usage(f.read().splitlines()))
        fa, fb = select(A, args, "A"), select(B, args, "B")
        archs_a = {k[0] for k in fa} - {None}
        archs_b = {k[0] for k in fb} - {None}
        if archs_a and archs_b and archs_a != archs_b:
            raise InputError(f"两边目标架构不一致:A {sorted(archs_a)},B {sorted(archs_b)};用 --arch 选同一个")
        pairs, only_a, only_b = [], [], []
        paired_by = "name"
        if len(fa) == 1 and len(fb) == 1:
            pairs = [(next(iter(fa)), next(iter(fb)))]
            paired_by = "single_function"
        else:
            names_b = {k[1]: k for k in fb}
            for ka in fa:
                kb = names_b.get(ka[1])
                if kb is None:
                    only_a.append(ka[1])
                else:
                    pairs.append((ka, kb))
            matched_b = {kb for _, kb in pairs}
            only_b = [k[1] for k in fb if k not in matched_b]
        if not pairs:
            raise InputError("两边没有同名函数可配对;用 --function 各筛出一个再比")
    except (InputError, re.error, OSError) as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False), file=sys.stderr)
        return 2

    results = []
    tot = {"a_full": Counter(), "a_base": Counter(), "b_full": Counter(), "b_base": Counter()}
    for ka, kb in pairs:
        r, a_full, a_base, b_full, b_base = compare_pair(ka, fa[ka], kb, fb[kb], A, B, args)
        results.append(r)
        tot["a_full"] += a_full
        tot["a_base"] += a_base
        tot["b_full"] += b_full
        tot["b_base"] += b_base

    checks = []
    n_changed = sum(r["status"] == "changed" for r in results)
    if args.expect == "changed":
        checks.append({"check": "expect changed", "ok": n_changed > 0,
                       "detail": f"{n_changed}/{len(results)} 个函数的规范化序列变了"})
    else:
        checks.append({"check": "expect same", "ok": n_changed == 0,
                       "detail": f"{n_changed}/{len(results)} 个函数的规范化序列变了"})
    for op, direction in [(o, "up") for o in args.expect_up] + [(o, "down") for o in args.expect_down]:
        key = "full" if "." in op else "base"
        a_n, b_n = tot[f"a_{key}"].get(op, 0), tot[f"b_{key}"].get(op, 0)
        ok = b_n > a_n if direction == "up" else b_n < a_n
        checks.append({"check": f"expect {direction} {op}", "ok": ok, "detail": f"A {a_n} -> B {b_n}"})
    passed = all(c["ok"] for c in checks)

    report = {
        "a": {"path": A["path"], "format": A["format"], "functions": len(A["functions"])},
        "b": {"path": B["path"], "format": B["format"], "functions": len(B["functions"])},
        "normalization": {"registers": True, "labels": True, "symbols": True,
                          "immediates": bool(args.ignore_immediates)},
        "summary": {"pairs": len(results), "paired_by": paired_by,
                    "changed": n_changed,
                    "noise_only": sum(r["status"] == "noise_only" for r in results),
                    "identical": sum(r["status"] == "identical" for r in results),
                    "only_in_a": only_a, "only_in_b": only_b},
        "functions": results,
        "checks": checks,
        "verdict": "pass" if passed else "fail",
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
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
