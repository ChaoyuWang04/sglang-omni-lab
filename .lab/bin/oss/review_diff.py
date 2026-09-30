#!/usr/bin/env python3
"""提 PR 前自审的机械部分:拍下相对基线的完整 diff 快照、判断快照是否过期、扫描新增行上的候选写法。

只依赖 Python 标准库与 git。

子命令:
  snapshot  以 merge-base(<基线 ref>, HEAD) 为基线(默认基线 ref 是 lab),取已提交、已暂存、未暂存
            与未跟踪(不含 gitignore 忽略的)全部改动,默认排除 .lab/ 与 .hlab/(与 deliver.sh 取净改动的
            范围一致),写出 <out-dir>/diff.patch 与 <out-dir>/snapshot.json。
            在临时暂存区里计算,不动真实暂存区与工作区。
            退出码:0 = 范围内的工作区与 HEAD 一致;1 = 有未提交或未跟踪的改动(deliver.sh 只取已提交的
            内容);2 = 用法或环境错误。
  check     按 snapshot.json 记下的基线与排除项重算 diff 指纹,和快照比。
            退出码:0 = 一致;1 = 已过期(被审的内容变了);2 = 用法或环境错误。
  scan      读一份 unified diff,按规则列出候选行(路径:行,增删哪一侧),并列出增删改的 Python def 与 class。
            候选不是结论,逐条读上下文再判。
            退出码:0 = 没有候选;1 = 有候选;2 = 用法或输入错误。

snapshot 与 check 会像 git add 一样把未提交内容的 blob 写进对象库(不被引用,gc 时回收),所以只对自己的
worktree 跑。

结果一律以 JSON 输出到 stdout。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

DEFAULT_BASE_REF = "lab"
DEFAULT_EXCLUDES = [".lab", ".hlab"]

# 与用户的 git 配置无关的 diff 选项:固定前缀、不走外部 diff 与 textconv、路径不转义。
DIFF_OPTS = [
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--no-relative",
    "--src-prefix=a/",
    "--dst-prefix=b/",
    "-M",
]

PY_EXT = {".py", ".pyi"}
CPP_EXT = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".cu", ".cuh"}

# 测试、参照实现、golden 一类的文件。只用来筛候选,命中之后读上下文。
TESTLIKE_RE = re.compile(
    r"(^|/)(tests?|testing)/"
    r"|(^|/)test_[^/]*$"
    r"|_tests?\.[^/]+$"
    r"|(^|/)conftest\.py$"
    r"|naive|golden"
    r"|(^|[/_])ref(erence)?([_./])"
)

# (规则名, 组, 检查哪一侧, 语言, 只看测试类文件, 正则)
RULES = [
    ("kwargs_string", "fragile", "+", PY_EXT, False,
     r"\*\*kwargs\b|\bkwargs(\.get\(|\[)|[\"'][A-Za-z_]\w*[\"']\s+(not\s+)?in\s+kwargs\b"),
    ("broad_except", "fragile", "+", PY_EXT, False,
     r"^\s*except\s*:|^\s*except\b[^:]*\b(Exception|BaseException)\b[^:]*:"),
    ("any_type", "fragile", "+", PY_EXT, False,
     r":\s*Any\b|->\s*Any\b"),
    ("simple_namespace_in_test", "fragile", "+", PY_EXT, True,
     r"\bSimpleNamespace\b"),
    ("hot_copy", "fragile", "+", PY_EXT, False,
     r"\.clone\(\)|\.copy_\(|\bdeepcopy\("),
    ("blocking_call", "fragile", "+", PY_EXT, False,
     r"\btime\.sleep\(|\brequests\.(get|post|put|delete|patch|head)\(|\burllib\.request\b|\burlopen\("),
    ("lock_in_async", "fragile", "+", PY_EXT, False,
     r"\basync\s+with\b|\.acquire\("),
    ("host_sync", "hostsync", "+", PY_EXT | CPP_EXT, False,
     r"\.item\(\)|\.cpu\(\)|\.tolist\(\)|\.numpy\(\)|\bsynchronize\(\)|\bcuda(Device|Stream)Synchronize\b"),
    ("tolerance", "relaxation", "+-", None, True,
     r"\b(atol|rtol|assert_close|allclose|assertAlmostEqual|toleran\w*|eps|ulps?)\b"),
    ("shape_set", "relaxation", "+-", None, True,
     r"\b(parametrize|skip\w*|xfail|SHAPES|shapes)\b"),
    ("commented_shape_row", "relaxation", "+", None, True,
     r"^\s*(#|//).*\d+\s*,\s*\d+"),
    ("removed_shape_row", "relaxation", "-", None, True,
     r"^\s*[(\[]\s*-?\d+(\s*,\s*-?\d+)+\s*,?\s*[)\]]\s*,?\s*$"),
    ("numerics_switch", "relaxation", "+-", None, False,
     r"allow_tf32|float32_matmul_precision|use_deterministic_algorithms|NONDETERMINISTIC"
     r"|allow_(fp16|bf16)_reduced_precision_reduction"),
]
GROUPS = sorted({r[1] for r in RULES})
COMPILED = [(name, group, sides, langs, testlike, re.compile(rx)) for name, group, sides, langs, testlike, rx in RULES]

DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\b")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class UsageError(Exception):
    pass


# ---------------------------------------------------------------- git


def run_git(repo, args, env=None, binary=False):
    try:
        proc = subprocess.run(["git", "-C", repo, *args], capture_output=True, env=env)
    except FileNotFoundError as exc:
        raise UsageError("git not found on PATH") from exc
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", errors="replace").strip()
        raise UsageError(f"git {' '.join(args)} failed: {msg}")
    return proc.stdout if binary else proc.stdout.decode("utf-8", errors="replace")


def pathspec(excludes):
    return ["--", "."] + [f":(exclude){e}" for e in excludes]


def repo_root(repo):
    if not os.path.isdir(repo):
        raise UsageError(f"not a directory: {repo}")
    return run_git(repo, ["rev-parse", "--show-toplevel"]).strip()


def rev(root, ref):
    return run_git(root, ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"]).strip()


def lines_z(text):
    return [p for p in text.split("\0") if p]


def compute_state(root, base_ref, excludes):
    """基线、head 与工作区全貌的 diff。用临时暂存区,不碰真实暂存区。"""
    try:
        base_sha = rev(root, base_ref)
    except UsageError as exc:
        raise UsageError(f"base ref {base_ref!r} does not resolve to a commit; pass --base-ref") from exc
    head = rev(root, "HEAD")
    merge_base = run_git(root, ["merge-base", base_ref, "HEAD"]).strip()
    branch = run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    real_index = run_git(root, ["rev-parse", "--git-path", "index"]).strip()
    if not os.path.isabs(real_index):
        real_index = os.path.join(root, real_index)
    ps = pathspec(excludes)
    tmpdir = tempfile.mkdtemp(prefix="review-diff-")
    try:
        tmp_index = os.path.join(tmpdir, "index")
        if os.path.exists(real_index):
            shutil.copyfile(real_index, tmp_index)
        env = dict(os.environ, GIT_INDEX_FILE=tmp_index)
        run_git(root, ["add", "-A", *ps], env=env)
        base_args = ["-c", "core.quotePath=false", "diff", "--cached", *DIFF_OPTS]
        patch = run_git(root, [*base_args, "--binary", merge_base, *ps], env=env, binary=True)
        raw = run_git(root, [*base_args, "--raw", "-z", merge_base, *ps], env=env)
        numstat = run_git(root, [*base_args, "--numstat", "-z", merge_base, *ps], env=env)
        not_committed = lines_z(run_git(root, [*base_args, "--name-only", "-z", "HEAD", *ps], env=env))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    untracked = lines_z(run_git(root, ["ls-files", "--others", "--exclude-standard", "-z", *ps]))
    files, submodules = parse_raw(raw)
    counts = parse_numstat(numstat)
    for f in files:
        added, removed = counts.get(f["path"], (None, None))
        f["added"], f["removed"] = added, removed
    totals = {
        "files": len(files),
        "added": sum(f["added"] or 0 for f in files),
        "removed": sum(f["removed"] or 0 for f in files),
    }
    return {
        "repo": root,
        "branch": branch,
        "base_ref": base_ref,
        "base_ref_sha": base_sha,
        "merge_base": merge_base,
        "head": head,
        "excludes": list(excludes),
        "diff_sha256": hashlib.sha256(patch).hexdigest(),
        "files": files,
        "totals": totals,
        "not_committed": not_committed,
        "untracked": untracked,
        "submodules_changed": submodules,
        "clean": not not_committed,
    }, patch


def parse_raw(text):
    """`git diff --raw -z` 的输出:每条是 `:模式 模式 sha sha 状态` 后跟 1 或 2 个路径。"""
    parts = text.split("\0")
    files, submodules = [], []
    i = 0
    while i < len(parts):
        meta = parts[i]
        if not meta.startswith(":"):
            i += 1
            continue
        fields = meta[1:].split()
        old_mode, new_mode, status = fields[0], fields[1], fields[4]
        if status[:1] in ("R", "C"):
            old_path, path = parts[i + 1], parts[i + 2]
            i += 3
        else:
            old_path, path = None, parts[i + 1]
            i += 2
        entry = {"status": status[:1], "path": path}
        if old_path is not None:
            entry["old_path"] = old_path
        files.append(entry)
        if "160000" in (old_mode, new_mode):
            submodules.append(path)
    return files, submodules


def parse_numstat(text):
    """`git diff --numstat -z`:普通条目是 `增\t删\t路径`,改名条目是 `增\t删\t` 后跟旧路径、新路径。"""
    parts = text.split("\0")
    out = {}
    i = 0
    while i < len(parts):
        item = parts[i]
        if not item:
            i += 1
            continue
        added, removed, path = item.split("\t", 2)
        if path == "":
            path = parts[i + 2]
            i += 3
        else:
            i += 1
        to_int = (lambda v: None if v == "-" else int(v))
        out[path] = (to_int(added), to_int(removed))
    return out


# ---------------------------------------------------------------- diff 解析


def unquote_path(p):
    """git 在 core.quotePath 打开时给非 ASCII 路径加引号并用八进制转义。"""
    if not (len(p) >= 2 and p[0] == '"' and p[-1] == '"'):
        return p
    body, out, i = p[1:-1], bytearray(), 0
    simple = {"n": b"\n", "t": b"\t", '"': b'"', "\\": b"\\", "a": b"\a", "b": b"\b", "f": b"\f", "r": b"\r", "v": b"\v"}
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if re.match(r"[0-7]{3}", body[i + 1:i + 4]):
                out.append(int(body[i + 1:i + 4], 8))
                i += 4
                continue
            if nxt in simple:
                out += simple[nxt]
                i += 2
                continue
        out += ch.encode("utf-8")
        i += 1
    return out.decode("utf-8", errors="replace")


def strip_prefix(p, prefix):
    # 非 git 的 diff 在路径后用 tab 接时间戳;git 会给含 tab 的路径加引号,所以没加引号的 tab 只能是分隔符。
    p = unquote_path(p if p.startswith('"') else p.split("\t", 1)[0])
    return p[len(prefix):] if p.startswith(prefix) else p


def parse_patch(text):
    """把 unified diff 拆成文件与行。每行记 (侧, 旧行号, 新行号, 内容)。

    文件头只在 hunk 之外识别,hunk 内按头部给的行数消费,所以内容以 `--- ` 开头的删除行不会被当成文件头。
    """
    files = []
    cur = None
    in_hunk = False
    old_no = new_no = old_left = new_left = 0
    for raw in text.splitlines():
        if in_hunk:
            if raw.startswith("\\"):
                continue
            tag, body = raw[:1], raw[1:]
            if tag == "+":
                cur["lines"].append(("+", None, new_no, body))
                new_no += 1
                new_left -= 1
            elif tag == "-":
                cur["lines"].append(("-", old_no, None, body))
                old_no += 1
                old_left -= 1
            else:
                old_no += 1
                new_no += 1
                old_left -= 1
                new_left -= 1
            if old_left <= 0 and new_left <= 0:
                in_hunk = False
            continue
        if raw.startswith("diff --git "):
            cur = {"path": None, "old_path": None, "binary": False, "deleted": False, "seen_plus": False, "lines": []}
            files.append(cur)
            m = re.match(r'diff --git ("?a/.*?"?) ("?b/.*"?)$', raw)
            if m:
                cur["old_path"], cur["path"] = strip_prefix(m.group(1), "a/"), strip_prefix(m.group(2), "b/")
            continue
        if raw.startswith("--- ") and (cur is None or cur["seen_plus"]):
            # 非 git 的 unified diff 没有 diff --git 行,每个文件从 --- 开始。
            cur = {"path": None, "old_path": None, "binary": False, "deleted": False, "seen_plus": False, "lines": []}
            files.append(cur)
        if cur is None:
            continue
        m = HUNK_RE.match(raw)
        if m:
            old_no, new_no = int(m.group(1)), int(m.group(3))
            old_left = int(m.group(2)) if m.group(2) is not None else 1
            new_left = int(m.group(4)) if m.group(4) is not None else 1
            in_hunk = old_left > 0 or new_left > 0
        elif raw.startswith("+++ "):
            cur["seen_plus"] = True
            p = raw[4:]
            if p.strip() != "/dev/null":
                cur["path"] = strip_prefix(p, "b/")
        elif raw.startswith("--- "):
            p = raw[4:]
            if p.strip() != "/dev/null":
                cur["old_path"] = strip_prefix(p, "a/")
        elif raw.startswith("rename from "):
            cur["old_path"] = unquote_path(raw[len("rename from "):])
        elif raw.startswith("rename to "):
            cur["path"] = unquote_path(raw[len("rename to "):])
        elif raw.startswith("deleted file mode"):
            cur["deleted"] = True
        elif raw.startswith("Binary files ") or raw.startswith("GIT binary patch"):
            cur["binary"] = True
    for f in files:
        if f["path"] is None:
            f["path"] = f["old_path"]
    return files


def ext_of(path):
    return os.path.splitext(path or "")[1].lower()


def scan_files(files, groups, max_text):
    hits, symbols = [], {}
    for f in files:
        path = f["path"] or ""
        ext = ext_of(path)
        testlike = bool(TESTLIKE_RE.search(path))
        for side, old_no, new_no, body in f["lines"]:
            line_no = new_no if side == "+" else old_no
            for name, group, sides, langs, only_test, rx in COMPILED:
                if group not in groups or side not in sides:
                    continue
                if langs is not None and ext not in langs:
                    continue
                if only_test and not testlike:
                    continue
                if rx.search(body):
                    hits.append({
                        "rule": name,
                        "group": group,
                        "path": path,
                        "line": line_no,
                        "side": side,
                        "text": body.strip()[:max_text],
                    })
            if ext in PY_EXT:
                for kind, srx in (("def", DEF_RE), ("class", CLASS_RE)):
                    m = srx.match(body)
                    if not m:
                        continue
                    sym = m.group(1)
                    dunder = sym.startswith("__") and sym.endswith("__")
                    if sym.startswith("_") and not dunder:
                        continue
                    key = (path, sym, kind)
                    entry = symbols.setdefault(key, {"path": path, "name": sym, "kind": kind, "lines": {"+": [], "-": []}})
                    entry["lines"][side].append(line_no)
    changed = []
    for entry in symbols.values():
        plus, minus = entry["lines"]["+"], entry["lines"]["-"]
        entry["change"] = "modified" if plus and minus else ("added" if plus else "removed")
        changed.append(entry)
    changed.sort(key=lambda e: (e["path"], e["name"]))
    return hits, changed


# ---------------------------------------------------------------- 子命令


def cmd_snapshot(args):
    root = repo_root(args.repo)
    excludes = args.exclude if args.exclude is not None else DEFAULT_EXCLUDES
    state, patch = compute_state(root, args.base_ref, excludes)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    patch_path = os.path.join(out_dir, "diff.patch")
    with open(patch_path, "wb") as fh:
        fh.write(patch)
    state = {
        "tool": "review_diff.py snapshot",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **state,
        "patch": patch_path,
    }
    with open(os.path.join(out_dir, "snapshot.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if state["clean"] else 1


def cmd_check(args):
    try:
        with open(args.snapshot, encoding="utf-8") as fh:
            old = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(f"cannot read snapshot {args.snapshot}: {exc}") from exc
    for key in ("repo", "base_ref", "excludes", "diff_sha256", "head", "merge_base"):
        if key not in old:
            raise UsageError(f"snapshot {args.snapshot} lacks {key!r}")
    root = repo_root(args.repo or old["repo"])
    new, _ = compute_state(root, old["base_ref"], old["excludes"])
    stale = new["diff_sha256"] != old["diff_sha256"]
    result = {
        "snapshot": os.path.abspath(args.snapshot),
        "stale": stale,
        "head_moved": new["head"] != old["head"],
        "merge_base_moved": new["merge_base"] != old["merge_base"],
        "old": {k: old[k] for k in ("head", "merge_base", "diff_sha256")},
        "new": {k: new[k] for k in ("head", "merge_base", "diff_sha256")},
        "clean_now": new["clean"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if stale else 0


def cmd_scan(args):
    if args.patch == "-":
        text = sys.stdin.read()
        label = "-"
    else:
        try:
            with open(args.patch, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            raise UsageError(f"cannot read patch {args.patch}: {exc}") from exc
        label = os.path.abspath(args.patch)
    groups = set(GROUPS) if args.rules == "all" else {g.strip() for g in args.rules.split(",") if g.strip()}
    unknown = groups - set(GROUPS)
    if unknown or not groups:
        raise UsageError(f"unknown rule groups {sorted(unknown)}; choose from {GROUPS} or 'all'")
    files = parse_patch(text)
    if text.strip() and not files:
        raise UsageError(f"{args.patch} does not look like a unified diff")
    hits, changed = scan_files(files, groups, args.max_text)
    counts = {}
    for h in hits:
        counts[h["rule"]] = counts.get(h["rule"], 0) + 1
    result = {
        "patch": label,
        "groups": sorted(groups),
        "files_scanned": len(files),
        "binary_files": [f["path"] for f in files if f["binary"]],
        "hits": hits,
        "counts_by_rule": counts,
        "changed_symbols": changed,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if hits else 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="review_diff.py",
        description="提 PR 前自审的机械部分:snapshot 拍下相对 merge-base 的完整 diff,check 判断快照是否过期,"
        "scan 列出新增行上的候选写法与增删改的 Python def/class。结果以 JSON 输出。",
        epilog="退出码:0 = 通过(snapshot:工作区已全部提交;check:快照未过期;scan:没有候选);"
        "1 = 不通过;2 = 用法或环境错误。",
    )
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("snapshot", help="拍下相对 merge-base(<base-ref>, HEAD) 的完整 diff(含未提交与未跟踪文件)")
    s.add_argument("--repo", default=".", help="被审的 worktree(默认当前目录)")
    s.add_argument("--base-ref", default=DEFAULT_BASE_REF, help=f"基线分支或提交(默认 {DEFAULT_BASE_REF})")
    s.add_argument("--out-dir", required=True, help="写 diff.patch 与 snapshot.json 的目录")
    s.add_argument("--exclude", action="append", default=None,
                   help="排除的路径,可重复;给了就替换默认值 .lab 与 .hlab")
    s.set_defaults(func=cmd_snapshot)

    c = sub.add_parser("check", help="重算 diff 指纹,判断 snapshot.json 是否过期")
    c.add_argument("--snapshot", required=True, help="snapshot 写出的 snapshot.json")
    c.add_argument("--repo", default=None, help="被审的 worktree(默认用快照里记的路径)")
    c.set_defaults(func=cmd_check)

    k = sub.add_parser("scan", help="列出 diff 新增行(与测试类文件的删除行)上的候选写法")
    k.add_argument("--patch", required=True, help="unified diff 文件,- 表示 stdin")
    k.add_argument("--rules", default="all", help=f"规则组,逗号分隔:{', '.join(GROUPS)};默认 all")
    k.add_argument("--max-text", type=int, default=200, help="每条命中保留的行内容长度(默认 200)")
    k.set_defaults(func=cmd_scan)
    return p


def main(argv=None):
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 2
    if not getattr(args, "func", None):
        parser.print_usage(sys.stderr)
        print("review_diff.py: error: choose a subcommand: snapshot, check or scan", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except (UsageError, OSError) as exc:
        # OSError:写不了 --out-dir 之类的环境错误,按约定退出码 2,而不是带 traceback 的 1。
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
