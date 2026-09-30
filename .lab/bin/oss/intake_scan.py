#!/usr/bin/env python3
"""扫描一个仓库,列出接入时要读的规矩文件与 agent 可见性缺口,输出 JSON。

只读,只依赖 Python 标准库。不运行仓库里的任何脚本;遍历不跟随软链接,软链接只记录并解析目标。
跳过 .git、node_modules、虚拟环境、构建产物、3rdparty 与 third_party、.lab 与 .hlab。

列出的节:
  guides        agent 守则:AGENTS.md、AGENTS.override.md、CLAUDE.md(含 .claude/CLAUDE.md)、CLAUDE.local.md、
                GEMINI.md、AI_POLICY.md、.cursorrules、.github/copilot-instructions.md;带作用范围、软链目标、@ 导入,
                有 agent 读不到的另列 mentioned_by(那个 agent 会读的守则里提到了它)
  rules         .claude/rules、.agents/rules、.cursor/rules 与 .github/instructions 下的规则,带 paths、globs、
                alwaysApply、applyTo
  skills        所有 skill(按真实目录去重),各自在哪些位置可见、Claude Code 与 Codex 能否自动发现、
                name 与目录名是否一致、fixable_by_wire_skills;skill 目录里不成 skill 的条目;工具包 oss-* 软链单列
  contributing  CONTRIBUTING、贡献文档目录、PR 与 issue 模板、CODEOWNERS、风格规范、子系统开发指南、
                「改之前先读」这类句子
  lint          .pre-commit-config.yaml 的 hook(repo、stages、local hook 的 entry)、Makefile 的格式化目标、
                格式化脚本
  ci            GitHub Actions 每个 workflow 的触发事件、runs-on、matrix 里的 runner 取值、标签与评论命令、
                标题校验行、提到的硬件(实验层自己的 oss-*.yml 标 lab_layer);其他 CI 系统的配置文件;
                标题校验脚本;CI 权限名单
  tests         测试目录、pytest marker、按硬件命名的测试清单、名字带 cpu 的文件
  hardware      CI 配置、测试清单与测试代码里提到 5090、GB202、sm_120、RTX PRO 6000 的行
  policy        守则、贡献文档、PR 模板里与 AI 协助、DCO、CLA 相关的行
  gaps          至少一个 agent 不会自动读到的守则、规则与 skill
  unreadable    遍历或读取时因权限等 OSError 跳过的路径与报错;这些路径下的内容没有进上面各节

可见性取值:auto(会话开始就会读到)、conditional(嵌套目录里的,是否加载取决于 agent 与会话的工作目录)、
unknown(各方说法不一,按看不到处理)、no。CLAUDE.md 与 AGENTS.md 是软链、互相导入、内容相同或只是一两句
「去读另一份」时,算两边都读得到。
约定与 _kit/wire-skills.sh 一致:Claude Code 读 CLAUDE.md 与 .claude/skills、.claude/rules;
Codex 读 AGENTS.md 与 .agents/skills。

fixable_by_wire_skills 按 wire-skills.sh 第 2 步模拟:根目录 .claude/skills、.agents/skills 中本身不是软链接的
是真实侧(还不存在也算,脚本会建);来源依次是根目录的 .claude/skills、.agents/skills、.agent/skills、
.codex/skills,来源里有 SKILL.md 的条目(oss-* 除外,留给工具包)在某个真实侧缺同名条目时补一个相对链接,
同名的先到先得。一侧整体软链到另一侧时两边共享目录,补在真实侧两边都看得到;一侧整体软链到别处时那一侧补不了;
子目录里嵌套的与不在这 4 处的 skill 补不了。有缺口、且补链后 Claude Code 与 Codex 都能自动发现,才记 true。

退出码:0 = 没有缺口;1 = 有缺口(gaps 非空:画像要写指针,或重跑 wire-skills.sh);2 = 用法或环境错误
(含根目录读不了)。子路径读不了只跳过并记进 unreadable,不改变退出码。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    tomllib = None

SCHEMA = "oss-project-intake/intake_scan/v1"
EXIT_CLEAN, EXIT_GAPS, EXIT_ENV = 0, 1, 2

EXCLUDE_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".tox", ".nox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "site-packages", "3rdparty", "third_party", "third-party",
    ".lab", ".hlab", "build", "dist", ".worktrees", ".eggs",
}
DOT_AGENT_DIRS = (".claude", ".agents", ".agent", ".codex", ".cursor")
# 与 _kit/wire-skills.sh 第 2 步一致:往哪两侧补链、从哪些来源补(按这个顺序,先到先得)
WIRE_SIDES = (".claude/skills", ".agents/skills")
WIRE_SOURCES = (".claude/skills", ".agents/skills", ".agent/skills", ".codex/skills")
GUIDE_NAMES = {
    "AGENTS.md", "AGENTS.override.md", "CLAUDE.md", "CLAUDE.local.md", "GEMINI.md",
    ".cursorrules", ".windsurfrules",
}
TEST_DIR_NAMES = ("test", "tests", "testing")
CI_PATH_PARTS = {"test", "tests", "testing", "ci", ".ci", ".github", ".buildkite", "jenkins",
                 "test_lists", "test-db", ".circleci", ".gitlab"}
OTHER_CI_DIRS = (".buildkite", "jenkins", ".circleci", ".gitlab", "ci", ".ci")
OTHER_CI_FILES = re.compile(r"^(Jenkinsfile.*|\.gitlab-ci\.ya?ml|azure-pipelines\.ya?ml|\.drone\.ya?ml)$")
KNOWN_EVENTS = {
    "push", "pull_request", "pull_request_target", "issue_comment", "workflow_dispatch",
    "workflow_call", "workflow_run", "schedule", "merge_group", "pull_request_review",
    "pull_request_review_comment", "release", "repository_dispatch", "issues",
}

# 5090 相关:型号、芯片代号 GB202、架构号 sm_120、同为 sm_120 的 RTX PRO 6000。
# 边界只排除字母数字与点,runner 标签里的 1-gpu-5090、测试清单名里的 l0_gb202 都能命中。
_B, _E = r"(?<![A-Za-z0-9.])", r"(?![A-Za-z0-9])"
HW_LOCAL = re.compile(
    rf"(?i){_B}(rtx[ _-]?5090|5090|gb202|sm_?120a?|compute_120a?|120a?-real|rtx[ _-]?pro[ _-]?6000){_E}"
)
HW_ANY = re.compile(
    rf"(?i){_B}(rtx[ _-]?pro[ _-]?6000|rtx[ _-]?\d{{4}}|5090|5080|4090|3090|gb20[0-9]|gb10|gb[23]00|"
    r"gh200|b[123]00|h[12]00|h20|h800|a100|a10g?|a30|a40|l40s?|l4|l20|t4|v100|mi\d{3}x?|"
    rf"sm_?\d{{2,3}}a?|xpu|npu|ascend|gaudi|hpu|tpu|cpu){_E}"
)
# AI 协助相关:大写的 AI 单独成词(排除 Wan-AI 这类名字);工具名排除 CLAUDE.md、.claude/ 这类文件与目录名
AI_PAT = re.compile(
    r"(?<![\w-])AI(?![\w/])|(?i:AI-assisted|AI-generated|\bLLM-generated|co-authored-by|assisted-by|"
    r"generated-by|generated with|coding agent|code-agent|autonomous|chatgpt|disclos)|"
    r"(?<![./\w-])(Claude|Codex|Gemini|Copilot|Cursor|CLAUDE|CODEX)(?![\w./-])"
)
DCO_PAT = re.compile(r"(?i)(signed-off-by|\bDCO\b|developer certificate of origin|--signoff|commit -s\b|sign-off)")
CLA_PAT = re.compile(r"(?i)(\bCLA\b|contributor license agreement|cla-assistant)")
MUST_READ_PAT = re.compile(
    r"(?i)(read (this |it |the \S+ )?(first |carefully )?before (modifying|editing|changing|touching|working)|"
    r"before modifying|must[- ]read|do not modify code in these areas|refuse the change)"
)
TITLE_PAT = re.compile(r"(?i)(pr[-_ ]?title|pull_request\.title|semantic-pull-request)")
STYLE_NAME = re.compile(r"(?i)(coding[_-]?guidelines?|coding[_-]?style|style[_-]?guide|code[_-]?style)")
DEVGUIDE_NAME = re.compile(r"(?i)(developer[_-]?guide|engineering[_-]?criteria|dev[_-]?guide)")
CONTRIB_DIR_NAMES = {"contributing", "developer_guide", "developer-guide", "developer_reference",
                     "contributor", "contributors", "contribute"}
FORMAT_SCRIPTS = {"format.sh", "autoformat.sh", "lint.sh", "codestyle.sh", ".lintrunner.toml",
                  "lintrunner.toml"}
MAKE_TARGET = re.compile(r"^(style|quality|lint|format|fmt|check|precommit|pre-commit|typecheck|mypy)\s*:")
CPU_HINT_EXT = (".md", ".mdx", ".rst", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".py", ".sh")
MAX_TEXT_BYTES = 1_000_000


class EnvError(Exception):
    """用法或环境错误,退出码 2。"""


# ---------------------------------------------------------------- 小工具

_TEXT_CACHE: dict[str, str] = {}
_READ_ERRORS: dict[str, OSError] = {}  # 读不了的文件,扫描结束时并进 unreadable


def read_text(path: str) -> str:
    if path in _TEXT_CACHE:
        return _TEXT_CACHE[path]
    _TEXT_CACHE[path] = _read_text(path)
    return _TEXT_CACHE[path]


def _read_text(path: str) -> str:
    try:
        if os.path.getsize(path) > MAX_TEXT_BYTES:
            return ""
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as e:
        _READ_ERRORS.setdefault(path, e)
        return ""


def parse_frontmatter(text: str) -> dict:
    """只解析这里用到的简单 YAML:key: value、key: [a, b]、key: 后接 - item 列表、布尔值。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    out: dict = {}
    key = None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        m = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
        if m and not line.startswith((" ", "\t")):
            key, val = m.group(1), m.group(2).strip()
            if val in ("", ">", ">-", "|", "|-"):
                out[key] = [] if val == "" else ""
            elif val.startswith("[") and val.endswith("]"):
                out[key] = [unquote(v) for v in val[1:-1].split(",") if v.strip()]
            elif val.lower() in ("true", "false"):
                out[key] = val.lower() == "true"
            else:
                out[key] = unquote(val)
            continue
        item = re.match(r"^\s+-\s+(.*)$", line)
        if key is not None and item and isinstance(out.get(key), list):
            out[key].append(unquote(item.group(1).strip()))
        elif key is not None and isinstance(out.get(key), str) and line.startswith((" ", "\t")):
            out[key] = (out[key] + " " + line.strip()).strip()  # 折叠的多行字符串
    return out


def unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def meaningful_lines(text: str) -> set[str]:
    """用于比较两份守则内容:去掉空行、@ 导入行与实验层的 Codex 接线行。"""
    return {
        l.strip() for l in text.splitlines()
        if l.strip() and not l.strip().startswith("@") and "~/1Project/oss/AGENTS.md" not in l
    }


class Collector:
    """按条数封顶的列表,记录是否截断。"""

    def __init__(self, cap: int):
        self.cap = cap
        self.items: list = []
        self.dropped = 0

    def add(self, item) -> None:
        if len(self.items) < self.cap:
            self.items.append(item)
        else:
            self.dropped += 1


def grep_lines(path: str, rel: str, pattern: re.Pattern, sink: Collector, text: str | None = None) -> None:
    text = _read_text(path) if text is None else text  # 批量扫描不进缓存
    for i, line in enumerate(text.splitlines(), 1):
        if pattern.search(line):
            sink.add({"file": rel, "line": i, "text": line.strip()[:200]})


# ---------------------------------------------------------------- 扫描

class Scanner:
    def __init__(self, root: str, max_hits: int):
        self.root = os.path.realpath(root)
        self.cap = max_hits
        self.files: list[tuple[str, str]] = []  # (abs, rel),不含软链接到目录的
        self.linked_dirs: list[str] = []  # 相对路径
        self.dot_dirs: list[str] = []  # .claude 等(含软链的),相对路径
        self.unreadable: dict[str, dict] = {}  # 真实路径 -> {path, error};同一处经软链再碰到只记一次

    def note_unreadable(self, path: str, err: OSError) -> None:
        real = os.path.realpath(path)
        if real not in self.unreadable:
            shown = self.rel(path) if self.inside(path) else path
            self.unreadable[real] = {"path": shown, "error": f"{type(err).__name__}: {err.strerror or err}"}

    def _walk_error(self, err: OSError) -> None:
        self.note_unreadable(err.filename or self.root, err)

    def listdir(self, path: str) -> list[str]:
        """排好序的目录项;读不了就记进 unreadable,当空目录处理。"""
        try:
            return sorted(os.listdir(path))
        except OSError as e:
            self.note_unreadable(path, e)
            return []

    def rel(self, path: str) -> str:
        r = os.path.relpath(path, self.root)
        return "." if r == "." else r.replace(os.sep, "/")

    def inside(self, path: str) -> bool:
        real = os.path.realpath(path)
        return real == self.root or real.startswith(self.root + os.sep)

    def link_target(self, path: str) -> str | None:
        if not os.path.islink(path):
            return None
        real = os.path.realpath(path)
        return self.rel(real) if self.inside(real) else real

    def walk(self) -> None:
        try:
            os.listdir(self.root)
        except OSError as e:
            raise EnvError(f"读不了根目录 {self.root}:{type(e).__name__}: {e.strerror or e}")
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False, onerror=self._walk_error):
            keep = []
            for d in dirnames:
                full = os.path.join(dirpath, d)
                if d in EXCLUDE_DIRS:
                    continue
                if os.path.islink(full):
                    self.linked_dirs.append(self.rel(full))
                    if d in DOT_AGENT_DIRS:
                        self.dot_dirs.append(self.rel(full))
                    continue
                if d in DOT_AGENT_DIRS:
                    self.dot_dirs.append(self.rel(full))
                keep.append(d)
            dirnames[:] = sorted(keep)
            for f in sorted(filenames):
                full = os.path.join(dirpath, f)
                self.files.append((full, self.rel(full)))

    def unique_files(self, rels: list[str]) -> list[str]:
        """按真实路径去重(CLAUDE.md 软链到 AGENTS.md 时只扫一次),保持顺序。"""
        seen, out = set(), []
        for rel in rels:
            real = os.path.realpath(os.path.join(self.root, rel))
            if real not in seen:
                seen.add(real)
                out.append(rel)
        return out

    # ------------------------------------------------------------ guides

    def scan_guides(self) -> list[dict]:
        guides = []
        for full, rel in self.files:
            name = os.path.basename(rel)
            parent = os.path.dirname(rel)
            if rel.lower() == ".github/copilot-instructions.md":
                scope = ""
            elif name in GUIDE_NAMES or name.lower() == "ai_policy.md":
                scope = os.path.dirname(parent) if os.path.basename(parent) == ".claude" else parent
            else:
                continue
            text = read_text(full)
            imports = []
            if name.startswith("CLAUDE"):
                imports = [m.group(1) for m in (re.match(r"^\s*@(\S+)", l) for l in text.splitlines()) if m]
            guides.append({
                "path": rel,
                "kind": name,
                "scope": scope or ".",
                "nested": bool(scope),
                "symlink_to": self.link_target(full),
                "imports": imports,
                "lines": len(text.splitlines()),
            })
        self._guide_visibility(guides)
        return guides

    def _real(self, rel: str) -> str:
        return os.path.realpath(os.path.join(self.root, rel))

    def _resolved_imports(self, g: dict) -> list[str | None]:
        """每个 @ 导入的真实路径;指向仓库外(如实验层接线 ~/1Project/oss/AGENTS.md)的记 None。"""
        out = []
        for target in g["imports"]:
            if target.startswith("~"):
                out.append(None)
                continue
            full = os.path.realpath(os.path.join(self.root, os.path.dirname(g["path"]), target))
            out.append(full if self.inside(full) else None)
        return out

    def _covers(self, reader: dict, g: dict) -> bool:
        """读 reader 的 agent 是否也就读到了 g 的内容。"""
        g_real, r_real = self._real(g["path"]), self._real(reader["path"])
        if g_real == r_real or g_real in self._resolved_imports(reader):
            return True
        wiring = os.path.join(self.root, ".lab")
        imports_ok = all(i is None or i == r_real or i.startswith(wiring + os.sep)
                         for i in self._resolved_imports(g))
        mine = meaningful_lines(read_text(g_real))
        if imports_ok and r_real in self._resolved_imports(g) and len(mine) <= 3:
            return True  # g 只是导入 reader 的薄壳
        if imports_ok and len(mine) <= 3 and os.path.basename(reader["path"]) in read_text(g_real):
            return True  # g 只是一两句「去读 reader」的指路
        return imports_ok and bool(mine) and mine <= meaningful_lines(read_text(r_real))

    def _guide_visibility(self, guides: list[dict]) -> None:
        family = {}
        for g in guides:
            fam = ("AGENTS" if g["kind"] in ("AGENTS.md", "AGENTS.override.md")
                   else "CLAUDE" if g["kind"] in ("CLAUDE.md", "CLAUDE.local.md") else None)
            g["_family"] = fam
            if fam:
                family.setdefault(g["scope"], {"AGENTS": [], "CLAUDE": []})[fam].append(g)
        root_claude_imports = set()
        for c in family.get(".", {}).get("CLAUDE", []):
            root_claude_imports.update(i for i in self._resolved_imports(c) if i)
        for g in guides:
            level = "conditional" if g["nested"] else "auto"
            fam = g.pop("_family")
            if fam == "AGENTS":
                g["codex"] = level
                g["claude"] = level if any(self._covers(c, g) for c in family[g["scope"]]["CLAUDE"]) else "no"
            elif fam == "CLAUDE":
                g["claude"] = level
                g["codex"] = level if any(self._covers(a, g) for a in family[g["scope"]]["AGENTS"]) else "no"
            else:
                g["claude"] = "auto" if self._real(g["path"]) in root_claude_imports else "no"
                g["codex"] = "no"
        for g in guides:  # 有 agent 读不到的,记下那个 agent 会读的守则里有没有指向它(如 AGENTS.md 写「去读 CLAUDE.md」)
            blind = [a for a in ("claude", "codex") if g[a] == "no"]
            if not blind:
                continue
            name = g["path"] if g["nested"] else os.path.basename(g["path"])
            g["mentioned_by"] = [
                o["path"] for o in guides
                if o is not g and any(o[a] == "auto" for a in blind)
                and self._real(o["path"]) != self._real(g["path"])
                and name in read_text(self._real(o["path"]))
            ]

    # ------------------------------------------------------------ rules 与 skills 的位置

    def locations(self, kind: str) -> list[dict]:
        locs = []
        for dot in self.dot_dirs:
            full = os.path.join(self.root, dot, kind)
            if not os.path.isdir(full):
                continue
            scope = os.path.dirname(dot) or "."
            via_link = os.path.islink(full) or os.path.islink(os.path.join(self.root, dot))
            real = os.path.realpath(full)
            locs.append({
                "path": f"{dot}/{kind}",
                "owner": os.path.basename(dot),
                "scope": scope,
                "nested": scope != ".",
                "symlink_to": (self.rel(real) if self.inside(real) else real) if via_link else None,
            })
        return locs

    def scan_rules(self) -> list[dict]:
        rules: dict[str, dict] = {}
        for loc in self.locations("rules"):
            base = os.path.join(self.root, loc["path"])
            for dirpath, _dirs, files in os.walk(base, onerror=self._walk_error):
                for f in sorted(files):
                    if not f.endswith((".md", ".mdc")) or f.lower().startswith("readme"):
                        continue
                    full = os.path.join(dirpath, f)
                    shown = loc["path"] + "/" + os.path.relpath(full, base).replace(os.sep, "/")
                    real = os.path.realpath(full)
                    entry = rules.get(real)
                    if entry is None:
                        fm = parse_frontmatter(read_text(full))
                        entry = rules[real] = {
                            "path": self.rel(real) if self.inside(real) else real,
                            "seen_at": [],
                            "paths": fm.get("paths"),
                            "globs": fm.get("globs"),
                            "alwaysApply": fm.get("alwaysApply"),
                            "path_field": fm.get("path"),
                            "description": fm.get("description"),
                        }
                    entry["seen_at"].append({"path": shown, "owner": loc["owner"], "nested": loc["nested"]})
        for full, rel in self.files:
            if rel.startswith(".github/instructions/") and rel.endswith(".instructions.md"):
                fm = parse_frontmatter(read_text(full))
                rules[os.path.realpath(full)] = {
                    "path": rel, "seen_at": [{"path": rel, "owner": "copilot", "nested": False}],
                    "applyTo": fm.get("applyTo"), "description": fm.get("description"),
                }
        out = []
        for entry in rules.values():
            claude = "no"
            for s in entry["seen_at"]:
                if s["owner"] == ".claude":
                    claude = "conditional" if s["nested"] else "auto"
                    if claude == "auto":
                        break
            entry["claude"] = claude
            entry["codex"] = "no"
            entry["always"] = not any(entry.get(k) for k in ("paths", "globs", "applyTo", "path_field")) \
                or entry.get("alwaysApply") is True
            out.append(entry)
        return sorted(out, key=lambda e: e["path"])

    def scan_skills(self) -> dict:
        skills: dict[str, dict] = {}
        not_discoverable = []
        toolkit = []
        locs = self.locations("skills")
        for loc in locs:
            base = os.path.join(self.root, loc["path"])
            for name in self.listdir(base):
                if name.startswith("."):
                    continue
                ep = os.path.join(base, name)
                shown = f"{loc['path']}/{name}"
                if name.startswith("oss-") and os.path.islink(ep) and not self.inside(ep):
                    toolkit.append(shown)
                    continue
                if os.path.isdir(ep):
                    md = os.path.join(ep, "SKILL.md")
                    try:
                        os.stat(md)
                    except (FileNotFoundError, NotADirectoryError):
                        pass
                    except OSError as e:  # 目录本身读不了:不是「没有 SKILL.md」
                        self.note_unreadable(ep, e)
                        continue
                    if not os.path.isfile(md):
                        not_discoverable.append({"path": shown, "reason": "目录里没有 SKILL.md"})
                        continue
                    real = os.path.realpath(ep)
                    entry = skills.get(real)
                    if entry is None:
                        fm = parse_frontmatter(read_text(md))
                        entry = skills[real] = {
                            "dir": self.rel(real) if self.inside(real) else real,
                            "name": fm.get("name"),
                            "name_mismatch": bool(fm.get("name")) and fm.get("name") != os.path.basename(real),
                            "has_description": bool(fm.get("description")),
                            "seen_at": [],
                        }
                    entry["seen_at"].append({
                        "path": shown, "owner": loc["owner"], "nested": loc["nested"],
                        "scope": loc["scope"], "dir_symlink": os.path.islink(ep),
                        "file_symlink": os.path.islink(md),
                    })
                elif os.path.isfile(ep) and name.lower().endswith(".md") and not name.lower().startswith("readme"):
                    not_discoverable.append({"path": shown, "reason": "散放的 .md,不是 skill 目录"})
        # 不在任何已知位置的 SKILL.md(如仓库根的 skills/、plugins/*/skills/)
        for full, rel in self.files:
            if os.path.basename(rel) != "SKILL.md":
                continue
            real = os.path.realpath(os.path.dirname(full))
            if real in skills:
                continue
            if any(real.startswith(os.path.realpath(os.path.join(self.root, l["path"])) + os.sep) for l in locs):
                continue  # 某个 skill 内部的嵌套 SKILL.md,随父 skill 走
            fm = parse_frontmatter(read_text(full))
            skills[real] = {
                "dir": self.rel(real), "name": fm.get("name"),
                "name_mismatch": bool(fm.get("name")) and fm.get("name") != os.path.basename(real),
                "has_description": bool(fm.get("description")), "seen_at": [],
            }

        wired = self.wire_skills_links()
        claude_side, codex_side = (os.path.realpath(os.path.join(self.root, s)) for s in WIRE_SIDES)
        out = []
        for real, entry in skills.items():
            seen = entry["seen_at"]
            entry["claude"] = self._skill_level(seen, ".claude")
            codex = self._skill_level(seen, ".agents")
            if codex == "no" and any(s["owner"] == ".codex" for s in seen):
                codex = "unknown"
            if codex == "auto" and all(s["file_symlink"] for s in seen if s["owner"] == ".agents"):
                codex = "unknown"  # 只有文件级 SKILL.md 软链,Codex 可能跳过
            entry["codex"] = codex
            gains = {side for (side, _name), target in wired.items() if target == real}
            entry["fixable_by_wire_skills"] = (
                (entry["claude"] != "auto" or entry["codex"] != "auto")
                and (entry["claude"] == "auto" or claude_side in gains)
                and (entry["codex"] == "auto" or codex_side in gains)
            )
            out.append(entry)
        return {
            "locations": locs,
            "skills": sorted(out, key=lambda e: e["dir"]),
            "not_discoverable": not_discoverable,
            "toolkit_links": toolkit,
        }

    def wire_skills_links(self) -> dict[tuple[str, str], str]:
        """模拟 wire-skills.sh 第 2 步会补的链接:{(所在侧的真实目录, 名字): 链接最终指向的真实目录}。

        一侧整体软链到另一侧时,两侧的真实目录相同,补在真实侧的链接两边都看得到。
        """
        plan: dict[tuple[str, str], str] = {}
        for to in WIRE_SIDES:
            if os.path.islink(os.path.join(self.root, to)):
                continue  # 整体是软链的一侧不补:它共享的是别处的目录
            to_real = os.path.realpath(os.path.join(self.root, to))
            for frm in WIRE_SOURCES:
                base = os.path.join(self.root, frm)
                if frm == to or not os.path.isdir(base):
                    continue
                for name in self.listdir(base):
                    entry = os.path.join(base, name)
                    if name.startswith((".", "oss-")) or not os.path.isfile(os.path.join(entry, "SKILL.md")):
                        continue
                    if (to_real, name) in plan or os.path.lexists(os.path.join(self.root, to, name)):
                        continue  # 同名条目已在(含断链、不成 skill 的目录),或前面的来源已补
                    plan[(to_real, name)] = os.path.realpath(entry)
        return plan

    @staticmethod
    def _skill_level(seen: list[dict], owner: str) -> str:
        levels = {("conditional" if s["nested"] else "auto") for s in seen if s["owner"] == owner}
        return "auto" if "auto" in levels else ("conditional" if levels else "no")

    # ------------------------------------------------------------ 贡献规矩

    def scan_contributing(self, guides: list[dict], rules: list[dict]) -> dict:
        contributing, dirs, pr_templates, issue_templates, codeowners, style, devguides = [], set(), [], [], [], [], []
        for full, rel in self.files:
            name = os.path.basename(rel)
            low = name.lower()
            parts = rel.split("/")
            if low.startswith("contributing") and len(parts) <= 4:
                contributing.append(rel)
            if low == "pull_request_template.md" or "/pull_request_template/" in rel.lower():
                pr_templates.append(rel)
            if "/issue_template/" in rel.lower() and rel.lower().startswith(".github/"):
                issue_templates.append(rel)
            if name == "CODEOWNERS":
                codeowners.append(rel)
            if low.endswith((".md", ".rst")) and STYLE_NAME.search(name):
                style.append(rel)
            if low.endswith((".md", ".rst")) and DEVGUIDE_NAME.search(name):
                devguides.append(rel)
            for i, p in enumerate(parts[:-1]):
                if p.lower() in CONTRIB_DIR_NAMES and i < 4:
                    dirs.add("/".join(parts[: i + 1]))
        must_read = Collector(self.cap)
        inside_rules = [r["path"] for r in rules if not os.path.isabs(r["path"])]
        for rel in self.unique_files([g["path"] for g in guides] + inside_rules + contributing):
            grep_lines(os.path.join(self.root, rel), rel, MUST_READ_PAT, must_read)
        return {
            "contributing": contributing,
            "contributing_dirs": sorted(dirs),
            "pr_templates": pr_templates,
            "issue_templates": issue_templates[: self.cap],
            "codeowners": codeowners,
            "style_guides": style,
            "developer_guides": devguides,
            "must_read_lines": must_read.items,
            "_dropped": {"must_read_lines": must_read.dropped},
        }

    # ------------------------------------------------------------ lint

    def scan_lint(self) -> dict:
        configs = []
        make_targets = []
        scripts = []
        for full, rel in self.files:
            name = os.path.basename(rel)
            if name == ".pre-commit-config.yaml":
                configs.append({"path": rel, **parse_precommit(read_text(full))})
            elif name == "Makefile" and "/" not in rel:
                for line in read_text(full).splitlines():
                    m = MAKE_TARGET.match(line)
                    if m:
                        make_targets.append(m.group(1))
            elif name in FORMAT_SCRIPTS and rel.count("/") <= 2:
                scripts.append(rel)
        return {"pre_commit": configs, "make_targets": sorted(set(make_targets)), "scripts": scripts}

    # ------------------------------------------------------------ CI

    def scan_ci(self, hw: Collector) -> dict:
        workflows = []
        other = []
        title_scripts = []
        permission_files = []
        for full, rel in self.files:
            name = os.path.basename(rel)
            low = name.lower()
            if rel.startswith(".github/workflows/") and low.endswith((".yml", ".yaml")) and rel.count("/") == 2:
                text = read_text(full)
                workflows.append({"file": rel, "lab_layer": name.startswith("oss-"), **parse_workflow(text, rel)})
                grep_lines(full, rel, HW_LOCAL, hw, text)
            elif rel.split("/")[0] in OTHER_CI_DIRS or OTHER_CI_FILES.match(name):
                other.append(rel)
                if low.endswith((".yml", ".yaml", ".groovy", ".json", ".sh", ".py", ".toml", ".txt")) \
                        or name.startswith("Jenkinsfile"):
                    grep_lines(full, rel, HW_LOCAL, hw)
            if re.search(r"(?i)pr[_-]?title", name):
                title_scripts.append(rel)
            if rel.startswith(".github/") and re.search(r"(?i)(permission|trusted|allowlist|whitelist)", name):
                permission_files.append(rel)
        return {
            "workflows": workflows,
            "other_ci_files": other[: self.cap],
            "other_ci_total": len(other),
            "title_check_scripts": title_scripts,
            "ci_permission_files": permission_files,
        }

    # ------------------------------------------------------------ 测试

    def scan_tests(self, hw: Collector) -> dict:
        test_dirs = []
        for d in TEST_DIR_NAMES:
            for base in (d, f"python/{d}"):
                full = os.path.join(self.root, base)
                if os.path.isdir(full) and not os.path.islink(full):
                    readmes = [f for f in self.listdir(full) if f.lower().startswith("readme")]
                    test_dirs.append({"path": base, "readme": readmes})
        markers = []
        for rel in ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"):
            markers += [{"file": rel, "marker": m} for m in parse_markers(os.path.join(self.root, rel))]
        for td in test_dirs:
            p = f"{td['path']}/pytest.ini"
            markers += [{"file": p, "marker": m} for m in parse_markers(os.path.join(self.root, p))]
        local_files, hw_files, cpu_files = [], [], []
        for full, rel in self.files:
            parts = rel.lower().split("/")
            name = parts[-1]
            ci_ish = any(p in CI_PATH_PARTS for p in parts[:-1])
            if ci_ish and not name.endswith(".py") and not rel.startswith(".github/workflows/"):
                stem = os.path.splitext(name)[0]
                if HW_LOCAL.search(stem):
                    local_files.append(rel)  # 名字里就是 5090、gb202、sm_120 的测试清单或基线
                elif HW_ANY.search(stem):
                    hw_files.append(rel)  # 按其他硬件命名的测试清单、runner 配置(如 l0_h100.yml)
            if "cpu" in name and name.endswith(CPU_HINT_EXT) \
                    and (ci_ish or any(p in ("docs", ".agents", "doc") for p in parts[:-1])):
                cpu_files.append(rel)
            if ci_ish and not rel.startswith(".github/workflows/") and parts[0] not in OTHER_CI_DIRS \
                    and name.endswith((".yml", ".yaml", ".json", ".txt", ".ini", ".cfg", ".toml", ".py", ".md")):
                grep_lines(full, rel, HW_LOCAL, hw)
        return {
            "test_dirs": test_dirs,
            "pytest_markers": markers,
            "local_gpu_named_files": local_files[: self.cap],
            "hardware_named_files": hw_files[: self.cap],
            "hardware_named_total": len(hw_files),
            "cpu_named_files": cpu_files[: self.cap],
        }

    # ------------------------------------------------------------ 政策

    def scan_policy(self, guides: list[dict], contrib: dict) -> dict:
        targets = [g["path"] for g in guides] + contrib["contributing"] + contrib["pr_templates"]
        for full, rel in self.files:
            parts = rel.split("/")
            if rel.startswith(".github/") and rel.count("/") == 1 and rel.lower().endswith(".md"):
                targets.append(rel)
            elif any(p.lower() in CONTRIB_DIR_NAMES for p in parts[:-1]) and rel.lower().endswith((".md", ".mdx", ".rst")):
                targets.append(rel)
        ordered = self.unique_files(targets)
        ai, dco, cla = Collector(self.cap), Collector(self.cap), Collector(self.cap)
        for rel in ordered:
            text = read_text(os.path.join(self.root, rel))
            grep_lines("", rel, AI_PAT, ai, text)
            grep_lines("", rel, DCO_PAT, dco, text)
            grep_lines("", rel, CLA_PAT, cla, text)
        signoff_hooks = []
        for full, rel in self.files:
            if os.path.basename(rel) == ".pre-commit-config.yaml":
                for line_no, line in enumerate(read_text(full).splitlines(), 1):
                    if re.search(r"(?i)(sign-?off|\bdco\b)", line):
                        signoff_hooks.append({"file": rel, "line": line_no, "text": line.strip()[:200]})
            elif rel.startswith(".github/") and re.search(r"(?i)(^|[/_.-])(dco|cla)([_.-]|$)", rel):
                signoff_hooks.append({"file": rel, "line": 0, "text": "按文件名命中"})
        return {
            "files_scanned": ordered,
            "ai": ai.items, "dco": dco.items, "cla": cla.items,
            "dco_cla_config": signoff_hooks[: self.cap],
            "_dropped": {"ai": ai.dropped, "dco": dco.dropped, "cla": cla.dropped},
        }


# ---------------------------------------------------------------- 解析器

def parse_precommit(text: str) -> dict:
    repos, hooks = [], []
    repo = None
    hook = None
    skip = []
    in_ci = False
    for line in text.splitlines():
        if re.match(r"^ci\s*:", line):
            in_ci = True
            continue
        if re.match(r"^\S", line):
            in_ci = False
        if in_ci:
            m = re.match(r"^\s+skip\s*:\s*\[(.*)\]", line)
            if m:
                skip = [unquote(s) for s in m.group(1).split(",") if s.strip()]
            continue
        m = re.match(r"^\s*-\s*repo\s*:\s*(\S+)", line)
        if m:
            repo = unquote(m.group(1))
            repos.append(repo)
            hook = None
            continue
        m = re.match(r"^\s*-\s*id\s*:\s*(\S+)", line)
        if m:
            hook = {"id": unquote(m.group(1)), "repo": repo}
            hooks.append(hook)
            continue
        if hook is None:
            continue
        m = re.match(r"^\s+(stages|entry|files|language|pass_filenames)\s*:\s*(.+)$", line)
        if m:
            val = m.group(2).strip()
            if val.startswith("["):
                val = [unquote(v) for v in val.strip("[]").split(",") if v.strip()]
            else:
                val = unquote(val)
            hook[m.group(1)] = val
    manual = [h["id"] for h in hooks if isinstance(h.get("stages"), list) and "manual" in h["stages"]]
    return {"repos": repos, "hooks": hooks, "manual_stage_hooks": manual, "ci_skip": skip}


def parse_workflow(text: str, rel: str) -> dict:
    lines = text.splitlines()
    events: list[str] = []
    in_on = False
    for line in lines:
        if re.match(r"^(on|\"on\"|'on'|true)\s*:", line):
            in_on = True
            rest = line.split(":", 1)[1]
            events += [e for e in re.findall(r"[a-z_]+", rest) if e in KNOWN_EVENTS]
            continue
        if in_on:
            if re.match(r"^\S", line):
                in_on = False
                continue
            m = re.match(r"^\s{1,4}([a-z_]+)\s*:", line) or re.match(r"^\s*-\s*([a-z_]+)\s*$", line)
            if m and m.group(1) in KNOWN_EVENTS:
                events.append(m.group(1))
    runs_on, runner_values, labels, commands, title_lines, hw_tokens = [], [], [], [], [], set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        m = re.match(r"^\s*(-\s*)?runs-on\s*:\s*(.*)$", line)
        if m:
            val = m.group(2).strip()
            if not val:
                indent = len(line) - len(line.lstrip())
                sub = []
                for nxt in lines[i + 1:]:
                    if not nxt.strip():
                        continue
                    if len(nxt) - len(nxt.lstrip()) <= indent:
                        break
                    sub.append(nxt.strip().lstrip("- ").strip())
                val = " ".join(sub)
            runs_on.append(val)
        for key, val in re.findall(r"(?<![\w-])(runner\w*|runs_on\w*)\s*:\s*['\"]?([A-Za-z0-9_.\-]+)", line):
            runner_values.append(val)
        if "label" in line.lower():
            for q in re.findall(r"'([^'\n]{1,60})'|\"([^\"\n]{1,60})\"", line):
                s = q[0] or q[1]
                if re.fullmatch(r"[A-Za-z0-9][\w:.+-]{0,39}", s) and s.lower() not in ("true", "false", "labels", "label"):
                    labels.append(s)
        if "comment" in line.lower():
            commands += re.findall(r"['\"](/[\w-]+(?: [^'\"\n]{0,40})?)['\"]", line)
        if TITLE_PAT.search(line):
            title_lines.append({"line": i + 1, "text": stripped[:200]})
        for tok in HW_ANY.findall(line):
            hw_tokens.add(tok.lower().replace(" ", "-"))
    return {
        "events": sorted(set(events)),
        "runs_on": sorted(set(runs_on)),
        "runner_values": sorted(set(runner_values)),
        "labels": sorted(set(labels)),
        "comment_commands": sorted(set(commands)),
        "title_check_lines": title_lines,
        "mentions_draft": "draft" in text,
        "hardware_tokens": sorted(hw_tokens),
    }


def parse_markers(path: str) -> list[str]:
    if not os.path.isfile(path):
        return []
    text = read_text(path)
    name = os.path.basename(path)
    if name == "pyproject.toml":
        if tomllib is not None:
            try:
                data = tomllib.loads(text)
            except (ValueError, TypeError):
                return []
            opts = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
            return [str(m).strip() for m in opts.get("markers", [])]
        return []
    out = []
    in_markers = False
    in_section = name == "pytest.ini"
    for line in text.splitlines():
        if re.match(r"^\[", line):
            in_section = line.strip() in ("[pytest]", "[tool:pytest]")
            in_markers = False
            continue
        if not in_section:
            continue
        m = re.match(r"^markers\s*=\s*(.*)$", line)
        if m:
            in_markers = True
            if m.group(1).strip():
                out.append(m.group(1).strip())
            continue
        if in_markers:
            if line.startswith((" ", "\t")) and line.strip():
                out.append(line.strip())
            elif line.strip():
                in_markers = False
    return out


# ---------------------------------------------------------------- 汇总

def build_gaps(guides: list[dict], rules: list[dict], skills: dict) -> list[dict]:
    gaps = []

    def missing(e):
        return [a for a in ("claude", "codex") if e.get(a) != "auto"]

    for g in guides:
        miss = missing(g)
        if miss:
            if g["nested"]:
                reason = f"嵌套守则,只管 {g['scope']}/ 下的改动;是否加载取决于 agent 与工作目录"
            elif g["kind"] in ("AGENTS.md", "AGENTS.override.md", "CLAUDE.md", "CLAUDE.local.md"):
                reason = "只有一个 agent 会读;另一个没有导入或软链"
            else:
                reason = "两个 agent 都不自动读"
            if g.get("mentioned_by"):
                reason += ";已被 " + "、".join(g["mentioned_by"]) + " 提到"
            gaps.append({"path": g["path"], "kind": "guide", "missing": miss, "reason": reason})
    for r in rules:
        scope = r.get("paths") or r.get("globs") or r.get("applyTo") or r.get("path_field") or "全部文件"
        gaps.append({"path": r["path"], "kind": "rule", "missing": missing(r),
                     "reason": f"规则作用范围:{scope}"})
    for s in skills["skills"]:
        miss = missing(s)
        if miss:
            where = ", ".join(x["path"] for x in s["seen_at"]) or s["dir"]
            gaps.append({"path": s["dir"], "kind": "skill", "missing": miss,
                         "reason": f"位于 {where}", "fixable_by_wire_skills": s["fixable_by_wire_skills"]})
    for nd in skills["not_discoverable"]:
        gaps.append({"path": nd["path"], "kind": "skill-like", "missing": ["claude", "codex"],
                     "reason": nd["reason"]})
    return gaps


def git_info(root: str) -> dict:
    def run(*args):
        try:
            r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    return {"head": run("rev-parse", "--short=12", "HEAD"), "branch": run("branch", "--show-current")}


def scan(root: str, max_hits: int) -> dict:
    _READ_ERRORS.clear()
    s = Scanner(root, max_hits)
    s.walk()
    hw = Collector(max_hits)
    guides = s.scan_guides()
    rules = s.scan_rules()
    skills = s.scan_skills()
    contrib = s.scan_contributing(guides, rules)
    lint = s.scan_lint()
    ci = s.scan_ci(hw)
    tests = s.scan_tests(hw)
    policy = s.scan_policy(guides, contrib)
    gaps = build_gaps(guides, rules, skills)
    for path, err in _READ_ERRORS.items():
        s.note_unreadable(path, err)
    unreadable = sorted(s.unreadable.values(), key=lambda u: u["path"])
    return {
        "schema": SCHEMA,
        "repo": s.root,
        "git": git_info(s.root),
        "summary": {
            "guides": len(guides),
            "nested_guides": sum(g["nested"] for g in guides),
            "rules": len(rules),
            "skills": len(skills["skills"]),
            "skills_both_agents": sum(x["claude"] == "auto" and x["codex"] == "auto" for x in skills["skills"]),
            "toolkit_links": len(skills["toolkit_links"]),
            "workflows": len(ci["workflows"]),
            "gaps": len(gaps),
            "unreadable": len(unreadable),
        },
        "guides": guides,
        "rules": rules,
        "skills": skills,
        "contributing": contrib,
        "lint": lint,
        "ci": ci,
        "tests": tests,
        "hardware": {"local_gpu_lines": hw.items, "dropped": hw.dropped},
        "policy": policy,
        "gaps": gaps,
        "unreadable": unreadable,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="扫描仓库的 agent 守则、rules、skill、贡献规矩、pre-commit、CI 与 5090 相关线索,输出 JSON。",
        epilog="退出码:0 没有可见性缺口;1 有缺口(见 gaps);2 用法或环境错误(含根目录读不了)。"
               "子路径因权限等读不了时跳过并列进 unreadable,不改变退出码。只读,不运行仓库里的脚本。",
    )
    ap.add_argument("repo", help="要扫描的仓库根目录")
    ap.add_argument("--max-hits", type=int, default=80, help="每类命中行的上限,默认 80;超出的只计数")
    args = ap.parse_args(argv)
    try:
        if not os.path.isdir(args.repo):
            raise EnvError(f"不是目录:{args.repo}")
        if args.max_hits < 1:
            raise EnvError("--max-hits 至少为 1")
        result = scan(args.repo, args.max_hits)
    except EnvError as e:
        print(json.dumps({"schema": SCHEMA, "error": str(e)}, ensure_ascii=False), file=sys.stderr)
        return EXIT_ENV
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return EXIT_GAPS if result["gaps"] else EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
