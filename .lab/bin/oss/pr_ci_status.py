#!/usr/bin/env python3
"""汇总上游 PR 当前 head 的 CI 事实,回答「全量 CI 跑没跑、为什么没跑、哪些 check 红或还在跑」。

只依赖 Python 标准库与已登录的 gh。全部是只读调用,仓库必须用 -R 写明(实验仓里 gh 默认指向实验仓):

  gh pr view <PR> -R <repo> --json ...                                  PR 元数据与 head 的 check
  gh api repos/<repo>/actions/runs?head_sha=<sha>&per_page=100&page=<k>   head 上每个 workflow 的 run
  gh api repos/<repo>/pulls/<PR>/commits?per_page=100&page=<k>            提交的 Signed-off-by 与 Verified
  gh api repos/<repo>/compare/<base>...<sha>                            落后与领先 base 的提交数

check 的归类(CheckRun 的 conclusion,StatusContext 的 state):
  pass             SUCCESS
  fail             FAILURE、TIMED_OUT、STARTUP_FAILURE、ERROR
  action_required  ACTION_REQUIRED
  cancelled        CANCELLED
  skipped/neutral/stale  同名
  pending          还没结束(QUEUED、IN_PROGRESS、WAITING、PENDING、EXPECTED 等)
detailsUrl 不指向 /actions/runs/ 的 check 标 external:日志不在 GitHub Actions,只报链接。

overall:
  not-triggered  head 上没有任何 Actions run 与 check,或每个 workflow 的最新 run 都在等批准(action_required)
  failing        有 check 为 fail 或 action_required
  pending        没有失败,但有 check 或 run 还没结束
  incomplete     没有失败与 pending,但有 cancelled、stale,有部分 workflow 的最新 run 在等批准,
                 或 --workflow 选定的 workflow 在 head 上没有 run
  passing        以上都不是
flags 另列机械可判的事实:draft、awaiting-approval、no-actions-runs、behind-base、commits-without-signoff、
unverified-commits、external-checks、missing-workflows。它们是否阻塞全量 CI,按项目画像的「CI 门槛」判断。

退出码:0 = passing;1 = 其余;2 = 用法错误、gh 不可用或返回错误、输出无法解析、-o 写不进去。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

DEFAULT_BEHIND_THRESHOLD = 20  # 出处见 skill 目录的 SOURCES.md
MAX_PAGES = 10

PR_FIELDS = [
    "number", "url", "state", "isDraft", "labels", "headRefOid", "headRefName", "baseRefName",
    "isCrossRepository", "mergeStateStatus", "statusCheckRollup",
]

CHECK_CONCLUSION_BUCKET = {
    "SUCCESS": "pass",
    "FAILURE": "fail",
    "TIMED_OUT": "fail",
    "STARTUP_FAILURE": "fail",
    "ERROR": "fail",
    "ACTION_REQUIRED": "action_required",
    "CANCELLED": "cancelled",
    "SKIPPED": "skipped",
    "NEUTRAL": "neutral",
    "STALE": "stale",
}
STATUS_STATE_BUCKET = {
    "SUCCESS": "pass",
    "FAILURE": "fail",
    "ERROR": "fail",
    "PENDING": "pending",
    "EXPECTED": "pending",
}
SIGNOFF_RE = re.compile(r"^Signed-off-by:\s*\S", re.MULTILINE)
RUN_URL_RE = re.compile(r"/actions/runs/(\d+)(?:/job/(\d+))?")


class GhError(Exception):
    pass


def gh(args):
    try:
        # 显式 UTF-8 且容错:输出里夹着非法字节时不抛 UnicodeDecodeError(那会以退出码 1 带 traceback 退出)。
        proc = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise GhError("gh not found on PATH") from exc
    if proc.returncode != 0:
        raise GhError("gh %s failed (exit %d): %s" % (" ".join(args), proc.returncode,
                                                       (proc.stderr or proc.stdout).strip()))
    return proc.stdout


def gh_json(args):
    out = gh(args)
    try:
        return json.loads(out or "null")
    except json.JSONDecodeError as exc:
        raise GhError("gh %s returned non-JSON output" % " ".join(args)) from exc


# ---------------------------------------------------------------- 采集

def fetch_runs(repo, sha):
    runs, total = [], None
    for page in range(1, MAX_PAGES + 1):
        data = gh_json(["api", "repos/%s/actions/runs?head_sha=%s&per_page=100&page=%d" % (repo, sha, page)])
        batch = (data or {}).get("workflow_runs") or []
        total = (data or {}).get("total_count", total)
        runs.extend(batch)
        if not batch or len(batch) < 100 or (total is not None and len(runs) >= total):
            break
    return runs


def fetch_commits(repo, pr):
    commits = []
    for page in range(1, MAX_PAGES + 1):
        batch = gh_json(["api", "repos/%s/pulls/%s/commits?per_page=100&page=%d" % (repo, pr, page)]) or []
        commits.extend(batch)
        if len(batch) < 100:
            break
    return commits


# ---------------------------------------------------------------- 归类

def normalize_check(item):
    kind = item.get("__typename") or ("StatusContext" if "context" in item else "CheckRun")
    if kind == "StatusContext":
        name = item.get("context") or ""
        url = item.get("targetUrl") or ""
        state = (item.get("state") or "").upper()
        bucket = STATUS_STATE_BUCKET.get(state, "pending")
        raw = state
        workflow = ""
    else:
        name = item.get("name") or ""
        url = item.get("detailsUrl") or ""
        status = (item.get("status") or "").upper()
        conclusion = (item.get("conclusion") or "").upper()
        if status and status != "COMPLETED":
            bucket, raw = "pending", status
        else:
            bucket = CHECK_CONCLUSION_BUCKET.get(conclusion, "pending" if not conclusion else "fail")
            raw = conclusion or status
        workflow = item.get("workflowName") or ""
    match = RUN_URL_RE.search(url)
    return {
        "name": name,
        "workflow": workflow,
        "bucket": bucket,
        "raw": raw,
        "url": url,
        "run_id": int(match.group(1)) if match else None,
        "job_id": int(match.group(2)) if match and match.group(2) else None,
        "external": match is None,
    }


def latest_per_workflow(runs):
    latest = {}
    counts = {}
    for run in runs:
        key = run.get("path") or run.get("name") or str(run.get("workflow_id"))
        counts[key] = counts.get(key, 0) + 1
        prev = latest.get(key)
        order = (run.get("created_at") or "", run.get("id") or 0)
        if prev is None or order > (prev.get("created_at") or "", prev.get("id") or 0):
            latest[key] = run
    rows = []
    for key, run in sorted(latest.items()):
        rows.append({
            "name": run.get("name"),
            "path": run.get("path"),
            "run_id": run.get("id"),
            "attempt": run.get("run_attempt"),
            "event": run.get("event"),
            "status": run.get("status"),
            "conclusion": run.get("conclusion"),
            "url": run.get("html_url"),
            "created_at": run.get("created_at"),
            "runs_for_head": counts[key],
        })
    return rows


def workflow_selected(row, selectors):
    path = row.get("path") or ""
    names = {row.get("name") or "", path, os.path.basename(path)}
    return any(sel in names for sel in selectors)


def summarize(pr, runs, commits, compare, compare_error, selectors, behind_threshold, repo):
    workflows = latest_per_workflow(runs)
    checks = [normalize_check(item) for item in (pr.get("statusCheckRollup") or [])]
    missing = []
    if selectors:
        chosen = [w for w in workflows if workflow_selected(w, selectors)]
        chosen_names = {w.get("name") for w in chosen}
        missing = [s for s in selectors if not any(workflow_selected(w, [s]) for w in workflows)]
        workflows = chosen
        checks = [c for c in checks if c["workflow"] in chosen_names]

    buckets = {}
    for check in checks:
        buckets.setdefault(check["bucket"], []).append(check)
    counts = {key: len(val) for key, val in sorted(buckets.items())}

    without_signoff, unverified = [], []
    for item in commits:
        sha = (item.get("sha") or "")[:12]
        commit = item.get("commit") or {}
        if not SIGNOFF_RE.search(commit.get("message") or ""):
            without_signoff.append(sha)
        verification = commit.get("verification") or {}
        if not verification.get("verified", False):
            unverified.append({"sha": sha, "reason": verification.get("reason")})

    awaiting = [w for w in workflows if w.get("conclusion") == "action_required"]
    runs_pending = [w for w in workflows if w.get("status") not in (None, "completed")]
    behind_by = (compare or {}).get("behind_by")

    flags = []
    if pr.get("isDraft"):
        flags.append({"flag": "draft", "detail": "PR is a draft; many gates skip drafts"})
    if awaiting:
        flags.append({"flag": "awaiting-approval",
                      "detail": "latest run is action_required (maintainer must approve workflows): "
                                + ", ".join(sorted(w.get("path") or w.get("name") or "?" for w in awaiting))})
    if not runs:
        flags.append({"flag": "no-actions-runs", "detail": "no GitHub Actions run exists for the head sha"})
    if isinstance(behind_by, int) and behind_by > behind_threshold:
        flags.append({"flag": "behind-base",
                      "detail": "%d commits behind %s (threshold %d)" % (behind_by, pr.get("baseRefName"),
                                                                         behind_threshold)})
    if without_signoff:
        flags.append({"flag": "commits-without-signoff", "detail": ", ".join(without_signoff)})
    if unverified:
        flags.append({"flag": "unverified-commits", "detail": ", ".join(u["sha"] for u in unverified)})
    external = [c for c in checks if c["external"]]
    if external:
        flags.append({"flag": "external-checks", "detail": ", ".join(c["name"] for c in external)})
    if missing:
        flags.append({"flag": "missing-workflows", "detail": ", ".join(missing)})

    only_awaiting = bool(workflows) and len(awaiting) == len(workflows)
    if (not runs and not checks) or (only_awaiting and not checks) or (selectors and not workflows and not checks):
        overall = "not-triggered"
    elif buckets.get("fail") or buckets.get("action_required"):
        overall = "failing"
    elif buckets.get("pending") or runs_pending:
        overall = "pending"
    elif buckets.get("cancelled") or buckets.get("stale") or missing or awaiting:
        overall = "incomplete"
    else:
        overall = "passing"

    def brief(items):
        return [{k: c[k] for k in ("name", "workflow", "raw", "url", "run_id", "job_id", "external")}
                for c in items]

    return {
        "repo": repo,
        "pr": pr.get("number"),
        "url": pr.get("url"),
        "state": pr.get("state"),
        "draft": bool(pr.get("isDraft")),
        "head_sha": pr.get("headRefOid"),
        "head_branch": pr.get("headRefName"),
        "base": pr.get("baseRefName"),
        "cross_repository": pr.get("isCrossRepository"),
        "labels": sorted(label.get("name") for label in (pr.get("labels") or [])),
        "merge_state": pr.get("mergeStateStatus"),
        "behind_by": behind_by,
        "ahead_by": (compare or {}).get("ahead_by"),
        "compare_error": compare_error,
        "workflows": workflows,
        "checks": {
            "counts": counts,
            "failing": brief(buckets.get("fail", []) + buckets.get("action_required", [])),
            "pending": brief(buckets.get("pending", [])),
            "cancelled": brief(buckets.get("cancelled", [])),
            "skipped": sorted(c["name"] for c in buckets.get("skipped", [])),
        },
        "commits": {"total": len(commits), "without_signoff": without_signoff, "unverified": unverified},
        "missing_workflows": missing,
        "flags": flags,
        "overall": overall,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize CI facts for the current head of an upstream PR (read-only gh calls).",
        epilog="Exit codes: 0 = passing; 1 = not-triggered, failing, pending or incomplete; "
               "2 = usage or gh error, or -o not writable.")
    parser.add_argument("pr", help="PR number")
    parser.add_argument("-R", "--repo", required=True, help="upstream OWNER/REPO (required: gh defaults to the lab repo)")
    parser.add_argument("--workflow", action="append", default=[],
                        help="only consider this workflow (file name, path or display name); repeatable")
    parser.add_argument("--behind-threshold", type=int, default=DEFAULT_BEHIND_THRESHOLD,
                        help="flag behind-base when the head is more than this many commits behind the base")
    parser.add_argument("-o", "--output", help="also write the JSON here")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        parser.error("--repo must look like OWNER/REPO")
    if not args.pr.isdigit():
        parser.error("pr must be a number")
    if shutil.which("gh") is None:
        print(json.dumps({"error": "gh not found on PATH"}))
        return 2

    try:
        pr = gh_json(["pr", "view", args.pr, "-R", args.repo, "--json", ",".join(PR_FIELDS)])
        if not isinstance(pr, dict) or not pr.get("headRefOid"):
            raise GhError("unexpected gh pr view output")
        sha = pr["headRefOid"]
        runs = fetch_runs(args.repo, sha)
        commits = fetch_commits(args.repo, args.pr)
        compare, compare_error = None, None
        try:
            compare = gh_json(["api", "repos/%s/compare/%s...%s" % (args.repo, pr.get("baseRefName"), sha)])
        except GhError as exc:
            compare_error = str(exc)
    except GhError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2

    result = summarize(pr, runs, commits, compare, compare_error, args.workflow, args.behind_threshold, args.repo)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError as exc:
            print(json.dumps({"error": "cannot write %s: %s" % (args.output, exc)}, ensure_ascii=False),
                  file=sys.stderr)
            return 2
    return 0 if result["overall"] == "passing" else 1


if __name__ == "__main__":
    sys.exit(main())
