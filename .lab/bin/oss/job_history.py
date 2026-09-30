#!/usr/bin/env python3
"""把某个 CI job 在上游默认分支最近 N 次完成的 run 里的结果列成表,给出与 PR 失败对照的分类、
最后一次通过与第一次失败的边界、按 runner 的分组。

只依赖 Python 标准库与已登录的 gh。全部是只读调用,仓库必须用 -R 写明:

  gh api repos/<repo>                                                         默认分支(没给 --branch 时)
  gh api repos/<repo>/actions/workflows/<W>/runs?branch=<B>&event=<E>&status=completed&per_page=..&page=..
  gh api repos/<repo>/actions/runs/<id>/jobs?per_page=100&filter=all&page=<k>          每个 run 全部 attempt 的 job
  gh api repos/<repo>/actions/jobs/<job>/logs                     只在给了 --signature 时,取失败 job 的日志

--workflow 写 workflow 文件名(如 pr-test.yml,也接受 .github/workflows/ 下的路径)或数字 ID,不接受显示名。
大仓库上带过滤条件的 run 列表会返回不完整的结果:同一查询连续调用,total_count 逐次变大、时间窗可能落后几周,
几次之后才稳定(`gh run list` 同样)。所以每个事件重复查询,直到连续两遍 total_count 相同或达到 --max-passes,
结果取并集;window 给出每遍的 total_count、是否收敛(converged)与窗口里最新、最旧的 created_at。
converged 为 false 时结论不可靠:加大 --max-passes 重跑。

只统计 --event 列出的事件(默认 push 与 schedule):按分支名过滤也会带出 fork 上分支恰好叫 main 的
pull_request run,它们不是默认分支的结果。

每个 run 里匹配到的 job(--job 精确匹配名字;加 --job-regex 按正则搜,用于 matrix 分片或被调用方加了前缀的名字)
取最后一次 attempt 的结论,合成这个 run 的 outcome:
  failure  任一匹配 job 失败(failure、timed_out、startup_failure);给了 --signature 时,还要日志匹配签名
  success  全部匹配 job 成功
  other    其余(cancelled、skipped、签名不匹配的失败等)
  absent   这个 run 里没有匹配的 job
同一 run 里某个匹配 job 先失败、重跑后成功,记为 rerun_flip(不稳定的证据)。

verdict(把 PR 上这个 job 的失败与默认分支对照;runs 按时间从新到旧):
  pr-caused     有可比的 run 且全部 success,没有 rerun_flip:默认分支上是绿的
  pre-existing  最新的一次或连续最新几次 failure,更早的全是 success(或窗口里全是 failure)
  unstable      failure 之后又出现 success,或有 rerun_flip
  new-job       窗口里所有 run 都没有这个 job
  inconclusive  有这个 job,但没有一次 success 或 failure 可比
boundary 给出最新那段连续 failure 里最早的一次(first_fail)与它之前最近的一次 success(last_pass);
last_pass 为 null 表示窗口不够长,加大 --limit。

退出码:0 = verdict 为 pr-caused(这个 job 在默认分支窗口内全绿);1 = 其余 verdict;
        2 = 用法错误、gh 不可用或返回错误、窗口里一个 run 都没有、-o 或 --md 写不进去。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from urllib.parse import quote

DEFAULT_LIMIT = 3  # 出处见 skill 目录的 SOURCES.md
DEFAULT_EVENTS = ("push", "schedule")
DEFAULT_MAX_PASSES = 6
MAX_PAGES = 20
FAIL_CONCLUSIONS = {"failure", "timed_out", "startup_failure"}


class GhError(Exception):
    pass


def gh(args):
    try:
        # 显式 UTF-8 且容错:CI 日志里常有非法字节,严格解码会抛 UnicodeDecodeError,以退出码 1 带 traceback 退出。
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

def default_branch(repo):
    data = gh_json(["api", "repos/%s" % repo]) or {}
    branch = data.get("default_branch")
    if not branch:
        raise GhError("cannot resolve default branch of %s" % repo)
    return branch


def list_runs(repo, workflow, branch, events, limit, max_passes):
    wf = quote(workflow.rsplit("/", 1)[-1], safe="")
    per_page = min(100, limit)
    seen, totals = {}, {}
    for event in events:
        counts = totals.setdefault(event, [])
        for _ in range(max_passes):
            fetched = 0
            for page in range(1, MAX_PAGES + 1):
                data = gh_json(["api", "repos/%s/actions/workflows/%s/runs?branch=%s&event=%s&status=completed"
                                "&per_page=%d&page=%d" % (repo, wf, quote(branch, safe=""), quote(event, safe=""),
                                                          per_page, page)]) or {}
                if page == 1:
                    counts.append(data.get("total_count"))
                batch = data.get("workflow_runs") or []
                fetched += len(batch)
                for row in batch:
                    if row.get("event") == event and row.get("head_branch") == branch:
                        seen[row["id"]] = row
                if not batch or len(batch) < per_page or fetched >= limit:
                    break
            if len(counts) >= 2 and counts[-1] == counts[-2]:
                break
    runs = sorted(seen.values(), key=lambda r: (r.get("created_at") or "", r.get("id") or 0), reverse=True)
    return runs[:limit], totals


def list_jobs(repo, run_id):
    jobs, total = [], None
    for page in range(1, MAX_PAGES + 1):
        data = gh_json(["api", "repos/%s/actions/runs/%s/jobs?per_page=100&filter=all&page=%d"
                        % (repo, run_id, page)]) or {}
        batch = data.get("jobs") or []
        total = data.get("total_count", total)
        jobs.extend(batch)
        if not batch or len(batch) < 100 or (total is not None and len(jobs) >= total):
            break
    return jobs


def job_log(repo, job_id):
    return gh(["api", "repos/%s/actions/jobs/%s/logs" % (repo, job_id)])


# ---------------------------------------------------------------- 归类

def match_jobs(jobs, pattern, use_regex):
    if use_regex:
        rx = re.compile(pattern)
        return [j for j in jobs if rx.search(j.get("name") or "")]
    return [j for j in jobs if (j.get("name") or "") == pattern]


def summarize_run(repo, run, jobs, pattern, use_regex, signature):
    matched = match_jobs(jobs, pattern, use_regex)
    by_name = OrderedDict()
    for job in sorted(matched, key=lambda j: (j.get("name") or "", j.get("run_attempt") or 1)):
        by_name.setdefault(job.get("name") or "", []).append(job)

    rows, finals, flip = [], [], False
    for name, attempts in by_name.items():
        final = attempts[-1]
        earlier = [a.get("conclusion") for a in attempts[:-1]]
        if final.get("conclusion") == "success" and any(c in FAIL_CONCLUSIONS for c in earlier):
            flip = True
        row = {
            "name": name,
            "job_id": final.get("id"),
            "attempt": final.get("run_attempt"),
            "conclusion": final.get("conclusion"),
            "earlier_attempts": earlier,
            "runner_name": final.get("runner_name"),
            "labels": final.get("labels") or [],
            "url": final.get("html_url"),
            "signature_match": None,
        }
        if signature is not None and final.get("conclusion") in FAIL_CONCLUSIONS:
            row["signature_match"] = bool(signature.search(job_log(repo, final.get("id"))))
        rows.append(row)
        finals.append(row)

    if not finals:
        outcome = "absent"
    else:
        failed = [r for r in finals if r["conclusion"] in FAIL_CONCLUSIONS]
        if signature is not None:
            failed = [r for r in failed if r["signature_match"]]
        if failed:
            outcome = "failure"
        elif all(r["conclusion"] == "success" for r in finals):
            outcome = "success"
        else:
            outcome = "other"

    return {
        "run_id": run.get("id"),
        "sha": run.get("head_sha"),
        "created_at": run.get("created_at"),
        "event": run.get("event"),
        "run_attempt": run.get("run_attempt"),
        "run_conclusion": run.get("conclusion"),
        "url": run.get("html_url"),
        "outcome": outcome,
        "rerun_flip": flip,
        "jobs": rows,
    }


def classify(rows):
    """rows 按时间从新到旧。"""
    if all(r["outcome"] == "absent" for r in rows):
        return "new-job"
    seq = [r for r in rows if r["outcome"] in ("success", "failure")]
    if not seq:
        return "inconclusive"
    flip = any(r["rerun_flip"] for r in rows)
    outcomes = [r["outcome"] for r in seq]
    if all(o == "success" for o in outcomes):
        return "unstable" if flip else "pr-caused"
    # newest-first:先是一段 failure,之后全是 success -> 默认分支从某次起坏掉
    lead = 0
    while lead < len(outcomes) and outcomes[lead] == "failure":
        lead += 1
    if lead > 0 and all(o == "success" for o in outcomes[lead:]) and not flip:
        return "pre-existing"
    return "unstable"


def boundary(rows):
    seq = [r for r in rows if r["outcome"] in ("success", "failure")]
    if not seq or seq[0]["outcome"] != "failure":
        return None
    idx = 0
    while idx + 1 < len(seq) and seq[idx + 1]["outcome"] == "failure":
        idx += 1

    def ref(r):
        return {"run_id": r["run_id"], "sha": r["sha"], "created_at": r["created_at"], "url": r["url"]}

    return {"first_fail": ref(seq[idx]), "last_pass": ref(seq[idx + 1]) if idx + 1 < len(seq) else None}


def group_runners(rows):
    by_runner, by_labels = {}, {}
    for r in rows:
        for job in r["jobs"]:
            c = job["conclusion"]
            key = "failure" if c in FAIL_CONCLUSIONS else ("success" if c == "success" else "other")
            name = job["runner_name"] or "(none)"
            labels = ",".join(job["labels"]) or "(none)"
            for table, k in ((by_runner, name), (by_labels, labels)):
                table.setdefault(k, {"success": 0, "failure": 0, "other": 0})[key] += 1
    return by_runner, by_labels


def markdown(result):
    lines = ["| run | created | sha | event | outcome | jobs (conclusion, runner) |",
             "|---|---|---|---|---|---|"]
    for r in result["runs"]:
        jobs = "; ".join("%s: %s%s @ %s" % (j["name"], j["conclusion"],
                                            " (earlier attempts: %s)" % "/".join(map(str, j["earlier_attempts"]))
                                            if any(c != "success" for c in j["earlier_attempts"]) else "",
                                            j["runner_name"] or "-") for j in r["jobs"]) or "-"
        lines.append("| [%s](%s) | %s | %s | %s | %s | %s |" % (
            r["run_id"], r["url"], (r["created_at"] or "")[:16], (r["sha"] or "")[:10], r["event"],
            r["outcome"], jobs.replace("|", "\\|")))
    lines.append("")
    lines.append("verdict: **%s**" % result["verdict"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tabulate a CI job's recent results on the upstream default branch and classify a PR failure.",
        epilog="Exit codes: 0 = verdict pr-caused (job green on the default branch); 1 = any other verdict; "
               "2 = usage or gh error, no runs in the window, or -o/--md not writable.")
    parser.add_argument("-R", "--repo", required=True, help="upstream OWNER/REPO (required: gh defaults to the lab repo)")
    parser.add_argument("--workflow", required=True, help="workflow file name (e.g. pr-test.yml) or numeric ID")
    parser.add_argument("--job", required=True, help="job name (exact) or, with --job-regex, a regular expression")
    parser.add_argument("--job-regex", action="store_true", help="treat --job as a regular expression (re.search)")
    parser.add_argument("--branch", help="branch to inspect (default: the repository's default branch)")
    parser.add_argument("--event", action="append",
                        help="run events to include; repeatable (default: push and schedule)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="number of completed runs to inspect")
    parser.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES,
                        help="repeat each run-listing query until total_count repeats, at most this many times")
    parser.add_argument("--signature", help="regex that a failed job's log must match to count as the same failure")
    parser.add_argument("--md", help="also write a Markdown table here")
    parser.add_argument("-o", "--output", help="also write the JSON here")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        parser.error("--repo must look like OWNER/REPO")
    if args.limit < 1 or args.max_passes < 2:
        parser.error("--limit must be >= 1 and --max-passes >= 2")
    try:
        if args.job_regex:
            re.compile(args.job)
        signature = re.compile(args.signature, re.MULTILINE) if args.signature else None
    except re.error as exc:
        parser.error("bad regular expression: %s" % exc)
    if shutil.which("gh") is None:
        print(json.dumps({"error": "gh not found on PATH"}))
        return 2

    events = args.event or list(DEFAULT_EVENTS)
    try:
        branch = args.branch or default_branch(args.repo)
        runs, totals = list_runs(args.repo, args.workflow, branch, events, args.limit, args.max_passes)
        if not runs:
            print(json.dumps({"error": "no completed %s runs of %s on %s" % ("/".join(events), args.workflow, branch)},
                             ensure_ascii=False))
            return 2
        rows = [summarize_run(args.repo, run, list_jobs(args.repo, run["id"]), args.job, args.job_regex,
                              signature) for run in runs]
    except GhError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2

    by_runner, by_labels = group_runners(rows)
    counts = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    result = {
        "repo": args.repo,
        "workflow": args.workflow,
        "branch": branch,
        "events": events,
        "job": args.job,
        "job_regex": args.job_regex,
        "signature": args.signature,
        "window": {"runs": len(rows), "newest": rows[0]["created_at"], "oldest": rows[-1]["created_at"],
                   "total_counts": totals,
                   "converged": all(len(v) >= 2 and v[-1] == v[-2] for v in totals.values())},
        "matched_job_names": sorted({j["name"] for r in rows for j in r["jobs"]}),
        "counts": counts,
        "verdict": classify(rows),
        "boundary": boundary(rows),
        "by_runner": by_runner,
        "by_labels": by_labels,
        "runs": rows,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    for path, content in ((args.output, text + "\n"), (args.md, markdown(result))):
        if not path:
            continue
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            print(json.dumps({"error": "cannot write %s: %s" % (path, exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
    return 0 if result["verdict"] == "pr-caused" else 1


if __name__ == "__main__":
    sys.exit(main())
