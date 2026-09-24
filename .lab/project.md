# sglang-omni 项目画像

> 上游 sgl-project/sglang-omni,默认分支 main。只写这个项目和别的项目不一样的地方;通用做法见 `~/1Project/oss/AGENTS.md`。

## 规矩

- 风格规范:`.claude/skills/code-review/coding-style.md`(根目录 AGENTS.md、CLAUDE.md 都指向它)。重点:不防不存在的情况、不用 getattr/hasattr 探字段、有 if 必有 else、名字表达物理含义、只用一次的小 helper 内联、注释只写 why 并署名 `# note (name):`、注释与 docstring 里不用反引号
- 贡献指南:`docs/developer_reference/main.md`;PR 模板 `.github/pull_request_template.md`
- 确定性检查:`pre-commit run --files <改动的文件>`,提交前 `pre-commit run --all-files`。本地 hook 里 `check_leading_underscore.py --fix` 会直接改名,`check_if_else.py` 只报告;两条都只管 `sglang_omni/`
- 测试文件只能放 `tests/unit_test`、`tests/test_model`、`tests/test_ci`、`tests/utils`(Test Layout 检查)

## 验证

- CPU 测试:`pytest tests/unit_test/cpu/ -v`;上游通用单测 `pytest tests/ -m "not benchmark and not accelerator"`
- GPU 测试:走 hlab 项目 `oss-sglang-omni`(待注册),环境与命令注册后补在这里。上游 GPU CI 跑在 H100,外部贡献者触发不了,所以 PR 的 Accuracy Test 与 Benchmark 两段靠我们自己在 5090 上测;5090 测不了的写明
- 解读:interviewprep 的 `opensource/多模态/sglang-omni/`,基准 `89e60d0bf216`(v0.1.6)

## 上游惯例

- PR 标题:`[领域] 祈使句`,如 `[Fix] …`、`[Perf] …`、`[Qwen3-TTS] …`
- 合并方式与规模:squash;近期合入的 PR 中位约 4 个文件、117 行,小 PR 合得快
- CI 门槛:fork PR 自动跑 Lint、Docs Check、Test Layout;GPU 与 CPU CI 要维护者加 `run-ci` 标签,draft 不跑。作者能在自己的 PR 上评论 `/rerun-failed-ci`
- 认领:在对应 issue 下留言并 @ 该条的负责人,等确认再开工;很多 Roadmap 条目已有人留言认领。另有 Slack `#sglang-omni-dev`
- 主要 reviewer:Ratish1、zhaochenyang20、luojiaxuan、AkazaAkane、JiaxinD
