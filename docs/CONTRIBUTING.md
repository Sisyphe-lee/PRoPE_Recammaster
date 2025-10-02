## 贡献指南（Git 流程与提交规范）

### 分支命名
- 功能：`feat/<short-name>`
- 修复：`fix/<short-name>`
- 文档：`docs/<short-name>`
- 重构/清理：`refactor/<short-name>`
- 实验/探索：`exp/<short-name>`

建议短名使用中划线，语义清晰：如 `feat/snr-weight-plot`。

### 提交流程（本地）
1) 更新主分支：
```bash
git checkout main
git pull --rebase
```
2) 创建工作分支：
```bash
git checkout -b feat/your-feature
```
3) 开发与自检：
```bash
uv run ruff check . --fix
uv run pytest -q
```
4) 提交：
```bash
git add -A
git commit -m "feat: short summary\n\n- change 1\n- change 2"
git push -u origin HEAD
```

### 同步主分支与解决冲突
- 在功能分支上保持与主分支同步，优先使用 rebase：
```bash
git fetch origin
git rebase origin/main
# 如遇冲突：按文件逐个解决 → git add → git rebase --continue
```
- rebase 优点：保持提交历史线性、易读。若分支已共享，出于安全可使用 merge：
```bash
git merge origin/main
```


### 代码质量清单（提交前）
- `uv run ruff check . --fix` 无错误；必要时 `uv run ruff format`。
- `uv run pytest -q` 通过（或至少不破坏现有测试）。
- 重要函数/类有类型注解与简明 docstring。
- README/文档同步更新（如新增 CLI 参数/行为变化）。

### 常见注意事项
- 大文件/二进制：避免提交模型权重与大媒体文件（放入 `outputs/` 或使用外部存储/DVC）。
- 依赖管理：新增依赖请通过 `uv add`，并说明引入理由；避免不必要的重型依赖。
- 兼容性：遵循 `requires-python`，避免使用过低版本不支持的语法。
- 性能敏感改动：注明时间/空间复杂度影响与基准验证方式。




### Changelog 模版
- 每次合并到 `main` 的重要变更，建议在 `CHANGELOG.md` 中追加一节，并遵循以下结构：
- 改动 @人员表明是谁写的 @yyb or @sjh
- 版本号与日期采用：`v<major>.<minor>.<patch> - YYYY-MM-DD`（示例：`v0.21.1(大改时用新的版本号) - 2025-09-22`）。
- 维度固定为：`新增`、`变更`、`修复`、`构建与工具链`、`其他`。若无内容可省略对应小节。

示例模版：

```markdown
## vX.Y.Z @yyb - YYYY-MM-DD

### 新增
- 新增点 1
- 新增点 2

### 变更
- 行为或接口变化（若有破坏性变更，需明确注明 BREAKING CHANGE）

### 修复
- 修复点 1（指明问题位置和影响范围）

### 构建与工具链
- 依赖/脚手架/CI 配置调整

### 其他
- 说明性条目（如：默认参数、示例更新、内部重构说明等）
```

### Commit 规范与模版
- 格式：`<type>(scope)?: <summary>`
- 常用 type：`feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`。
- 示例：
  - `feat(cli): add train subcommand`
  - `fix(samplers): correct Heun step size`
  - `docs: add development QA and workflow`


> 统一遵循 Conventional Commits；标题尽量使用祈使句，≤ 72 字符；正文用项目符号；必要时加“影响范围/验证方式/迁移指南/关联”。

1) 小修补（patch/minor fix）
```text
fix(<scope>): short summary

- what changed succinctly
- impact: <module/behavior>
- verification: <cmd or note>
```

示例：
```text
fix(sampling): prevent NaN when sigma=0

- clamp min sigma to 1e-5 in edm scheduler
- impact: avoids rare NaN on edge cases
- verification: pytest tests/test_samplers.py::test_edm_no_nan
```

2) 文档更新（docs-only）
```text
docs(<scope>): improve <section>

- add tutorial/examples
- fix links; clarify usage
```

示例：
```text
docs(readme): add quickstart for hydra configs

- add +profiles usage and links to docs/ROADMAP.md
```

3) 局部重构（不改行为）
```text
refactor(<scope>): simplify <component> w/o behavior change

- extract helpers; rename variables
- improve typing & docstrings
- no functional change; tests green
```

示例：
```text
refactor(trainer): split checkpoint io; improve types

- extract save/load into checkpoint_manager
- replace Optional[...] with X | None; add return types
- tests: passed locally `uv run pytest -q`
```

4) 小功能（向后兼容）
```text
feat(<scope>): add <feature>

- brief capabilities
- default behavior unchanged unless noted
- add minimal tests/docs if applicable
```

示例：
```text
feat(examples): add convergence video & unified axes

- periodic sampling (sample_interval/steps/count)
- export convergence_*.mp4 via imageio-ffmpeg
- unify axes: x=GT range; y=max density + margin
```

5) 重大变更/版本发布（默认/接口改变）
```text
feat!: <scope>: <breaking summary>

- breaking: what & why
- migration: how to adapt configs/calls
- verification: tests/benchmarks

Refs: vX.Y.0 (planned) or release notes link
```

示例：
```text
feat!: sampling: default sampler -> 'dpm' (DPMSolver1)

- breaking: default changed from 'euler' to 'dpm'
- migration: set --sampler=euler to keep old behavior
- verification: improved stability on CIFAR; tests updated

Refs: CHANGELOG "v0.22 - 2025-09-30"
```

6) 依赖与工具链（建议单独提交）
```text
chore(deps): add <pkg> to dev extras

- add imageio & imageio-ffmpeg for video export
- lockfile updated
```

7) 复合改动的拆分建议
- 优先拆为两类提交：功能/修复 与 依赖/脚手架（`feat|fix` vs `chore(build|deps)`）。
- 若必须合并在一个提交：标题用最主要的类型（常为 `feat`/`fix`），正文按分组列出。


<!-- 最佳实践：
- 为每个条目提供最小可验证信息（涉及文件/模块、可复现方式或影响面）。
- 重大接口或行为变更需在 README/Docs 同步说明，并在此处链接到对应文档段落。
- 若一次 PR 体量大，可在“其他”中给出导航索引，帮助审阅者快速理解。 -->



<!-- ### Pull Request（PR）建议
- PR 小而频：专注单一主题，便于评审。
- 自检通过：在本地确保 `ruff` 与 `pytest` 通过。
- 描述清晰：概述动机、修改点、影响范围、测试方式与风险。
- 变更可追踪：尽量避免顺手格式化大段无关代码，降低 diff 噪音。

### 发布（未来阶段）
- 打包：`python -m build`
- 验证：`pip install dist/*.whl && nano-diffusers --help`
- 发布：`twine upload dist/*` -->

<!-- 如对流程有疑问，请先查阅 `docs/DEVELOPMENT_QA.md`，或在 PR 中提出讨论。 -->


