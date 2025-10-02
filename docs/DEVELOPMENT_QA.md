## 开发 Q&A（工具、文件与最佳实践流程）

### 1. 这些文件分别做什么？
- `pyproject.toml`：项目元数据与依赖的“单一事实来源”，同时集中配置 `ruff`、`pytest` 等工具。
- `.pre-commit-config.yaml`：`pre-commit` 钩子列表；提交前自动运行 `ruff`/格式化等。
- `README.md`：总览与上手指南，含安装、CLI、构建发布与开发流程速查。
- `nanodiffusers/cli.py`：命令行入口，实现 `main()`；通过 `pyproject.toml` 的 `[project.scripts]` 暴露为 `nano-diffusers`。
- `.gitignore`：忽略提交的文件/目录（如 `.venv/`、`dist/`、`outputs/`）。
- `tests/`：`pytest` 测试目录（建议添加以保障关键模块）。

### 2. uv 的作用是什么？
- 统一“创建虚拟环境 + 安装/解析依赖 + 跑命令”的一体化工具。
- 常用命令：
  - `uv sync`：安装/同步依赖（使用锁文件时可复现）。
  - `uv add <pkg>`：添加依赖（`--dev` 添加到开发依赖）。
  - `uv run <cmd>`：在虚拟环境中运行命令（如 `uv run pytest`）。

### 3. ruff 与 pylint 有何区别？
- `ruff`：更快，覆盖 pycodestyle/pyflakes/isort/pyupgrade 等，支持格式化；常用作第一层守门。
- `pylint`：规则更严格/更语义化，适合大型或对规范要求很高的项目；本仓库中为可选补充。

### 4. pytest 用来做什么？
- 编写和运行单元/集成测试，保障回归；结合 CI 可实现自动化质量门禁。
- 常用：`uv run pytest -q`。

### 5. pre-commit 的价值与用法（详解）
- 作用：在每次 `git commit` 之前自动运行一组“钩子”（hooks），例如 Ruff 检查与格式化，阻止不符合规范的改动进入提交，保持代码库一致性。
- 工作机制：根据仓库根目录的 `.pre-commit-config.yaml` 定义要运行的 hooks（本项目内置了 `ruff` 与 `ruff-format`）。

常用操作：
- 初始化安装（一次性）：
```bash
pre-commit install
```
- 对已暂存文件运行（commit 时会自动触发）：
```bash
pre-commit run
```
- 对整个仓库所有文件运行（全量体检）：
```bash
pre-commit run -a
```
- 跳过 pre-commit（不建议常用，仅应在紧急场景或者已知无害情况下）：
```bash
git commit -m "msg" --no-verify
```

与 Ruff 的协作：
- 钩子里已配置 `ruff --fix` 与 `ruff-format`，会在提交前尽量自动修复风格问题。
- 若钩子失败，请先本地手动修复后再提交：
```bash
uv run ruff check . --fix
uv run ruff format
```

为项目添加更多钩子：
1) 编辑 `.pre-commit-config.yaml`，追加需要的 repo/hook（例如 `check-yaml`、`end-of-file-fixer`、`trailing-whitespace` 等）。
2) 运行：`pre-commit autoupdate`（可选，更新到各钩子的最新稳定版本）。
3) 运行：`pre-commit run -a` 验证。

常见问题排查：
- “pre-commit 命令找不到”：确保已安装 `pre-commit`（`uv add --dev pre-commit` 或 `pip install pre-commit`），并确认当前 shell PATH 正确。
- “每次都很慢”：首次运行会安装 hooks 的虚拟环境，后续有缓存会更快；尽量只在改动较多时跑全量 `-a`，日常提交只跑暂存文件。
- “修复后仍失败”：可能有未暂存的文件或额外错误，先 `git add -A` 再重试；或单独运行对应工具（如 `ruff`）查看具体报错行。

### 9. 总览对照表（问题 → 工具 → 阶段 → 并行性 → 流程位置）

| 名词/工具 | 解决的问题 | 主要应用阶段 | 可并行 | 在开发流程中的位置 |
|---|---|---|---|---|
| uv | 快速安装/锁定依赖、统一虚拟环境与命令运行 | 环境初始化、日常开发 | 是（和 ruff/pytest 并行执行不同命令） | 初始化：`uv sync`；运行：`uv run <cmd>` |
| pyproject.toml | 标准化项目元数据与依赖、集中工具配置 | 项目配置、构建/发布 | 否（配置文件本身） | 全流程基座；构建依赖于它 |
| ruff | 代码风格/静态检查、快速自动修复、格式化 | 编码后、提交前 | 是（与 pytest 并行） | Lint：`uv run ruff check . --fix`；Format：`uv run ruff format` |
| pylint（可选） | 更严格的语义检查 | 编码后、提交前/CI | 是（与 ruff/pytest 并行） | 严格检查：`uv run pylint nanodiffusers` |
| pytest | 单元/集成测试，保障回归 | 编码后、提交前/CI | 是（与 ruff 并行） | 测试：`uv run pytest -q` |
| pre-commit | 提交前自动执行检查/修复 | 提交时（本地钩子） | 钩子内串行；多钩子整体可并行感知低 | 提交：`pre-commit install` 后自动生效；`pre-commit run -a` 全量 |
| GitHub Actions（CI） | Push/PR 上自动化 lint/test/build | 团队协作/稳定阶段 | 是（CI job 并行矩阵） | 远端：`.github/workflows/ci.yml` |
| python -m build | 产出 wheel/sdist | 发布前/本地自检 | 否 | 构建：`python -m build` |
| twine | 上传 PyPI | 发布阶段 | 否 | 发布：`twine upload dist/*` |
| README.md | 用户与贡献者入口文档 | 全流程 | 否 | 查阅安装/CLI/构建/流程 |
| .pre-commit-config.yaml | pre-commit 钩子清单 | 提交前 | 否 | 维护钩子：新增/升级 |
| .gitignore | 忽略不应提交的文件 | 全流程 | 否 | 保持仓库整洁 |
| nanodiffusers/cli.py | 提供 `nano-diffusers` CLI 入口 | 使用/演示/验证 | 否 | 运行：`nano-diffusers --help` |

建议的一次提交内的并行化安排：
- 本地可并行运行：`ruff check` 与 `pytest`（分别在两个终端或通过任务并发）；完成后再统一 `git commit`。
- 构建/打包通常放在功能完成后的“里程碑点”再执行，避免频繁构建影响节奏。

参考顺序（线性视角）：
1) 初始化：`uv sync` → `pre-commit install`
2) 编码
3) 并行质量保障：`uv run ruff check . --fix` 与 `uv run pytest -q`
4) commit & push：`git add -A && git commit -m ... && git push`
5) 按需构建/安装自检：`python -m build && pip install dist/*.whl`；CLI 验证：`nano-diffusers --help`

### 10. 常用诊断命令与调试实践（结合本仓库实例）

- `source $HOME/.local/bin/env`
  - 作用：将 uv 安装的可执行加入当前 shell PATH（`~/.local/bin`），保证 `uv/uvx` 等命令可用。
  - 何时：首次安装 uv 后、或新开 shell 时。
  - 典型问题：找不到 `uv` 命令 ⇒ 执行该命令或重启 shell。

- `uv run ruff check . --fix`
  - 作用：在 uv 虚拟环境内运行 Ruff 对项目进行静态检查并自动修复可修复项。
  - 何时：提交前本地质量门禁；调试风格/导入顺序/类型现代化（如 `Dict/List/Tuple`）。
  - 典型输出：E402 导入顺序、UP006/UP035 类型提示升级、E501 行过长等；可通过 `--fix` 自动修部分问题，长行需手动拆分。

- `python -m py_compile <file.py>`
  - 作用：进行语法级编译检查，快速发现缩进、拼写等语法错误。
  - 何时：出现 Ruff 难以定位的缩进/语法错误时；或在大改动后快速 sanity check。
  - 典型输出：无输出表示通过；有错误会抛出具体文件与行号。

- `python -m pyflakes <file.py>`（可选）
  - 作用：轻量静态检查器，发现未使用变量、未定义名称等语义问题（不修改代码）。
  - 何时：不想引入完整 linter 时的快速扫描；或与 Ruff 互补。
  - 提示：若提示模块不存在，可 `uv add --dev pyflakes` 后使用。

- `uv run pytest -q`
  - 作用：运行测试用例，验证改动未破坏行为；`-q` 精简输出。
  - 何时：提交流程的固定环节，或在本地重构后立即验证。

实战示例（本仓库中的一次 debug 流程）：
- 症状：重构 `nanodiffusers/training/checkpoint_manager.py` 后 Ruff 报大量错误（导入顺序、类型提示、长行、缩进）。
- 操作：
  1) `uv run ruff check nanodiffusers/training/checkpoint_manager.py --fix`：先自动修；
  2) 针对未自动修复项：
     - 类型提示现代化：`Dict/List/Optional` 改为 `dict/list | None`；
     - 长行拆分：格式化 f-string、参数换行；
     - 导入顺序：确保 `from __future__ import annotations` 紧随模块文档字符串后，其他导入置顶；
  3) `python -m py_compile nanodiffusers/training/checkpoint_manager.py`：验证语法无误；
  4) 再次 `uv run ruff check ...`：确认零报错后再提交。

经验窍门：
- 大改后先跑 `py_compile` 再跑 Ruff，能更快定位纯语法问题；
- 对重复的大规模报错，可分块修：先导入顺序和语法，再类型现代化，最后处理长行；
- 结合 `pre-commit run -a` 做全仓库体检，确保遗漏文件也被检查到。

### 6. GitHub Actions 什么时候启用？
- 当开始多人协作或对稳定性有较高要求时启用。自动在 PR/Push 上跑 lint 与测试，保证主分支健康。
- 最小工作流示例已放在 README 对应章节，可复制到 `.github/workflows/ci.yml`。

### 7. 每次开发应当走怎样的“最佳实践流程”？
1) 初始化（一次性）
```bash
uv sync
pre-commit install
```
2) 日常开发循环
```bash
git checkout -b feat/your-feature

# 编码 → 本地质量保障
uv run ruff check . --fix
uv run pytest -q

git add -A
git commit -m "feat: ..."
git push origin HEAD
```
3) 按需：验证 CLI 与构建
```bash
nano-diffusers --help
python -m build && pip install dist/*.whl
```

备注：`torch` 不作为硬依赖，请按 PyTorch 官网根据平台/CUDA 单独安装。

### 8. 常见问题
- Q：为什么不用 `requirements.txt`？
  - A：`pyproject.toml` 为现代标准，集中声明依赖与工具配置；如需“锁定”，可结合 uv 的锁文件/同步机制。
- Q：本地和 CI 的 Python 版本不一致怎么办？
  - A：优先对齐版本；或在 `pyproject.toml` 的 `requires-python` 指定下限，同时在 CI 的 `setup-python` 对齐版本。
- Q：Ruff 报了很多 `List/Dict/Tuple`？
  - A：建议逐步替换为内置（`list/dict/tuple`）与联合类型 `X | None`，提升类型提示现代化程度。




## 11. 代码中的 TODO/@ 使用规范（含 Todo Tree 配置）

### 11.1 标签与优先级
- 推荐标签：`TODO`（待做）、`FIXME`（需尽快修复）、`REVIEW`（待评审）、`NOTE`（说明）、`QUESTION`（问题）、`PERF`（性能）、`DEBT`（技术债）。
- 优先级：`[P0]` 阻塞、`[P1]` 重要、`[P2]` 可延后。

### 11.2 标注格式（强烈建议统一）
- 基本格式：`TAG: @owner YYYY-MM-DD [PX] 简要动作 - 细节/上下文（可选）`
- 例：
```python
# TODO: @yyb 2025-09-23 [P0] 修复 1D 采样溢出 - 见 examples/9_1d_toy_training.py 的 max density 计算
# FIXME: @sjh 2025-09-23 [P1] EMA 恢复时 device 同步
# REVIEW: @yyb 2025-09-24 [P2] Heun 步长推导是否正确
```

支持多语言：
```bash
# TODO: @yyb 2025-09-23 [P1] scripts/run_example.sh 增加超参转发
```
```yaml
# TODO: @sjh 2025-09-23 [P2] configs/config.yaml 的 profiles 改为可运行示例
```

规范要点：
- `@owner` 为当前负责人，仅用于代码内协作提示（暂不对接 Issue/PR）。
- 日期为创建日，便于清理与回溯；必要时补充上下文路径/文件。
- 一行尽量短；复杂背景移到文档或注释块的下一行。

### 11.3 生命周期与清理
- 新增：按上述格式创建。
- 激活：开发中可在同文件附近添加 `IN-PROGRESS` 临时注释；合并前删除临时标记。
- 完成：删除或改为 `DONE:` 并附上提交哈希/简要说明（可选）。

### 11.4 VS Code Todo Tree 配置（建议）
`settings.json` 示例：
```json
{
  "todo-tree.general.tags": [
    "TODO",
    "FIXME",
    "REVIEW",
    "NOTE",
    "QUESTION",
    "PERF",
    "DEBT"
  ],
  "todo-tree.regex.regex": "(//|#|<!--|;|/\\*|^)\\s*($TAGS)(:?)",
  "todo-tree.tree.showCountsInTree": true,
  "todo-tree.highlights.useColourScheme": true,
  "todo-tree.general.statusBar": "tags",
  "todo-tree.tree.expanded": true,
  "todo-tree.general.groupBy": "tags"
}
```

常见正则增强（可选，高亮负责人与优先级）：
```json
{
  "todo-tree.regex.regex": "(//|#|<!--|;|/\\*|^)\\s*($TAGS)(:?)\\s*(?:@(?<owner>[a-zA-Z0-9_-]+))?\\s*(?<date>[0-9]{4}-[0-9]{2}-[0-9]{2})?\\s*(?<prio>\\[P[0-2]\\])?"
}
```

### 11.5 示例精选（本仓库）
```python
# PERF: @yyb 2025-09-23 [P1] 采样循环向量化（减少 Python 层开销）
# DEBT: @sjh 2025-09-24 [P2] types 重命名与收敛到协议类型
```

提示：`todo.txt` 作为“天为单位”的速记本；代码内 TODO 仅记录与该处实现强相关的短任务与提醒。
