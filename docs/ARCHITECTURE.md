## 项目架构与约定（Architecture）

### 设计目标
- 对外 API 简洁稳定（用户只接触 models 与 CLI）。
- 内部实现高内聚、低耦合；能在不破坏外部 API 的情况下演进。
- 配置单一事实来源，可复现、可组合。

### 分层概览（当前约定）
- models（对外门面）：面向用户的高层模型与配置（例如 `BabyDenoiser`, `NanoDiT`）。
- methods：训练目标/算法（EDM、Flow Matching、DDPM 等）。
- sampling：采样器与调度。
- utils：日志、输出、工具。
- configs：统一的配置定义与默认集（可与 Hydra 对接）。

提示：遵循“能放进一个文件就不拆”的原则。仅当文件职责明显增多、复用需求上升、导入依赖变复杂时再拆分。

### 模型与实现的组织
- 对外仅从 `nanodiffusers.models` 导出模型门面类与其配置。
- 模型内部可包含其专属实现细节与小型模块（必要时再拆）。
- 通用的工具/小积木放在 `utils` 或后续的 `common` 区域，避免重复实现。

依赖方向（约束）：
- methods → 依赖 models 的门面接口（不触及内部实现细节）。
- sampling → 独立于具体模型，依赖通用张量 API。
- models → 可以使用 utils、（必要时）内部小模块。

### 配置策略
- 外部/持久化/需要落盘的配置集中在 `configs`，并提供默认与文档化字段。
- 模块内部的临时参数对象可以存在，但不命名为 `*Config`、不导出、不写磁盘（用 `*Options`/`*Params`）。
- 与 Hydra 的对接：高层配置（实验/训练）在 `configs/`，CLI 通过 Hydra 或本地 YAML 加载并映射到各模型配置。

### 输出与检查点
- `OutputManager` 统一目录与 IO；
- `CheckpointManager` 仅做策略（何时保存/命名/恢复），实际写盘委托 `OutputManager`。

### 代码风格与质量
- 使用 Ruff（lint + format），pre-commit 在提交前自动检查与格式化。
- pytest 覆盖核心路径；建议新增功能时同步补充最小测试。

### 何时拆分文件/目录
- 文件 > ~300–500 行且承担多职责；
- 同一代码片段开始在多处复制；
- 循环导入/初始化顺序问题频发；
- 重构或测试成本显著升高。

### 迁移与演进
- 新增模型：放入 `models/`，对外只导出门面与配置；内部实现先用单文件，后续按需拆分。
- 新增配置：优先在 `configs/`，并更新 README/DEVELOPMENT_QA。
- 结构性调整：在 CHANGELOG 记录，并保持对外 API 兼容（必要时提供迁移指引）。
