## Debug Agent 快速体验指南

### 项目简介
Debug Agent 提供两类核心能力：
- general task：对通用工程项目进行多轮自动调试与修复（支持单体与前后端等复杂技术栈）。
- paperbench：对论文项目进行固定层级迭代（L0-L4）的复现与代码改进。

本仓库根目录已提供四个可直接运行的示例脚本：
- demo_complex_debug.py（general task——图书馆系统——50个bug；“complex_library_system copy”为带缺陷版本，“complex_library_system”为修复参考）
- demo_photo_debug.py（general task——照片抠图示例，多技术栈）
- rice_paper_demo.py（paperbench——RICE）
- forget_paper_demo.py（paperbench——What Will My Model Forget）

---

### 环境准备
1) Python 3.8+
2) 安装依赖（如遇报错，至少安装 requests 和 pydeps）：
   - pip install -r requirements.txt
   - 或手动：pip install requests pydeps
3) 可选检查：python check_dependencies.py（检测可选库与外部工具，如 code2flow、Graphviz）
4) 网络建议：国内环境建议配置代理，避免 LLM 请求超时

---

### .env 配置（放在仓库根目录）
必须项：
- AGENT_API_KEY=你的OpenRouter或其它兼容API的Key
- AGENT_BASE_URL=https://openrouter.ai/api/v1
- AGENT_MODEL=anthropic/claude-sonnet-4

可选项（未设置将使用默认值）：
- AGENT_TEMPERATURE=0.7
- AGENT_MAX_TOKENS=16384

说明：系统会自动从根目录加载 .env；也可直接以环境变量方式注入。

---

### 快速开始（general task）

#### A) 复杂图书馆系统（50个bug）
命令：
```
python demo_complex_debug.py
```
说明：
- 目标项目默认指向 `complex_library_system`（修复参考）。若要体验自动修复流程，请把脚本中的 `repo_path` 改为带缺陷的 `complex_library_system copy`，或将其重命名为 `complex_library_system` 再运行。
- 运行后会先演示初始错误，再询问是否继续迭代修复（输入 y）。
- 调试产物默认保存到目标项目下的 `debug_output/`（包含 `debug_report.json`、`modification_history.json`、`repo_index.json` 等）。

#### B) 照片抠图（复杂技术栈示例）
命令：
```
python demo_photo_debug.py
```
默认目标项目：`test_input/photo_cutout_tool`

建议准备（便于每轮收集终端信息 terminal.txt）：
- 前端：
  - cd test_input/photo_cutout_tool/frontend
  - npm install
  - npm start
  - 注：该前端为最小化示例，可能提示缺少 index.js，属模拟环境
- 后端：
  - cd test_input/photo_cutout_tool/backend
  - pip install -r requirements.txt
  - uvicorn main:app --reload
- 手动进行最小化交互（点击页面按钮触发API）。每次迭代将三段输出拼接写入根目录 `terminal.txt`：
  - 前端 stdout
  - 后端 stdout
  - 浏览器控制台 console（复制粘贴）
- 调试时选择“输入txt文件路径”的交互方式，将 `terminal.txt` 的绝对路径传入系统。

产物位置：同样输出到目标项目 `debug_output/`。

---

### 快速开始（paperbench）

#### A) RICE 论文复现
命令：
```
python rice_paper_demo.py
```
运行后选择“1. 标准复现流程”。

默认配置（如你移动了仓库或路径不同，请在脚本顶部配置区同步修改）：
- repo_path：test_input/rice
- paper_guide：test_papers/paper_test_1_reproduction_guide.md
- additional_guides：支持多个markdown，建议包含 OpenAI 为该论文提供的辅助文档，并将网页内容一并粘贴到 md（当前先手动整理；工程层面可用 mineru + re + firecrawl 实现）。

输出位置：在 `repo_path` 下自动新建 `paper_reproduction_output_时间戳/`，包含 `reproduction_result.json` 与过程日志。

#### B) What Will My Model Forget 论文复现
命令：
```
python forget_paper_demo.py
```
运行后选择“1. 标准复现流程”。

默认配置（如你移动了仓库或路径不同，请在脚本顶部配置区同步修改）：
- repo_path：test_input/will model forget
- paper_guide：test_papers/paper_test_2_reproduction_guide.md
- additional_guides：同 RICE，传入包含网页内容的 md 文档。

输出位置：在 `repo_path` 下自动新建 `paper_reproduction_output_时间戳/`。

---

### 自定义新用例注意事项
1) general task（单技术栈）
- 参照 `demo_complex_debug.py`：设置 `repo_path`、`main_file`、`expected_behavior`。
- 建议启用自动模式：`DebugSystem(auto_mode=True)`，减少交互。

2) general task（复杂技术栈）
- 每次迭代在根目录 `terminal.txt` 写入前端 stdout + 后端 stdout + 浏览器 console。
- 运行时选择“输入txt文件路径”方式，把 `terminal.txt` 的绝对路径传给系统。
- 需进行人工API测试（为了演示可点击按钮；规范应使用 APIFOX）。
- 当前仅作工程规模与流程演示，暂不做全面适配。

3) paperbench
- 必备参数：`repo_path`、`paper_guide`（复现指南），`additional_guides`（建议包含 OpenAI 辅助文档与网页内容的 md）。
- 可调：`max_iterations`（建议 5，对应 L0-L4）。
- 输出目录会自动生成并归档迭代结果与摘要。

---

### 产物位置速查
- general task：目标项目的 `debug_output/`
  - debug_report.json
  - modification_history.json
  - repo_index.json
- paperbench：目标 `repo_path` 下的 `paper_reproduction_output_时间戳/`
  - reproduction_result.json

---

### 常见问题（FAQ）
- 缺少API配置：请确保根目录 `.env` 已设置 `AGENT_API_KEY` 与 `AGENT_BASE_URL`。
- LLM 请求失败或限流：稍后重试，或降低 temperature / max_tokens；检查网络连通性。
- 路径不存在：部分示例脚本包含绝对路径（来自默认开发环境）。如你移动了仓库或操作系统不同，请修改脚本顶部的配置项为你本机的绝对/相对路径。
- 前端缺少文件：`photo_cutout_tool` 的前端为最小化示例，出现缺少 `index.js` 等提示属正常；本示例重点在“调试流程”与“多源 stdout 收集”。

---

### 运行命令汇总
```
python demo_complex_debug.py
python demo_photo_debug.py
python rice_paper_demo.py
python forget_paper_demo.py
python check_dependencies.py
```

---

### 说明
- 本 README 仅提供快速体验指引，未对脚本进行改动。
- 体验图书馆系统“自动修复”时，请将脚本中的 `repo_path` 指向带缺陷项目 `complex_library_system copy`，或调整目录名后运行。


