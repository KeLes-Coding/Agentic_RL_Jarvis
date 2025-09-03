# Jarvis - LLM 驱动的安卓设备操控Agent

Jarvis 是一个由大型语言模型（LLM）驱动的 AI Agent，旨在通过模仿人类用户的交互方式来操作安卓设备，以完成指定的高级任务。

## 功能特性

* **多模态感知**: 结合屏幕截图和 UI 布局（XML）来全面理解当前设备状态。
* **LLM 驱动的决策**: 利用大型语言模型（支持 OpenAI, Gemini, Claude）进行“思考”，根据观察到的信息决定下一步操作。
* **人类行为模拟**: 执行如点击、滑动、输入文本、返回、返回主页等一系列原子操作来与设备交互。
* **可扩展的 LLM 支持**: 轻松在不同的 LLM 提供商之间切换，并支持通过代理连接。
* **鲁棒性**:
    * **错误重试机制**: 当遇到 LLM 输出格式错误等可恢复问题时，Agent 会自动重试，而不是直接中断任务。
    * **JSON 自动修复**: 能够处理并修复来自 LLM 的不完全规范的 JSON 响应。
* **详尽的执行记录**:
    * **完整轨迹追踪**: 自动为每次任务运行创建独立的文件夹，详细记录每一步的截图、UI 布局、LLM 思考过程和最终行动。
    * **对话历史**: 保存每一步与 LLM 的完整对话（Prompt 和 Response），便于深度分析和调试。
    * **成本统计**: 在任务总结中精确记录 Token 的消耗量。
* **并行设备管理**: 能够同时检测多个连接的安卓设备，并为每台设备启动一个独立的Agent进程来执行任务。

## 工作流程

Jarvis V2 的核心是一个“观察-思考-行动”的循环：

1.  **观察 (Observe)**: `Observer` 模块获取当前安卓设备的屏幕截图和 UI 层次结构（XML 文件）。
2.  **思考 (Think)**: `JarvisAgent` 将观察到的信息连同任务目标一起发送给 `LLMClient`。LLM 根据这些信息，按照预设的指令（`prompts.py`），以 JSON 格式返回它的思考过程和下一步要执行的具体动作。如果此步骤出现格式错误，**重试机制将被激活**。
3.  **行动 (Act)**: `Actuator` 模块接收来自 LLM 的动作指令，并将其解析为具体的 ADB (Android Debug Bridge) 命令在设备上执行。
4.  **记录 (Record)**: `InfoPoolManager` 会记录下循环中的每一步，包括观察数据、**完整的LLM对话**、行动结果，并保存在本次任务的专属文件夹中。

这个循环会持续进行，直到任务完成、达到预设的最大步骤数或重试次数耗尽。

## 安装指南

1.  **克隆代码库**:
    ```
    git clone https://github.com/KeLes-Coding/Jarvis_V2.git
    cd Jarvis_V2
    ```

2.  **安装依赖**:
    ```
    pip install -r requirements.txt
    ```

3.  **安卓调试桥 (ADB)**:
    确保您已经安装了 Android SDK Platform-Tools，并且 `adb` 命令在您的系统路径 (PATH) 中。您可以通过运行 `adb devices` 来验证。

## 配置

在运行项目之前，您需要配置您的个人设置：

1.  **创建配置文件**:
    复制 `config.template.yaml` 并将其重命名为 `config.yaml`。
    ```
    cp config.template.yaml config.yaml
    ```

2.  **编辑 `config.yaml`**:
    打开 `config.yaml` 文件并填入以下信息：
    * `adb.executable_path`: 如果 `adb` 不在您的系统 PATH 中，请在此处指定其完整路径。
    * `proxy`: 如果您需要通过代理访问 LLM 服务，请启用并设置代理服务器地址。
    * `agent.retry_on_error`: 配置错误重试机制。
        * `enabled`: `true` 开启重试，`false` 关闭。
        * `attempts`: 遇到错误时最大重试的次数。
    * `llm.api_mode`: 选择您要使用的 LLM 提供商 (`openai`, `gemini`, 或 `claude`)。
    * `llm.fix_json_enabled`: 设置为 `true` 以允许系统自动尝试修复不规范的 JSON 响应。
    * `llm.providers`: 在您选择的提供商下，填入您的 `api_key` 和要使用的 `model` 名称。
    * `main.tasks`: 在这里定义要执行的任务列表（见下文）。

    **注意**: `config.yaml` 已被添加到 `.gitignore` 中，以防止您的敏感信息（如 API Key）被意外提交。

## 如何运行

1.  **连接安卓设备**:
    使用 USB 数据线将您的安卓设备连接到电脑，并确保已开启“开发者选项”中的“USB 调试”功能，并已授权您的电脑进行调试。

2.  **定义任务**:
    打开 `config.yaml` 文件，在 `main` 部分下的 `tasks` 列表中定义一个或多个任务。Agent Manager 会将这些任务依次分配给所有可用的设备。
    ```
    # config.yaml
    
    main:
      # 在这里定义你的任务列表
      tasks:
        - "打开Bilibili，搜索“原神”，然后进入第一个视频并点赞。"
        - "打开设置，将屏幕亮度调到最亮。"
        - "查询今天北京的天气怎么样。"
      
      # 设备提供者配置...
      device_providers:
        ...
    ```

3.  **启动代理**:
    运行 `agent_manager.py`。程序会自动检测所有连接的设备，并为每台设备分配一个代理来执行任务。
    ```
    python agent_manager.py
    ```

4.  **查看结果**:
    每次运行的产出，包括日志、截图和详细步骤，都会保存在 `jarvis/runs/` 目录下的一个以时间戳和任务名命名的文件夹中。该文件夹包含：
    * `agent_run.log`: 本次任务的详细运行日志。
    * `summary.json`: 任务的最终总结，包括开始/结束时间、最终状态、总步数以及**Token消耗统计**。
    * `execution_trace.json`: 一个包含所有步骤详细信息的完整轨迹文件，便于程序化分析。
    * `step_xxx/` 文件夹（每个步骤一个）:
        * `screenshot.png`: 该步骤开始时的屏幕截图。
        * `layout.xml`: 原始的 UI 布局文件。
        * `simplified_layout.txt`: 简化后供 LLM 读取的 UI 元素列表。
        * `llm_dialogue.json`: **完整的 LLM 对话记录**，包含发送的 `prompt` 和收到的原始 `response`，是调试 LLM 行为的关键。
        * `step_details.json`: 该步骤的结构化详情，包括 LLM 解析后的思考和行动。

## 项目结构

```
.
├── agent_manager.py            # 主入口，管理和启动Agent进程
├── config.template.yaml        # 配置文件模板
├── requirements.txt            # Python依赖
├── jarvis/
│   ├── agent.py                # 核心Agent类，负责编排整个工作流
│   ├── info_pool.py            # 信息池管理器，负责记录和保存运行轨迹
│   ├── logger_setup.py         # 日志系统初始化
│   ├── __init__.py
│   ├── llm/
│   │   ├── client.py           # LLM客户端，支持多家服务并处理代理
│   │   ├── prompts.py          # 存放引导LLM思考的系统提示
│   │   └── __init__.py
│   ├── modules/
│   │   ├── observer.py         # 观察者模块，负责获取设备屏幕和UI信息
│   │   ├── actuator.py         # 执行器模块，负责在设备上执行操作
│   │   └── __init__.py
│   └── runs/                   # (自动创建) 存放所有任务的运行产出
└── ...
