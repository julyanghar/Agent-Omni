# Daily-Omni Benchmark 评估

本目录包含 Daily-Omni benchmark 的所有评估相关代码。

## 目录结构

```
daily_omni/
├── __init__.py
├── run_evaluation.py        # Python 评估入口脚本
├── agent_omni_adapter.py   # Agent-Omni 适配器
├── run_daily_omni.sh        # Bash 启动脚本（推荐使用）
├── README.md               # 本文档
└── test_model_api/         # Daily-Omni 评估框架
    ├── __init__.py
    ├── main_tester.py       # 主测试框架
    ├── test_utils.py       # 工具函数和模型调用接口
    └── test_config.py      # 配置文件
```

## 快速开始

### 方式一：使用 Bash 脚本（推荐）

```bash
# 在 Agent-Omni 根目录下运行
bash eval/daily_omni/run_daily_omni.sh
```

### 方式二：使用环境变量配置

```bash
# 设置环境变量后运行
export MODEL_TYPE=agent_omni
export EXECUTION_MODE=sequential
export QA_FILE=/path/to/qa.json
export MAX_ITEMS=10
export VIDEO_DIR=/path/to/videos

bash eval/daily_omni/run_daily_omni.sh
```

### 方式三：直接使用 Python 脚本

```bash
# 在 Agent-Omni 根目录下运行
python eval/daily_omni/run_evaluation.py \
    --model agent_omni \
    --mode sequential \
    --qa_file /path/to/qa.json \
    --max_items 10 \
    --video_dir /path/to/videos
```

## 参数说明

### Bash 脚本环境变量

- `MODEL_TYPE`: 模型类型（默认: `agent_omni`）
  - 可选值: `agent_omni`, `gemini_av`, `gemini_visual`, `gpt4o_visual`, `gpt4o_text`, `deepseek_text`
- `EXECUTION_MODE`: 执行模式（默认: `sequential`）
  - 可选值: `sequential`, `parallel`
- `QA_FILE`: QA JSON 文件路径（可选）
- `MAX_ITEMS`: 最大处理数量（可选，用于测试）
- `VIDEO_DIR`: 视频文件基础目录（可选）

### Python 脚本参数

- `--model`: 模型类型（默认: `agent_omni`）
- `--mode`: 执行模式（默认: `sequential`）
- `--qa_file`: QA JSON 文件路径（可选）
- `--max_items`: 最大处理数量（可选）
- `--video_dir`: 视频文件基础目录（可选）

## 配置文件

编辑 `test_model_api/test_config.py` 来设置默认配置：

- `BASE_VIDEO_DIR`: 视频文件基础目录
- `DEFAULT_QA_JSON_PATH`: 默认 QA JSON 文件路径
- `DEFAULT_EXECUTION_MODE`: 默认执行模式

## 输出结果

评估完成后会输出：

- 总体准确率
- 按 QA 类型的准确率
- 按视频类别的准确率
- 按视频时长的准确率
- 详细的统计信息

## 注意事项

1. 确保在 Agent-Omni 根目录下运行脚本
2. 确保视频文件路径正确
3. Agent-Omni 评估时间较长，建议先用 `--max_items` 进行小规模测试
4. 推荐使用 `sequential` 模式，并行模式可能不稳定

