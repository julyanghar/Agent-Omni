#!/bin/bash
# Agent-Omni 评估示例脚本
# 
# 使用方法:
#   1. 设置你的文件路径和配置（在下方配置区域）
#   2. 运行: bash eval/daily_omni/run_agent_omni_example.sh
#   或
#   ./eval/daily_omni/run_agent_omni_example.sh

# ========== 配置区域 ==========
# 请修改以下配置为你的实际路径

# Agent-Omni 配置文件路径（可选，如果不设置会使用默认路径: test/config.yaml 或 src/config.yaml）
export CONFIG_PATH=${CONFIG_PATH:-"/home/yilin/Agent-Omni/test/config.yaml"}
export GOOGLE_API_KEY="AIzaSyDjCRT3t8okZw3iMFvesGKzNHRiOTOV5-Q"
# QA JSON 文件路径（必需）
export QA_FILE=${QA_FILE:-"/home/yilin/Daily-Omni/qa.json"}

# 视频文件基础目录（可选，如果不设置会使用配置文件中的默认值）
export VIDEO_DIR=${VIDEO_DIR:-"/home/yilin/Daily-Omni/Videos"}

# 评估结果输出目录（可选，如果不设置会使用默认路径: eval/daily_omni/eval_results）
export OUTPUT_DIR=${OUTPUT_DIR:-""}

# 最大处理数量（用于测试，设置为空或大数字表示处理全部）
export MAX_ITEMS=${MAX_ITEMS:-"2"}

# 执行模式: sequential (顺序) 或 parallel (并行)
# 注意: Agent-Omni 建议使用 sequential 模式
export EXECUTION_MODE=${EXECUTION_MODE:-"sequential"}

# Python 调试模式: 1 启用 debugpy 连接，0 或空则关闭
# 启用后需要确保 VS Code/Cursor 调试器正在监听指定端口
export DEBUG_MODE=${DEBUG_MODE:-"1"}
export DEBUG_PORT=${DEBUG_PORT:-"5679"}

# ========== 执行评估 ==========

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AGENT_OMNI_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# 切换到 Agent-Omni 根目录
cd "$AGENT_OMNI_ROOT"

# 设置 CONFIG_PATH 环境变量（如果指定了）
if [ -n "$CONFIG_PATH" ]; then
    export CONFIG_PATH="$CONFIG_PATH"
fi

# 构建命令
if [ "$DEBUG_MODE" = "1" ]; then
    echo "启动 Python 调试模式，连接到端口 $DEBUG_PORT..."
    echo "请确保 VS Code/Cursor 调试器正在监听端口 $DEBUG_PORT"
    echo ""
    CMD="python -m debugpy --connect $DEBUG_PORT eval/daily_omni/run_evaluation.py --model agent_omni --mode $EXECUTION_MODE"
else
    CMD="python eval/daily_omni/run_evaluation.py --model agent_omni --mode $EXECUTION_MODE"
fi

# 添加可选参数
if [ -n "$QA_FILE" ]; then
    CMD="$CMD --qa_file $QA_FILE"
fi

if [ -n "$MAX_ITEMS" ]; then
    CMD="$CMD --max_items $MAX_ITEMS"
fi

if [ -n "$VIDEO_DIR" ]; then
    CMD="$CMD --video_dir $VIDEO_DIR"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

# 打印配置信息
echo "=========================================="
echo "Daily-Omni Benchmark 评估 (Agent-Omni)"
echo "=========================================="
echo "模型类型: agent_omni"
echo "执行模式: $EXECUTION_MODE"
if [ "$DEBUG_MODE" = "1" ]; then
    echo "Python 调试模式: 已启用 (端口 $DEBUG_PORT)"
fi
if [ -n "$CONFIG_PATH" ]; then
    echo "配置文件: $CONFIG_PATH"
fi
if [ -n "$QA_FILE" ]; then
    echo "QA 文件: $QA_FILE"
fi
if [ -n "$MAX_ITEMS" ]; then
    echo "最大处理数量: $MAX_ITEMS"
fi
if [ -n "$VIDEO_DIR" ]; then
    echo "视频目录: $VIDEO_DIR"
fi
if [ -n "$OUTPUT_DIR" ]; then
    echo "结果输出目录: $OUTPUT_DIR"
fi
echo "=========================================="
echo ""

# 执行命令
eval $CMD

