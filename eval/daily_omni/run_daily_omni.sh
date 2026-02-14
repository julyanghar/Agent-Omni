#!/bin/bash
# Daily-Omni Benchmark 评估脚本
# 
# 使用方法:
#   bash eval/daily_omni/run_daily_omni.sh
#   或
#   ./eval/daily_omni/run_daily_omni.sh
#
# 可以通过环境变量或修改此脚本来配置参数

# 设置默认值（可根据需要修改）
MODEL_TYPE=${MODEL_TYPE:-"agent_omni"}
EXECUTION_MODE=${EXECUTION_MODE:-"sequential"}
QA_FILE=${QA_FILE:-""}
MAX_ITEMS=${MAX_ITEMS:-"10"}
VIDEO_DIR=${VIDEO_DIR:-""}

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AGENT_OMNI_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# 切换到 Agent-Omni 根目录
cd "$AGENT_OMNI_ROOT"

# 构建命令
CMD="python eval/daily_omni/run_evaluation.py --model $MODEL_TYPE --mode $EXECUTION_MODE"

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

# 打印配置信息
echo "=========================================="
echo "Daily-Omni Benchmark 评估"
echo "=========================================="
echo "模型类型: $MODEL_TYPE"
echo "执行模式: $EXECUTION_MODE"
if [ -n "$QA_FILE" ]; then
    echo "QA 文件: $QA_FILE"
fi
if [ -n "$MAX_ITEMS" ]; then
    echo "最大处理数量: $MAX_ITEMS"
fi
if [ -n "$VIDEO_DIR" ]; then
    echo "视频目录: $VIDEO_DIR"
fi
echo "=========================================="
echo ""

# 执行命令
eval $CMD

