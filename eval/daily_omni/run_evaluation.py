#!/usr/bin/env python
"""
Agent-Omni 在 Daily-Omni benchmark 上的评估入口脚本

使用方法:
    python eval/daily_omni/run_evaluation.py --model agent_omni --mode sequential --qa_file <qa.json路径> --max_items <可选：限制处理数量>
    
或者直接运行（使用默认配置）:
    python eval/daily_omni/run_evaluation.py
"""

import sys
import os
import argparse

# 添加 Agent-Omni 根目录到路径
_agent_omni_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, _agent_omni_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="在 Daily-Omni benchmark 上评估 Agent-Omni",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置评估
  python eval/daily_omni/run_evaluation.py
  
  # 指定 QA 文件路径
  python eval/daily_omni/run_evaluation.py --qa_file /path/to/qa.json
  
  # 限制处理前 10 个问题（用于测试）
  python eval/daily_omni/run_evaluation.py --qa_file /path/to/qa.json --max_items 10
  
  # 使用并行模式
  python eval/daily_omni/run_evaluation.py --qa_file /path/to/qa.json --mode parallel
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default='agent_omni',
        choices=['agent_omni', 'gemini_av', 'gemini_visual', 'gpt4o_visual', 'gpt4o_text', 'deepseek_text'],
        help="要评估的模型类型（默认: agent_omni）"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default='sequential',
        choices=['sequential', 'parallel'],
        help="执行模式：sequential（顺序）或 parallel（并行，默认: sequential）"
    )
    
    parser.add_argument(
        "--qa_file", 
        type=str, 
        default=None,
        help="QA JSON 文件路径（如果未指定，将使用 test_config.py 中的默认路径）"
    )
    
    parser.add_argument(
        "--max_items", 
        type=int, 
        default=None,
        help="最大处理的问题数量（用于测试，默认: None，处理所有问题）"
    )
    
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="视频文件基础目录（如果未指定，将使用 test_config.py 中的默认路径）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="评估结果输出目录（如果未指定，将使用默认路径: eval/daily_omni/eval_results）"
    )
    
    args = parser.parse_args()
    
    # 动态导入评估框架（只在需要时导入，避免不必要的依赖）
    from eval.daily_omni.test_model_api.main_tester import run_tests
    
    # 确定 QA 文件路径
    if args.qa_file:
        qa_json_path = args.qa_file
    else:
        import eval.daily_omni.test_model_api.test_config as test_config
        qa_json_path = test_config.DEFAULT_QA_JSON_PATH
        print(f"使用默认 QA 文件路径: {qa_json_path}")
    
    # 检查文件是否存在
    if not os.path.exists(qa_json_path):
        print(f"错误: QA 文件不存在: {qa_json_path}")
        print("请使用 --qa_file 参数指定正确的文件路径")
        sys.exit(1)
    
    print("=" * 60)
    print("Agent-Omni Daily-Omni Benchmark 评估")
    print("=" * 60)
    print(f"模型类型: {args.model}")
    print(f"执行模式: {args.mode}")
    print(f"QA 文件: {qa_json_path}")
    if args.max_items:
        print(f"最大处理数量: {args.max_items}")
    print("=" * 60)
    print()
    
    # 确定视频目录
    if args.video_dir:
        video_base_dir = os.path.abspath(args.video_dir)
    else:
        # 使用配置文件中的默认路径
        import eval.daily_omni.test_model_api.test_config as test_config
        video_base_dir = os.path.abspath(test_config.BASE_VIDEO_DIR)
    
    print(f"视频目录: {video_base_dir}")
    
    # 确定输出目录
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        # 使用默认路径: eval/daily_omni/eval_results
        _agent_omni_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        output_dir = os.path.join(_agent_omni_root, 'eval', 'daily_omni', 'eval_results')
        output_dir = os.path.abspath(output_dir)
    
    print(f"结果输出目录: {output_dir}")
    
    # 运行评估
    try:
        run_tests(
            model_type=args.model,
            execution_mode=args.mode,
            qa_json_path=qa_json_path,
            max_items=args.max_items,
            video_base_dir=video_base_dir,
            output_dir=output_dir
        )
    except KeyboardInterrupt:
        print("\n\n评估被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

