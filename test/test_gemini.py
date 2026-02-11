#!/usr/bin/env python
"""
测试 Google Gemini 2.5-Flash 集成

使用方法：
1. 设置环境变量：export GOOGLE_API_KEY="your-api-key"
2. 运行：python test_gemini.py
"""

import os
import sys
# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models import load_model
from src.config import load_config
from langchain_core.messages import HumanMessage, SystemMessage

def test_gemini_basic():
    """测试基本的 Gemini 模型加载和调用"""
    print("=" * 50)
    print("测试 1: 基本模型加载和调用")
    print("=" * 50)
    
    try:
        # 检查 API key
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("❌ 错误: GOOGLE_API_KEY 环境变量未设置")
            print("   请运行: export GOOGLE_API_KEY='your-api-key'")
            return False
        
        print(f"✓ API Key 已设置: {api_key[:10]}...")
        
        # 加载配置
        config_path = os.path.join(project_root, "configs", "model_configs", "gemini-2.5-flash.yaml")
        config = load_config(config_path)
        print(f"✓ 配置文件加载成功: {config_path}")
        
        # 加载模型
        model = load_model(config)
        print("✓ 模型加载成功")
        
        # 测试调用
        messages = [HumanMessage(content="What is 2+2? Please answer in one sentence.")]
        print("\n发送测试消息...")
        response = model.invoke(messages)
        print(f"✓ 响应接收成功")
        print(f"\n模型回答: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gemini_with_system_prompt():
    """测试带系统提示的调用"""
    print("\n" + "=" * 50)
    print("测试 2: 带系统提示的调用")
    print("=" * 50)
    
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "configs", "model_configs", "gemini-2.5-flash.yaml")
        config = load_config(config_path)
        model = load_model(config)
        
        messages = [
            SystemMessage(content="You are a helpful math tutor."),
            HumanMessage(content="What is the square root of 16?")
        ]
        
        print("发送带系统提示的消息...")
        response = model.invoke(messages)
        print(f"✓ 响应接收成功")
        print(f"\n模型回答: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gemini_batch():
    """测试批量调用"""
    print("\n" + "=" * 50)
    print("测试 3: 批量调用")
    print("=" * 50)
    
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "configs", "model_configs", "gemini-2.5-flash.yaml")
        config = load_config(config_path)
        model = load_model(config)
        
        messages_batch = [
            [HumanMessage(content="What is 1+1?")],
            [HumanMessage(content="What is 2+2?")],
            [HumanMessage(content="What is 3+3?")]
        ]
        
        print("发送批量消息...")
        responses = model.batch(messages_batch)
        print(f"✓ 批量响应接收成功 ({len(responses)} 条)")
        
        for i, response in enumerate(responses, 1):
            print(f"\n问题 {i} 的回答: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gemini_media_invoke():
    """测试媒体调用（文本）"""
    print("\n" + "=" * 50)
    print("测试 4: 媒体调用接口（文本）")
    print("=" * 50)
    
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "configs", "model_configs", "gemini-2.5-flash.yaml")
        config = load_config(config_path)
        model = load_model(config)
        
        content = {
            "system_prompt": "You are a helpful assistant.",
            "question": "Summarize the following text in one sentence.",
            "text": "Artificial intelligence is transforming the way we work and live."
        }
        
        print("发送媒体调用...")
        response = model.media_invoke(content)
        print(f"✓ 响应接收成功")
        print(f"\n模型回答: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("Google Gemini 2.5-Flash 集成测试")
    print("=" * 50)
    print()
    
    results = []
    
    # 运行测试
    results.append(("基本调用", test_gemini_basic()))
    results.append(("系统提示", test_gemini_with_system_prompt()))
    results.append(("批量调用", test_gemini_batch()))
    results.append(("媒体调用", test_gemini_media_invoke()))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！Gemini 集成成功！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    exit(main())

