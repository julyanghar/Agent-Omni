"""
演示如何查询 LangChain 模型对象的可用方法和调用方式
"""

import inspect
from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel

def query_model_methods(model_instance, model_name="Model"):
    """
    查询模型实例的所有可用方法
    
    参数:
        model_instance: 模型实例
        model_name: 模型名称（用于显示）
    """
    print(f"\n{'='*60}")
    print(f"查询 {model_name} 的可用方法")
    print(f"{'='*60}\n")
    
    # 1. 检查继承关系
    print("1. 继承关系检查:")
    print(f"   类型: {type(model_instance)}")
    print(f"   是否继承自 Runnable: {isinstance(model_instance, Runnable)}")
    print(f"   是否继承自 BaseChatModel: {isinstance(model_instance, BaseChatModel)}")
    print(f"   MRO (方法解析顺序): {type(model_instance).__mro__[:5]}...")  # 只显示前5个
    
    # 2. 查看所有方法
    print("\n2. 主要可用方法:")
    methods = [m for m in dir(model_instance) if not m.startswith('_')]
    runnable_methods = ['invoke', 'batch', 'stream', 'astream', 'abatch', 'ainvoke']
    for method in runnable_methods:
        if hasattr(model_instance, method):
            print(f"   ✓ {method}")
        else:
            print(f"   ✗ {method} (不存在)")
    
    # 3. 查看 invoke 方法的签名
    print("\n3. invoke() 方法签名:")
    if hasattr(model_instance, 'invoke'):
        sig = inspect.signature(model_instance.invoke)
        print(f"   {model_instance.invoke.__name__}{sig}")
        if model_instance.invoke.__doc__:
            doc_lines = model_instance.invoke.__doc__.split('\n')[:3]
            print(f"   文档: {doc_lines[0]}")
    
    # 4. 查看 batch 方法的签名
    print("\n4. batch() 方法签名:")
    if hasattr(model_instance, 'batch'):
        sig = inspect.signature(model_instance.batch)
        print(f"   {model_instance.batch.__name__}{sig}")
        if model_instance.batch.__doc__:
            doc_lines = model_instance.batch.__doc__.split('\n')[:3]
            print(f"   文档: {doc_lines[0]}")
    
    # 5. 查看方法的完整文档
    print("\n5. batch() 方法完整文档:")
    if hasattr(model_instance, 'batch'):
        print("   " + "\n   ".join(model_instance.batch.__doc__.split('\n')[:10]))
    
    print(f"\n{'='*60}\n")


def demonstrate_usage():
    """
    演示如何使用这些方法
    """
    print("\n使用示例:")
    print("""
    # 1. 单次调用
    messages = [SystemMessage(content="Hello"), HumanMessage(content="Hi")]
    response = model.invoke(messages)
    
    # 2. 批量调用
    messages_batch = [
        [SystemMessage(content="Hello"), HumanMessage(content="Hi")],
        [SystemMessage(content="Hello"), HumanMessage(content="Hello")]
    ]
    responses = model.batch(messages_batch)
    
    # 3. 带配置的批量调用
    from langchain_core.runnables.config import RunnableConfig
    config = RunnableConfig(max_concurrency=5)
    responses = model.batch(messages_batch, config=config)
    """)


if __name__ == "__main__":
    # 注意：这里需要实际加载一个模型才能运行
    # 以下是示例代码，展示如何查询
    
    print("="*60)
    print("LangChain 模型方法查询工具")
    print("="*60)
    
    print("\n要使用此脚本，请先加载一个模型:")
    print("""
    from src.models import load_model
    from src.config import config
    
    model = load_model(config["model"]["master_agent"])
    query_model_methods(model, "Master Model")
    """)
    
    demonstrate_usage()
    
    print("\n提示：")
    print("1. 在 Python REPL 中使用 help(model.batch) 查看详细文档")
    print("2. 在 IDE 中将鼠标悬停在 model 上查看类型信息")
    print("3. 查看 LangChain 官方文档获取最新信息")

