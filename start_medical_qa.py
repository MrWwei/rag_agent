#!/usr/bin/env python3
"""
医疗问答系统快速启动脚本
一键启动交互式医疗问答功能
"""

import os
import sys
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from medical_qa_system import MedicalQASystem


def check_environment(enable_rag=True):
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 错误: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("请设置阿里云百炼大模型API密钥:")
        print("export DASHSCOPE_API_KEY='your-api-key'")
        return False
    else:
        print("✅ API密钥已配置")
    
    # 只在启用RAG时检查向量存储
    if enable_rag:
        vector_store_path = project_root / "vector_store"
        if not vector_store_path.exists():
            print("❌ 错误: 向量存储不存在")
            print("请先运行以下命令构建知识库:")
            print("python src/build_knowledge_base.py")
            print("或者选择使用纯大模型模式（不需要向量存储）")
            return False
        else:
            print("✅ 向量存储已就绪")
    else:
        print("🔄 使用纯大模型模式，跳过向量存储检查")
    
    return True


def quick_test(enable_rag=True):
    """快速测试问答功能"""
    print(f"\n🧪 快速测试 ({'RAG模式' if enable_rag else '纯大模型模式'})...")
    
    qa_system = MedicalQASystem(enable_rag=enable_rag)
    
    test_question = "高血压的定义是什么？"
    print(f"测试问题: {test_question}")
    
    try:
        result = qa_system.answer_question(test_question, k=2)
        
        if enable_rag and result['retrieval_success']:
            print("✅ RAG测试成功!")
            print(f"检索到 {len(result['search_results'])} 个相关文档")
            print(f"答案长度: {len(result['answer'])} 字符")
        elif not enable_rag:
            print("✅ 纯大模型测试成功!")
            print(f"答案长度: {len(result['answer'])} 字符")
        else:
            print("⚠️  测试完成但未检索到相关文档")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return True


def start_interactive_qa(enable_rag=True):
    """启动交互式问答"""
    mode = "RAG增强模式" if enable_rag else "纯大模型模式"
    print(f"\n🚀 启动医疗问答系统 ({mode})...")
    print("="*50)
    
    try:
        qa_system = MedicalQASystem(enable_rag=enable_rag)
        qa_system.interactive_qa()
    except KeyboardInterrupt:
        print("\n\n👋 感谢使用医疗问答系统!")
    except Exception as e:
        print(f"❌ 启动失败: {e}")


def choose_mode():
    """选择运行模式"""
    print("\n🔧 请选择运行模式:")
    print("1. RAG增强模式 (使用专业医疗知识库)")
    print("2. 纯大模型模式 (仅使用大模型内置知识)")
    
    while True:
        choice = input("\n请选择模式 (1/2): ").strip()
        if choice == "1":
            return True
        elif choice == "2":
            return False
        else:
            print("无效选择，请输入 1 或 2")


def show_demo_questions():
    """显示示例问题"""
    demo_questions = [
        "高血压的诊断标准是什么？",
        "糖尿病有哪些类型？",
        "冠心病的主要症状有哪些？",
        "阿司匹林的用法用量是多少？",
        "二甲双胍有什么副作用？",
        "高血压患者饮食要注意什么？",
        "如何预防糖尿病并发症？",
        "心绞痛和心肌梗死有什么区别？"
    ]
    
    print("\n💡 示例问题 (您可以询问以下类型的问题):")
    print("-" * 50)
    for i, question in enumerate(demo_questions, 1):
        print(f"{i:2d}. {question}")
    print("-" * 50)


def main():
    """主函数"""
    print("🏥 医疗问答系统 - 快速启动")
    print("="*50)
    print("基于RAG检索增强生成的智能医疗问答助手")
    print("结合专业医疗知识库和大语言模型，提供准确的医疗信息")
    print("="*50)
    
    # 选择运行模式
    enable_rag = choose_mode()
    
    # 检查环境
    if not check_environment(enable_rag):
        if enable_rag:
            print("\n❌ RAG模式环境检查失败")
            print("💡 提示: 可以选择使用纯大模型模式")
            use_pure_llm = input("是否切换到纯大模型模式? (y/n): ").strip().lower()
            if use_pure_llm in ['y', 'yes', '是']:
                enable_rag = False
                if not check_environment(enable_rag):
                    return
            else:
                return
        else:
            print("\n❌ 环境检查失败，请解决上述问题后重试")
            return
    
    # 快速测试
    if not quick_test(enable_rag):
        print("\n❌ 系统测试失败，请检查配置")
        return
    
    # 显示示例问题
    show_demo_questions()
    
    print("\n⚠️  重要提醒:")
    print("本系统提供的信息仅供参考，不能替代专业医疗建议。")
    print("如有健康问题，请及时咨询专业医生。")
    
    # 启动交互式问答
    start_interactive_qa(enable_rag)


if __name__ == "__main__":
    main()
