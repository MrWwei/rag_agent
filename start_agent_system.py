"""
智能体系统启动器
演示不同类型的智能体构建方案
"""

import os
import sys
import time
from typing import Dict, Any
from src.medical_agent import MedicalAgent


class AgentSystemDemo:
    """智能体系统演示类"""
    
    def __init__(self):
        self.agents = {}
        self.current_agent = None
        
    def initialize_agents(self):
        """初始化不同类型的智能体"""
        print("🚀 初始化智能体系统...")
        
        # 1. ReAct智能体（推理+行动）
        print("   初始化 ReAct 智能体...")
        try:
            self.agents["react"] = MedicalAgent(
                vector_store_path="./vector_store",
                model="qwen-plus",
                enable_rag=True,
                max_iterations=5,
                temperature=0.1
            )
            print("   ✅ ReAct 智能体初始化成功")
        except Exception as e:
            print(f"   ❌ ReAct 智能体初始化失败: {e}")
        
        # # 2. 简化智能体（仅RAG+LLM）
        # print("   初始化简化智能体...")
        # try:
        #     self.agents["simple"] = MedicalAgent(
        #         vector_store_path="./vector_store",
        #         model="qwen-plus",
        #         enable_rag=True,
        #         max_iterations=1,  # 只进行一次推理
        #         temperature=0.0
        #     )
        #     print("   ✅ 简化智能体初始化成功")
        # except Exception as e:
        #     print(f"   ❌ 简化智能体初始化失败: {e}")
            
        print(f"\n✅ 智能体系统初始化完成！可用智能体: {list(self.agents.keys())}\n")
    
    def explain_agent_architectures(self):
        """解释不同的智能体架构"""
        explanation = """
🧠 智能体（Agent）构建方案详解

═══════════════════════════════════════════════════════════════

📖 **1. ReAct 模式（Reasoning + Acting）**
┌─────────────────────────────────────────────────────────────┐
│ 特点：思考-行动-观察的循环模式                               │
│ 流程：问题 → 思考 → 选择工具 → 执行 → 观察结果 → 再思考...    │
│ 优势：能处理复杂的多步骤任务，推理过程透明                   │
│ 适用：需要多步推理和工具调用的复杂问题                       │
└─────────────────────────────────────────────────────────────┘

🛠️ **2. Function Calling 模式**
┌─────────────────────────────────────────────────────────────┐
│ 特点：基于大模型的函数调用能力                               │
│ 流程：问题 → 模型选择函数 → 执行函数 → 返回结果             │
│ 优势：简单直接，容易实现                                     │
│ 适用：有明确工具集的任务自动化                               │
└─────────────────────────────────────────────────────────────┘

📋 **3. Plan-and-Execute 模式**
┌─────────────────────────────────────────────────────────────┐
│ 特点：先制定详细计划，再逐步执行                             │
│ 流程：问题 → 制定计划 → 逐步执行 → 调整计划 → 完成任务     │
│ 优势：适合长期任务，执行过程可控                             │
│ 适用：复杂的项目管理、多步骤任务                             │
└─────────────────────────────────────────────────────────────┘

👥 **4. Multi-Agent 协作模式**
┌─────────────────────────────────────────────────────────────┐
│ 特点：多个专门化智能体协同工作                               │
│ 流程：任务分解 → 分配给专门智能体 → 协作执行 → 结果整合     │
│ 优势：专业化分工，处理复杂系统                               │
│ 适用：大型系统，需要不同专业知识的任务                       │
└─────────────────────────────────────────────────────────────┘

🔄 **5. 记忆增强模式（Memory-Augmented）**
┌─────────────────────────────────────────────────────────────┐
│ 特点：具备长期记忆和上下文管理能力                           │
│ 流程：问题 → 检索记忆 → 推理 → 执行 → 更新记忆             │
│ 优势：支持长期对话，个性化服务                               │
│ 适用：客服、个人助手、持续学习场景                           │
└─────────────────────────────────────────────────────────────┘

🎯 **6. 目标导向模式（Goal-Oriented）**
┌─────────────────────────────────────────────────────────────┐
│ 特点：围绕明确目标进行规划和执行                             │
│ 流程：设定目标 → 分解子目标 → 执行策略 → 评估结果           │
│ 优势：目标明确，执行高效                                     │
│ 适用：明确目标的任务，如销售、客户服务                       │
└─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════

🏥 **本项目采用的方案：ReAct + RAG + Tools**

核心组件：
• 🧠 ReAct推理引擎：思考-行动-观察循环
• 📚 RAG检索系统：医疗知识库检索
• 🛠️ 工具库：6种专业医疗工具
• 💬 对话管理：多轮对话支持
• 🔍 上下文管理：智能上下文感知

技术栈：
• LangChain：智能体框架
• OpenAI API：大语言模型
• ChromaDB：向量数据库
• Sentence Transformers：文本嵌入

═══════════════════════════════════════════════════════════════
        """
        
        print(explanation)
    
    def demo_react_agent(self):
        """演示ReAct智能体"""
        if "react" not in self.agents:
            print("❌ ReAct智能体未初始化")
            return
            
        agent = self.agents["react"]
        print("\n" + "="*60)
        print("🧠 ReAct 智能体演示")
        print("="*60)
        
        # 演示功能
        print(agent.demonstrate_capabilities())
        
        # 示例对话
        test_queries = [
            "我最近总是感到胸闷，有时还伴有轻微胸痛，这严重吗？",
            # "阿司匹林的副作用有哪些？适合什么人群服用？",
            # "糖尿病患者应该如何安排日常饮食？"
        ]
        
        print("\n🎭 **智能体对话演示**\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- 示例 {i} ---")
            print(f"👤 用户: {query}")
            
            try:
                response = agent.chat(query)
                print(f"🤖 智能体: {response}")
            except Exception as e:
                print(f"❌ 处理失败: {e}")
            
            print("\n" + "-"*50)
            time.sleep(1)  # 避免API调用过快
    
    def interactive_chat(self, agent_type: str = "react"):
        """交互式聊天"""
        if agent_type not in self.agents:
            print(f"❌ 智能体 '{agent_type}' 不存在")
            return
            
        agent = self.agents[agent_type]
        print(f"\n💬 开始与 {agent_type.upper()} 智能体对话")
        print("输入 'quit' 退出，'reset' 重置对话，'help' 查看帮助\n")
        
        while True:
            try:
                user_input = input("👤 您: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 再见！")
                    break
                elif user_input.lower() == 'reset':
                    agent.reset_conversation()
                    continue
                elif user_input.lower() == 'help':
                    print(agent.demonstrate_capabilities())
                    continue
                elif not user_input:
                    continue
                
                print("\n🤖 智能体思考中...")
                response = agent.chat(user_input)
                print(f"🤖 智能体: {response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 对话已中断，再见！")
                break
            except Exception as e:
                print(f"❌ 出现错误: {e}\n")
    
    def show_agent_comparison(self):
        """显示不同智能体的对比"""
        comparison = """
📊 **智能体类型对比**

┌─────────────────┬──────────────┬──────────────┬──────────────┐
│ 特性             │ ReAct智能体   │ Function Call│ 简化智能体    │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ 推理能力         │ ★★★★★        │ ★★★          │ ★★           │
│ 工具使用         │ ★★★★★        │ ★★★★★        │ ★★           │
│ 执行效率         │ ★★★          │ ★★★★         │ ★★★★★        │
│ 结果准确性       │ ★★★★★        │ ★★★★         │ ★★★          │
│ 可解释性         │ ★★★★★        │ ★★★          │ ★★★          │
│ 资源消耗         │ ★★           │ ★★★          │ ★★★★★        │
│ 复杂任务处理     │ ★★★★★        │ ★★★★         │ ★★           │
│ 开发难度         │ ★★★          │ ★★           │ ★★★★★        │
└─────────────────┴──────────────┴──────────────┴──────────────┘

💡 **选择建议:**
• 复杂医疗咨询 → ReAct智能体
• 简单信息查询 → Function Call
• 快速响应场景 → 简化智能体
        """
        
        print(comparison)
    
    def run_demo(self):
        """运行完整演示"""
        print("🏥 医疗智能体系统演示")
        print("="*60)
        
        # 初始化智能体
        self.initialize_agents()
        
        while True:
            print("\n📋 **菜单选项:**")
            print("1. 📖 智能体架构详解")
            print("2. 🧠 ReAct智能体演示")
            print("3. 💬 交互式对话")
            print("4. 📊 智能体对比")
            print("5. 🚪 退出")
            
            try:
                choice = input("\n请选择 (1-5): ").strip()
                
                if choice == "1":
                    self.explain_agent_architectures()
                elif choice == "2":
                    self.demo_react_agent()
                elif choice == "3":
                    agent_type = input("选择智能体类型 (react/simple) [react]: ").strip()
                    if not agent_type:
                        agent_type = "react"
                    self.interactive_chat(agent_type)
                elif choice == "4":
                    self.show_agent_comparison()
                elif choice == "5":
                    print("👋 感谢使用智能体演示系统！")
                    break
                else:
                    print("❌ 无效选择，请重试")
                    
            except KeyboardInterrupt:
                print("\n👋 程序已中断，再见！")
                break
            except Exception as e:
                print(f"❌ 出现错误: {e}")


def main():
    """主函数"""
    # 检查环境
    print("🔍 检查运行环境...")
    
    # 检查必要的文件
    required_files = [
        "src/medical_agent.py",
        "src/agent_tools.py", 
        "src/rag_retriever.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        print("请确保所有必要的文件都存在")
        return
    
    print("✅ 环境检查通过\n")
    
    # 启动演示系统
    demo_system = AgentSystemDemo()
    demo_system.run_demo()


if __name__ == "__main__":
    main()
