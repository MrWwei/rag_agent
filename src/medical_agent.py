"""
医疗智能体模块
实现基于ReAct模式的医疗智能体，支持工具调用和多轮对话
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from src.agent_tools import tool_registry
from src.rag_retriever import MedicalRAGRetriever
import os

class MedicalAgent:
    """医疗智能体 - 基于ReAct模式实现"""
    
    def __init__(self, 
                 vector_store_path: str = "./vector_store",
                 model: str = "qwen-plus",
                 enable_rag: bool = True,
                 max_iterations: int = 5,
                 temperature: float = 0.1):
        """
        初始化医疗智能体
        
        Args:
            vector_store_path: 向量存储路径
            model: 使用的大模型
            enable_rag: 是否启用RAG检索
            max_iterations: 最大思考-行动迭代次数
            temperature: 模型温度参数
        """
        self.vector_store_path = vector_store_path
        self.model = model
        self.enable_rag = enable_rag
        self.max_iterations = max_iterations
        self.temperature = temperature
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),  # 这里需要设置实际的API密钥
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        # self.client = OpenAI(
        #     api_key="sk-64ff01544b8941eaab50072f7d2201d9",  # 这里需要设置实际的API密钥
        #     base_url="https://api.deepseek.com"
        # )
        # self.client = OpenAI(
        #     api_key="sk-TbRGdYJdI5XWafL5e4uNwFyQgJcPGTZMV2phnBHFhitciVTg", # 在这里将 MOONSHOT_API_KEY 替换为你从 Kimi 开放平台申请的 API Key
        #     base_url="https://api.moonshot.cn/v1",
        # )
        # self.model = 'kimi-k2-0905-preview'
        # 初始化RAG检索器（如果启用）
        if self.enable_rag:
            try:
                self.rag_retriever = MedicalRAGRetriever(vector_store_path)
            except Exception as e:
                print(f"RAG检索器初始化失败: {e}")
                self.enable_rag = False
                self.rag_retriever = None
        else:
            self.rag_retriever = None
        
        # 获取可用工具
        self.tools = tool_registry.get_tools_schema()
        
        # 对话历史
        self.conversation_history = []
        
        # 系统提示词
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        prompt = """你是一个专业的医疗智能助手，具备以下能力：

1. **知识检索**: 可以搜索医疗知识库获取准确信息
2. **症状分析**: 能够分析症状并提供初步诊断建议  
3. **药物咨询**: 提供药物信息和用药指导
4. **健康建议**: 给出针对性的健康生活建议
5. **紧急评估**: 评估症状紧急程度，指导就医时机
6. **科室推荐**: 根据症状推荐合适的医院科室

**工作模式 - ReAct (Reasoning + Acting):**
当用户提出问题时，你需要：
1. **思考(Think)**: 分析问题，确定需要什么信息
2. **行动(Act)**: 使用合适的工具获取信息
3. **观察(Observe)**: 分析工具返回的结果
4. **重复**: 如果需要更多信息，重复上述过程
5. **回答**: 基于收集的信息给出综合回答

**重要原则:**
- 始终强调医疗建议仅供参考，不能替代专业医疗诊断
- 遇到紧急情况，优先建议立即就医
- 提供信息时要准确、客观、易懂
- 保护患者隐私，避免询问过于敏感的个人信息
- 不提供具体的诊断结论，只提供参考信息

**回答格式:**
使用清晰的结构化回答，包含：
- 问题理解
- 相关信息（通过工具获取）
- 分析和建议
- 注意事项和就医建议

现在开始为用户提供专业的医疗咨询服务。"""
        
        return prompt
    
    def think(self, user_query: str, context: str = "") -> str:
        """思考阶段 - 分析问题并制定行动计划"""
        thinking_prompt = f"""
用户问题: {user_query}

{f"上下文信息: {context}" if context else ""}

请分析这个问题，思考需要采取什么行动来回答用户的问题。
你可以使用的工具包括：
- medical_knowledge_search: 搜索医疗知识
- symptom_analysis: 症状分析
- drug_information: 药物信息查询
- health_advice: 健康建议
- emergency_assessment: 紧急程度评估
- department_recommendation: 科室推荐

请说明：
1. 你对用户问题的理解
2. 需要使用哪些工具来获取信息
3. 预期的工作流程

思考过程:"""
        
        return thinking_prompt
    
    def process_query(self, user_query: str) -> str:
        """处理用户查询 - 主要的ReAct循环"""
        print(f"\n🤔 收到用户问题: {user_query}")
        
        # 初始化对话
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # ReAct循环
        for iteration in range(self.max_iterations):
            print(f"\n--- 第 {iteration + 1} 轮思考 ---")
            
            try:
                # 调用大模型进行推理
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=1500
                )
                
                assistant_message = response.choices[0].message
                messages.append(assistant_message.model_dump())
                # import pdb; pdb.set_trace()
                
                # 检查是否需要调用工具
                if assistant_message.tool_calls:
                    print(f"🔧 需要调用 {len(assistant_message.tool_calls)} 个工具")
                    
                    # 执行工具调用
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        print(f"   调用工具: {tool_name}")
                        print(f"   参数: {tool_args}")
                        
                        # 执行工具
                        tool_result = tool_registry.execute_tool(tool_name, **tool_args)
                        
                        print(f"   结果: {tool_result[:200]}...")
                        
                        # 将工具结果添加到对话中
                        messages.append({
                            "role": "tool",
                            "content": tool_result,
                            "tool_call_id": tool_call.id
                        })
                    
                    # 继续对话让模型处理工具结果
                    continue
                
                else:
                    # 没有工具调用，返回最终回答
                    final_answer = assistant_message.content
                    print(f"\n✅ 获得最终回答")
                    
                    # 保存对话历史
                    self.conversation_history.append({
                        "user_query": user_query,
                        "assistant_response": final_answer,
                        "timestamp": time.time(),
                        "iterations": iteration + 1
                    })
                    
                    return final_answer
                    
            except Exception as e:
                print(f"❌ 第 {iteration + 1} 轮处理出错: {e}")
                if iteration == self.max_iterations - 1:
                    return f"抱歉，处理您的问题时遇到了错误: {str(e)}。请重新描述您的问题。"
                continue
        
        # 达到最大迭代次数
        return "抱歉，问题比较复杂，我需要更多时间来分析。请您简化问题或分步骤询问。"
    
    def get_rag_context(self, query: str, top_k: int = 3) -> str:
        """获取RAG上下文信息"""
        if not self.enable_rag or not self.rag_retriever:
            return ""
        
        try:
            relevant_docs = self.rag_retriever.search_relevant_docs(query, top_k=top_k)
            if relevant_docs:
                return "\n\n".join(relevant_docs)
            return ""
        except Exception as e:
            print(f"RAG检索错误: {e}")
            return ""
    
    def chat(self, user_input: str) -> str:
        """对话接口 - 支持多轮对话"""
        return self.process_query(user_input)
    
    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []
        print("对话历史已重置")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """获取对话摘要"""
        if not self.conversation_history:
            return {"total_conversations": 0, "summary": "暂无对话记录"}
        
        total_conversations = len(self.conversation_history)
        avg_iterations = sum(conv.get("iterations", 1) for conv in self.conversation_history) / total_conversations
        
        recent_topics = []
        for conv in self.conversation_history[-5:]:  # 最近5次对话
            query = conv["user_query"][:50]
            recent_topics.append(query)
        
        return {
            "total_conversations": total_conversations,
            "average_iterations": round(avg_iterations, 2),
            "recent_topics": recent_topics,
            "rag_enabled": self.enable_rag
        }
    
    def demonstrate_capabilities(self) -> str:
        """演示智能体的功能"""
        demo_text = """
🏥 医疗智能体功能演示

我是一个专业的医疗智能助手，具备以下核心功能：

📚 **知识检索能力**
- 搜索海量医疗知识库
- 获取疾病、症状、治疗相关信息
- 提供循证医学依据

🔍 **症状分析能力**  
- 分析症状组合
- 提供初步诊断参考
- 考虑患者基本信息

💊 **药物咨询能力**
- 查询药物基本信息
- 提供用法用量指导
- 说明注意事项和副作用

🍎 **健康指导能力**
- 提供生活方式建议
- 制定健康管理方案
- 疾病预防指导

🚨 **紧急评估能力**
- 评估症状紧急程度
- 指导就医时机选择
- 提供急救建议

🏥 **科室推荐能力**
- 根据症状推荐科室
- 优化就医流程
- 提高诊疗效率

**工作原理 - ReAct模式:**
思考 → 行动 → 观察 → 思考 → 行动...

您可以尝试询问：
- "我最近总是头痛，应该怎么办？"
- "阿司匹林的副作用有哪些？"
- "高血压患者的饮食建议"
- "胸痛需要立即就医吗？"

⚠️ **重要提醒**: 我的建议仅供参考，不能替代专业医疗诊断。遇到健康问题请及时就医。
        """
        
        return demo_text
