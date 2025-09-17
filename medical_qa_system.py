"""
医疗问答系统
结合RAG检索、大模型和智能体，提供专业的医疗问答服务
"""

import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.rag_retriever import MedicalRAGRetriever
from src.medical_agent import MedicalAgent


class MedicalQASystem:
    """医疗问答系统 - 支持RAG、LLM和Agent三种模式"""
    
    def __init__(self, 
                 vector_store_path: str = "./vector_store",
                 model: str = "qwen-plus",
                 max_context_length: int = 4000,
                 mode: str = "rag",
                 enable_rag: bool = True):
        """
        初始化医疗问答系统
        
        Args:
            vector_store_path: 向量存储路径
            model: 使用的大模型名称
            max_context_length: 最大上下文长度
            mode: 工作模式 ("rag", "llm", "agent")
            enable_rag: 是否启用RAG检索（仅在rag和agent模式中生效）
        """
        self.vector_store_path = vector_store_path
        self.model = model
        self.max_context_length = max_context_length
        self.mode = mode.lower()
        self.enable_rag = enable_rag and (self.mode in ["rag", "agent"])
        
        # 验证模式
        valid_modes = ["rag", "llm", "agent"]
        if self.mode not in valid_modes:
            raise ValueError(f"无效的模式: {self.mode}。支持的模式: {valid_modes}")
        
        # 根据模式初始化相应组件
        if self.mode == "agent":
            # 智能体模式
            self.agent = MedicalAgent(
                vector_store_path=vector_store_path,
                model=model,
                enable_rag=self.enable_rag
            )
            self.retriever = self.agent.retriever if self.enable_rag else None
            self.client = self.agent.client
        else:
            # RAG或LLM模式
            self.agent = None
            
            # 根据RAG开关决定是否初始化RAG检索器
            if self.enable_rag:
                self.retriever = MedicalRAGRetriever(vector_store_path)
            else:
                self.retriever = None
                print("🔄 RAG检索已关闭，使用纯大模型模式")
            
            # 初始化大模型客户端
            self.client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        
        # 系统提示词
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        if self.mode == "agent":
            # 智能体模式使用简化的提示词，因为主要逻辑在agent中
            return """你是一位专业的医疗智能助手，具有工具调用和推理能力。
请基于提供的信息回答用户问题，并始终提醒用户咨询专业医生。"""
        elif self.enable_rag:
            return """你是一位专业的医疗知识问答助手。请遵循以下原则：

1. **专业性**：基于提供的医疗知识库内容回答问题，确保信息准确性
2. **安全性**：不提供具体的诊断或治疗建议，建议用户咨询专业医生
3. **结构化**：回答要条理清晰，分点说明
4. **完整性**：尽量提供全面的信息，包括相关的背景知识
5. **谨慎性**：对于不确定的信息，明确说明并建议进一步咨询

回答格式要求：
- 首先基于知识库内容提供准确信息
- 如果涉及诊断或治疗，提醒用户咨询专业医生
- 提供相关的预防措施或注意事项
- 如果知识库中没有相关信息，诚实说明并建议咨询专业人士

请注意：你的回答仅供参考，不能替代专业医疗建议。"""
        else:
            return """你是一位专业的医疗知识问答助手。请遵循以下原则：

1. **专业性**：基于你的医疗知识回答问题，确保信息准确性
2. **安全性**：不提供具体的诊断或治疗建议，强烈建议用户咨询专业医生
3. **结构化**：回答要条理清晰，分点说明
4. **完整性**：尽量提供全面的信息，包括相关的背景知识
5. **谨慎性**：对于不确定的信息，明确说明并建议进一步咨询专业医生

重要提醒：
- 你的回答基于一般医疗知识，不能替代专业医疗建议
- 任何健康问题都应咨询专业医生进行个性化诊断和治疗
- 不要提供具体的药物剂量或治疗方案
- 如遇紧急情况，建议立即就医

请注意：你的回答仅供参考，不能替代专业医疗建议。"""

    def retrieve_context(self, question: str, k: int = 3) -> tuple[str, List[Dict[str, Any]]]:
        """
        检索相关上下文
        
        Args:
            question: 用户问题
            k: 检索文档数量
            
        Returns:
            (合并的上下文文本, 检索结果列表)
        """
        # 如果RAG被禁用，返回空上下文
        if not self.enable_rag or not self.retriever:
            return "", []
        
        # 使用RAG检索相关文档
        search_results = self.retriever.similarity_search(question, k)
        
        if not search_results:
            return "未找到相关医疗知识。", []
        
        # 构建上下文
        context_parts = []
        for result in search_results:
            source = result['source'].split('/')[-1] if '/' in result['source'] else result['source']
            similarity = result['similarity_score']
            content = result['content']
            
            # 截断过长的内容
            if len(content) > self.max_context_length // k:
                content = content[:self.max_context_length // k] + "..."
            
            context_parts.append(f"【来源: {source} | 相似度: {similarity:.3f}】\n{content}")
        
        context = "\n\n" + "="*50 + "\n\n".join(context_parts)
        return context, search_results
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        使用大模型生成答案
        
        Args:
            question: 用户问题
            context: 检索到的上下文 (如果RAG关闭则为空)
            
        Returns:
            生成的答案
        """
        # 根据是否启用RAG构建不同的用户消息
        if self.enable_rag and context:
            user_message = f"""基于以下医疗知识库内容，回答用户的问题。

知识库内容:
{context}

用户问题: {question}

请基于上述知识库内容，提供专业、准确、安全的回答。"""
        else:
            user_message = f"""请回答以下医疗相关问题，基于你的医疗知识提供专业、准确、安全的回答。

用户问题: {question}

请提供详细的回答，并强调需要咨询专业医生的重要性。"""

        try:
            # 调用大模型
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,  # 降低随机性，提高准确性
                max_tokens=1500   # 限制回答长度
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"调用大模型时出错: {e}")
            return f"抱歉，生成答案时出现了错误。请稍后再试。\n\n基于检索到的信息，相关内容如下：\n{context}"
    
    def answer_question(self, question: str, k: int = 3, show_context: bool = False) -> Dict[str, Any]:
        """
        回答医疗问题
        
        Args:
            question: 用户问题
            k: 检索文档数量  
            show_context: 是否在结果中显示检索上下文
            
        Returns:
            包含答案和相关信息的字典
        """
        mode_name = f"{self.mode.upper()}模式" + ("(RAG增强)" if self.enable_rag else "")
        print(f"\n=== 医疗问答系统 ({mode_name}) ===")
        print(f"问题: {question}")
        
        if self.mode == "agent":
            # 智能体模式
            return self._answer_with_agent(question)
        else:
            # RAG或LLM模式
            return self._answer_with_rag_or_llm(question, k, show_context)
    
    def _answer_with_agent(self, question: str) -> Dict[str, Any]:
        """使用智能体模式回答问题"""
        try:
            agent_result = self.agent.chat(question)
            
            return {
                'question': question,
                'answer': agent_result['response'],
                'search_results': [],
                'retrieval_success': False,
                'sources': [],
                'context': None,
                'mode': f"Agent模式{'(RAG增强)' if self.enable_rag else ''}",
                'rag_enabled': self.enable_rag,
                'agent_info': {
                    'tool_calls': agent_result.get('tool_calls', []),
                    'iterations': agent_result.get('iterations', 0),
                    'tools_used': [tc['tool_call']['tool_name'] for tc in agent_result.get('tool_calls', [])]
                }
            }
        except Exception as e:
            return {
                'question': question,
                'answer': f"智能体处理过程中出现错误: {str(e)}",
                'search_results': [],
                'retrieval_success': False,
                'sources': [],
                'context': None,
                'mode': f"Agent模式{'(RAG增强)' if self.enable_rag else ''}",
                'rag_enabled': self.enable_rag,
                'error': str(e)
            }
    
    def _answer_with_rag_or_llm(self, question: str, k: int, show_context: bool) -> Dict[str, Any]:
        """使用RAG或LLM模式回答问题"""
        # 1. 检索相关上下文 (如果启用RAG)
        context, search_results = self.retrieve_context(question, k)
        
        # 2. 生成答案
        answer = self.generate_answer(question, context)
        
        # 3. 组织结果
        mode_name = "RAG模式" if self.enable_rag else "LLM模式"
        result = {
            'question': question,
            'answer': answer,
            'search_results': search_results,
            'retrieval_success': len(search_results) > 0,
            'sources': [result['source'] for result in search_results],
            'context': context if show_context else None,
            'mode': mode_name,
            'rag_enabled': self.enable_rag
        }
        
        return result
    
    def batch_answer(self, questions: List[str], k: int = 3) -> List[Dict[str, Any]]:
        """
        批量回答问题
        
        Args:
            questions: 问题列表
            k: 每个问题检索的文档数量
            
        Returns:
            答案列表
        """
        results = []
        for question in questions:
            result = self.answer_question(question, k)
            results.append(result)
        return results
    
    def interactive_qa(self):
        """交互式问答模式"""
        mode_name = f"{self.mode.upper()}模式" + ("(RAG增强)" if self.enable_rag else "")
        print(f"=== 医疗问答系统 ({mode_name}) ===")
        print("欢迎使用医疗知识问答系统！")
        print("您可以询问关于疾病、症状、治疗、药物等医疗相关问题。")
        print("输入 'quit' 或 'exit' 退出系统。")
        
        if self.mode == "agent":
            print("当前使用智能体模式，具有工具调用和推理能力。")
        elif self.enable_rag:
            print("当前使用RAG增强模式，基于专业医疗知识库回答。")
        else:
            print("当前使用纯大模型模式，基于模型内置知识回答。")
        print("="*50)
        
        if self.enable_rag and (not self.retriever or not self.retriever.vector_store):
            print("错误：向量存储未初始化，请先运行 build_knowledge_base.py 构建知识库")
            return
        
        # 智能体模式的对话历史
        conversation_history = [] if self.mode == "agent" else None
        
        while True:
            try:
                question = input("\n请输入您的医疗问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("感谢使用医疗问答系统，再见！")
                    break
                
                if not question:
                    print("请输入有效的问题。")
                    continue
                
                
                # 根据模式回答问题
                if self.mode == "agent":
                    # 智能体模式，支持多轮对话
                    agent_result = self.agent.chat(question, conversation_history)
                    
                    print(f"\n【智能体回答】")
                    print(agent_result['response'])
                    
                    # 显示工具使用信息
                    if agent_result.get('tool_calls'):
                        print(f"\n【工具使用】")
                        print(f"使用了 {len(agent_result['tool_calls'])} 个工具:")
                        for i, tool_call in enumerate(agent_result['tool_calls'], 1):
                            tool_name = tool_call['tool_call']['tool_name']
                            reason = tool_call['tool_call'].get('reason', '未说明')
                            success = tool_call['result'].get('success', False)
                            status = "✅" if success else "❌"
                            print(f"  {i}. {tool_name} {status} - {reason}")
                    
                    print(f"\n【执行统计】推理迭代: {agent_result['iterations']}次")
                    
                    # 更新对话历史
                    conversation_history = agent_result['conversation_history']
                    
                else:
                    # RAG或LLM模式
                    result = self.answer_question(question, k=3)
                    
                    # 显示答案
                    print(f"\n【回答】")
                    print(result['answer'])
                    
                    # 只在RAG模式下显示信息来源和检索结果
                    if self.enable_rag and result['sources']:
                        print(f"\n【信息来源】")
                        for i, source in enumerate(set(result['sources']), 1):
                            print(f"{i}. {source}")
                    
                        # 显示检索结果的相似度
                        if result['search_results']:
                            print(f"\n【检索信息】")
                            print(f"找到 {len(result['search_results'])} 个相关文档")
                            for i, res in enumerate(result['search_results'], 1):
                                print(f"{i}. {res['source']} (相似度: {res['similarity_score']:.3f})")
                    elif not self.enable_rag:
                        print(f"\n【信息来源】大模型内置知识")
                
                print("\n" + "-"*50)
                
            except KeyboardInterrupt:
                print("\n\n感谢使用医疗问答系统，再见！")
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
    
    def toggle_rag_mode(self, enable_rag: bool = None):
        """
        切换RAG模式（仅在rag和llm模式下有效）
        
        Args:
            enable_rag: 是否启用RAG，None则切换当前状态
        """
        if self.mode == "agent":
            print("智能体模式的RAG设置在初始化时确定，无法动态切换")
            return
        
        if enable_rag is None:
            self.enable_rag = not self.enable_rag
        else:
            self.enable_rag = enable_rag
        
        # 根据新状态初始化或清理RAG检索器
        if self.enable_rag:
            if not self.retriever:
                self.retriever = MedicalRAGRetriever(self.vector_store_path)
            print("✅ RAG模式已开启 - 将使用专业医疗知识库")
        else:
            print("🔄 RAG模式已关闭 - 将使用纯大模型模式")
        
        # 更新系统提示词
        self.system_prompt = self._build_system_prompt()
    
    def switch_mode(self, new_mode: str, enable_rag: bool = True):
        """
        切换工作模式
        
        Args:
            new_mode: 新模式 ("rag", "llm", "agent")
            enable_rag: 是否启用RAG（仅对rag和agent模式有效）
        """
        valid_modes = ["rag", "llm", "agent"]
        if new_mode.lower() not in valid_modes:
            print(f"❌ 无效的模式: {new_mode}。支持的模式: {valid_modes}")
            return
        
        old_mode = self.mode
        self.mode = new_mode.lower()
        self.enable_rag = enable_rag and (self.mode in ["rag", "agent"])
        
        # 重新初始化组件
        if self.mode == "agent":
            print("🤖 切换到智能体模式...")
            self.agent = MedicalAgent(
                vector_store_path=self.vector_store_path,
                model=self.model,
                enable_rag=self.enable_rag
            )
            self.retriever = self.agent.retriever if self.enable_rag else None
            self.client = self.agent.client
        else:
            print(f"🔄 切换到{self.mode.upper()}模式...")
            self.agent = None
            
            if self.enable_rag:
                if not self.retriever:
                    self.retriever = MedicalRAGRetriever(self.vector_store_path)
            else:
                self.retriever = None
        
        # 更新系统提示词
        self.system_prompt = self._build_system_prompt()
        
        mode_name = f"{self.mode.upper()}模式" + ("(RAG增强)" if self.enable_rag else "")
        print(f"✅ 已从{old_mode.upper()}模式切换到{mode_name}")
    
    def get_current_mode(self) -> str:
        """获取当前模式状态"""
        return f"{self.mode.upper()}模式" + ("(RAG增强)" if self.enable_rag else "")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """获取当前模式的能力描述"""
        if self.mode == "agent":
            return self.agent.get_capabilities()
        else:
            return {
                "mode": self.get_current_mode(),
                "capabilities": [
                    "医疗问答",
                    "文档检索" if self.enable_rag else "知识问答",
                    "批量处理",
                    "质量评估"
                ],
                "limitations": [
                    "不提供具体医疗诊断",
                    "不能替代专业医疗咨询",
                    "建议结果仅供参考"
                ]
            }
    
    def evaluate_answer_quality(self, question: str, answer: str, search_results: List[Dict]) -> Dict[str, Any]:
        """
        评估答案质量
        
        Args:
            question: 问题
            answer: 答案
            search_results: 检索结果
            
        Returns:
            质量评估结果
        """
        evaluation = {
            'has_retrieval_results': len(search_results) > 0,
            'num_sources': len(search_results),
            'avg_similarity': sum(r['similarity_score'] for r in search_results) / len(search_results) if search_results else 0,
            'answer_length': len(answer),
            'has_safety_disclaimer': any(keyword in answer.lower() for keyword in ['咨询医生', '专业医疗', '仅供参考']),
            'coverage_score': self._calculate_coverage_score(question, answer, search_results)
        }
        
        return evaluation
    
    def _calculate_coverage_score(self, question: str, answer: str, search_results: List[Dict]) -> float:
        """计算答案覆盖度分数"""
        if not search_results:
            return 0.0
        
        # 简单的关键词覆盖度计算
        question_keywords = set(question.lower().split())
        answer_keywords = set(answer.lower().split())
        
        # 计算问题关键词在答案中的覆盖率
        coverage = len(question_keywords.intersection(answer_keywords)) / len(question_keywords) if question_keywords else 0
        
        return min(coverage, 1.0)


def main():
    """主函数 - 演示医疗问答系统"""
    # 创建医疗问答系统
    qa_system = MedicalQASystem()
    
    if not qa_system.retriever.vector_store:
        print("向量存储未找到，请先运行 build_knowledge_base.py 构建知识库")
        return
    
    # 测试问题
    test_questions = [
        "高血压的诊断标准是什么？",
        "糖尿病有哪些类型？如何分类？",
        "冠心病的主要治疗方法有哪些？",
        "阿司匹林的用法用量是多少？有什么注意事项？",
        "二甲双胍有哪些不良反应？",
        "高血压患者的生活方式应该注意什么？"
    ]
    
    print("=== 医疗问答系统测试 ===")
    
    # 测试问答功能
    for question in test_questions[:3]:  # 测试前3个问题
        print("\n" + "="*80)
        result = qa_system.answer_question(question, k=3)
        
        print(f"\n【问题】{result['question']}")
        print(f"\n【回答】\n{result['answer']}")
        
        if result['sources']:
            print(f"\n【信息来源】")
            for source in set(result['sources']):
                print(f"- {source}")
        
        # 评估答案质量
        quality = qa_system.evaluate_answer_quality(
            result['question'], 
            result['answer'], 
            result['search_results']
        )
        print(f"\n【质量评估】")
        print(f"- 检索结果数: {quality['num_sources']}")
        print(f"- 平均相似度: {quality['avg_similarity']:.3f}")
        print(f"- 答案长度: {quality['answer_length']} 字符")
        print(f"- 包含安全提醒: {'是' if quality['has_safety_disclaimer'] else '否'}")
        print(f"- 覆盖度分数: {quality['coverage_score']:.3f}")
    
    print("\n" + "="*80)
    print("\n如需交互式问答，请运行: qa_system.interactive_qa()")


if __name__ == "__main__":
    main()
