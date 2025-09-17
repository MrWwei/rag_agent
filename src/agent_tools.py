"""
智能体工具模块
定义医疗智能体可以使用的各种工具
"""

import json
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.rag_retriever import MedicalRAGRetriever


class ToolRegistry:
    """工具注册器 - 管理所有可用的工具"""
    
    def __init__(self):
        self.tools = {}
        self.register_default_tools()
    
    def register_tool(self, name: str, func, description: str, parameters: Dict[str, Any]):
        """注册工具"""
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
    
    def register_default_tools(self):
        """注册默认工具"""
        # 医疗知识搜索工具
        self.register_tool(
            "medical_knowledge_search",
            self.medical_knowledge_search,
            "搜索医疗知识库，获取相关医疗信息",
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询词"
                    },
                    "top_k": {
                        "type": "integer", 
                        "description": "返回结果数量",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        )
        
        # 症状分析工具
        self.register_tool(
            "symptom_analysis",
            self.symptom_analysis,
            "分析症状，提供初步诊断建议",
            {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "症状列表"
                    },
                    "patient_info": {
                        "type": "object",
                        "properties": {
                            "age": {"type": "integer"},
                            "gender": {"type": "string"},
                            "medical_history": {"type": "array", "items": {"type": "string"}}
                        },
                        "description": "患者基本信息（可选）"
                    }
                },
                "required": ["symptoms"]
            }
        )
        
        # 药物查询工具
        self.register_tool(
            "drug_information",
            self.drug_information,
            "查询药物信息，包括用法用量、副作用等",
            {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "药物名称"
                    }
                },
                "required": ["drug_name"]
            }
        )
        
        # 健康建议工具
        self.register_tool(
            "health_advice",
            self.health_advice,
            "根据病症提供健康生活建议",
            {
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "疾病或健康状况"
                    },
                    "lifestyle_factors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "生活方式因素（可选）"
                    }
                },
                "required": ["condition"]
            }
        )
        
        # 紧急情况评估工具
        self.register_tool(
            "emergency_assessment",
            self.emergency_assessment,
            "评估症状的紧急程度，判断是否需要立即就医",
            {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "当前症状"
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["mild", "moderate", "severe"],
                        "description": "症状严重程度"
                    }
                },
                "required": ["symptoms"]
            }
        )
        
        # 医院科室推荐工具
        self.register_tool(
            "department_recommendation",
            self.department_recommendation,
            "根据症状推荐合适的医院科室",
            {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "症状描述"
                    },
                    "suspected_condition": {
                        "type": "string",
                        "description": "疑似疾病（可选）"
                    }
                },
                "required": ["symptoms"]
            }
        )
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """获取工具的OpenAI函数调用格式"""
        tools_schema = []
        for name, tool_info in self.tools.items():
            tools_schema.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool_info["description"],
                    "parameters": tool_info["parameters"]
                }
            })
        return tools_schema
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """执行工具"""
        if tool_name not in self.tools:
            return f"工具 '{tool_name}' 不存在"
        
        try:
            result = self.tools[tool_name]["function"](**kwargs)
            return result
        except Exception as e:
            return f"执行工具 '{tool_name}' 时发生错误: {str(e)}"
    
    # 工具实现
    def medical_knowledge_search(self, query: str, top_k: int = 3) -> str:
        """搜索医疗知识库"""
        try:
            # 初始化RAG检索器
            retriever = MedicalRAGRetriever()
            results = retriever.search_relevant_docs(query, top_k=top_k)
            
            if not results:
                return "未找到相关医疗信息"
            
            search_results = []
            for i, result in enumerate(results, 1):
                search_results.append(f"结果{i}: {result}")
            
            return "\n\n".join(search_results)
        except Exception as e:
            return f"搜索医疗知识时出错: {str(e)}"
    
    def symptom_analysis(self, symptoms: List[str], patient_info: Optional[Dict] = None) -> str:
        """症状分析"""
        symptom_text = "、".join(symptoms)
        
        # 基于症状的初步分析
        analysis = f"症状分析报告:\n"
        analysis += f"主要症状: {symptom_text}\n\n"
        
        if patient_info:
            analysis += f"患者信息:\n"
            if "age" in patient_info:
                analysis += f"- 年龄: {patient_info['age']}岁\n"
            if "gender" in patient_info:
                analysis += f"- 性别: {patient_info['gender']}\n"
            if "medical_history" in patient_info:
                history = "、".join(patient_info['medical_history'])
                analysis += f"- 既往病史: {history}\n"
            analysis += "\n"
        
        # 简单的症状匹配逻辑
        common_conditions = {
            "发热": ["感冒", "流感", "感染"],
            "咳嗽": ["感冒", "支气管炎", "肺炎"],
            "头痛": ["偏头痛", "紧张性头痛", "高血压"],
            "胸痛": ["心绞痛", "肌肉拉伤", "焦虑"],
            "腹痛": ["胃炎", "肠胃炎", "阑尾炎"]
        }
        
        possible_conditions = set()
        for symptom in symptoms:
            for key, conditions in common_conditions.items():
                if key in symptom:
                    possible_conditions.update(conditions)
        
        if possible_conditions:
            analysis += f"可能的疾病: {', '.join(possible_conditions)}\n\n"
        
        analysis += "注意: 此分析仅供参考，请及时就医获得专业诊断。"
        
        return analysis
    
    def drug_information(self, drug_name: str) -> str:
        """药物信息查询"""
        # 简化的药物信息数据库
        drug_db = {
            "阿司匹林": {
                "作用": "解热镇痛、抗血小板聚集",
                "用法用量": "口服，每次75-100mg，每日1次",
                "副作用": "胃肠道不适、出血风险增加",
                "注意事项": "餐后服用，注意出血风险"
            },
            "布洛芬": {
                "作用": "解热镇痛抗炎",
                "用法用量": "口服，每次200-400mg，每日2-3次",
                "副作用": "胃肠道不适、头晕",
                "注意事项": "餐后服用，避免长期使用"
            },
            "对乙酰氨基酚": {
                "作用": "解热镇痛",
                "用法用量": "口服，每次500mg，每4-6小时一次",
                "副作用": "过量可导致肝损伤",
                "注意事项": "注意日用量不超过4g"
            }
        }
        
        drug_name = drug_name.strip()
        if drug_name in drug_db:
            info = drug_db[drug_name]
            result = f"药物: {drug_name}\n\n"
            result += f"作用: {info['作用']}\n"
            result += f"用法用量: {info['用法用量']}\n"
            result += f"副作用: {info['副作用']}\n"
            result += f"注意事项: {info['注意事项']}\n\n"
            result += "警告: 请在医生指导下使用药物，不要自行调整剂量。"
            return result
        else:
            return f"未找到药物 '{drug_name}' 的信息。建议咨询医生或药师获取详细信息。"
    
    def health_advice(self, condition: str, lifestyle_factors: Optional[List[str]] = None) -> str:
        """健康建议"""
        advice_db = {
            "高血压": {
                "饮食": "低盐饮食，多吃蔬菜水果",
                "运动": "适量有氧运动，如散步、游泳",
                "生活": "规律作息，控制体重，戒烟限酒"
            },
            "糖尿病": {
                "饮食": "控制碳水化合物摄入，定时定量进餐",
                "运动": "餐后30分钟适量运动",
                "生活": "监测血糖，按时服药，足部护理"
            },
            "冠心病": {
                "饮食": "低脂低胆固醇饮食",
                "运动": "循序渐进的有氧运动",
                "生活": "控制情绪，避免过度劳累"
            }
        }
        
        result = f"针对 '{condition}' 的健康建议:\n\n"
        
        if condition in advice_db:
            advice = advice_db[condition]
            result += f"饮食建议: {advice['饮食']}\n"
            result += f"运动建议: {advice['运动']}\n"
            result += f"生活建议: {advice['生活']}\n\n"
        else:
            result += "一般健康建议:\n"
            result += "- 保持均衡饮食\n"
            result += "- 适量运动\n"
            result += "- 规律作息\n"
            result += "- 定期体检\n\n"
        
        if lifestyle_factors:
            result += f"基于您的生活方式因素 ({', '.join(lifestyle_factors)})，"
            result += "建议进一步咨询医生制定个性化健康方案。\n\n"
        
        result += "重要提醒: 以上建议仅供参考，具体治疗方案请遵医嘱。"
        
        return result
    
    def emergency_assessment(self, symptoms: List[str], severity: str = "moderate") -> str:
        """紧急情况评估"""
        emergency_symptoms = [
            "胸痛", "呼吸困难", "意识模糊", "剧烈头痛", 
            "高热", "大出血", "严重腹痛", "中毒症状"
        ]
        
        urgent_symptoms = [
            "持续发热", "剧烈咳嗽", "严重呕吐", 
            "关节疼痛", "皮疹", "失眠"
        ]
        
        assessment = "紧急程度评估:\n\n"
        assessment += f"症状: {', '.join(symptoms)}\n"
        assessment += f"严重程度: {severity}\n\n"
        
        emergency_count = sum(1 for symptom in symptoms 
                            for emergency in emergency_symptoms 
                            if emergency in symptom)
        
        urgent_count = sum(1 for symptom in symptoms 
                         for urgent in urgent_symptoms 
                         if urgent in symptom)
        
        if emergency_count > 0 or severity == "severe":
            level = "紧急"
            recommendation = "建议立即就医或拨打急救电话120"
            color = "🔴"
        elif urgent_count > 0 or severity == "moderate":
            level = "较急"
            recommendation = "建议24小时内就医"
            color = "🟡"
        else:
            level = "一般"
            recommendation = "可预约门诊就医，注意观察症状变化"
            color = "🟢"
        
        assessment += f"评估结果: {color} {level}\n"
        assessment += f"建议: {recommendation}\n\n"
        assessment += "注意: 此评估仅供参考，如有疑虑请及时就医。"
        
        return assessment
    
    def department_recommendation(self, symptoms: List[str], suspected_condition: Optional[str] = None) -> str:
        """科室推荐"""
        department_map = {
            "心": "心内科",
            "胸": "心内科",
            "呼吸": "呼吸科",
            "咳嗽": "呼吸科",
            "腹": "消化科",
            "胃": "消化科",
            "头": "神经内科",
            "关节": "骨科",
            "皮": "皮肤科",
            "眼": "眼科",
            "耳": "耳鼻喉科"
        }
        
        result = "科室推荐:\n\n"
        result += f"症状: {', '.join(symptoms)}\n"
        
        if suspected_condition:
            result += f"疑似疾病: {suspected_condition}\n"
        
        result += "\n推荐科室:\n"
        
        recommended_departments = set()
        
        for symptom in symptoms:
            for key, department in department_map.items():
                if key in symptom:
                    recommended_departments.add(department)
        
        if suspected_condition:
            for key, department in department_map.items():
                if key in suspected_condition:
                    recommended_departments.add(department)
        
        if recommended_departments:
            for dept in recommended_departments:
                result += f"- {dept}\n"
        else:
            result += "- 内科（建议先到内科初诊）\n"
        
        result += "\n提醒: 如不确定，可先挂号内科，由医生进一步转诊。"
        
        return result


# 全局工具注册器实例
tool_registry = ToolRegistry()
