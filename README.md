# 医疗知识 RAG 智能问答系统

本项目实现了一个基于 LangChain + 大语言模型的医疗知识检索增强生成（RAG）智能问答系统，结合专业医疗知识库和大模型，提供准确、安全的医疗信息查询服务。

## ✨ 主要特性

- **专业医疗知识库**: 包含高血压、糖尿病、冠心病、常用药物等专业医疗文档
- **RAG检索增强**: 基于向量相似度的智能文档检索，确保答案的准确性和可追溯性
- **大模型生成**: 集成阿里云百炼大模型，生成专业、结构化的医疗问答
- **安全提醒**: 自动添加医疗安全提醒，强调专业医疗咨询的重要性
- **交互式问答**: 支持命令行交互式问答，用户体验友好
- **质量评估**: 内置答案质量评估机制，确保回答的可靠性

## 🏗️ 项目结构
```
langchain_rag/
├── documents/              # 医疗知识文档库
│   ├── 高血压.md          # 高血压相关知识
│   ├── 糖尿病.md          # 糖尿病相关知识  
│   ├── 冠心病.md          # 冠心病相关知识
│   └── 常用药物.md        # 常用药物信息
├── src/                   # 核心源代码
│   ├── build_knowledge_base.py  # 知识库构建脚本
│   ├── rag_retriever.py        # RAG检索器
│   └── test_rag.py             # RAG功能测试
├── vector_store/          # 向量存储数据库
├── medical_qa_system.py   # 🆕 医疗问答系统核心
├── demo_medical_qa.py     # 🆕 演示脚本
├── start_medical_qa.py    # 🆕 快速启动脚本
├── chat_llm_api.py        # 大模型API调用示例
├── requirements.txt       # 项目依赖
└── README.md             # 项目说明
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装Python依赖
pip install -r requirements.txt

# 设置大模型API密钥 (阿里云百炼)
export DASHSCOPE_API_KEY='your-api-key-here'
```

### 2. 构建知识库
```bash
# 构建医疗知识向量数据库
python src/build_knowledge_base.py
```

### 3. 启动问答系统
```bash
# 快速启动交互式问答
python start_medical_qa.py

# 或运行完整演示
python demo_medical_qa.py
```

## 💡 使用示例

### 交互式问答
```python
from medical_qa_system import MedicalQASystem

# 创建问答系统
qa_system = MedicalQASystem()

# 单次问答
result = qa_system.answer_question("高血压的诊断标准是什么？")
print(result['answer'])

# 交互式模式
qa_system.interactive_qa()
```

### 批量问答
```python
# 批量处理问题
questions = [
    "糖尿病有哪些类型？",
    "冠心病的主要症状？",
    "阿司匹林的用法用量？"
]
results = qa_system.batch_answer(questions)
```

## 🔍 系统架构

```
用户问题 → RAG检索器 → 向量数据库检索 → 相关文档
                                           ↓
用户答案 ← 大语言模型 ← 结构化提示词 ← 检索上下文
```

### 核心组件

1. **MedicalRAGRetriever**: 医疗知识检索器
   - 基于sentence-transformers的多语言嵌入模型
   - Chroma向量数据库存储和检索
   - 支持相似度搜索和元数据过滤

2. **MedicalQASystem**: 医疗问答系统
   - 集成RAG检索和大模型生成
   - 智能上下文构建和长度控制
   - 专业医疗提示词工程

3. **质量评估模块**: 
   - 检索结果相关性评估
   - 答案完整性和安全性检查
   - 信息来源可追溯性验证

## 📋 示例问题

系统可以回答以下类型的医疗问题：

**疾病相关:**
- "高血压的诊断标准是什么？"
- "糖尿病有哪些类型和症状？"
- "冠心病的危险因素有哪些？"

**药物相关:**
- "阿司匹林的用法用量是多少？"
- "二甲双胍有什么副作用？"
- "胰岛素注射的注意事项？"

**预防保健:**
- "如何预防高血压？"
- "糖尿病患者饮食要注意什么？"
- "冠心病患者的运动建议？"

## ⚠️ 重要声明

**本系统提供的信息仅供参考和学习使用，不能替代专业医疗建议、诊断或治疗。如有健康问题，请及时咨询专业医生。**

## 🛠️ 技术栈

- **LangChain**: RAG框架和文档处理
- **ChromaDB**: 向量数据库
- **sentence-transformers**: 多语言嵌入模型
- **OpenAI API**: 大语言模型调用 (兼容阿里云百炼)
- **Python**: 核心开发语言

## 📈 扩展功能

- [ ] Web界面集成
- [ ] 医疗图像问答
- [ ] 多轮对话支持  
- [ ] 个性化推荐
- [ ] 医疗知识图谱集成

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目仅供学习和研究使用。
