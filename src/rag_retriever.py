"""
医疗知识RAG检索系统
实现基于向量相似度的文档检索功能
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.schema import Document
import numpy as np

class MedicalRAGRetriever:
    """医疗知识RAG检索器"""
    
    def __init__(self, 
                 vector_store_path: str = "./vector_store",
                 embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化RAG检索器
        
        Args:
            vector_store_path: 向量存储路径
            embeddings_model: 嵌入模型名称
        """
        self.vector_store_path = vector_store_path
        self.embeddings_model = embeddings_model
        
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 加载向量存储
        self.vector_store = self.load_vector_store()
    
    def load_vector_store(self) -> Optional[Chroma]:
        """
        加载向量存储
        
        Returns:
            Chroma向量存储实例或None
        """
        try:
            if not os.path.exists(self.vector_store_path):
                print(f"向量存储路径不存在: {self.vector_store_path}")
                return None
                
            vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )
            
            count = vector_store._collection.count()
            print(f"向量存储加载成功，共有 {count} 个向量")
            return vector_store
            
        except Exception as e:
            print(f"加载向量存储时出错: {e}")
            return None
    
    def embedding_query(self, query: str) -> List[float]:
        """
        对查询文本进行嵌入
        
        Args:
            query: 查询文本
            
        Returns:
            查询文本的嵌入向量
        """
        try:
            # 使用嵌入模型对查询进行向量化
            query_embedding = self.embeddings.embed_query(query)
            print(f"查询文本嵌入完成，向量维度: {len(query_embedding)}")
            print(f"查询文本: '{query}'")
            return query_embedding
            
        except Exception as e:
            print(f"嵌入查询时出错: {e}")
            return []
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 3, 
                         score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        基于相似度的文档检索
        
        Args:
            query: 查询文本
            k: 返回的最相似文档数量
            score_threshold: 相似度阈值
            
        Returns:
            检索结果列表，每个结果包含文档内容和相似度分数
        """
        if not self.vector_store:
            print("向量存储未初始化")
            return []
        
        try:
            print(f"\n=== 开始检索 ===")
            print(f"查询: {query}")
            print(f"检索数量: {k}")
            
            # 进行相似度检索，返回文档和分数
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # 格式化检索结果
            formatted_results = []
            for i, (doc, score) in enumerate(results):
                if score >= score_threshold:
                    result = {
                        'rank': i + 1,
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': float(score),
                        'source': doc.metadata.get('source', 'Unknown')
                    }
                    formatted_results.append(result)
            
            print(f"检索完成，找到 {len(formatted_results)} 个相关文档")
            return formatted_results
            
        except Exception as e:
            print(f"检索时出错: {e}")
            return []
    
    def semantic_search(self, 
                       query: str, 
                       k: int = 5, 
                       filter_dict: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        语义搜索，支持元数据过滤
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: 元数据过滤条件
            
        Returns:
            搜索结果列表
        """
        if not self.vector_store:
            print("向量存储未初始化")
            return []
        
        try:
            print(f"\n=== 语义搜索 ===")
            print(f"查询: {query}")
            
            # 使用filter参数进行过滤搜索
            if filter_dict:
                print(f"过滤条件: {filter_dict}")
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k
                )
            
            # 格式化结果
            formatted_results = []
            for i, (doc, score) in enumerate(results):
                result = {
                    'rank': i + 1,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score),
                    'source': doc.metadata.get('source', 'Unknown'),
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"语义搜索时出错: {e}")
            return []
    
    def retrieve_relevant_docs(self, 
                              query: str, 
                              k: int = 3) -> str:
        """
        检索相关文档并合并为上下文
        
        Args:
            query: 查询文本
            k: 检索文档数量
            
        Returns:
            合并的相关文档内容
        """
        results = self.similarity_search(query, k)
        
        if not results:
            return "没有找到相关文档。"
        
        # 合并检索到的文档内容
        context_parts = []
        for result in results:
            source = result['source'].split('/')[-1] if '/' in result['source'] else result['source']
            context_parts.append(f"来源: {source}\n{result['content']}")
        
        context = "\n\n---\n\n".join(context_parts)
        return context
    
    def search_relevant_docs(self, query: str, top_k: int = 3) -> List[str]:
        """
        搜索相关文档（为了兼容性）
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            相关文档内容列表
        """
        results = self.similarity_search(query, k=top_k)
        
        if not results:
            return []
        
        # 返回文档内容列表
        docs = []
        for result in results:
            source = result['source'].split('/')[-1] if '/' in result['source'] else result['source']
            doc_content = f"来源: {source}\n{result['content']}"
            docs.append(doc_content)
        
        return docs
    
    def analyze_query_embedding(self, query: str) -> Dict[str, Any]:
        """
        分析查询嵌入向量的特征
        
        Args:
            query: 查询文本
            
        Returns:
            嵌入向量分析结果
        """
        embedding = self.embedding_query(query)
        
        if not embedding:
            return {}
        
        embedding_array = np.array(embedding)
        
        analysis = {
            'query': query,
            'embedding_dim': len(embedding),
            'mean': float(np.mean(embedding_array)),
            'std': float(np.std(embedding_array)),
            'min': float(np.min(embedding_array)),
            'max': float(np.max(embedding_array)),
            'norm': float(np.linalg.norm(embedding_array))
        }
        
        return analysis
    
    def print_search_results(self, results: List[Dict[str, Any]], show_content: bool = True):
        """
        格式化打印搜索结果
        
        Args:
            results: 搜索结果列表
            show_content: 是否显示完整内容
        """
        if not results:
            print("没有找到相关结果。")
            return
        
        print(f"\n=== 检索结果 ({len(results)} 个) ===")
        
        for result in results:
            print(f"\n排名 {result['rank']}:")
            print(f"来源: {result['source']}")
            print(f"相似度分数: {result['similarity_score']:.4f}")
            
            if show_content:
                print(f"内容:\n{result['content']}")
            else:
                print(f"内容预览:\n{result.get('content_preview', result['content'][:200] + '...')}")
            
            print("-" * 50)

class MedicalRAGSystem:
    """完整的医疗RAG系统"""
    
    def __init__(self, vector_store_path: str = "./vector_store"):
        """
        初始化医疗RAG系统
        
        Args:
            vector_store_path: 向量存储路径
        """
        self.retriever = MedicalRAGRetriever(vector_store_path)
    
    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """
        完整的RAG查询流程
        
        Args:
            question: 用户问题
            k: 检索文档数量
            
        Returns:
            查询结果字典
        """
        print(f"\n=== 医疗知识查询 ===")
        print(f"问题: {question}")
        
        # 1. 嵌入查询
        query_embedding = self.retriever.embedding_query(question)
        
        # 2. 检索相关文档
        search_results = self.retriever.similarity_search(question, k)
        
        # 3. 获取相关上下文
        context = self.retriever.retrieve_relevant_docs(question, k)
        
        # 4. 组织结果
        result = {
            'question': question,
            'query_embedding_info': self.retriever.analyze_query_embedding(question),
            'search_results': search_results,
            'context': context,
            'retrieval_success': len(search_results) > 0
        }
        
        return result

def main():
    """主函数 - 演示RAG系统功能"""
    # 设置项目根目录
    project_root = Path(__file__).parent.parent
    vector_store_path = project_root / "vector_store"
    
    # 创建RAG系统
    rag_system = MedicalRAGSystem(str(vector_store_path))
    
    if not rag_system.retriever.vector_store:
        print("向量存储未找到，请先运行 build_knowledge_base.py 构建知识库")
        return
    
    # 测试查询
    test_queries = [
        "高血压的诊断标准是什么？",
        "糖尿病有哪些类型？",
        "冠心病的治疗方法",
        "阿司匹林的用法用量",
        "二甲双胍的不良反应"
    ]
    
    print("=== 医疗RAG系统测试 ===")
    
    for query in test_queries:
        # 执行查询
        result = rag_system.query(query, k=3)
        
        # 显示检索结果
        rag_system.retriever.print_search_results(result['search_results'], show_content=False)
        
        # 显示嵌入信息
        embedding_info = result['query_embedding_info']
        if embedding_info:
            print(f"\n嵌入向量信息:")
            print(f"- 维度: {embedding_info['embedding_dim']}")
            print(f"- 平均值: {embedding_info['mean']:.6f}")
            print(f"- 标准差: {embedding_info['std']:.6f}")
            print(f"- 向量范数: {embedding_info['norm']:.6f}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
