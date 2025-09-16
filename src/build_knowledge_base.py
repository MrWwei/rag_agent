"""
医疗知识库构建模块
使用LangChain处理医疗文档，进行文本分割和向量化
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

class MedicalKnowledgeBuilder:
    """医疗知识库构建器"""
    
    def __init__(self, 
                 documents_path: str = "./documents",
                 embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 vector_store_path: str = "./vector_store"):
        """
        初始化知识库构建器
        
        Args:
            documents_path: 文档目录路径
            embeddings_model: 嵌入模型名称
            vector_store_path: 向量存储路径
        """
        self.documents_path = documents_path
        self.embeddings_model = embeddings_model
        self.vector_store_path = vector_store_path
        
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", "!", "?", "，", "、", " ", ""]
        )
    
    def load_documents(self) -> List[Document]:
        """
        加载文档目录中的所有markdown文件
        
        Returns:
            加载的文档列表
        """
        try:
            # 使用DirectoryLoader加载markdown文件
            loader = DirectoryLoader(
                self.documents_path,
                glob="*.md",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            documents = loader.load()
            
            print(f"成功加载 {len(documents)} 个文档")
            for doc in documents:
                print(f"- {doc.metadata.get('source', 'Unknown')}")
            
            return documents
            
        except Exception as e:
            print(f"加载文档时出错: {e}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档分割成较小的文本块
        
        Args:
            documents: 原始文档列表
            
        Returns:
            分割后的文档块列表
        """
        try:
            split_docs = self.text_splitter.split_documents(documents)
            print(f"文档分割完成，共生成 {len(split_docs)} 个文本块")
            
            # 显示分割统计信息
            chunk_lengths = [len(doc.page_content) for doc in split_docs]
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            print(f"平均文本块长度: {avg_length:.1f} 字符")
            print(f"最短文本块: {min(chunk_lengths)} 字符")
            print(f"最长文本块: {max(chunk_lengths)} 字符")
            
            return split_docs
            
        except Exception as e:
            print(f"分割文档时出错: {e}")
            return []
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        创建向量存储
        
        Args:
            documents: 文档列表
            
        Returns:
            Chroma向量存储实例
        """
        try:
            print("开始创建向量存储...")
            print(f"使用嵌入模型: {self.embeddings_model}")
            
            # 创建Chroma向量存储
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.vector_store_path
            )
            
            # 持久化存储
            vector_store.persist()
            print(f"向量存储已创建并保存至: {self.vector_store_path}")
            print(f"向量数据库中共有 {vector_store._collection.count()} 个向量")
            
            return vector_store
            
        except Exception as e:
            print(f"创建向量存储时出错: {e}")
            return None
    
    def load_vector_store(self) -> Chroma:
        """
        加载已存在的向量存储
        
        Returns:
            Chroma向量存储实例
        """
        try:
            if not os.path.exists(self.vector_store_path):
                print(f"向量存储路径不存在: {self.vector_store_path}")
                return None
                
            vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )
            
            print(f"向量存储加载成功，共有 {vector_store._collection.count()} 个向量")
            return vector_store
            
        except Exception as e:
            print(f"加载向量存储时出错: {e}")
            return None
    
    def build_knowledge_base(self) -> Chroma:
        """
        构建完整的医疗知识库
        
        Returns:
            Chroma向量存储实例
        """
        print("=== 开始构建医疗知识库 ===")
        
        # 1. 加载文档
        documents = self.load_documents()
        if not documents:
            print("没有找到文档，退出构建过程")
            return None
        
        # 2. 分割文档
        split_docs = self.split_documents(documents)
        if not split_docs:
            print("文档分割失败，退出构建过程")
            return None
        
        # 3. 创建向量存储
        vector_store = self.create_vector_store(split_docs)
        if not vector_store:
            print("向量存储创建失败")
            return None
        
        print("=== 医疗知识库构建完成 ===")
        return vector_store
    
    def get_document_info(self, documents: List[Document]) -> Dict[str, Any]:
        """
        获取文档信息统计
        
        Args:
            documents: 文档列表
            
        Returns:
            文档信息字典
        """
        info = {
            "total_documents": len(documents),
            "total_characters": sum(len(doc.page_content) for doc in documents),
            "sources": list(set(doc.metadata.get('source', 'Unknown') for doc in documents))
        }
        
        return info

def main():
    """主函数"""
    # 设置项目根目录
    project_root = Path(__file__).parent.parent
    documents_path = project_root / "documents"
    vector_store_path = project_root / "vector_store"
    
    # 创建知识库构建器
    builder = MedicalKnowledgeBuilder(
        documents_path=str(documents_path),
        vector_store_path=str(vector_store_path)
    )
    
    # 构建知识库
    vector_store = builder.build_knowledge_base()
    
    if vector_store:
        print("\n=== 知识库构建成功 ===")
        print(f"向量存储位置: {vector_store_path}")
        print("可以开始使用RAG检索系统了!")
    else:
        print("\n=== 知识库构建失败 ===")

if __name__ == "__main__":
    main()
