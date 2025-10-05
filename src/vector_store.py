"""
Vector Store Manager for Thai RAG Chatbot
จัดการ embeddings และ vector database ด้วย FAISS และ local embedding models
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings


class LocalThaiEmbeddings(Embeddings):
    """
    Local Embedding class สำหรับภาษาไทย
    ใช้ Sentence Transformers models ที่รองรับภาษาไทย
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            model_name: ชื่อโมเดล embedding ที่จะใช้
                       แนะนำ:
                       - paraphrase-multilingual-MiniLM-L12-v2 (เร็ว, รองรับหลายภาษา)
                       - distiluse-base-multilingual-cased (ดี, รองรับหลายภาษา)
                       - paraphrase-multilingual-mpnet-base-v2 (ดีที่สุด, ช้าหน่อย)
        """
        self.model_name = model_name
        print(f"🔄 กำลังโหลดโมเดล embedding: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            print(f"✅ โหลดโมเดล {model_name} สำเร็จ")
        except Exception as e:
            print(f"❌ ไม่สามารถโหลดโมเดล {model_name}: {e}")
            # ลองใช้โมเดลสำรอง
            backup_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            print(f"🔄 กำลังลองใช้โมเดลสำรอง: {backup_model}")
            self.model = SentenceTransformer(backup_model)
            self.model_name = backup_model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """สร้าง embeddings สำหรับรายการข้อความ"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            print(f"❌ ไม่สามารถสร้าง embeddings: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """สร้าง embedding สำหรับคำถาม"""
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        except Exception as e:
            print(f"❌ ไม่สามารถสร้าง embedding สำหรับคำถาม: {e}")
            raise


class ThaiVectorStoreManager:
    """คลาสจัดการ Vector Store สำหรับภาษาไทย"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 vector_store_path: str = "./vectorstore"):
        """
        Args:
            embedding_model: ชื่อโมเดล embedding
            vector_store_path: เส้นทางเก็บ vector store
        """
        self.embedding_model_name = embedding_model
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(exist_ok=True)
        
        # สร้าง embedding model
        self.embeddings = LocalThaiEmbeddings(embedding_model)
        
        # Vector store
        self.vector_store: Optional[FAISS] = None
        self.documents: List[Document] = []
        
        # Metadata สำหรับการจัดการ
        self.metadata_path = self.vector_store_path / "metadata.pkl"
        self.faiss_index_path = self.vector_store_path / "index.faiss"
        self.faiss_pkl_path = self.vector_store_path / "index.pkl"
    
    def add_documents(self, documents: List[Document]) -> None:
        """เพิ่มเอกสารลงใน vector store"""
        if not documents:
            print("⚠️ ไม่มีเอกสารที่จะเพิ่ม")
            return
        
        print(f"🔄 กำลังเพิ่ม {len(documents)} เอกสารลงใน vector store...")
        
        # กรองเอกสารที่ซ้ำกันออก (deduplication ก่อน embed)
        unique_docs = []
        seen_contents = set()
        
        for doc in documents:
            # ใช้ content ทั้งหมดเป็น signature
            content_hash = hash(doc.page_content.strip())
            if content_hash not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(content_hash)
            else:
                print(f"  ⏭️ ข้าม chunk ที่ซ้ำ (hash: {content_hash})")
        
        print(f"  ✂️ กรองแล้ว: {len(documents)} → {len(unique_docs)} เอกสาร (ลบซ้ำ {len(documents) - len(unique_docs)} ชิ้น)")
        
        if not unique_docs:
            print("⚠️ ไม่มีเอกสารใหม่หลังกรอง")
            return
        
        if self.vector_store is None:
            # สร้าง vector store ใหม่
            self.vector_store = FAISS.from_documents(unique_docs, self.embeddings)
            self.documents = unique_docs.copy()
        else:
            # เพิ่มเอกสารลงใน vector store ที่มีอยู่
            self.vector_store.add_documents(unique_docs)
            self.documents.extend(unique_docs)
        
        print(f"✅ เพิ่มเอกสารเสร็จสิ้น รวม {len(self.documents)} เอกสาร")
    
    def save_vector_store(self) -> None:
        """บันทึก vector store ลงดิสก์"""
        if self.vector_store is None:
            print("⚠️ ไม่มี vector store ที่จะบันทึก")
            return
        
        try:
            print("🔄 กำลังบันทึก vector store...")
            
            # บันทึก FAISS index
            self.vector_store.save_local(str(self.vector_store_path))
            
            # บันทึก metadata
            metadata = {
                "embedding_model": self.embedding_model_name,
                "total_documents": len(self.documents),
                "documents": self.documents
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"✅ บันทึก vector store เสร็จสิ้น ที่ {self.vector_store_path}")
            
        except Exception as e:
            print(f"❌ ไม่สามารถบันทึก vector store: {e}")
            raise
    
    def load_vector_store(self) -> bool:
        """โหลด vector store จากดิสก์"""
        try:
            if not self.faiss_index_path.exists() or not self.metadata_path.exists():
                print("⚠️ ไม่พบ vector store ที่บันทึกไว้")
                return False
            
            print("🔄 กำลังโหลด vector store...")
            
            # โหลด metadata
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # ตรวจสอบ embedding model
            if metadata['embedding_model'] != self.embedding_model_name:
                print(f"⚠️ Embedding model ไม่ตรงกัน: {metadata['embedding_model']} vs {self.embedding_model_name}")
                print("กำลังโหลดด้วย embedding model ใหม่...")
            
            # โหลด FAISS vector store
            self.vector_store = FAISS.load_local(
                str(self.vector_store_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            self.documents = metadata['documents']
            
            print(f"✅ โหลด vector store เสร็จสิ้น - {len(self.documents)} เอกสาร")
            return True
            
        except Exception as e:
            print(f"❌ ไม่สามารถโหลด vector store: {e}")
            return False
    
    def search_similar(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """ค้นหาเอกสารที่คล้ายกับคำถาม"""
        if self.vector_store is None:
            print("⚠️ ไม่มี vector store สำหรับการค้นหา")
            return []
        
        try:
            # ค้นหาด้วย similarity score
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # กรองตาม score threshold
            filtered_results = [(doc, score) for doc, score in results if score >= score_threshold]
            
            return filtered_results
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการค้นหา: {e}")
            return []
    
    def get_retriever(self, k: int = 5, search_type: str = "similarity"):
        """สร้าง retriever สำหรับใช้ใน RAG"""
        if self.vector_store is None:
            raise ValueError("ไม่มี vector store สำหรับการสร้าง retriever")
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """สถิติของ vector store"""
        if self.vector_store is None:
            return {"status": "empty"}
        
        return {
            "status": "ready",
            "total_documents": len(self.documents),
            "embedding_model": self.embedding_model_name,
            "vector_store_path": str(self.vector_store_path),
            "index_size": self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0
        }
    
    def clear_vector_store(self) -> None:
        """ล้าง vector store"""
        self.vector_store = None
        self.documents = []
        
        # ลบไฟล์ที่บันทึกไว้
        for file_path in [self.faiss_index_path, self.faiss_pkl_path, self.metadata_path]:
            if file_path.exists():
                file_path.unlink()
        
        print("✅ ล้าง vector store เสร็จสิ้น")


# โมเดล embedding ที่แนะนำสำหรับภาษาไทย
RECOMMENDED_THAI_MODELS = {
    "fast": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "balanced": "sentence-transformers/distiluse-base-multilingual-cased", 
    "best": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "thai_specific": "airesearch/wangchanberta-base-att-spm-uncased"  # ถ้ามี
}


# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # สร้าง vector store manager
    manager = ThaiVectorStoreManager(
        embedding_model=RECOMMENDED_THAI_MODELS["fast"],
        vector_store_path="./test_vectorstore"
    )
    
    # สร้างเอกสารทดสอบ
    test_docs = [
        Document(
            page_content="ประเทศไทยมีเมืองหลวงคือกรุงเทพมหานคร",
            metadata={"source": "test1.txt", "topic": "geography"}
        ),
        Document(
            page_content="อาหารไทยมีรสชาติหวาน เปรี้ยว เค็ม เผ็ด",
            metadata={"source": "test2.txt", "topic": "food"}
        ),
        Document(
            page_content="พระบรมมหาราชวังตั้งอยู่ในกรุงเทพมหานคร",
            metadata={"source": "test3.txt", "topic": "landmarks"}
        )
    ]
    
    # เพิ่มเอกสาร
    manager.add_documents(test_docs)
    
    # บันทึก
    manager.save_vector_store()
    
    # ทดสอบการค้นหา
    results = manager.search_similar("เมืองหลวงของไทย", k=2)
    print("\n🔍 ผลการค้นหา:")
    for doc, score in results:
        print(f"Score: {score:.4f}")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
    
    # แสดงสถิติ
    stats = manager.get_stats()
    print(f"\n📊 สถิติ: {stats}")
    
    # ล้างข้อมูลทดสอบ
    manager.clear_vector_store()
    import shutil
    if Path("./test_vectorstore").exists():
        shutil.rmtree("./test_vectorstore")