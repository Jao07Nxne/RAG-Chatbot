"""
RAG System for Thai Chatbot with Chat History Support
ระบบ RAG ที่รองรับการสนทนาแบบต่อเนื่องและใช้ Local LLM
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain.schema import Document, BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

import ollama
import requests


class OllamaLLM(LLM):
    """Custom LLM class สำหรับ Ollama"""
    
    model_name: str = "llama3.1"
    temperature: float = 0.1
    max_tokens: int = 2048
    
    def __init__(self, model_name: str = "llama3.1", temperature: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.temperature = temperature
        
        # ตรวจสอบการเชื่อมต่อ Ollama
        self._check_ollama_connection()
    
    def _check_ollama_connection(self) -> bool:
        """ตรวจสอบการเชื่อมต่อกับ Ollama"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                
                if self.model_name not in available_models:
                    print(f"⚠️ โมเดล {self.model_name} ไม่พบใน Ollama")
                    print(f"📋 โมเดลที่มีอยู่: {available_models}")
                    if available_models:
                        print(f"🔄 จะใช้โมเดล {available_models[0]} แทน")
                        self.model_name = available_models[0]
                
                print(f"✅ เชื่อมต่อ Ollama สำเร็จ - ใช้โมเดล: {self.model_name}")
                return True
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ ไม่สามารถเชื่อมต่อ Ollama: {e}")
            print("💡 กรุณาตรวจสอบว่า Ollama ทำงานอยู่ (ollama serve)")
            return False
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, 
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """เรียกใช้ Ollama LLM"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "คุณเป็นผู้ช่วยที่ตอบคำถามเป็นภาษาไทยเท่านั้น และใช้เฉพาะข้อมูลที่ให้มา ห้ามใช้ความรู้ทั่วไป"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_p": 0.9,  # ควบคุมความหลากหลาย
                    "repeat_penalty": 1.1,  # ลดการซ้ำ
                    "stop": ["</s>", "<|end|>", "Human:", "คำถาม:"]  # หยุดที่จุดที่เหมาะสม
                }
            )
            
            result = response['message']['content']
            
            # ตรวจสอบคำตอบ
            if not result or len(result.strip()) < 5:
                return "ขออภัย ไม่สามารถสร้างคำตอบที่เหมาะสมได้ กรุณาลองถามใหม่อีกครั้ง"
            
            return result.strip()
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการเรียกใช้ LLM: {e}")
            return f"ขออภัย เกิดข้อผิดพลาดในการประมวลผล: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "ollama"


class ThaiRAGSystem:
    """ระบบ RAG สำหรับภาษาไทยพร้อม Chat History"""
    
    def __init__(self, 
                 vector_store_manager,
                 llm_model: str = "llama3.1",
                 temperature: float = 0.1,
                 max_tokens: int = 2048):
        """
        Args:
            vector_store_manager: ThaiVectorStoreManager instance
            llm_model: ชื่อโมเดล Ollama ที่จะใช้
            temperature: ความสร้างสรรค์ของการตอบ (0-1)
            max_tokens: จำนวนคำสูงสุดในการตอบ
        """
        self.vector_store_manager = vector_store_manager
        
        # สร้าง LLM
        self.llm = OllamaLLM(
            model_name=llm_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # สร้าง Memory สำหรับเก็บประวัติการสนทนา
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # สร้าง Prompt Template
        self.prompt_template = self._create_prompt_template()
        
        # สร้าง RAG Chain
        self.rag_chain = None
        self._setup_rag_chain()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """สร้าง Prompt Template สำหรับภาษาไทย"""
        template = """คุณเป็นผู้ช่วยปัญญาประดิษฐ์ที่เชี่ยวชาญในการตอบคำถามจากเอกสารภาษาไทยอย่างแม่นยำ

ข้อมูลที่เกี่ยวข้องจากเอกสาร:
{context}

ประวัติการสนทนา:
{chat_history}

คำถาม: {question}

กฎการตอบ (ต้องปฏิบัติอย่างเคร่งครัด):
1. ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอื่น
2. ใช้เฉพาะข้อมูลจากเอกสารที่ให้มาเป็นหลักในการตอบ
3. หากไม่มีข้อมูลที่เกี่ยวข้องในเอกสาร ให้ตอบว่า "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่อัพโหลด"
4. ห้ามสร้างข้อมูลเพิ่มเติมหรือตอบจากความรู้ทั่วไป
5. ตอบให้ชัดเจน เฉพาะเจาะจง และตรงประเด็น
6. หากข้อมูลไม่เพียงพอ ให้ระบุว่าต้องการข้อมูลเพิ่มเติม
7. ใช้ข้อมูลจากเอกสารเพื่อสนับสนุนคำตอบเท่านั้น

คำตอบ (ภาษาไทยเท่านั้น):"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
    
    def _setup_rag_chain(self):
        """ตั้งค่า RAG Chain"""
        if self.vector_store_manager.vector_store is None:
            print("⚠️ ไม่มี vector store สำหรับการสร้าง RAG chain")
            return
        
        # สร้าง retriever
        retriever = self.vector_store_manager.get_retriever(k=5)
        
        # สร้าง ConversationalRetrievalChain
        self.rag_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.prompt_template},
            verbose=True
        )
        
        print("✅ สร้าง RAG Chain สำเร็จ")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        ถามคำถามและได้รับคำตอบพร้อมแหล่งข้อมูล
        
        Args:
            question: คำถามภาษาไทย
            
        Returns:
            Dict containing answer, sources, and metadata
        """
        if self.rag_chain is None:
            return {
                "answer": "ระบบยังไม่พร้อมใช้งาน กรุณาอัพโหลดเอกสารก่อน",
                "sources": [],
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            print(f"🤔 กำลังประมวลผลคำถาม: {question}")
            
            # ตรวจสอบว่ามี vector store หรือไม่
            if not self.vector_store_manager.vector_store:
                return {
                    "answer": "ขออภัย ไม่มีเอกสารในระบบ กรุณาอัพโหลดเอกสารก่อน",
                    "sources": [],
                    "question": question,
                    "timestamp": datetime.now().isoformat()
                }
            
            # ค้นหาเอกสารที่เกี่ยวข้อง
            relevant_docs = self.vector_store_manager.vector_store.similarity_search(
                question, 
                k=4  # ดึง 4 documents ที่เกี่ยวข้องมากที่สุด
            )
            
            # ตรวจสอบคุณภาพของ context
            if not relevant_docs:
                return {
                    "answer": "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่อัพโหลด",
                    "sources": [],
                    "question": question,
                    "timestamp": datetime.now().isoformat()
                }
            
            # สร้าง context จากเอกสารที่เกี่ยวข้อง
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Debug: แสดง context
            print(f"📄 Context ที่พบ: {context[:200]}...")
            
            # เรียกใช้ RAG Chain ด้วย ConversationalRetrievalChain
            result = self.rag_chain({
                "question": question,
                "chat_history": self.memory.chat_memory.messages
            })
            
            # จัดรูปแบบแหล่งข้อมูล
            sources = []
            seen_sources = set()
            for doc in relevant_docs:
                source_info = {
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata,
                    "filename": doc.metadata.get("filename", "ไม่ทราบ"),
                    "chunk_index": doc.metadata.get("chunk_index", 0)
                }
                
                # หลีกเลี่ยงการซ้ำ
                source_key = f"{source_info['filename']}_{source_info['chunk_index']}"
                if source_key not in seen_sources:
                    sources.append(source_info)
                    seen_sources.add(source_key)
            
            # ตรวจสอบคำตอบ
            answer = result.get("answer", result.get("text", "ไม่สามารถสร้างคำตอบได้"))
            
            # ถ้าคำตอบสั้นเกินไปหรือไม่มีเนื้อหา
            if len(answer.strip()) < 10:
                answer = "ขออภัย ไม่สามารถสร้างคำตอบที่มีความหมายจากข้อมูลในเอกสารได้ กรุณาลองถามคำถามใหม่หรือให้รายละเอียดเพิ่มเติม"
            
            print(f"✅ คำตอบ: {answer[:100]}...")
            
            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "chat_history_length": len(self.memory.chat_memory.messages)
            }
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการตอบคำถาม: {e}")
            return {
                "answer": f"ขออภัย เกิดข้อผิดพลาด: {str(e)}",
                "sources": [],
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """ดึงประวัติการสนทนา"""
        history = []
        messages = self.memory.chat_memory.messages
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]
                
                history.append({
                    "question": human_msg.content,
                    "answer": ai_msg.content,
                    "timestamp": datetime.now().isoformat()  # อาจจะเก็บ timestamp จริงใน metadata
                })
        
        return history
    
    def clear_chat_history(self):
        """ล้างประวัติการสนทนา"""
        self.memory.clear()
        print("✅ ล้างประวัติการสนทนาเสร็จสิ้น")
    
    def update_vector_store(self, vector_store_manager):
        """อัพเดท vector store และ RAG chain"""
        self.vector_store_manager = vector_store_manager
        self._setup_rag_chain()
        print("✅ อัพเดท vector store เสร็จสิ้น")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """สถิติของระบบ"""
        vector_stats = self.vector_store_manager.get_stats()
        
        return {
            "vector_store": vector_stats,
            "llm_model": self.llm.model_name,
            "chat_history_length": len(self.memory.chat_memory.messages) // 2,
            "rag_chain_ready": self.rag_chain is not None,
            "timestamp": datetime.now().isoformat()
        }


# โมเดล LLM ที่แนะนำสำหรับภาษาไทย
RECOMMENDED_THAI_LLM_MODELS = {
    "small_fast": "gemma2:2b",        # เล็ก, เร็ว, RAM น้อย (4GB) - แนะนำสำหรับเริ่มต้น
    "balanced": "llama3.1:8b",        # สมดุล, RAM ปานกลาง (8GB)
    "large": "llama3.1:70b",          # ใหญ่, คุณภาพสูง (80GB)
    "coding": "codellama:7b",         # สำหรับโค้ด
    "chat": "mistral:7b"              # การสนทนา
}


# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # นำเข้าโมดูลที่จำเป็น
    from vector_store import ThaiVectorStoreManager
    from document_processor import ThaiDocumentProcessor
    from langchain.schema import Document
    
    # สร้างเอกสารทดสอบ
    test_docs = [
        Document(
            page_content="ประเทศไทยมีเมืองหลวงคือกรุงเทพมหานคร มีประชากรประมาณ 70 ล้านคน",
            metadata={"source": "thailand_info.txt", "topic": "general"}
        ),
        Document(
            page_content="อาหารไทยที่มีชื่อเสียงได้แก่ ต้มยำกุ้ง ผัดไทย และมะม่วงข้าวเหนียว",
            metadata={"source": "thai_food.txt", "topic": "food"}
        )
    ]
    
    # สร้าง vector store
    vector_manager = ThaiVectorStoreManager()
    vector_manager.add_documents(test_docs)
    
    # สร้าง RAG system
    rag_system = ThaiRAGSystem(
        vector_store_manager=vector_manager,
        llm_model="llama3.1:8b",  # ใช้โมเดลที่เหมาะสม
        temperature=0.1
    )
    
    # ทดสอบการถามคำถาม
    questions = [
        "เมืองหลวงของไทยคืออะไร?",
        "ประเทศไทยมีประชากรเท่าไหร่?",
        "อาหารไทยที่มีชื่อเสียงมีอะไรบ้าง?",
        "จากที่พูดไปก่อนหน้านี้ เมืองหลวงมีประชากรเท่าไหร่?"  # ทดสอบ chat history
    ]
    
    print("🤖 เริ่มทดสอบ RAG System")
    print("=" * 50)
    
    for question in questions:
        result = rag_system.ask_question(question)
        
        print(f"\n❓ คำถาม: {result['question']}")
        print(f"🤖 คำตอบ: {result['answer']}")
        print(f"📚 จำนวนแหล่งข้อมูล: {len(result['sources'])}")
        
        if result['sources']:
            print("📖 แหล่งข้อมูล:")
            for i, source in enumerate(result['sources'], 1):
                print(f"   {i}. {source['filename']} (chunk {source['chunk_index']})")
        
        print("-" * 30)
    
    # แสดงประวัติการสนทนา
    print("\n💬 ประวัติการสนทนา:")
    history = rag_system.get_chat_history()
    for i, chat in enumerate(history, 1):
        print(f"{i}. Q: {chat['question']}")
        print(f"   A: {chat['answer'][:100]}...")
    
    # แสดงสถิติ
    stats = rag_system.get_system_stats()
    print(f"\n📊 สถิติระบบ: {stats}")