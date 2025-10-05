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
                        "content": "คุณเป็นผู้ช่วยที่ตอบคำถามเป็นภาษาไทยเท่านั้น ตอบสั้น กระชับ ตรงประเด็น ห้ามซ้ำคำ ห้ามพูดเยิ่นเย้อ"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": min(self.max_tokens, 512),  # จำกัดความยาวคำตอบ
                    "top_p": 0.9,
                    "top_k": 40,  # จำกัด choices
                    "repeat_penalty": 1.5,  # เพิ่มค่า penalty การซ้ำ
                    "stop": [
                        "</s>", 
                        "<|end|>", 
                        "Human:", 
                        "คำถาม:",
                        "\n\nคำถาม",  # หยุดถ้าเจอคำถามใหม่
                        "ขออภัย ไม่พบข้อมูล",  # หยุดหลังจากประโยคปิด
                        "\n\n\n"  # หยุดถ้าเจอบรรทัดว่าง 3 บรรทัด
                    ]
                }
            )
            
            result = response['message']['content']
            
            # ตัดส่วนที่ซ้ำออก (ถ้ามีการซ้ำมากกว่า 50 ตัวอักษร)
            lines = result.split('\n')
            seen = set()
            unique_lines = []
            for line in lines:
                line_stripped = line.strip()
                if len(line_stripped) > 20:  # เฉพาะบรรทัดที่มีเนื้อหา
                    if line_stripped not in seen:
                        unique_lines.append(line)
                        seen.add(line_stripped)
                else:
                    unique_lines.append(line)
            
            result = '\n'.join(unique_lines)
            
            # ตรวจสอบคำตอบ
            if not result or len(result.strip()) < 5:
                return "ขออภัย ไม่สามารถสร้างคำตอบที่เหมาะสมได้ กรุณาลองถามใหม่อีกครั้ง"
            
            # ตัดให้สั้นถ้ายาวเกินไป (max 1000 chars)
            if len(result) > 1000:
                result = result[:997] + "..."
            
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
1. ตอบเป็นภาษาไทยเท่านั้น
2. ใช้เฉพาะข้อมูลจากเอกสารที่ให้มา - ห้ามแต่งเรื่อง
3. **ถ้าถามรหัสวิชา (เช่น 05506231)** → ต้องค้นหารหัสนั้นให้เจอก่อนตอบ ห้ามตอบมั่ว
4. **ถ้าถามปีที่ X ภาคการศึกษาที่ Y** → ต้องหาส่วนที่ตรงปี/ภาคนั้นเท่านั้น
5. **ถ้าถาม "มีอะไรบ้าง" หรือ "ทั้งหมด"** → ระบุให้ครบทุกรายการที่พบ
6. ถ้าไม่พบข้อมูล → ตอบว่า "ไม่พบข้อมูล..." ห้ามตอบเรื่องอื่น
7. ตอบสั้น กระชับ ตรงประเด็น 3-5 ประโยค (ยกเว้นถ้าถามรายการให้ระบุครบ)
8. ห้ามทำซ้ำข้อความ ห้ามต่อท้ายด้วยข้อมูลที่ไม่เกี่ยวข้อง
9. ถ้ามีรายการ → แสดงเป็น bullet points
10. หยุดทันทีเมื่อตอบครบประเด็น

คำตอบ:"""
        
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
            
            # ค้นหาเอกสารที่เกี่ยวข้อง - เพิ่ม k เป็น 5 เพื่อหาข้อมูลครบถ้วน
            relevant_docs = self.vector_store_manager.vector_store.similarity_search(
                question, 
                k=5  # เพิ่มจาก 3 เป็น 5 เพื่อดูข้อมูลมากขึ้น
            )
            
            # ตรวจสอบคุณภาพของ context
            if not relevant_docs:
                return {
                    "answer": "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่อัพโหลด",
                    "sources": [],
                    "question": question,
                    "timestamp": datetime.now().isoformat()
                }
            
            # 🎯 กรองเอกสารที่เกี่ยวข้องตามคำถาม (Metadata Filtering)
            import re
            
            # ตรวจจับรหัสวิชาในคำถาม (รองรับทั้ง 05506231 และ 0550 6231)
            question_course_code = re.search(r'\b(\d{4})\s*(\d{4})\b|\b(\d{8})\b', question)
            
            # ตรวจจับชั้นปี/ภาคการศึกษาในคำถาม
            question_year = re.search(r'ปีที่\s*(\d+)', question)
            question_semester = re.search(r'ภาคการศึกษาที่\s*(\d+)', question)
            
            filtered_docs = []
            
            # กรองตาม metadata ถ้าพบคำสำคัญในคำถาม
            for doc in relevant_docs:
                metadata = doc.metadata
                content = doc.page_content
                
                # ถ้าถามรหัสวิชาเฉพาะ → ต้องมีรหัสนั้นใน content (รองรับทั้งติดกันและมีช่องว่าง)
                if question_course_code:
                    # ดึงรหัสจาก regex (group 1+2 หรือ group 3)
                    if question_course_code.group(3):  # 8 หลักติด
                        target_code = question_course_code.group(3)
                    else:  # 4+4 หลักแยก
                        target_code = question_course_code.group(1) + question_course_code.group(2)
                    
                    # ค้นหาในหลายรูปแบบ: "05506231" หรือ "0550 6231" หรือ "0550-6231"
                    pattern1 = target_code  # ติดกัน
                    pattern2 = target_code[:4] + r'\s*' + target_code[4:]  # มีช่องว่าง
                    
                    if (target_code in content or 
                        re.search(pattern2, content) or
                        target_code in metadata.get("course_codes", "")):
                        filtered_docs.append(doc)
                        continue
                
                # ถ้าถามปี/ภาค → ต้องตรงกับ metadata (เข้มงวด!)
                elif question_year or question_semester:
                    # 🔥 ต้องตรงทั้งปีและภาค (ถ้ามีทั้งคู่)
                    if question_year and question_semester:
                        # ตรงทั้งคู่ถึงจะผ่าน
                        if (metadata.get("year") == question_year.group(1) and 
                            metadata.get("semester") == question_semester.group(1)):
                            # Priority สูงถ้ามี tag curriculum table
                            if metadata.get("is_curriculum_table") == "yes":
                                filtered_docs.insert(0, doc)  # ใส่หน้าสุด!
                            else:
                                filtered_docs.append(doc)
                            continue
                    # ถ้ามีแค่ปี หรือแค่ภาค
                    elif question_year and metadata.get("year") == question_year.group(1):
                        filtered_docs.append(doc)
                        continue
                    elif question_semester and metadata.get("semester") == question_semester.group(1):
                        filtered_docs.append(doc)
                        continue
                
                # ถ้าไม่มีเงื่อนไขพิเศษ → เอาทุก doc
                else:
                    filtered_docs.append(doc)
            
            # ถ้ากรองแล้วไม่เหลือเลย → ใช้ docs เดิม
            if not filtered_docs:
                filtered_docs = relevant_docs
                print(f"⚠️ ไม่พบเอกสารที่ตรงเงื่อนไข ใช้ทั้งหมด {len(relevant_docs)} chunks")
            else:
                print(f"✅ กรองแล้วเหลือ {len(filtered_docs)} chunks ที่ตรงเงื่อนไข")
            
            # Deduplication
            unique_docs = []
            seen_contents = set()
            for doc in filtered_docs:
                # ใช้ 100 ตัวอักษรแรกเป็น signature
                signature = doc.page_content[:100].strip()
                if signature not in seen_contents:
                    unique_docs.append(doc)
                    seen_contents.add(signature)
            
            # ใช้ top 2 chunks (ลดเพื่อป้องกัน Ollama ล่ม)
            relevant_docs = unique_docs[:2]
            
            # 🔥 สร้าง context โดยใช้เนื้อหาเต็ม (ไม่ตัด) สำหรับตาราง
            # gemma2:2b รองรับ ~2048 tokens ≈ 1500-1800 chars
            context_parts = []
            total_chars = 0
            max_context_chars = 1500  # ปลอดภัยสำหรับ gemma2:2b
            
            for doc in relevant_docs:
                # คำนวณว่ายังใส่ได้อีกเท่าไหร่
                remaining = max_context_chars - total_chars
                
                if remaining <= 0:
                    break
                
                # ใช้เนื้อหาเต็มถ้ายังไม่เกิน limit
                content = doc.page_content[:remaining]
                context_parts.append(content)
                total_chars += len(content)
            
            context = "\n\n---\n\n".join(context_parts)
            
            print(f"� ใช้ context รวม {total_chars} chars จาก {len(context_parts)} chunks")
            
            # Debug: แสดง context
            print(f"📄 Context ที่พบ ({len(context)} chars): {context[:200]}...")
            
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
    "chat": "mistral:7b",              # การสนทนา
    "scb10x": "llama3.2-typhoon2-1b-instruct:latest"
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