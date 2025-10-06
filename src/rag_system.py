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
                    "num_predict": self.max_tokens,  # ใช้ค่าที่ตั้งจาก UI โดยตรง
                    "top_p": 0.9,
                    "top_k": 40,  # จำกัด choices
                    "repeat_penalty": 1.1,  # ลดค่า penalty - อนุญาตให้ซ้ำรูปแบบเดียวกันได้ (สำหรับรายการ)
                    "stop": [
                        "</s>", 
                        "<|end|>", 
                        "Human:", 
                        "\n\nคำถาม:",  # หยุดถ้าเจอคำถามใหม่
                        "\n\n---\n"  # หยุดถ้าเจอ separator
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
        template = """คุณเป็นผู้ช่วยปัญญาประดิษฐ์ที่เชี่ยวชาญในการตอบคำถามจากหลักสูตรการศึกษาภาษาไทย

ข้อมูลที่เกี่ยวข้องจากเอกสาร:
{context}

ประวัติการสนทนา:
{chat_history}

คำถาม: {question}

กฎการตอบ (ปฏิบัติอย่างเคร่งครัด):

**สำหรับคำถามเกี่ยวกับรหัสวิชา (เช่น "05506012 คือวิชาอะไร"):**
- ✅ ค้นหารหัสวิชาในข้อมูลที่ให้มา
- ✅ ตอบด้วย: "รหัสวิชา [รหัส] คือ [ชื่อวิชาภาษาไทย] ([ชื่อภาษาอังกฤษถ้ามี]) จำนวน [หน่วยกิต] หน่วยกิต"
- ❌ ห้ามตอบด้วยเลขอะไรก็ตามที่ไม่ใช่ชื่อวิชา

**สำหรับคำถามเกี่ยวกับรายวิชาในภาค/ปี (เช่น "ปีที่ 1 ภาคการศึกษาที่ 1 มีอะไรบ้าง"):**
- ✅ **คำสั่งสำคัญ: ต้องแสดงรายการวิชาครบทั้งหมด 100%**
- ✅ ตอบเป็นรายการ bullet point ทุกวิชา:
  • รหัสวิชา ชื่อวิชา (หน่วยกิต)
  • รหัสวิชา ชื่อวิชา (หน่วยกิต)
  • (แสดงต่อไปจนครบทุกวิชา)
- ✅ บอกจำนวนรวมหน่วยกิตท้ายสุด
- ❌ **ห้ามตอบแค่บางวิชา** - ต้องระบุให้ครบทุกวิชาที่มีในข้อมูล
- ❌ **ห้ามใช้ "..." หรือ "และอื่นๆ"** - ต้องเขียนครบ

**ตัวอย่างคำตอบที่ถูกต้อง:**
"ในปีที่ 1 ภาคการศึกษาที่ 2 มีรายวิชาดังนี้:

• 05506233 แคลคูลัสสำหรับวิทยาการคอมพิวเตอร์ (3 หน่วยกิต)
• 05506001 คณิตศาสตร์ดิสครีต (3 หน่วยกิต)
• 05506004 การเขียนโปรแกรมเชิงออบเจกต์ (3 หน่วยกิต)
• 05506008 โครงสร้างและสถาปัตยกรรมคอมพิวเตอร์ (3 หน่วยกิต)
• 05506011 ปฏิสัมพันธ์ระหว่างมนุษย์และคอมพิวเตอร์ (3 หน่วยกิต)
• 90644008 ภาษาอังกฤษพื้นฐาน 2 (3 หน่วยกิต)
• 90xxxxxx วิชาเลือกหมวดวิชาศึกษาทั่วไป (3 หน่วยกิต)

รวม 21 หน่วยกิต"

**สำหรับคำถามทั่วไป:**
- ✅ ใช้เฉพาะข้อมูลจากเอกสาร
- ✅ ตอบสั้น กระชับ 3-5 ประโยค
- ❌ ห้ามแต่งข้อมูล ห้ามซ้ำคำ
- ❌ ถ้าไม่พบข้อมูล → ตอบว่า "ไม่พบข้อมูลในเอกสาร"

คำตอบ (ตอบเป็นภาษาไทยเท่านั้น):"""
        
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
            
            # ค้นหาเอกสารที่เกี่ยวข้อง - เพิ่ม k เป็น 10 สำหรับคำถามเกี่ยวกับรายวิชา
            relevant_docs = self.vector_store_manager.vector_store.similarity_search(
                question, 
                k=10  # เพิ่มเป็น 10 เพื่อดูข้อมูลครบถ้วนสำหรับตาราง
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
            
            # 🔍 กรองตาม metadata และ content ถ้าพบคำสำคัญในคำถาม
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
                    pattern2 = target_code[:4] + r'\s+' + target_code[4:]  # มีช่องว่าง
                    pattern3 = target_code[:4] + r'\s*-\s*' + target_code[4:]  # มีขีด
                    
                    if (target_code in content or 
                        re.search(pattern2, content) or
                        re.search(pattern3, content) or
                        target_code in metadata.get("course_codes", "")):
                        # พบรหัสวิชา! priority สูงมาก
                        filtered_docs.insert(0, doc)
                        print(f"✅ พบรหัสวิชา {target_code} ใน chunk: {content[:100]}")
                        continue
                
                # ถ้าถามปี/ภาค → ค้นหาทั้ง metadata และ content โดยตรง
                elif question_year or question_semester:
                    match_score = 0  # คะแนนความตรง
                    
                    # ถ้าถามทั้งปีและภาค
                    if question_year and question_semester:
                        year_str = question_year.group(1)
                        semester_str = question_semester.group(1)
                        
                        # ตรวจสอบ metadata ก่อน (แม่นยำที่สุด)
                        meta_year_match = metadata.get("year") == year_str
                        meta_semester_match = metadata.get("semester") == semester_str
                        
                        if meta_year_match and meta_semester_match:
                            match_score = 100  # สมบูรณ์แบบ!
                        elif meta_year_match or meta_semester_match:
                            match_score = 50  # ตรงบางส่วน
                        
                        # ถ้า metadata ไม่ตรง → ค้นหาใน content โดยตรง
                        else:
                            # ค้นหาว่ามีคำว่า "ปีที่ X ภาคการศึกษาที่ Y" ใน content หรือไม่
                            year_in_content = re.search(rf'ปีที่\s*{year_str}', content)
                            semester_in_content = re.search(rf'ภาคการศึกษาที่\s*{semester_str}', content)
                            
                            if year_in_content and semester_in_content:
                                match_score = 80  # เจอใน content ทั้งคู่
                            elif year_in_content or semester_in_content:
                                match_score = 40  # เจอใน content บางส่วน
                        
                        # ถ้ามีคะแนน → เอามา
                        if match_score > 0:
                            # Priority สูงถ้าคะแนนสูง
                            if match_score >= 80:
                                filtered_docs.insert(0, doc)
                                print(f"✅ [คะแนน {match_score}] พบ ปี{year_str} ภาค{semester_str}: {content[:100]}")
                            else:
                                filtered_docs.append(doc)
                                print(f"⚠️ [คะแนน {match_score}] พบบางส่วน ปี{year_str} ภาค{semester_str}")
                            continue
                    
                    # ถ้าถามเฉพาะปี หรือเฉพาะภาค
                    elif question_year:
                        year_str = question_year.group(1)
                        if (metadata.get("year") == year_str or 
                            re.search(rf'ปีที่\s*{year_str}', content)):
                            filtered_docs.append(doc)
                            print(f"✅ พบ ปี{year_str}")
                            continue
                    elif question_semester:
                        semester_str = question_semester.group(1)
                        if (metadata.get("semester") == semester_str or 
                            re.search(rf'ภาคการศึกษาที่\s*{semester_str}', content)):
                            filtered_docs.append(doc)
                            print(f"✅ พบ ภาค{semester_str}")
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
            num_chunks = 5 if (question_year or question_semester or "อะไรบ้าง" in question or "ทั้งหมด" in question) else 3
            relevant_docs = unique_docs[:num_chunks]
            
            # 🔥 สร้าง context โดยใช้เนื้อหาเต็ม (ไม่ตัด) สำหรับตาราง
            # gemma2:2b รองรับ ~2048 tokens ≈ 1500-1800 chars
            context_parts = []
            total_chars = 0
            max_context_chars = 2000  # เพิ่มขึ้นสำหรับตารางข้อมูล
            
            for doc in relevant_docs:
                # คำนวณว่ายังใส่ได้อีกเท่าไหร่
                remaining = max_context_chars - total_chars
                
                if remaining <= 0:
                    break
                
                # ใช้เนื้อหาเต็มถ้ายังไม่เกิน limit
                content = doc.page_content[:remaining]
                
                # เพิ่ม metadata hint ถ้ามี
                metadata_hint = ""
                if doc.metadata.get("year") and doc.metadata.get("semester"):
                    metadata_hint = f"[ปีที่ {doc.metadata['year']} ภาคการศึกษาที่ {doc.metadata['semester']}]\n"
                
                context_parts.append(metadata_hint + content)
                total_chars += len(metadata_hint) + len(content)
            
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
    "gemma2":"latest",
    "large": "llama3.1:70b"         # ใหญ่, คุณภาพสูง (80GB)
    
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