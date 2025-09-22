"""
Document Processor for Thai RAG Chatbot
รองรับการอ่านและประมวลผลเอกสารภาษาไทย
"""

import os
import re
from typing import List, Dict, Any
from pathlib import Path
import chardet

# Document reading libraries
import PyPDF2
from docx import Document
from pptx import Presentation

# Thai language processing
import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_words

# LangChain text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangChainDocument


class ThaiDocumentProcessor:
    """คลาสสำหรับประมวลผลเอกสารภาษาไทย"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: ขนาดของแต่ละ chunk (characters)
            chunk_overlap: ส่วนที่ซ้อนทับระหว่าง chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # สร้าง text splitter ที่เหมาะกับภาษาไทย
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ".",     # Sentence endings
                "!",     # Exclamation
                "?",     # Question
                ";",     # Semicolon
                ",",     # Comma
                " ",     # Spaces
                ""       # Characters
            ]
        )
    
    def detect_encoding(self, file_path: str) -> str:
        """ตัวอย่างการตรวจสอบ encoding ของไฟล์"""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'
    
    def read_text_file(self, file_path: str) -> str:
        """อ่านไฟล์ text ธรรมดา"""
        encoding = self.detect_encoding(file_path)
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            # ลองใช้ encoding อื่น
            for enc in ['utf-8', 'cp874', 'iso-8859-11']:
                try:
                    with open(file_path, 'r', encoding=enc) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"ไม่สามารถอ่านไฟล์ {file_path} ได้")
    
    def read_pdf_file(self, file_path: str) -> str:
        """อ่านไฟล์ PDF"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise ValueError(f"ไม่สามารถอ่านไฟล์ PDF {file_path}: {str(e)}")
    
    def read_docx_file(self, file_path: str) -> str:
        """อ่านไฟล์ Word (.docx)"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise ValueError(f"ไม่สามารถอ่านไฟล์ Word {file_path}: {str(e)}")
    
    def read_pptx_file(self, file_path: str) -> str:
        """อ่านไฟล์ PowerPoint (.pptx)"""
        try:
            prs = Presentation(file_path)
            text = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            return text
        except Exception as e:
            raise ValueError(f"ไม่สามารถอ่านไฟล์ PowerPoint {file_path}: {str(e)}")
    
    def clean_thai_text(self, text: str) -> str:
        """ทำความสะอาดข้อความภาษาไทย"""
        # ลบ whitespace ที่ไม่จำเป็น
        text = re.sub(r'\s+', ' ', text)
        
        # ลบอักขระพิเศษที่ไม่ต้องการ
        text = re.sub(r'[^\u0E00-\u0E7F\w\s\.\,\!\?\;\:\-\(\)\"\'\/]', '', text)
        
        # ลบบรรทัดว่าง
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def preprocess_text(self, text: str) -> str:
        """ประมวลผลข้อความก่อนการแบ่ง chunks"""
        # ทำความสะอาด
        text = self.clean_thai_text(text)
        
        # ตัด word ภาษาไทยเพื่อปรับปรุงการแบ่ง chunk
        # (อาจจะใช้หรือไม่ใช้ก็ได้ ขึ้นอยู่กับความต้องการ)
        
        return text
    
    def read_document(self, file_path: str) -> str:
        """อ่านเอกสารตามประเภทไฟล์"""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.txt':
            return self.read_text_file(str(file_path))
        elif file_extension == '.pdf':
            return self.read_pdf_file(str(file_path))
        elif file_extension in ['.docx', '.doc']:
            return self.read_docx_file(str(file_path))
        elif file_extension in ['.pptx', '.ppt']:
            return self.read_pptx_file(str(file_path))
        else:
            # ลองอ่านเป็น text file
            try:
                return self.read_text_file(str(file_path))
            except:
                raise ValueError(f"ไม่รองรับไฟล์ประเภท {file_extension}")
    
    def process_document(self, file_path: str, metadata: Dict[str, Any] = None) -> List[LangChainDocument]:
        """
        ประมวลผลเอกสารและแบ่งเป็น chunks
        
        Args:
            file_path: เส้นทางไฟล์
            metadata: ข้อมูลเพิ่มเติมของเอกสาร
            
        Returns:
            List[LangChainDocument]: รายการ chunks พร้อม metadata
        """
        # อ่านเอกสาร
        text = self.read_document(file_path)
        
        # ประมวลผลข้อความ
        processed_text = self.preprocess_text(text)
        
        if not processed_text.strip():
            raise ValueError(f"ไม่มีข้อความในไฟล์ {file_path}")
        
        # แบ่งเป็น chunks
        chunks = self.text_splitter.split_text(processed_text)
        
        # สร้าง LangChain Documents
        documents = []
        file_name = Path(file_path).name
        
        base_metadata = {
            "source": file_path,
            "filename": file_name,
            "total_chunks": len(chunks)
        }
        
        if metadata:
            base_metadata.update(metadata)
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_size": len(chunk)
            })
            
            documents.append(LangChainDocument(
                page_content=chunk,
                metadata=chunk_metadata
            ))
        
        return documents
    
    def process_multiple_documents(self, file_paths: List[str]) -> List[LangChainDocument]:
        """ประมวลผลหลายเอกสารพร้อมกัน"""
        all_documents = []
        
        for file_path in file_paths:
            try:
                documents = self.process_document(file_path)
                all_documents.extend(documents)
                print(f"✅ ประมวลผลไฟล์ {Path(file_path).name} เสร็จสิ้น - {len(documents)} chunks")
            except Exception as e:
                print(f"❌ ไม่สามารถประมวลผลไฟล์ {Path(file_path).name}: {str(e)}")
        
        return all_documents
    
    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """สถิติของข้อความ"""
        words = word_tokenize(text, engine='newmm')
        thai_words_count = len([w for w in words if re.match(r'[\u0E00-\u0E7F]+', w)])
        
        return {
            "total_characters": len(text),
            "total_words": len(words),
            "thai_words": thai_words_count,
            "lines": len(text.split('\n')),
            "estimated_chunks": (len(text) // self.chunk_size) + 1
        }


# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    processor = ThaiDocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    # ทดสอบอ่านไฟล์
    try:
        # สร้างไฟล์ทดสอบ
        test_text = """
        ตัวอย่างเอกสารภาษาไทยสำหรับทดสอบ RAG Chatbot
        
        ประเทศไทยมีชื่อเป็นทางการว่า ราชอาณาจักรไทย เป็นประเทศในเอเชียตะวันออกเฉียงใต้ 
        มีพรมแดนติดกับพม่า ลาว กัมพูชา และมาเลเซีย มีเนื้อที่ประมาณ 513,120 ตารางกิโลเมตร
        
        กรุงเทพมหานครเป็นเมืองหลวงและเมืองที่ใหญ่ที่สุดของประเทศไทย 
        ประชากรประมาณ 70 ล้านคน ใช้ภาษาไทยเป็นภาษาราชการ
        
        ระบบการเมืองเป็นราชาธิปไตยภายใต้รัธรรมนูญ โดยมีพระมหากษัตริย์เป็นประมุข
        """
        
        with open("test_thai.txt", "w", encoding="utf-8") as f:
            f.write(test_text)
        
        # ทดสอบการประมวลผล
        docs = processor.process_document("test_thai.txt")
        
        print(f"จำนวน chunks: {len(docs)}")
        for i, doc in enumerate(docs):
            print(f"\nChunk {i+1}:")
            print(f"Content: {doc.page_content[:100]}...")
            print(f"Metadata: {doc.metadata}")
        
        # ลบไฟล์ทดสอบ
        os.remove("test_thai.txt")
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")