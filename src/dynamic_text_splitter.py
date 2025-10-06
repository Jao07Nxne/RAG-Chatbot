"""
Dynamic Text Splitter for Thai RAG Chatbot
แบ่ง chunks โดยปรับ strategy ตามประเภทเนื้อหา
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple
from content_classifier import ContentClassifier, ContentType


class DynamicTextSplitter:
    """Text Splitter ที่ปรับ strategy ตามประเภทเนื้อหา"""
    
    def __init__(self):
        """สร้าง splitters สำหรับแต่ละประเภทเนื้อหา"""
        
        # Strategy 1: General Content (เนื้อหาทั่วไป)
        self.general_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=[
                "\n\n",        # Paragraph breaks
                "\n",          # Line breaks
                ". ",          # Sentence endings
                "。",          # Thai/Asian full stop
                " ",           # Spaces
                ""             # Characters
            ]
        )
        
        # Strategy 2: Curriculum Table (ตารางรายวิชา) - สำคัญที่สุด!
        self.table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,    # ใหญ่มาก! รองรับตารางทั้งภาค
            chunk_overlap=500,  # Overlap สูง 16%
            length_function=len,
            separators=[
                "\n\n3.1.4",           # 🔥 Section header "3.1.4 แผนการศึกษา"
                "\n\nปีที่ ",          # 🔥 "ปีที่ 1", "ปีที่ 2"
                "\n\nภาคการศึกษาที่ ", # 🔥 "ภาคการศึกษาที่ 1"
                "\n\n\n",              # Multiple newlines (end of table)
                "\n\n",                # Paragraph breaks
                "\n",                  # Line breaks
                ""
            ]
        )
        
        # Strategy 3: Course Description (คำอธิบายรายวิชา)
        self.course_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            separators=[
                r"\n\n\d{8}",  # 🔥 Course code (05506011, 05506012) - raw string!
                "\n\n",        # Paragraph breaks
                "\n",          # Line breaks
                " ",
                ""
            ]
        )
        
        # Strategy 4: Appendix (ภาคผนวก)
        self.appendix_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )
    
    def split_text(
        self, 
        text: str, 
        page_num: int = None
    ) -> Tuple[List[str], ContentType]:
        """
        แบ่ง text เป็น chunks โดยเลือก strategy ตามประเภทเนื้อหา
        
        Args:
            text: ข้อความที่ต้องการแบ่ง
            page_num: หมายเลขหน้า (optional)
        
        Returns:
            Tuple of (chunks, content_type)
        """
        # 🔍 Debug: แสดงตัวอย่างข้อความ
        preview = text[:200].replace('\n', ' ')
        print(f"\n🔍 กำลังจำแนกเนื้อหา...")
        print(f"   Preview: {preview}...")
        
        # จำแนกประเภทเนื้อหา
        content_type = ContentClassifier.classify(text, page_num)
        
        print(f"   ผลลัพธ์: {content_type}")
        
        # เลือก splitter ตามประเภท
        if content_type == "curriculum_table":
            splitter = self.table_splitter
            strategy_emoji = "📊"
            strategy_name = "Table Strategy"
            params = "(3000/500)"
        elif content_type == "course_description":
            splitter = self.course_splitter
            strategy_emoji = "📋"
            strategy_name = "Course Strategy"
            params = "(1500/300)"
        elif content_type == "appendix":
            splitter = self.appendix_splitter
            strategy_emoji = "📄"
            strategy_name = "Appendix Strategy"
            params = "(800/150)"
        else:
            splitter = self.general_splitter
            strategy_emoji = "📝"
            strategy_name = "General Strategy"
            params = "(1000/200)"
        
        print(f"{strategy_emoji} ใช้ {strategy_name} {params}")
        
        # แบ่ง chunks
        chunks = splitter.split_text(text)
        
        # แสดงสถิติ
        total_chars = sum(len(chunk) for chunk in chunks)
        avg_chunk_size = total_chars // len(chunks) if chunks else 0
        
        print(f"   → ได้ {len(chunks)} chunks จาก '{content_type}' "
              f"(เฉลี่ย {avg_chunk_size} chars/chunk)")
        
        return chunks, content_type
    
    def get_splitter_for_type(self, content_type: ContentType) -> RecursiveCharacterTextSplitter:
        """
        ดึง splitter สำหรับประเภทเนื้อหาเฉพาะ
        
        Args:
            content_type: ประเภทเนื้อหา
        
        Returns:
            RecursiveCharacterTextSplitter ที่เหมาะสม
        """
        if content_type == "curriculum_table":
            return self.table_splitter
        elif content_type == "course_description":
            return self.course_splitter
        elif content_type == "appendix":
            return self.appendix_splitter
        else:
            return self.general_splitter
