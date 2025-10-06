"""
Content Classifier for Dynamic Chunking Strategy
จำแนกประเภทเนื้อหาเพื่อเลือก chunking strategy ที่เหมาะสม
"""

import re
from typing import Literal

ContentType = Literal["general", "curriculum_table", "course_description", "appendix"]


class ContentClassifier:
    """จำแนกประเภทเนื้อหาเพื่อเลือก chunking strategy"""
    
    @staticmethod
    def classify(text: str, page_num: int = None) -> ContentType:
        """
        จำแนกประเภทเนื้อหา
        
        Args:
            text: ข้อความที่ต้องการจำแนก
            page_num: หมายเลขหน้า (optional)
        
        Returns:
            ประเภทเนื้อหา: general, curriculum_table, course_description, appendix
        """
        
        # 🔥 1. ตรวจสอบ Curriculum Table ก่อน (มีลำดับความสำคัญสูงสุด!)
        # ต้องเช็คก่อน appendix เพราะตารางอาจมีคำว่า "ภาคผนวก" ด้วย
        if ContentClassifier._is_curriculum_table(text):
            return "curriculum_table"
        
        # 2. ตรวจสอบ Course Description
        if ContentClassifier._is_course_description(text):
            return "course_description"
        
        # 3. ตรวจสอบ Appendix (เช็คทีหลัง เพื่อไม่ให้ทับตาราง)
        if ContentClassifier._is_appendix(text, page_num):
            return "appendix"
        
        # 4. Default: General content
        return "general"
    
    @staticmethod
    def _is_curriculum_table(text: str) -> bool:
        """
        ตรวจสอบว่าเป็นตารางแผนการศึกษาหรือไม่
        
        เงื่อนไข:
        - มีหัวข้อ "ปีที่ X ภาคการศึกษาที่ Y" (ยืดหยุ่น)
        - มีรหัสวิชา 3 รายการขึ้นไป
        - มีคำว่า "หน่วยกิต"
        - มีรูปแบบ "รวม XX หน่วยกิต"
        """
        # เงื่อนไข 1: มีหัวข้อ "ปีที่ X" และ "ภาคการศึกษาที่ Y" (แยกเช็ค - ยืดหยุ่นกว่า)
        has_year = bool(re.search(
            r'ปี\s*ที่\s*\d+|ชั้นปี\s*\d+', 
            text,
            re.IGNORECASE
        ))
        
        has_semester = bool(re.search(
            r'ภาค\s*การศึกษา\s*ที่\s*\d+|ภาค\s*\d+|เทอม\s*\d+',
            text,
            re.IGNORECASE
        ))
        
        has_year_semester = has_year and has_semester
        
        # เงื่อนไข 2: มีรหัสวิชา 3 รายการขึ้นไป (8 หลัก เช่น 05506232)
        course_codes = re.findall(r'\b\d{8}\b', text)
        has_multiple_courses = len(course_codes) >= 3
        
        # เงื่อนไข 3: มีคำว่า "หน่วยกิต"
        has_credits = 'หน่วยกิต' in text
        
        # เงื่อนไข 4: มีรูปแบบ "รวม XX หน่วยกิต"
        has_total = bool(re.search(r'รวม\s+\d+\s+หน่วยกิต', text))
        
        # 🔥 ลดเกณฑ์: ต้องผ่านอย่างน้อย 2/4 เงื่อนไข (แต่ต้องมีรหัสวิชา!)
        score = sum([
            has_year_semester,
            has_multiple_courses,
            has_credits,
            has_total
        ])
        
        # ถ้ามีรหัสวิชา 3+ และหน่วยกิต → เป็นตาราง!
        is_table = (score >= 2 and has_multiple_courses) or score >= 3
        
        if is_table:
            print(f"   ✅ ตรวจพบตารางหลักสูตร (คะแนน {score}/4, รหัสวิชา {len(course_codes)} รายการ)")
        
        return is_table
    
    @staticmethod
    def _is_course_description(text: str) -> bool:
        """
        ตรวจสอบว่าเป็นคำอธิบายรายวิชาหรือไม่
        
        เงื่อนไข:
        - ขึ้นต้นด้วยรหัสวิชา 8 หลัก
        - มีคำว่า "วัตถุประสงค์" หรือ "เนื้อหารายวิชา"
        - มีชื่อวิชาภาษาอังกฤษในวงเล็บ
        """
        # เงื่อนไข 1: ขึ้นต้นด้วยรหัสวิชา 8 หลัก
        starts_with_code = bool(re.match(r'^\s*\d{8}\s+', text))
        
        # เงื่อนไข 2: มีคำว่า "วัตถุประสงค์" หรือ "เนื้อหารายวิชา"
        has_objective = 'วัตถุประสงค์' in text or 'เนื้อหารายวิชา' in text or 'เนื้อหา' in text[:200]
        
        # เงื่อนไข 3: มีชื่อวิชาภาษาอังกฤษในวงเล็บ
        has_english_name = bool(re.search(r'\([A-Z][a-zA-Z\s]+\)', text[:300]))
        
        is_course = starts_with_code and (has_objective or has_english_name)
        
        if is_course:
            # ดึงรหัสวิชาเพื่อแสดง
            code_match = re.match(r'^\s*(\d{8})\s+', text)
            code = code_match.group(1) if code_match else "unknown"
            print(f"   ✅ ตรวจพบคำอธิบายรายวิชา (รหัส {code})")
        
        return is_course
    
    @staticmethod
    def _is_appendix(text: str, page_num: int = None) -> bool:
        """
        ตรวจสอบว่าเป็นภาคผนวกหรือไม่
        
        เงื่อนไข:
        - มีคำว่า "ภาคผนวก" หรือ "Appendix"
        - มีคำว่า "แผนที่หลักสูตร" หรือ "Curriculum Map"
        - หน้ามากกว่า 45 (ถ้ามี page_num)
        - **แต่ห้ามมีรหัสวิชา 3+ รายการ** (คงเป็นตาราง)
        """
        # ตรวจสอบว่ามีรหัสวิชาเยอะไหม (ถ้ามี = คงเป็นตาราง ไม่ใช่ appendix)
        course_codes = re.findall(r'\b\d{8}\b', text)
        if len(course_codes) >= 3:
            # มีรหัสวิชาเยอะ → น่าจะเป็นตาราง ไม่ใช่ appendix
            return False
        
        # มีคำสำคัญ
        has_appendix_keyword = bool(re.search(
            r'ภาคผนวก|Appendix|แผนที่หลักสูตร|Curriculum\s+Map|เอกสารอ้างอิง|รายชื่ออาจารย์',
            text,
            re.IGNORECASE
        ))
        
        # หน้ามากกว่า 45 (สันนิษฐาน)
        is_late_page = page_num and page_num > 45
        
        is_appendix = has_appendix_keyword or is_late_page
        
        if is_appendix:
            print(f"   ✅ ตรวจพบภาคผนวก (รหัสวิชา: {len(course_codes)} รายการ)")
        
        return is_appendix
    
    @staticmethod
    def get_strategy_info(content_type: ContentType) -> dict:
        """
        ดึงข้อมูล strategy สำหรับแต่ละประเภท
        
        Returns:
            dict with chunk_size, chunk_overlap, description
        """
        strategies = {
            "general": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "description": "เนื้อหาทั่วไป (บทความ, คำอธิบาย)"
            },
            "curriculum_table": {
                "chunk_size": 3000,
                "chunk_overlap": 500,
                "description": "ตารางรายวิชา (ครบถ้วน, ไม่ตัด)"
            },
            "course_description": {
                "chunk_size": 1500,
                "chunk_overlap": 300,
                "description": "คำอธิบายรายวิชา (1 chunk = 1 วิชา)"
            },
            "appendix": {
                "chunk_size": 800,
                "chunk_overlap": 150,
                "description": "ภาคผนวก (ประหยัด)"
            }
        }
        
        return strategies.get(content_type, strategies["general"])
