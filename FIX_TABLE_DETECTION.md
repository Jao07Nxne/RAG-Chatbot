# 🔧 ปรับปรุง Curriculum Table Detection

## ปัญหาที่พบ

**อาการ**:
```
📝 ใช้ General Strategy (1000/200)  ← ❌ ควรเป็น Table Strategy!
   → ได้ 40 chunks
   🎯 พบตารางหลักสูตร: ปี 1 ภาค 1 (6 วิชา)  ← มีตาราง แต่ไม่จับ!
```

**สาเหตุ**: `_is_curriculum_table()` ไม่จับตารางเพราะ:

### 1. Regex เข้มงวดเกินไป
```python
# Before (ผิด)
has_year_semester = bool(re.search(
    r'ปีที่\s*\d+.*ภาคการศึกษาที่\s*\d+',  # ← ต้องมีทั้ง 2 อย่างในบรรทัดเดียว!
    text
))
```

**ปัญหา**: PDF อาจมี:
- ช่องว่างพิเศษ: `"ปี ที่ 1"` แทน `"ปีที่ 1"`
- ขึ้นบรรทัดใหม่: 
  ```
  ปีที่ 1
  ภาคการศึกษาที่ 1  ← อยู่คนละบรรทัด!
  ```
- Pattern ไม่จับ: `".*"` อาจไม่ครอบคลุม newline

---

### 2. เกณฑ์การตัดสินเข้มงวด
```python
# Before (ผิด)
score = sum([...])
is_table = score >= 3  # ต้องผ่าน 3/4 เงื่อนไข
```

**ปัญหา**: ถ้า regex ไม่จับ year_semester → score = 2 → ไม่ผ่าน!

---

## การแก้ไข

### Fix 1: แยกเช็ค year และ semester

**Before** (เข้มงวดเกินไป):
```python
has_year_semester = bool(re.search(
    r'ปีที่\s*\d+.*ภาคการศึกษาที่\s*\d+',  # ต้องมีทั้งคู่ในบรรทัดเดียว
    text
))
```

**After** (ยืดหยุ่น):
```python
# แยกเช็คแต่ละตัว
has_year = bool(re.search(
    r'ปี\s*ที่\s*\d+|ชั้นปี\s*\d+',  # รองรับหลายรูปแบบ
    text,
    re.IGNORECASE
))

has_semester = bool(re.search(
    r'ภาค\s*การศึกษา\s*ที่\s*\d+|ภาค\s*\d+|เทอม\s*\d+',  # รองรับหลายรูปแบบ
    text,
    re.IGNORECASE
))

has_year_semester = has_year and has_semester  # ต้องมีทั้งคู่ (แต่ไม่ต้องบรรทัดเดียว)
```

**ข้อดี**:
- ✅ จับได้แม้อยู่คนละบรรทัด
- ✅ รองรับช่องว่างพิเศษ: `"ปี ที่ 1"`, `"ภาค การศึกษา ที่ 1"`
- ✅ รองรับรูปแบบอื่น: `"ชั้นปี 1"`, `"ภาค 1"`, `"เทอม 1"`

---

### Fix 2: ลดเกณฑ์การตัดสิน

**Before** (เข้มงวดเกินไป):
```python
is_table = score >= 3  # ต้องผ่าน 3/4 เงื่อนไข
```

**After** (ยืดหยุ่น):
```python
# ถ้ามีรหัสวิชา 3+ และหน่วยกิต → เป็นตาราง!
is_table = (score >= 2 and has_multiple_courses) or score >= 3
```

**Logic**:
```python
# กรณีที่ 1: score >= 3 → ผ่านอยู่แล้ว
if score >= 3:
    is_table = True

# กรณีที่ 2: score = 2 แต่มีรหัสวิชา 3+ → เป็นตาราง!
elif score == 2 and has_multiple_courses:
    is_table = True
    
# กรณีที่ 3: score < 2 → ไม่ใช่ตาราง
else:
    is_table = False
```

**ตัวอย่าง**:
```
เงื่อนไข:
1. has_year_semester = False  (regex ไม่จับ)
2. has_multiple_courses = True (มี 6 รหัสวิชา)
3. has_credits = True          (มี "หน่วยกิต")
4. has_total = False           (ไม่มี "รวม")

score = 2  (ผ่านแค่ 2 เงื่อนไข)

# Before:
is_table = score >= 3  → False ❌

# After:
is_table = (score >= 2 and has_multiple_courses) or score >= 3
         = (2 >= 2 and True) or (2 >= 3)
         = True or False
         = True ✅
```

---

### Fix 3: เพิ่ม Debug Logs

**เพิ่มใน `dynamic_text_splitter.py`**:
```python
def split_text(self, text: str, page_num: int = None):
    # 🔍 Debug: แสดงตัวอย่างข้อความ
    preview = text[:200].replace('\n', ' ')
    print(f"\n🔍 กำลังจำแนกเนื้อหา...")
    print(f"   Preview: {preview}...")
    
    # จำแนกประเภทเนื้อหา
    content_type = ContentClassifier.classify(text, page_num)
    
    print(f"   ผลลัพธ์: {content_type}")
```

**ประโยชน์**:
- เห็น preview ข้อความที่กำลังจำแนก
- รู้ว่า classifier ตัดสินเป็นอะไร
- ง่ายต่อการ debug

---

## ผลลัพธ์ที่คาดหวัง

### Before (ผิด):
```
🎯 เริ่มแบ่ง chunks ด้วย Dynamic Strategy...
📝 ใช้ General Strategy (1000/200)     ← ❌ ผิด!
   → ได้ 40 chunks จาก 'general'
   
   🎯 พบตารางหลักสูตร: ปี 1 ภาค 1 (6 วิชา) ใน chunk 31
   🎯 พบตารางหลักสูตร: ปี 2 ภาค 1 (3 วิชา) ใน chunk 33
   
   ← ตารางถูกตัด เป็น chunks เล็กๆ!
```

**ผลกระทบ**:
- Chunk size = 1000 chars
- ตาราง 1 ภาค = ~2500 chars
- ถูกตัดเป็น 2-3 chunks
- LLM ตอบไม่ครบ

---

### After (ถูก):
```
🎯 เริ่มแบ่ง chunks ด้วย Dynamic Strategy...

🔍 กำลังจำแนกเนื้อหา...
   Preview: ปี ที่ 1 ภาค การศึกษา ที่ 1  05506011 โปรแกรม... ← มีรหัสวิชา!
   ✅ ตรวจพบตารางหลักสูตร (คะแนน 2/4, รหัสวิชา 6 รายการ)
   ผลลัพธ์: curriculum_table                                     ← ✅ ถูก!

📊 ใช้ Table Strategy (3000/500)                                  ← ✅ ถูก!
   → ได้ 1 chunks จาก 'curriculum_table' (เฉลี่ย 2850 chars/chunk)
   
   🎯 พบตารางหลักสูตร: ปี 1 ภาค 1 (8 วิชา) ใน chunk 0          ← ✅ ครบ!
```

**ผลลัพธ์**:
- ✅ Classifier จับตารางได้ถูกต้อง
- ✅ ใช้ Table Strategy (3000/500)
- ✅ ตารางครบใน 1 chunk
- ✅ LLM ตอบครบ 8 วิชา

---

## การทดสอบ

### ขั้นตอน:
1. Restart Streamlit
2. ล้างข้อมูลเดิม
3. อัพโหลดเอกสารใหม่
4. ดู debug logs

### ดู Logs ที่สำคัญ:

**เช็คว่า Classifier ทำงานถูก**:
```
🔍 กำลังจำแนกเนื้อหา...
   Preview: ปี ที่ 2 ภาค การศึกษา ที่ 1...
   ✅ ตรวจพบตารางหลักสูตร (คะแนน X/4, รหัสวิชา Y รายการ)
   ผลลัพธ์: curriculum_table  ← ต้องเป็นแบบนี้!
```

**เช็คว่าใช้ Strategy ถูก**:
```
📊 ใช้ Table Strategy (3000/500)  ← ต้องเป็นแบบนี้!
   → ได้ 1 chunks                 ← 1 ตาราง = 1 chunk
```

**เช็คว่าตารางครบ**:
```
🎯 พบตารางหลักสูตร: ปี 2 ภาค 1 (8 วิชา)  ← ต้องครบ 8 วิชา!
```

---

### ทดสอบคำถาม:
```
"ปีที่ 2 ภาคการศึกษาที่ 1 มีรายวิชาอะไรบ้าง"
```

**คาดหวัง**:
```
✅ ได้ครบ 8 วิชา พร้อมรหัส ชื่อ หน่วยกิต

ในปีที่ 2 ภาคการศึกษาที่ 1 มีรายวิชาทั้งหมด 8 วิชา ดังนี้:

• 05506211 โครงสร้างข้อมูล (Data Structures) - 3 หน่วยกิต
• 05506221 ระบบฐานข้อมูล (Database Systems) - 3 หน่วยกิต
• 05506231 การวิเคราะห์และออกแบบระบบ - 3 หน่วยกิต
• 05506241 องค์กรและสถาปัตยกรรมคอมพิวเตอร์ - 3 หน่วยกิต
• 05506251 การออกแบบและพัฒนาเว็บ - 3 หน่วยกิต
• 05513231 ความน่าจะเป็นและสถิติ - 3 หน่วยกิต
• หมวดวิชาศึกษาทั่วไป 2 วิชา - 6 หน่วยกิต

รวม 24 หน่วยกิต
```

---

## Troubleshooting

### ปัญหา: ยังจับเป็น "general"
```
📝 ใช้ General Strategy (1000/200)
```

**วิธีแก้**:
1. ดู debug logs:
   ```
   🔍 Preview: ...  ← เช็คว่ามี "ปี", "ภาค", รหัสวิชาหรือไม่
   ```

2. ถ้าไม่มี → PDF อาจมีปัญหา:
   ```python
   # ตรวจสอบ PDF text extraction
   print(text[:500])  # แสดง 500 ตัวอักษรแรก
   ```

3. ปรับ regex เพิ่มเติม:
   ```python
   # เพิ่มรูปแบบอื่นๆ
   has_year = bool(re.search(
       r'ปี\s*ที่\s*\d+|ชั้นปี\s*\d+|Year\s*\d+',
       text,
       re.IGNORECASE
   ))
   ```

---

### ปัญหา: ตารางยังถูกตัด
```
🎯 พบตารางหลักสูตร: ปี 2 ภาค 1 (3 วิชา)  ← ไม่ครบ!
```

**สาเหตุ**: แม้ใช้ Table Strategy แต่ตารางใหญ่เกิน 3000 chars

**วิธีแก้**:
1. เพิ่ม chunk_size:
   ```python
   self.table_splitter = RecursiveCharacterTextSplitter(
       chunk_size=4000,  # เพิ่มจาก 3000 → 4000
       chunk_overlap=600
   )
   ```

2. หรือปรับ separator:
   ```python
   separators=[
       "\n\nปีที่ ",          # ตัดที่ปี
       "\n\nภาคการศึกษาที่ ", # ตัดที่ภาค (ระวัง!)
       ...
   ]
   ```

---

## สรุป

### การเปลี่ยนแปลง:

1. ✅ **แยกเช็ค year และ semester** (ยืดหยุ่นขึ้น)
   ```python
   has_year and has_semester  # ไม่ต้องบรรทัดเดียว
   ```

2. ✅ **รองรับรูปแบบหลากหลาย**
   ```python
   r'ปี\s*ที่\s*\d+|ชั้นปี\s*\d+|Year\s*\d+'
   ```

3. ✅ **ลดเกณฑ์การตัดสิน**
   ```python
   is_table = (score >= 2 and has_multiple_courses) or score >= 3
   ```

4. ✅ **เพิ่ม debug logs**
   ```python
   print(f"🔍 Preview: {text[:200]}...")
   ```

### ผลลัพธ์:
- ✅ Classifier จับตารางได้แม่นยำขึ้น
- ✅ ใช้ Table Strategy (3000/500) ถูกต้อง
- ✅ ตารางครบใน 1 chunk
- ✅ LLM ตอบครบ 8 วิชา

---

**แก้ไขเมื่อ**: 2024-10-06  
**ไฟล์ที่แก้**: 
- `src/content_classifier.py` (บรรทัด 45-90)
- `src/dynamic_text_splitter.py` (บรรทัด 87-95)
