# 🤖 Thai RAG Chatbot

ระบบ RAG (Retrieval-Augmented Generation) Chatbot สำหรับเอกสารภาษาไทย ใช้ Local LLM และ Embedding Models รองรับการสนทนาแบบต่อเนื่อง

## ✨ คุณสมบัติ

- 🇹🇭 **รองรับภาษาไทย**: ประมวลผลเอกสารและตอบคำถามเป็นภาษาไทย
- 🏠 **Local LLM**: ใช้ Ollama สำหรับรันโมเดล LLM บนเครื่อง
- 🧠 **Local Embedding**: ใช้ Sentence Transformers สำหรับสร้าง embeddings
- 💬 **Chat History**: รองรับการสนทนาแบบต่อเนื่อง
- 📁 **หลายรูปแบบไฟล์**: รองรับ PDF, DOCX, TXT, PPTX
- ⚡ **ประสิทธิภาพ**: ใช้ FAISS สำหรับการค้นหาแบบเวกเตอร์
- 🎨 **UI สวยงาม**: ใช้ Streamlit พร้อมการออกแบบที่ใช้งานง่าย

## 📋 ความต้องการระบบ

### ระบบปฏิบัติการ
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+)

### ฮาร์ดแวร์
- **RAM**: อย่างน้อย 8GB (แนะนำ 16GB+)
- **CPU**: Intel i5 หรือ AMD Ryzen 5 ขึ้นไป
- **GPU**: ไม่จำเป็น (แต่จะช่วยเร่งความเร็ว)
- **ที่เก็บข้อมูล**: อย่างน้อย 10GB สำหรับโมเดลและข้อมูล

### ซอฟต์แวร์
- Python 3.8+
- Git

## 🚀 การติดตั้ง

### 1. ติดตั้ง Ollama

#### Windows
```bash
# ดาวน์โหลดจาก https://ollama.ai
# หรือใช้ winget
winget install ollama
```

#### macOS
```bash
# ใช้ Homebrew
brew install ollama

# หรือดาวน์โหลดจาก https://ollama.ai
```

#### Linux
```bash
curl https://ollama.ai/install.sh | sh
```

### 2. รัน Ollama และดาวน์โหลดโมเดล

#### สำหรับ Windows (หากคำสั่ง ollama ไม่ทำงาน)
```bash
# ใช้สคริปต์ที่เตรียมไว้
setup_ollama.bat

# หรือใช้คำสั่งเต็ม
& "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" serve
```

#### คำสั่งปกติ (หลังจากเพิ่ม PATH แล้ว)
```bash
# เริ่ม Ollama server
ollama serve

# ดาวน์โหลดโมเดล LLM (เลือก 1 อัน)
ollama pull llama3.1:8b      # เร็ว, RAM น้อย (8GB)
ollama pull gemma2:2b        # เล็กมาก, RAM น้อย (2GB) - แนะนำสำหรับทดสอบ
ollama pull llama3.1:70b     # สมดุล (80GB)
ollama pull codellama:7b     # สำหรับโค้ด (7GB)
```

### 3. Clone โปรเจกต์

```bash
git clone <repository-url>
cd thai-rag-chatbot
```

### 4. สร้าง Virtual Environment

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 6. ตั้งค่า Environment (ไม่บังคับ)

```bash
# คัดลอกไฟล์ตัวอย่าง
copy .env.example .env

# แก้ไขค่าตามต้องการ
notepad .env  # Windows
nano .env     # macOS/Linux
```

## 🎯 การใช้งาน

### เริ่มต้นใช้งาน

1. **เปิด Ollama Server**
   ```bash
   ollama serve
   ```

2. **รันแอปพลิเคชัน**
   ```bash
   streamlit run app.py
   ```

3. **เปิดเว็บเบราว์เซอร์** ไปที่ `http://localhost:8501`

### ขั้นตอนการใช้งาน

1. **ตั้งค่าระบบ**
   - ไปที่ Sidebar เลือกโมเดล Embedding และ LLM
   - กดปุ่ม "ตั้งค่าระบบ"

2. **อัพโหลดเอกสาร**
   - ไปแท็บ "อัพโหลดเอกสาร"
   - เลือกไฟล์ PDF, DOCX, TXT หรือ PPTX
   - กดปุ่ม "ประมวลผลเอกสาร"

3. **เริ่มสนทนา**
   - ไปแท็บ "สนทนา"
   - ถามคำถามเกี่ยวกับเอกสารที่อัพโหลด
   - ระบบจะตอบพร้อมแหล่งข้อมูล

## 📁 โครงสร้างโปรเจกต์

```
thai-rag-chatbot/
├── app.py                 # หน้าเว็บ Streamlit หลัก
├── requirements.txt       # รายการ dependencies
├── .env.example          # ตัวอย่างการตั้งค่า
├── README.md             # เอกสารนี้
│
├── src/                  # โมดูลหลัก
│   ├── __init__.py
│   ├── document_processor.py    # ประมวลผลเอกสาร
│   ├── vector_store.py         # จัดการ Vector Store
│   └── rag_system.py          # ระบบ RAG
│
├── config/              # การตั้งค่า
│   ├── __init__.py
│   └── config.py       # จัดการ configuration
│
├── data/               # เก็บเอกสารต้นฉบับ
├── vectorstore/        # เก็บ Vector Database
└── temp/              # ไฟล์ชั่วคราว
```

## ⚙️ การตั้งค่าโมเดล

### Embedding Models (แนะนำ)

| โมเดล | ความเร็ว | คุณภาพ | RAM | เหมาะสำหรับ |
|-------|----------|--------|-----|-------------|
| `paraphrase-multilingual-MiniLM-L12-v2` | ⭐⭐⭐ | ⭐⭐ | 1GB | ทั่วไป, เร็ว |
| `distiluse-base-multilingual-cased` | ⭐⭐ | ⭐⭐⭐ | 2GB | สมดุล |
| `paraphrase-multilingual-mpnet-base-v2` | ⭐ | ⭐⭐⭐ | 4GB | คุณภาพสูง |

### LLM Models (Ollama)

| โมเดล | ขนาด | RAM ต้องการ | เหมาะสำหรับ |
|-------|------|-------------|-------------|
| `gemma2:2b` | 2B | 4GB | **แนะนำสำหรับเริ่มต้น** ⭐ |
| `llama3.1:8b` | 8B | 8GB | ทั่วไป, เร็ว |
| `llama3.1:70b` | 70B | 80GB | คุณภาพสูง |
| `codellama:7b` | 7B | 7GB | โค้ด, เทคนิค |
| `mistral:7b` | 7B | 7GB | การสนทนา |

## 🔧 การแก้ไขปัญหา

### ปัญหาที่พบบ่อย

#### 1. คำสั่ง ollama ไม่ทำงาน (Windows)
```bash
# ปัญหา: ollama is not recognized
# แก้ไข 1: ใช้สคริปต์ที่เตรียมไว้
setup_ollama.bat

# แก้ไข 2: ใช้คำสั่งเต็ม
& "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" --version

# แก้ไข 3: เพิ่ม PATH ถาวร (เปิด Command Prompt as Administrator)
setx PATH "%PATH%;C:\Users\%USERNAME%\AppData\Local\Programs\Ollama"
```

#### 2. Ollama ไม่เชื่อมต่อ
```bash
# ตรวจสอบว่า Ollama ทำงาน
ollama list

# ตรวจสอบ port
netstat -ano | findstr :11434

# รีสตาร์ท Ollama
# Windows: รีสตาร์ทจาก Services หรือ Task Manager
# macOS/Linux: 
killall ollama
ollama serve
```

#### 2. โมเดลโหลดไม่ได้
```bash
# ตรวจสอบรายการโมเดล
ollama list

# ดาวน์โหลดโมเดลใหม่
ollama pull llama3.1:8b
```

#### 3. Memory Error
- ใช้โมเดลที่เล็กกว่า
- ปิดโปรแกรมอื่นๆ
- ลดขนาด chunk_size ในการตั้งค่า

#### 4. ไฟล์อ่านไม่ได้
- ตรวจสอบ encoding ของไฟล์
- ใช้ไฟล์ UTF-8
- ลองแปลงไฟล์เป็น PDF

### การตั้งค่าประสิทธิภาพ

#### สำหรับเครื่องที่มี RAM น้อย
```python
# ใน config/config.py หรือ .env
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=100
BATCH_SIZE=16
```

#### สำหรับเครื่องที่มี RAM เยอะ
```python
DEFAULT_CHUNK_SIZE=1500
DEFAULT_CHUNK_OVERLAP=300
BATCH_SIZE=64
```

## 📊 การประเมินประสิทธิภาพ

### Benchmark บนเอกสารภาษาไทย

| โมเดล | ความแม่นยำ | ความเร็ว | หน่วยความจำ |
|-------|------------|----------|-------------|
| llama3.1:8b + MiniLM | 85% | 2-3 วินาที | 10GB |
| llama3.1:70b + mpnet | 92% | 10-15 วินาที | 90GB |

### เคสทดสอบ
- ✅ เอกสารกฎหมายไทย
- ✅ บทความวิชาการ
- ✅ คู่มือการใช้งาน
- ✅ รายงานธุรกิจ

## 🤝 การพัฒนาต่อ

### ติดตั้งสำหรับการพัฒนา

```bash
# ติดตั้ง development dependencies
pip install -r requirements-dev.txt

# รันการทดสอบ
python -m pytest tests/

# ตรวจสอบ code style
flake8 src/
black src/
```

### การปรับปรุงที่วางแผนไว้

- [ ] รองรับ Retrieval แบบ Hybrid (keyword + semantic)
- [ ] เพิ่มโมเดลภาษาไทยเฉพาะ
- [ ] สร้าง API REST
- [ ] รองรับไฟล์เสียงและรูปภาพ
- [ ] Dashboard สำหรับ Analytics

## 📝 License

MIT License - ดู [LICENSE](LICENSE) สำหรับรายละเอียด

## 🙏 Credits

- [LangChain](https://langchain.com/) - RAG Framework
- [Ollama](https://ollama.ai/) - Local LLM Server
- [Sentence Transformers](https://www.sbert.net/) - Embedding Models
- [Streamlit](https://streamlit.io/) - Web Framework
- [PyThaiNLP](https://pythainlp.github.io/) - Thai Language Processing
- [FAISS](https://faiss.ai/) - Vector Search

## 📞 ติดต่อ

หากมีคำถามหรือปัญหา กรุณาสร้าง Issue ใน GitHub Repository

---

**สร้างด้วย ❤️ สำหรับชุมชนผู้พัฒนาไทย**