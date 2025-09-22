"""
Streamlit UI for Thai RAG Chatbot
หน้าเว็บแอปพลิเคชันสำหรับ RAG Chatbot ภาษาไทย
"""

import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
from pathlib import Path
import time

# นำเข้าโมดูลของเรา
import sys
sys.path.append('./src')

from document_processor import ThaiDocumentProcessor
from vector_store import ThaiVectorStoreManager, RECOMMENDED_THAI_MODELS
from rag_system import ThaiRAGSystem, RECOMMENDED_THAI_LLM_MODELS


# กำหนดค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS สำหรับปรับแต่งหน้าเว็บ
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #444444;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    .chat-container {
        background-color: #2d3748;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .chat-message {
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        max-width: 85%;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        margin-left: auto;
        position: relative;
    }
    
    .user-message::before {
        content: "👤";
        position: absolute;
        left: -2.5rem;
        top: 0.5rem;
        background-color: #ff6b6b;
        border-radius: 50%;
        width: 2rem;
        height: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #ffd93d, #ff9500);
        color: #2d3748;
        margin-right: auto;
        position: relative;
    }
    
    .bot-message::before {
        content: "🤖";
        position: absolute;
        left: -2.5rem;
        top: 0.5rem;
        background-color: #ffd93d;
        border-radius: 50%;
        width: 2rem;
        height: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    
    .message-content {
        margin-left: 0.5rem;
        font-weight: 500;
    }
    
    .source-card {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        font-size: 0.9rem;
        box-shadow: 0 3px 10px rgba(79, 172, 254, 0.3);
    }
    
    /* ปรับแต่ง expander */
    .streamlit-expander {
        background-color: #2d3748 !important;
        border: 1px solid #4a5568 !important;
        border-radius: 10px !important;
        margin: 0.5rem 0 !important;
    }
    
    .streamlit-expander > div > div > div > div > p {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    
    .streamlit-expander[data-testid="stExpander"][aria-expanded="true"] {
        background-color: #2d3748 !important;
    }
    
    /* ปรับแต่งหัวข้อ expander */
    details summary {
        background-color: #4a5568 !important;
        color: #e2e8f0 !important;
        padding: 0.8rem !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        font-weight: 500 !important;
    }
    
    details summary:hover {
        background-color: #5a6578 !important;
    }
    
    details[open] summary {
        border-bottom-left-radius: 0 !important;
        border-bottom-right-radius: 0 !important;
        border-bottom: 1px solid #4a5568 !important;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .chat-input {
        background-color: #4a5568;
        border: 2px solid #718096;
        border-radius: 25px;
        color: white;
        padding: 0.8rem 1.5rem;
    }
    
    .chat-input::placeholder {
        color: #a0aec0;
    }
    
    /* Dark theme สำหรับ chat container */
    .stChatInputContainer {
        background-color: #2d3748;
        border-radius: 15px;
        padding: 1rem;
    }
    
    .chat-input-container {
        background-color: #2d3748;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #4a5568;
    }
    
    /* ปรับแต่ง text area */
    .stTextArea > div > div > textarea {
        background-color: #4a5568 !important;
        color: white !important;
        border: 2px solid #718096 !important;
        border-radius: 10px !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: #a0aec0 !important;
    }
    
    /* ปรับแต่งปุ่ม */
    .stButton > button {
        border-radius: 20px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """กำหนดค่าเริ่มต้นสำหรับ session state"""
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = None  # จะสร้างเมื่อผู้ใช้เลือกค่า
    
    if 'vector_store_manager' not in st.session_state:
        st.session_state.vector_store_manager = None
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False


def clean_html_content(text: str) -> str:
    """ทำความสะอาดข้อความจาก HTML tags และ entities"""
    if not text:
        return ""
    
    # แทนที่ HTML tags
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    
    # แทนที่ HTML entities อื่นๆ
    text = text.replace('&', '&amp;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#x27;')
    
    # ลบ HTML tags ที่อาจรั่วมา
    import re
    text = re.sub(r'<[^>]+>', '', text)
    
    return text.strip()


def setup_sidebar():
    """ตั้งค่า Sidebar"""
    st.sidebar.title("⚙️ การตั้งค่า")
    
    # การตั้งค่าโมเดล
    st.sidebar.subheader("🤖 การตั้งค่าโมเดล")
    
    # เลือก Embedding Model
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        options=list(RECOMMENDED_THAI_MODELS.keys()),
        format_func=lambda x: f"{x}: {RECOMMENDED_THAI_MODELS[x].split('/')[-1]}",
        help="เลือกโมเดลสำหรับการสร้าง embeddings"
    )
    
    # เลือก LLM Model
    llm_model = st.sidebar.selectbox(
        "LLM Model (Ollama)",
        options=list(RECOMMENDED_THAI_LLM_MODELS.keys()),
        format_func=lambda x: f"{x}: {RECOMMENDED_THAI_LLM_MODELS[x]}",
        help="เลือกโมเดล LLM สำหรับการตอบคำถาม"
    )
    
    # พารามิเตอร์ของโมเดล
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="ความสร้างสรรค์ของการตอบ (0 = เป็นระบบ, 1 = สร้างสรรค์)"
    )
    
    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=512,
        max_value=4096,
        value=2048,
        step=256,
        help="จำนวนคำสูงสุดในการตอบ"
    )
    
    st.sidebar.divider()
    
    # การตั้งค่าการประมวลผลเอกสาร
    st.sidebar.subheader("📄 การตั้งค่าเอกสาร")
    
    st.sidebar.markdown("""
    **คำแนะนำการตั้งค่า:**
    - **Chunk Size เล็ก (200-500)**: เหมาะกับคำถามเฉพาะเจาะจง
    - **Chunk Size กลาง (800-1200)**: สมดุลระหว่างความแม่นยำและบริบท
    - **Chunk Size ใหญ่ (1500-2000)**: เหมาะกับคำถามที่ต้องการบริบทมาก
    """)
    
    chunk_size = st.sidebar.slider(
        "Chunk Size (ขนาดแต่ละส่วน)",
        min_value=200,
        max_value=2000,
        value=1000,
        step=100,
        help="ขนาดของแต่ละส่วนข้อความ (characters) - ค่าใหญ่ = บริบทมากขึ้น แต่ใช้เวลานานขึ้น"
    )
    
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap (ส่วนซ้อนทับ)",
        min_value=0,
        max_value=min(500, chunk_size//2),
        value=min(200, chunk_size//5),
        step=50,
        help="ส่วนที่ซ้อนทับระหว่างส่วนข้อความ (characters) - ช่วยรักษาบริบทต่อเนื่อง"
    )
    
    # คำนวณจำนวน chunks โดยประมาณ
    if chunk_size > 0:
        estimated_chunks_per_1000_chars = max(1, int(1000 / (chunk_size - chunk_overlap)))
        st.sidebar.caption(f"ประมาณ {estimated_chunks_per_1000_chars} chunks ต่อ 1,000 ตัวอักษร")
    
    # ปุ่มตั้งค่าระบบ
    if st.sidebar.button("🔧 ตั้งค่าระบบ"):
        setup_system(embedding_model, llm_model, temperature, max_tokens, chunk_size, chunk_overlap)
    
    st.sidebar.divider()
    
    # สถิติระบบ
    display_system_stats()
    
    st.sidebar.divider()
    
    # ปุ่มจัดการ
    if st.sidebar.button("🗑️ ล้างข้อมูลทั้งหมด"):
        clear_all_data()
    
    if st.sidebar.button("💬 ล้างประวัติการสนทนา"):
        clear_chat_history()


def setup_system(embedding_model: str, llm_model: str, temperature: float, max_tokens: int, chunk_size: int, chunk_overlap: int):
    """ตั้งค่าระบบ"""
    with st.spinner("กำลังตั้งค่าระบบ..."):
        try:
            # สร้าง Document Processor ด้วยค่าที่ผู้ใช้เลือก
            st.session_state.document_processor = ThaiDocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # สร้าง Vector Store Manager
            st.session_state.vector_store_manager = ThaiVectorStoreManager(
                embedding_model=RECOMMENDED_THAI_MODELS[embedding_model],
                vector_store_path="./vectorstore"
            )
            
            # ลองโหลด vector store ที่มีอยู่
            if st.session_state.vector_store_manager.load_vector_store():
                st.success("✅ โหลด vector store ที่มีอยู่สำเร็จ")
            
            # สร้าง RAG System
            st.session_state.rag_system = ThaiRAGSystem(
                vector_store_manager=st.session_state.vector_store_manager,
                llm_model=RECOMMENDED_THAI_LLM_MODELS[llm_model],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            st.session_state.system_ready = True
            st.success("✅ ตั้งค่าระบบเสร็จสิ้น")
            
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการตั้งค่าระบบ: {str(e)}")


def display_system_stats():
    """แสดงสถิติระบบ"""
    st.sidebar.subheader("📊 สถิติระบบ")
    
    # แสดงค่าปัจจุบันของ chunk settings
    if st.session_state.document_processor:
        st.sidebar.info(f"""
        **การตั้งค่าปัจจุบัน:**
        - Chunk Size: {st.session_state.document_processor.chunk_size} characters
        - Chunk Overlap: {st.session_state.document_processor.chunk_overlap} characters
        """)
    
    if st.session_state.rag_system:
        stats = st.session_state.rag_system.get_system_stats()
        
        st.sidebar.metric(
            "เอกสารในระบบ", 
            stats['vector_store'].get('total_documents', 0)
        )
        
        st.sidebar.metric(
            "ประวัติการสนทนา", 
            stats.get('chat_history_length', 0)
        )
        
        st.sidebar.markdown(f"**โมเดล LLM:** {stats.get('llm_model', 'ไม่ทราบ')}")
        st.sidebar.markdown(f"**สถานะ:** {'พร้อมใช้งาน' if stats.get('rag_chain_ready', False) else 'ไม่พร้อม'}")
    else:
        st.sidebar.info("กรุณาตั้งค่าระบบก่อน")


def upload_documents():
    """ส่วนการอัพโหลดเอกสาร"""
    st.subheader("📁 อัพโหลดเอกสาร")
    
    # ตรวจสอบว่าตั้งค่าระบบแล้วหรือยัง
    if not st.session_state.system_ready or not st.session_state.document_processor:
        st.warning("⚠️ กรุณาตั้งค่าระบบและเลือก Chunk Size, Chunk Overlap ใน Sidebar ก่อน")
        return
    
    # แสดงการตั้งค่าปัจจุบัน
    st.info(f"""
    **การตั้งค่าการประมวลผลปัจจุบัน:**
    - Chunk Size: {st.session_state.document_processor.chunk_size} characters
    - Chunk Overlap: {st.session_state.document_processor.chunk_overlap} characters
    """)
    
    uploaded_files = st.file_uploader(
        "เลือกไฟล์เอกสาร",
        type=['txt', 'pdf', 'docx', 'doc', 'pptx'],
        accept_multiple_files=True,
        help="รองรับไฟล์ .txt, .pdf, .docx, .pptx"
    )
    
    if uploaded_files and st.button("📤 ประมวลผลเอกสาร"):
        process_uploaded_files(uploaded_files)


def process_uploaded_files(uploaded_files):
    """ประมวลผลไฟล์ที่อัพโหลด"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_documents = []
    processed_count = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"กำลังประมวลผล: {uploaded_file.name}")
            
            # บันทึกไฟล์ชั่วคราว
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # ประมวลผลเอกสาร
            documents = st.session_state.document_processor.process_document(
                tmp_path,
                metadata={"uploaded_filename": uploaded_file.name}
            )
            
            all_documents.extend(documents)
            processed_count += 1
            
            # ลบไฟล์ชั่วคราว
            os.unlink(tmp_path)
            
            # อัพเดท progress
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.session_state.processed_files.append({
                "filename": uploaded_file.name,
                "chunks": len(documents),
                "size": uploaded_file.size
            })
            
        except Exception as e:
            st.error(f"❌ ไม่สามารถประมวลผล {uploaded_file.name}: {str(e)}")
    
    if all_documents:
        # เพิ่มเอกสารลงใน vector store
        status_text.text("กำลังสร้าง embeddings และบันทึกลง vector store...")
        st.session_state.vector_store_manager.add_documents(all_documents)
        st.session_state.vector_store_manager.save_vector_store()
        
        # อัพเดท RAG system
        st.session_state.rag_system.update_vector_store(st.session_state.vector_store_manager)
        
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"✅ ประมวลผลเสร็จสิ้น: {processed_count} ไฟล์, {len(all_documents)} chunks")
        
        # แสดงรายละเอียดไฟล์ที่ประมวลผล
        with st.expander("📋 รายละเอียดไฟล์ที่ประมวลผล"):
            for file_info in st.session_state.processed_files[-processed_count:]:
                st.markdown(f"**{file_info['filename']}** - {file_info['chunks']} chunks, {file_info['size']} bytes")


def chat_interface():
    """ส่วนการสนทนา"""
    st.subheader("💬 สนทนากับเอกสาร")
    
    if not st.session_state.system_ready:
        st.warning("⚠️ กรุณาตั้งค่าระบบและอัพโหลดเอกสารก่อน")
        return
    
    # แสดงประวัติการสนทนาใน chat container สวยๆ
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for chat in st.session_state.chat_history:
        # ข้อความผู้ใช้
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-content">
                <strong>คำถาม:</strong><br>
                {chat['question']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ข้อความบอท
        # ทำความสะอาดข้อความให้ปลอดภัยจาก HTML tags
        clean_answer = clean_html_content(chat['answer'])
        st.markdown(f"""
        <div class="chat-message bot-message">
            <div class="message-content">
                <strong>คำตอบ:</strong><br>
                {clean_answer}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # แสดงแหล่งที่มาใน expander
        if 'sources' in chat and chat['sources']:
            with st.expander(f"📚 ดูแหล่งที่มาข้อมูล ({len(chat['sources'])} แหล่ง)", expanded=False):
                for i, source in enumerate(chat['sources'], 1):
                    # ทำความสะอาดข้อมูลแหล่งที่มาจาก HTML tags
                    clean_content = clean_html_content(source['content'])
                    filename = source.get('filename', 'ไม่ทราบชื่อไฟล์')
                    chunk_index = source.get('chunk_index', 'ไม่ทราบ')
                    
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>📄 แหล่งที่มา {i}:</strong> {filename} (ส่วนที่ {chunk_index})<br>
                        <div style="margin-top: 0.5rem; font-style: italic;">
                            {clean_content[:300]}{'...' if len(clean_content) > 300 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ช่องป้อนคำถามในสไตล์สวยๆ
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    st.divider()
    
    with st.form("question_form", clear_on_submit=True):
        question = st.text_area(
            "💭 ถามคำถามเกี่ยวกับเอกสาร:",
            placeholder="พิมพ์คำถามของคุณที่นี่... เช่น เอกสารนี้เล่าเรื่องอะไร? หรือ มีข้อมูลเกี่ยวกับหัวข้อใดบ้าง?",
            height=80,
            key="question_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            submit_button = st.form_submit_button("🚀 ส่งคำถาม", use_container_width=True, type="primary")
        with col2:
            if st.form_submit_button("🗑️ ล้างข้อความ", use_container_width=True):
                st.rerun()
        with col3:
            if st.form_submit_button("💡 ตัวอย่างคำถาม", use_container_width=True):
                st.info("""
                **ตัวอย่างคำถามที่ดี:**
                - เอกสารนี้เกี่ยวกับเรื่องอะไร?
                - สรุปประเด็นสำคัญของเอกสาร
                - มีข้อมูลเกี่ยวกับ [หัวข้อ] หรือไม่?
                - วิธีการ [ทำสิ่งนั้น] คืออะไร?
                """)
        
        if submit_button and question.strip():
            ask_question(question.strip())
    
    st.markdown('</div>', unsafe_allow_html=True)


def ask_question(question: str):
    """ถามคำถามและแสดงผล"""
    with st.spinner("🤔 กำลังคิด..."):
        result = st.session_state.rag_system.ask_question(question)
        
        # เพิ่มลงประวัติการสนทนา
        st.session_state.chat_history.append(result)
        
        # รีเฟรชหน้าเพื่อแสดงคำตอบใหม่
        st.rerun()


def clear_chat_history():
    """ล้างประวัติการสนทนา"""
    st.session_state.chat_history = []
    if st.session_state.rag_system:
        st.session_state.rag_system.clear_chat_history()
    st.success("✅ ล้างประวัติการสนทนาเสร็จสิ้น")


def clear_all_data():
    """ล้างข้อมูลทั้งหมด"""
    st.session_state.chat_history = []
    st.session_state.processed_files = []
    st.session_state.system_ready = False
    
    if st.session_state.vector_store_manager:
        st.session_state.vector_store_manager.clear_vector_store()
    
    st.session_state.vector_store_manager = None
    st.session_state.rag_system = None
    
    st.success("✅ ล้างข้อมูลทั้งหมดเสร็จสิ้น")


def main():
    """ฟังก์ชันหลัก"""
    # Header
    st.markdown('<h1 class="main-header">🤖 Thai RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        ระบบ RAG Chatbot สำหรับเอกสารภาษาไทย | ใช้ Local LLM และ Embedding Models
    </div>
    """, unsafe_allow_html=True)
    
    # กำหนดค่าเริ่มต้น
    initialize_session_state()
    
    # Sidebar
    setup_sidebar()
    
    # เนื้อหาหลัก
    tab1, tab2, tab3 = st.tabs(["📁 อัพโหลดเอกสาร", "💬 สนทนา", "ℹ️ ข้อมูลระบบ"])
    
    with tab1:
        upload_documents()
        
        # แสดงไฟล์ที่ประมวลผลแล้ว
        if st.session_state.processed_files:
            st.subheader("📋 ไฟล์ที่ประมวลผลแล้ว")
            for file_info in st.session_state.processed_files:
                st.markdown(f"- **{file_info['filename']}** ({file_info['chunks']} chunks)")
    
    with tab2:
        chat_interface()
    
    with tab3:
        display_system_info()


def display_system_info():
    """แสดงข้อมูลระบบ"""
    st.subheader("ℹ️ ข้อมูลระบบ")
    
    if st.session_state.rag_system:
        stats = st.session_state.rag_system.get_system_stats()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🗃️ Vector Store")
            st.json(stats['vector_store'])
        
        with col2:
            st.markdown("### 🤖 LLM")
            st.markdown(f"**โมเดล:** {stats['llm_model']}")
            st.markdown(f"**การสนทนา:** {stats['chat_history_length']} รอบ")
            st.markdown(f"**สถานะ RAG:** {'✅ พร้อม' if stats['rag_chain_ready'] else '❌ ไม่พร้อม'}")
    
    st.divider()
    
    # คำแนะนำการใช้งาน
    st.subheader("📖 คำแนะนำการใช้งาน")
    
    with st.expander("🚀 วิธีเริ่มต้นใช้งาน"):
        st.markdown("""
        1. **ตั้งค่าระบบ**: ไปที่ Sidebar เลือกโมเดลและกดปุ่ม "ตั้งค่าระบบ"
        2. **อัพโหลดเอกสาร**: ในแท็บ "อัพโหลดเอกสาร" เลือกไฟล์และกดประมวลผล
        3. **เริ่มสนทนา**: ไปแท็บ "สนทนา" และเริ่มถามคำถาม
        """)
    
    with st.expander("🛠️ การติดตั้ง Ollama"):
        st.markdown("""
        **สำหรับ Windows:**
        ```bash
        # ดาวน์โหลด Ollama จาก https://ollama.ai
        # ติดตั้งแล้วรันคำสั่ง:
        ollama serve
        
        # ดาวน์โหลดโมเดล:
        ollama pull llama3.1:8b
        ```
        
        **สำหรับ macOS/Linux:**
        ```bash
        curl https://ollama.ai/install.sh | sh
        ollama serve
        ollama pull llama3.1:8b
        ```
        """)
    
    with st.expander("📝 โมเดลที่แนะนำ"):
        st.markdown("""
        **Embedding Models:**
        - **Fast**: paraphrase-multilingual-MiniLM-L12-v2 (เร็ว, RAM น้อย)
        - **Balanced**: distiluse-base-multilingual-cased (สมดุล)
        - **Best**: paraphrase-multilingual-mpnet-base-v2 (ดีที่สุด)
        
        **LLM Models:**
        - **Small**: llama3.1:8b (8GB RAM)
        - **Balanced**: llama3.1:70b (80GB RAM)
        - **Coding**: codellama:7b (สำหรับโค้ด)
        """)


if __name__ == "__main__":
    main()