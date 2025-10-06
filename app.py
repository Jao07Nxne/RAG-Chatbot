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

# 🎨 CSS สำหรับปรับแต่งหน้าเว็บให้สวยงาม
st.markdown("""
<style>
    /* ===== GLOBAL STYLES ===== */
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Sarabun', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* ===== HEADER ===== */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2d4a6f 100%);
        border-right: 2px solid #667eea;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffd93d !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* ===== CHAT CONTAINER ===== */
    .chat-container {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* ===== CHAT MESSAGES ===== */
    .chat-message {
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        max-width: 80%;
        line-height: 1.8;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        animation: slideIn 0.3s ease-out;
        position: relative;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* ===== USER MESSAGE ===== */
    .user-message {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 50%, #ff8787 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
        box-shadow: 0 8px 20px rgba(255, 107, 107, 0.4);
    }
    
    .user-message::before {
        content: "👤";
        position: absolute;
        right: -3rem;
        top: 0.5rem;
        background: linear-gradient(135deg, #ff6b6b, #ff8787);
        border-radius: 50%;
        width: 2.5rem;
        height: 2.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.5);
    }
    
    /* ===== BOT MESSAGE ===== */
    .bot-message {
        background: linear-gradient(135deg, #ffd93d 0%, #ffb700 50%, #ffe66d 100%);
        color: #1a202c;
        margin-right: auto;
        border-bottom-left-radius: 5px;
        box-shadow: 0 8px 20px rgba(255, 217, 61, 0.4);
    }
    
    .bot-message::before {
        content: "🤖";
        position: absolute;
        left: -3rem;
        top: 0.5rem;
        background: linear-gradient(135deg, #ffd93d, #ffe66d);
        border-radius: 50%;
        width: 2.5rem;
        height: 2.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 4px 12px rgba(255, 217, 61, 0.5);
    }
    
    /* ===== BUTTONS ===== */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* ===== INPUT FIELDS ===== */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background-color: rgba(45, 55, 72, 0.8);
        color: white;
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
        outline: none;
    }
    
    /* ===== SELECT BOX ===== */
    .stSelectbox>div>div>div {
        background-color: rgba(45, 55, 72, 0.8);
        color: white;
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox>div>div>div:hover {
        border-color: #667eea;
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(45, 55, 72, 0.6), rgba(26, 32, 44, 0.6));
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, rgba(45, 55, 72, 0.8), rgba(26, 32, 44, 0.8));
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* ===== SPINNER ===== */
    .stSpinner>div {
        border-top-color: #667eea !important;
        border-right-color: #764ba2 !important;
    }
    
    /* ===== SUCCESS/ERROR MESSAGES ===== */
    .stSuccess {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);
        animation: slideIn 0.3s ease-out;
    }
    
    .stError {
        background: linear-gradient(135deg, #f56565, #e53e3e);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(245, 101, 101, 0.4);
        animation: slideIn 0.3s ease-out;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border-radius: 10px;
        font-weight: 600;
        color: #ffd93d;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a202c;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
        border: 2px solid #1a202c;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    /* ===== LOADING ANIMATION ===== */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* ===== INFO BOX ===== */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            padding: 1.5rem 1rem;
        }
        
        .chat-message {
            max-width: 90%;
            padding: 1rem;
        }
        
        .user-message::before,
        .bot-message::before {
            width: 2rem;
            height: 2rem;
            font-size: 1rem;
        }
    }
    
    /* ===== MESSAGE CONTENT ===== */
    .message-content {
        margin-left: 0.5rem;
        font-weight: 500;
    }
    
    /* ===== SOURCE CARD ===== */
    .source-card {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        font-size: 0.9rem;
        box-shadow: 0 3px 10px rgba(79, 172, 254, 0.3);
        transition: all 0.3s ease;
    }
    
    .source-card:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.5);
    }
    
    /* ===== STATS CARD ===== */
    .stats-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* ===== CHAT INPUT CONTAINER ===== */
    .stChatInputContainer,
    .chat-input-container {
        background: linear-gradient(135deg, rgba(45, 55, 72, 0.6), rgba(26, 32, 44, 0.6));
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .chat-input {
        background-color: rgba(74, 85, 104, 0.8);
        border: 2px solid #718096;
        border-radius: 25px;
        color: white;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .chat-input::placeholder {
        color: #a0aec0;
    }
    
    .chat-input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
        outline: none;
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
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; 
                text-align: center; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);">
        <h2 style="color: white; margin: 0; font-size: 1.8rem;">⚙️ การตั้งค่า</h2>
        <p style="color: rgba(255, 255, 255, 0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            ปรับแต่งระบบตามความต้องการ
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
        max_value=8192,  # เพิ่มเป็น 8192 รองรับคำตอบยาว
        value=4096,      # ตั้งค่า default สูงๆ
        step=512,
        help="จำนวน tokens สูงสุดในการตอบ (ไม่ใช่จำนวนขั้นต่ำ - LLM จะตอบสั้นตามความเหมาะสม)"
    )
    
    st.sidebar.divider()
    
    # การตั้งค่าการประมวลผลเอกสาร
    st.sidebar.subheader("📄 การตั้งค่าเอกสาร")
    
    # Dynamic Chunking Toggle
    use_dynamic_chunking = st.sidebar.checkbox(
        "✨ ใช้ Dynamic Chunking",
        value=True,  # เปิดโดย default
        help="แบ่ง chunks ตามประเภทเนื้อหา (ตาราง=3000, ทั่วไป=1000) - แนะนำเปิด!"
    )
    
    if use_dynamic_chunking:
        st.sidebar.success("""
        **Dynamic Chunking เปิดอยู่** ✅
        - 📊 ตารางรายวิชา: 3000 chars
        - 📋 คำอธิบายรายวิชา: 1500 chars
        - 📝 เนื้อหาทั่วไป: 1000 chars
        """)
        # ปิด manual chunk settings
        chunk_size = None
        chunk_overlap = None
    else:
        st.sidebar.warning("**Manual Chunking** - ตั้งค่าเอง")
        
        st.sidebar.markdown("""
        **คำแนะนำการตั้งค่า:**
        - **Chunk Size เล็ก (500-1000)**: เหมาะกับคำถามเฉพาะเจาะจง
        - **Chunk Size กลาง (1000-2000)**: สมดุลระหว่างความแม่นยำและบริบท
        - **Chunk Size ใหญ่ (2000-3000)**: เหมาะกับตารางและบริบทมาก
        """)
        
        chunk_size = st.sidebar.slider(
            "Chunk Size (ขนาดแต่ละส่วน)",
            min_value=500,
            max_value=3000,  # เพิ่มเป็น 3000
            value=1500,
            step=100,
            help="ขนาดของแต่ละส่วนข้อความ (characters)"
        )
        
        chunk_overlap = st.sidebar.slider(
            "Chunk Overlap (ส่วนซ้อนทับ)",
            min_value=100,
            max_value=min(600, chunk_size//2),
            value=min(400, chunk_size//3),
            step=50,
            help="ส่วนที่ซ้อนทับระหว่างส่วนข้อความ (characters)"
        )
        
        # คำนวณจำนวน chunks โดยประมาณ
        if chunk_size > 0:
            estimated_chunks_per_1000_chars = max(1, int(1000 / (chunk_size - chunk_overlap)))
            st.sidebar.caption(f"ประมาณ {estimated_chunks_per_1000_chars} chunks ต่อ 1000 ตัวอักษร")
    
    # ปุ่มตั้งค่าระบบ
    if st.sidebar.button("🔧 ตั้งค่าระบบ"):
        setup_system(
            embedding_model=embedding_model,
            llm_model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_dynamic_chunking=use_dynamic_chunking
        )
    
    st.sidebar.divider()
    
    # สถิติระบบ
    display_system_stats()
    
    st.sidebar.divider()
    
    # ปุ่มจัดการ
    if st.sidebar.button("🗑️ ล้างข้อมูลทั้งหมด"):
        clear_all_data()
    
    if st.sidebar.button("💬 ล้างประวัติการสนทนา"):
        clear_chat_history()


def setup_system(
    embedding_model: str, 
    llm_model: str, 
    temperature: float, 
    max_tokens: int, 
    chunk_size: int = None, 
    chunk_overlap: int = None,
    use_dynamic_chunking: bool = True
):
    """ตั้งค่าระบบ พร้อม Dynamic Chunking"""
    with st.spinner("กำลังตั้งค่าระบบ..."):
        try:
            # สร้าง Document Processor
            if use_dynamic_chunking:
                st.info("✨ ใช้ Dynamic Chunking Strategy")
                st.session_state.document_processor = ThaiDocumentProcessor(
                    use_dynamic_chunking=True
                )
            else:
                st.info(f"📝 ใช้ Manual Chunking ({chunk_size}/{chunk_overlap})")
                st.session_state.document_processor = ThaiDocumentProcessor(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    use_dynamic_chunking=False
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
            import traceback
            st.error(traceback.format_exc())


def display_system_stats():
    """แสดงสถิติระบบ"""
    st.sidebar.subheader("📊 สถิติระบบ")
    
    # แสดงค่าปัจจุบันของ chunk settings
    if st.session_state.document_processor:
        if st.session_state.document_processor.use_dynamic_chunking:
            st.sidebar.success("""
            **Dynamic Chunking เปิดอยู่** ✅
            - 📊 ตาราง: 3000/500
            - 📋 คำอธิบาย: 1500/300  
            - 📝 ทั่วไป: 1000/200
            """)
        else:
            st.sidebar.info(f"""
            **Manual Chunking:**
            - Chunk Size: {st.session_state.document_processor.chunk_size}
            - Overlap: {st.session_state.document_processor.chunk_overlap}
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
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                border-radius: 15px; padding: 1.5rem; margin-bottom: 2rem; 
                border: 1px solid rgba(102, 126, 234, 0.3);">
        <h3 style="color: #ffd93d; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>📁</span>
            <span>อัพโหลดและประมวลผลเอกสาร</span>
        </h3>
        <p style="color: #a0aec0; margin: 0; font-size: 1rem;">
            อัพโหลดเอกสารของคุณเพื่อเริ่มการสนทนา รองรับไฟล์ PDF, DOCX, TXT และอื่นๆ
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ตรวจสอบว่าตั้งค่าระบบแล้วหรือยัง
    if not st.session_state.system_ready or not st.session_state.document_processor:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255, 193, 7, 0.2), rgba(255, 152, 0, 0.2)); 
                    border-radius: 12px; padding: 1.5rem; border-left: 4px solid #ffd93d;">
            <p style="color: #ffe66d; margin: 0; font-size: 1.1rem; font-weight: 600;">
                ⚠️ กรุณาตั้งค่าระบบและเลือก Chunk Size, Chunk Overlap ใน Sidebar ก่อน
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # แสดงการตั้งค่าปัจจุบัน
    st.markdown(f"""
    <div class="info-box">
        <h4 style="color: #ffd93d; margin-bottom: 0.5rem;">⚙️ การตั้งค่าการประมวลผลปัจจุบัน</h4>
        <p style="color: #a0aec0; margin: 0;">
            📏 <strong>Chunk Size:</strong> {st.session_state.document_processor.chunk_size} characters<br>
            🔗 <strong>Chunk Overlap:</strong> {st.session_state.document_processor.chunk_overlap} characters
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📎 เลือกไฟล์เอกสารของคุณ",
        type=['txt', 'pdf', 'docx', 'doc', 'pptx'],
        accept_multiple_files=True,
        help="รองรับไฟล์ .txt, .pdf, .docx, .pptx | สามารถเลือกหลายไฟล์พร้อมกันได้"
    )
    
    if uploaded_files:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(72, 187, 120, 0.2), rgba(56, 161, 105, 0.2)); 
                    border-radius: 10px; padding: 1rem; margin: 1rem 0; border-left: 4px solid #48bb78;">
            <p style="color: #9ae6b4; margin: 0;">
                ✅ <strong>เลือกไฟล์แล้ว {len(uploaded_files)} ไฟล์</strong><br>
                <small>กดปุ่ม "ประมวลผลเอกสาร" เพื่อเริ่มต้น</small>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📤 ประมวลผลเอกสาร", type="primary", use_container_width=True):
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
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                border-radius: 15px; padding: 1.5rem; margin-bottom: 2rem; 
                border: 1px solid rgba(102, 126, 234, 0.3);">
        <h3 style="color: #ffd93d; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>💬</span>
            <span>สนทนากับเอกสารของคุณ</span>
        </h3>
        <p style="color: #a0aec0; margin: 0; font-size: 1rem;">
            ถามคำถามเกี่ยวกับเอกสารที่คุณอัพโหลด และระบบจะตอบด้วยข้อมูลจากเอกสารพร้อมแหล่งอ้างอิง
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.system_ready:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(245, 101, 101, 0.2), rgba(229, 62, 62, 0.2)); 
                    border-radius: 12px; padding: 1.5rem; border-left: 4px solid #f56565;">
            <p style="color: #fc8181; margin: 0; font-size: 1.1rem; font-weight: 600;">
                ⚠️ กรุณาตั้งค่าระบบและอัพโหลดเอกสารก่อนเริ่มการสนทนา
            </p>
        </div>
        """, unsafe_allow_html=True)
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
    # Enhanced Header with Animation
    st.markdown('''
    <h1 class="main-header">
        <span style="position: relative; z-index: 1;">
            🤖 CS RAG Chatbot
        </span>
    </h1>
    ''', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2.5rem;">
        <p style="font-size: 1.2rem; color: #a0aec0; font-weight: 300; margin-bottom: 0.5rem;">
            ระบบ RAG Chatbot สำหรับเอกสารภาษาไทย
        </p>
        <p style="font-size: 1rem; color: #718096; font-weight: 300;">
            <span style="background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600;">
                Powered by
            </span> 
            Local LLM ⚡ Ollama 🚀 LangChain 🎯 FAISS
        </p>
        <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <span style="background: linear-gradient(135deg, #667eea20, #764ba220); padding: 0.5rem 1rem; border-radius: 20px; color: #ffd93d; font-size: 0.9rem; border: 1px solid #667eea40;">
                ✨ Thai Language Support
            </span>
            <span style="background: linear-gradient(135deg, #667eea20, #764ba220); padding: 0.5rem 1rem; border-radius: 20px; color: #ffd93d; font-size: 0.9rem; border: 1px solid #667eea40;">
                🔒 Private & Secure
            </span>
            <span style="background: linear-gradient(135deg, #667eea20, #764ba220); padding: 0.5rem 1rem; border-radius: 20px; color: #ffd93d; font-size: 0.9rem; border: 1px solid #667eea40;">
                ⚡ Fast & Accurate
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # กำหนดค่าเริ่มต้น
    initialize_session_state()
    
    # Sidebar
    setup_sidebar()
    
    # เนื้อหาหลักด้วย Tabs สวยงาม
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background: linear-gradient(135deg, rgba(45, 55, 72, 0.6), rgba(26, 32, 44, 0.6));
            padding: 1rem;
            border-radius: 15px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
            border-radius: 10px;
            color: white;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);
            transform: translateY(-2px);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.4), rgba(118, 75, 162, 0.4));
            transform: translateY(-2px);
        }
    </style>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📁 อัพโหลดเอกสาร", "💬 สนทนา", "ℹ️ ข้อมูลระบบ"])
    
    with tab1:
        upload_documents()
        
        # แสดงไฟล์ที่ประมวลผลแล้วในรูปแบบสวยงาม
        if st.session_state.processed_files:
            st.markdown("""
            <div style="margin-top: 2rem;">
                <h3 style="color: #ffd93d; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span>📋</span>
                    <span>ไฟล์ที่ประมวลผลแล้ว</span>
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, file_info in enumerate(st.session_state.processed_files, 1):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15)); 
                            border-radius: 12px; padding: 1.2rem; margin: 0.8rem 0; 
                            border-left: 4px solid #667eea; transition: all 0.3s ease;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <p style="color: #ffd93d; margin: 0; font-size: 1.1rem; font-weight: 600;">
                                📄 {file_info['filename']}
                            </p>
                            <p style="color: #a0aec0; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                                🧩 {file_info['chunks']} chunks | 📦 ประมวลผลเสร็จสิ้น
                            </p>
                        </div>
                        <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                                    padding: 0.5rem 1rem; border-radius: 20px; color: white; 
                                    font-weight: 600; font-size: 0.9rem;">
                            #{i}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
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