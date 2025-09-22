# สำหรับการรันแอปพลิเคชัน
@echo off
echo Starting Thai RAG Chatbot...
echo.

echo Checking Ollama...
ollama list
if %errorlevel% neq 0 (
    echo Error: Ollama is not running!
    echo Please start Ollama first: ollama serve
    pause
    exit /b 1
)

echo.
echo Starting Streamlit app...
streamlit run app.py

pause