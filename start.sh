#!/bin/bash
# สำหรับการรันแอปพลิเคชันบน macOS/Linux

echo "Starting Thai RAG Chatbot..."
echo

echo "Checking Ollama..."
if ! ollama list >/dev/null 2>&1; then
    echo "Error: Ollama is not running!"
    echo "Please start Ollama first: ollama serve"
    exit 1
fi

echo
echo "Starting Streamlit app..."
streamlit run app.py