@echo off
REM Setup Ollama Path and Run Commands
REM สคริปต์สำหรับช่วยใช้งาน Ollama บน Windows

echo Setting up Ollama environment...

REM Add Ollama to PATH for this session
set "OLLAMA_PATH=C:\Users\%USERNAME%\AppData\Local\Programs\Ollama"
set "PATH=%PATH%;%OLLAMA_PATH%"

REM Check if Ollama is installed
if not exist "%OLLAMA_PATH%\ollama.exe" (
    echo Error: Ollama not found at %OLLAMA_PATH%
    echo Please install Ollama from https://ollama.ai
    pause
    exit /b 1
)

echo Ollama found at: %OLLAMA_PATH%

REM Check if Ollama server is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Ollama server...
    start /B "" "%OLLAMA_PATH%\ollama.exe" serve
    echo Waiting for server to start...
    timeout /t 5 /nobreak >nul
) else (
    echo Ollama server is already running
)

REM Check available models
echo.
echo Available models:
"%OLLAMA_PATH%\ollama.exe" list

echo.
echo Ollama setup complete!
echo You can now use: ollama [command]
echo.
echo Common commands:
echo   ollama list                    - Show installed models
echo   ollama pull llama3.1:8b       - Download llama3.1 8B model
echo   ollama pull gemma2:2b         - Download smaller gemma2 2B model
echo   ollama run llama3.1:8b        - Chat with model
echo.

REM Keep the window open
cmd /k "set PATH=%PATH%;%OLLAMA_PATH%"