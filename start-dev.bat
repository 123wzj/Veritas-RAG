@echo off
setlocal

set "ROOT_DIR=E:\PythonProject\Veritas-RAG"
set "BACKEND_DIR=%ROOT_DIR%\backend"
set "FRONTEND_DIR=%ROOT_DIR%\frontend"
set "CONDA_ACTIVATE=D:\Software\anaconda3\Scripts\activate.bat"
set "BACKEND_PYTHONPATH=%BACKEND_DIR%;%ROOT_DIR%"

start "veritas-rag-backend" cmd /k "title veritas-rag-backend && call "%CONDA_ACTIVATE%" cook-rag-1 && set "PYTHONPATH=%BACKEND_PYTHONPATH%" && cd /d "%BACKEND_DIR%" && python main.py"
start "veritas-rag-frontend" cmd /k "title veritas-rag-frontend && cd /d "%FRONTEND_DIR%" && npm run dev"

endlocal
