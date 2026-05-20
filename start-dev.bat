@echo off
setlocal

set "ROOT_DIR=E:\PythonProject\cook-rag"
set "BACKEND_DIR=%ROOT_DIR%\backend"
set "FRONTEND_DIR=%ROOT_DIR%\frontend"
set "CONDA_ACTIVATE=D:\Software\anaconda3\Scripts\activate.bat"
set "BACKEND_PYTHONPATH=%BACKEND_DIR%;%ROOT_DIR%"

start "cook-rag-backend" cmd /k "title cook-rag-backend && call "%CONDA_ACTIVATE%" cook-rag-1 && set "PYTHONPATH=%BACKEND_PYTHONPATH%" && cd /d "%BACKEND_DIR%" && python main.py"
start "cook-rag-frontend" cmd /k "title cook-rag-frontend && cd /d "%FRONTEND_DIR%" && npm run dev"

endlocal
