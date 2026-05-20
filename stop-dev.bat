@echo off
setlocal

taskkill /FI "WINDOWTITLE eq veritas-rag-backend" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq veritas-rag-frontend" /T /F >nul 2>&1

echo Veritas-RAG 开发服务已关闭。

endlocal
