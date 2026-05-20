@echo off
setlocal

taskkill /FI "WINDOWTITLE eq cook-rag-backend" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq cook-rag-frontend" /T /F >nul 2>&1

echo cook-rag 开发服务已关闭。

endlocal
