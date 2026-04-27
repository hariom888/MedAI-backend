@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_pipeline.ps1" %*
set EXIT_CODE=%ERRORLEVEL%

endlocal & exit /b %EXIT_CODE%
