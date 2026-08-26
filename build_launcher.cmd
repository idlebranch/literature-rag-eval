@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv\Scripts\python.exe not found.
  exit /b 1
)

if not exist "launcher.pyw" (
  echo [ERROR] launcher.pyw not found.
  exit /b 1
)

".venv\Scripts\python.exe" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name LiteratureRAG-Launcher ^
  --distpath dist ^
  --workpath build\launcher ^
  --specpath build ^
  launcher.pyw

if errorlevel 1 exit /b %errorlevel%
echo Built: %CD%\dist\LiteratureRAG-Launcher.exe
endlocal
