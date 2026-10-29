@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%..\.."

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv\Scripts\python.exe not found.
  exit /b 1
)

if not exist "%SCRIPT_DIR%launcher.pyw" (
  echo [ERROR] launcher.pyw not found.
  exit /b 1
)

".venv\Scripts\python.exe" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name LiteratureRAG-Launcher ^
  --distpath "%SCRIPT_DIR%dist" ^
  --workpath "%SCRIPT_DIR%build\launcher" ^
  --specpath "%SCRIPT_DIR%build" ^
  "%SCRIPT_DIR%launcher.pyw"

if errorlevel 1 exit /b %errorlevel%
echo Built: %SCRIPT_DIR%dist\LiteratureRAG-Launcher.exe
endlocal
