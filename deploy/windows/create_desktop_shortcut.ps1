$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$Launcher = Join-Path $ScriptDir "dist\LiteratureRAG-Launcher.exe"
if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
    throw "Launcher EXE not found: $Launcher. Run build_launcher.cmd first."
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Literature RAG.lnk"
$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $Launcher
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Description = "启动本地文献 RAG 演示"
$Shortcut.Save()

Write-Host "Created: $ShortcutPath"
