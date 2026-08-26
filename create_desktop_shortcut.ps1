$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Launcher = Join-Path $ProjectRoot "dist\LiteratureRAG-Launcher.exe"
if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
    throw "Launcher EXE not found: $Launcher. Run build_launcher.cmd first."
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Literature RAG.lnk"
$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $Launcher
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Description = "Launch the local Literature RAG Production Demo"
$Shortcut.Save()

Write-Host "Created: $ShortcutPath"
