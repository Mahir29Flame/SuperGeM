[CmdletBinding()]
param(
    # Change these defaults only if your venv or icon lives somewhere else.
    [string]$VenvPath = ".venv",
    [string]$IconPath = "assets\icon.ico"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Prefer the currently activated environment; otherwise use the uv default .venv.
if ($env:VIRTUAL_ENV) {
    $EnvironmentPath = $env:VIRTUAL_ENV
}
else {
    if ([System.IO.Path]::IsPathRooted($VenvPath)) {
        $EnvironmentPath = $VenvPath
    }
    else {
        $EnvironmentPath = Join-Path $ProjectRoot $VenvPath
    }
}

$ScriptsPath = Join-Path $EnvironmentPath "Scripts"
$PythonExe = Join-Path $ScriptsPath "python.exe"
$BuilderExe = Join-Path $ScriptsPath "streamlit-desktop-app.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Python not found in venv: $PythonExe. Pass -VenvPath with your uv venv path."
}
if (-not (Test-Path $BuilderExe)) {
    throw "streamlit-desktop-app not found in this venv. Install it with: uv pip install streamlit-desktop-app"
}
if (-not (Test-Path (Join-Path $ProjectRoot "app.py"))) {
    throw "app.py was not found in $ProjectRoot"
}
if (-not (Test-Path (Join-Path $ProjectRoot "READPDF.py"))) {
    throw "READPDF.py was not found beside app.py."
}
if (-not (Test-Path (Join-Path $ProjectRoot "Gemini"))) {
    throw "Gemini folder was not found."
}
if (-not (Test-Path (Join-Path $ProjectRoot "assets"))) {
    throw "assets folder was not found."
}

if ([System.IO.Path]::IsPathRooted($IconPath)) {
    $ResolvedIcon = $IconPath
}
else {
    $ResolvedIcon = Join-Path $ProjectRoot $IconPath
}
if (-not (Test-Path $ResolvedIcon)) {
    throw "Icon not found: $ResolvedIcon. Run with -IconPath 'path\to\your.ico'."
}
if ([System.IO.Path]::GetExtension($ResolvedIcon) -ne ".ico") {
    throw "The app icon must be a real .ico file: $ResolvedIcon"
}

# Verify this is an actual ICO file, not a PNG renamed to .ico.
$FileStream = [System.IO.File]::OpenRead($ResolvedIcon)
try {
    $Header = New-Object byte[] 4
    [void]$FileStream.Read($Header, 0, 4)
}
finally {
    $FileStream.Dispose()
}
if ($Header[0] -ne 0 -or $Header[1] -ne 0 -or $Header[2] -ne 1 -or $Header[3] -ne 0) {
    throw "'$ResolvedIcon' does not appear to be a valid ICO file (ICO header is missing)."
}

Write-Host "Building SuperGeM from: $ProjectRoot" -ForegroundColor Cyan
Write-Host "Using Python: $PythonExe" -ForegroundColor DarkGray
Write-Host "Using icon:   $ResolvedIcon" -ForegroundColor DarkGray

# The key fixes are --paths (so PyInstaller can locate the project-root module)
# and --hidden-import READPDF (so the local module is included in the bundle).
$BuildArgs = @(
    "build",
    (Join-Path $ProjectRoot "app.py"),
    "--name", "SuperGeM",
    "--icon", $ResolvedIcon,
    "--pyinstaller-options",
    "--onefile",
    "--windowed",
    "--noconfirm",
    "--clean",
    "--paths", $ProjectRoot,
    "--hidden-import", "READPDF",
    "--collect-all", "streamlit",
    "--collect-all", "langchain",
    "--collect-all", "langchain_classic",
    "--collect-all", "langchain_community",
    "--collect-all", "langchain_core",
    "--collect-all", "langchain_google_genai",
    "--collect-all", "langchain_text_splitters",
    "--collect-all", "pymupdf",
    "--collect-all", "faiss",
    "--collect-all", "ddgs",
    "--collect-all", "wikipedia",
    "--collect-all", "arxiv",
    "--collect-all", "serpapi",
    "--add-data", "$ProjectRoot\Gemini;Gemini",
    "--add-data", "$ProjectRoot\assets;assets",
    "--copy-metadata", "streamlit",
    "--copy-metadata", "langchain",
    "--copy-metadata", "langchain-classic",
    "--copy-metadata", "langchain-community",
    "--copy-metadata", "langchain-core",
    "--copy-metadata", "langchain-google-genai",
    "--copy-metadata", "langchain-text-splitters"
)

& $BuilderExe @BuildArgs
if ($LASTEXITCODE -ne 0) {
    throw "SuperGeM build failed (exit code $LASTEXITCODE)."
}

$OutputExe = Join-Path $ProjectRoot "dist\SuperGeM.exe"
if (Test-Path $OutputExe) {
    Write-Host "Build complete: $OutputExe" -ForegroundColor Green
}
else {
    Write-Warning "Build command succeeded, but dist\SuperGeM.exe was not found. Check the build output for its actual location."
}
