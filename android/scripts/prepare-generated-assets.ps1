[CmdletBinding()]
param(
    [string]$BuildPython
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$AndroidRoot = Join-Path $ProjectRoot "android"
$FontSource = Join-Path $AndroidRoot ".toolchains\fonts\ZCOOLKuaiLe-Regular.ttf"
$FontOutput = Join-Path $AndroidRoot ".generated-assets\fonts"
$ExpectedFontSha256 = "812a6fc1fe54b6d73a419245c32dfeba8aa33104d5be90d1cf6af082007cb71d"
$FontUrl = "https://raw.githubusercontent.com/google/fonts/main/ofl/zcoolkuaile/ZCOOLKuaiLe-Regular.ttf"

if ([string]::IsNullOrWhiteSpace($BuildPython)) {
    $Candidate = Join-Path $env:USERPROFILE "miniconda3\envs\python-torch\python.exe"
    if (Test-Path -LiteralPath $Candidate) {
        $BuildPython = $Candidate
    } else {
        $BuildPython = "python"
    }
}

New-Item -ItemType Directory -Force -Path `
    (Split-Path -Parent $FontSource), $FontOutput | Out-Null
if (-not (Test-Path -LiteralPath $FontSource)) {
    Write-Host "Downloading the OFL mobile UI font..."
    Invoke-WebRequest -Uri $FontUrl -OutFile $FontSource
}
$ActualFontSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $FontSource).Hash.ToLowerInvariant()
if ($ActualFontSha256 -ne $ExpectedFontSha256) {
    throw "Mobile UI font checksum mismatch: $ActualFontSha256"
}

& $BuildPython (Join-Path $ProjectRoot "tools\generate_mobile_ui_font.py") `
    --source $FontSource `
    --output-dir $FontOutput
if ($LASTEXITCODE -ne 0) {
    throw "Mobile UI font generation failed with exit code $LASTEXITCODE"
}

& $BuildPython (Join-Path $ProjectRoot "tools\export_android_model.py")
if ($LASTEXITCODE -ne 0) {
    throw "SAB-T120 ONNX export failed with exit code $LASTEXITCODE"
}

Write-Host "Generated Android assets are ready."
