param(
    [switch]$Capture,
    [ValidateRange(280, 4096)]
    [int]$Width = 560,
    [ValidateRange(560, 8192)]
    [int]$Height = 1120,
    [string]$Output
)

$ErrorActionPreference = "Stop"
$androidRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $androidRoot
$gradlew = Join-Path $androidRoot "gradlew.bat"

Push-Location $androidRoot
try {
    if ($Capture) {
        if ([string]::IsNullOrWhiteSpace($Output)) {
            $Output = Join-Path $projectRoot "runs\mobile_ui_preview\current.png"
        }
        $absoluteOutput = [System.IO.Path]::GetFullPath($Output)
        & $gradlew --no-daemon :desktop:capturePreview `
            "-PpreviewWidth=$Width" `
            "-PpreviewHeight=$Height" `
            "-PpreviewOutput=$absoluteOutput"
    } else {
        & $gradlew --no-daemon :desktop:run `
            "-PpreviewWidth=$Width" `
            "-PpreviewHeight=$Height"
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Local UI preview failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
