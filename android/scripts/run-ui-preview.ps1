param(
    [switch]$Capture,
    [switch]$Showcase,
    [ValidateRange(280, 4096)]
    [int]$Width = 560,
    [ValidateRange(560, 8192)]
    [int]$Height = 1120,
    [ValidateRange(1, 600)]
    [int]$CaptureFrames = 12,
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
        $arguments = @(
            "--no-daemon",
            ":desktop:capturePreview",
            "-PpreviewWidth=$Width",
            "-PpreviewHeight=$Height",
            "-PpreviewCaptureFrames=$CaptureFrames",
            "-PpreviewOutput=$absoluteOutput"
        )
        if ($Showcase) {
            $arguments += "-PpreviewShowcase=true"
        }
        & $gradlew @arguments
    } else {
        $arguments = @(
            "--no-daemon",
            ":desktop:run",
            "-PpreviewWidth=$Width",
            "-PpreviewHeight=$Height"
        )
        if ($Showcase) {
            $arguments += "-PpreviewShowcase=true"
        }
        & $gradlew @arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Local UI preview failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
