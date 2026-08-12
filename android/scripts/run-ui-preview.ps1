param(
    [switch]$Capture,
    [switch]$Showcase,
    [ValidateSet(
        "default",
        "home",
        "solo",
        "score-low",
        "score-high",
        "duel",
        "demo",
        "reaction",
        "reaction-overlap",
        "settings",
        "history",
        "exit",
        "new",
        "result"
    )]
    [string]$Screen = "home",
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
$env:GRADLE_USER_HOME = Join-Path $androidRoot ".gradle-user-home"

Push-Location $androidRoot
try {
    if ($Capture) {
        if ([string]::IsNullOrWhiteSpace($Output)) {
            $filename = if ($Screen -eq "default" -or $Screen -eq "home") {
                "current.png"
            } else {
                "$Screen.png"
            }
            $Output = Join-Path `
                $projectRoot `
                "runs\mobile_ui_preview\$filename"
        }
        $absoluteOutput = [System.IO.Path]::GetFullPath($Output)
        $arguments = @(
            "--no-daemon",
            "--offline",
            ":desktop:capturePreview",
            "-PpreviewWidth=$Width",
            "-PpreviewHeight=$Height",
            "-PpreviewScreen=$Screen",
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
            "--offline",
            ":desktop:run",
            "-PpreviewWidth=$Width",
            "-PpreviewHeight=$Height",
            "-PpreviewScreen=$Screen"
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
