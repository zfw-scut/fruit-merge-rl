[CmdletBinding()]
param(
    [switch]$BootstrapSdk
)

$ErrorActionPreference = "Stop"
$AndroidRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$SdkBootstrap = Join-Path $PSScriptRoot "bootstrap-android-sdk.ps1"
$GradleWrapper = Join-Path $AndroidRoot "gradlew.bat"

if ($BootstrapSdk -or -not (Test-Path -LiteralPath (Join-Path $AndroidRoot "local.properties"))) {
    & $SdkBootstrap
}
if (-not (Test-Path -LiteralPath $GradleWrapper)) {
    throw "Gradle wrapper is missing. Run the repository toolchain bootstrap first."
}

$env:GRADLE_USER_HOME = Join-Path $AndroidRoot ".gradle-user-home"
Push-Location $AndroidRoot
try {
    & $GradleWrapper --no-daemon :app:assembleDebug
    if ($LASTEXITCODE -ne 0) {
        throw "Android build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$ApkPath = Join-Path $AndroidRoot "app\build\outputs\apk\debug\app-debug.apk"
if (-not (Test-Path -LiteralPath $ApkPath)) {
    throw "Gradle succeeded but APK was not found at $ApkPath"
}

$ReleaseDirectory = Join-Path $AndroidRoot "release"
$ReleaseApk = Join-Path $ReleaseDirectory "FruitMergeAI-v0.1.0-debug.apk"
New-Item -ItemType Directory -Force -Path $ReleaseDirectory | Out-Null
Copy-Item -LiteralPath $ApkPath -Destination $ReleaseApk -Force

$Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ReleaseApk).Hash
Write-Host "APK: $ReleaseApk"
Write-Host "SHA256: $Hash"
