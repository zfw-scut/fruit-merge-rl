[CmdletBinding()]
param(
    [switch]$BootstrapSdk,
    [switch]$RerunTasks,
    [string]$BuildPython
)

$ErrorActionPreference = "Stop"
$AndroidRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$PrepareAssets = Join-Path $PSScriptRoot "prepare-generated-assets.ps1"
$SdkBootstrap = Join-Path $PSScriptRoot "bootstrap-android-sdk.ps1"

& $PrepareAssets -BuildPython $BuildPython

if ($BootstrapSdk -or -not (Test-Path -LiteralPath (Join-Path $AndroidRoot "local.properties"))) {
    & $SdkBootstrap
}

$Gradle = Get-ChildItem `
    (Join-Path $env:USERPROFILE ".gradle\wrapper\dists\gradle-8.10.2-bin") `
    -Filter gradle.bat -Recurse -ErrorAction SilentlyContinue `
    | Select-Object -First 1 -ExpandProperty FullName
if ([string]::IsNullOrWhiteSpace($Gradle)) {
    throw "Gradle 8.10.2 is unavailable; run the Gradle wrapper once or install Gradle 8.10.2."
}

Push-Location $AndroidRoot
try {
    $Arguments = @("--no-daemon")
    if ($RerunTasks) {
        $Arguments += "--rerun-tasks"
    }
    $Arguments += ":app:assembleDebug"
    & $Gradle @Arguments
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
$ReleaseApk = Join-Path $ReleaseDirectory "FruitMergeAI-SAB-T120-debug.apk"
New-Item -ItemType Directory -Force -Path $ReleaseDirectory | Out-Null
Copy-Item -LiteralPath $ApkPath -Destination $ReleaseApk -Force
Write-Host "APK: $ReleaseApk"
Write-Host "SHA256: $((Get-FileHash -Algorithm SHA256 -LiteralPath $ReleaseApk).Hash)"
