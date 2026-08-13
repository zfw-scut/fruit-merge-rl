[CmdletBinding()]
param(
    [string]$SdkRoot
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($SdkRoot)) {
    $SdkRoot = Join-Path $PSScriptRoot "..\.toolchains\android-sdk"
}
$SdkRoot = [System.IO.Path]::GetFullPath($SdkRoot)
$ToolsVersion = "15859902"
$ToolsSha256 = "90ae805d20434428bffcb699c290860f19bb5f66a67e6b330067e3de801fb04a"
$ToolsUrl = "https://dl.google.com/android/repository/commandlinetools-win-$ToolsVersion`_latest.zip"
$ToolchainsRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.toolchains"))
$ArchivePath = Join-Path $ToolchainsRoot "commandlinetools-win-$ToolsVersion.zip"
$ExtractRoot = Join-Path $ToolchainsRoot "cmdline-tools-extract"
$SdkManager = Join-Path $SdkRoot "cmdline-tools\latest\bin\sdkmanager.bat"

function Assert-StrictChildPath {
    param(
        [Parameter(Mandatory)]
        [string]$Child,
        [Parameter(Mandatory)]
        [string]$Parent
    )

    $ResolvedChild = [System.IO.Path]::GetFullPath($Child)
    $ResolvedParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    $RequiredPrefix = $ResolvedParent + [System.IO.Path]::DirectorySeparatorChar
    if (-not $ResolvedChild.StartsWith(
        $RequiredPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Unsafe path '$ResolvedChild': expected a child of '$ResolvedParent'"
    }
}

New-Item -ItemType Directory -Force -Path $ToolchainsRoot, $SdkRoot | Out-Null

if (-not (Test-Path -LiteralPath $SdkManager)) {
    if (-not (Test-Path -LiteralPath $ArchivePath)) {
        Write-Host "Downloading Android command-line tools $ToolsVersion..."
        Invoke-WebRequest -Uri $ToolsUrl -OutFile $ArchivePath
    }

    $ActualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ArchivePath).Hash.ToLowerInvariant()
    if ($ActualSha256 -ne $ToolsSha256) {
        throw "Android command-line tools checksum mismatch: $ActualSha256"
    }

    if (Test-Path -LiteralPath $ExtractRoot) {
        Assert-StrictChildPath -Child $ExtractRoot -Parent $ToolchainsRoot
        Remove-Item -LiteralPath $ExtractRoot -Recurse -Force
    }
    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $ExtractRoot
    $LatestRoot = Join-Path $SdkRoot "cmdline-tools\latest"
    New-Item -ItemType Directory -Force -Path $LatestRoot | Out-Null
    Copy-Item -Path (Join-Path $ExtractRoot "cmdline-tools\*") -Destination $LatestRoot -Recurse -Force
}

$env:ANDROID_HOME = $SdkRoot
$env:ANDROID_SDK_ROOT = $SdkRoot

Write-Host "Accepting Android SDK licenses..."
$LicenseAnswers = 1..20 | ForEach-Object { "y" }
$LicenseAnswers | & $SdkManager --sdk_root=$SdkRoot --licenses | Out-Host

Write-Host "Installing reproducible Android SDK packages..."
& $SdkManager --sdk_root=$SdkRoot "platform-tools" "platforms;android-35" "build-tools;35.0.0"
if ($LASTEXITCODE -ne 0) {
    throw "sdkmanager failed with exit code $LASTEXITCODE"
}

$EscapedSdkRoot = $SdkRoot.Replace("\", "\\")
$LocalProperties = Join-Path $PSScriptRoot "..\local.properties"
Set-Content -LiteralPath $LocalProperties -Encoding ASCII -Value "sdk.dir=$EscapedSdkRoot"

Write-Host "Android SDK ready at $SdkRoot"
