[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9.-]+$')]
    [string]$SshHost = 'connect.westb.seetacloud.com',

    [ValidateRange(1, 65535)]
    [int]$SshPort = 17899,

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$SshUser = 'root',

    [ValidateRange(1, 65535)]
    [int]$RemoteDashboardPort = 8765,

    [ValidateRange(1, 65535)]
    [int[]]$LocalPortCandidates = @(8765, 18765, 28765),

    [switch]$UpdateCredential,
    [switch]$Uninstall
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

function Get-CurrentDesktopPath {
    $desktop = [Environment]::GetFolderPath(
        [Environment+SpecialFolder]::DesktopDirectory
    )
    if ([string]::IsNullOrWhiteSpace($desktop)) {
        $registryPath = (
            'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\' +
            'User Shell Folders'
        )
        $rawDesktop = Get-ItemPropertyValue `
            -LiteralPath $registryPath `
            -Name Desktop `
            -ErrorAction SilentlyContinue
        if (-not [string]::IsNullOrWhiteSpace($rawDesktop)) {
            $desktop = [Environment]::ExpandEnvironmentVariables($rawDesktop)
        }
    }
    if ([string]::IsNullOrWhiteSpace($desktop)) {
        $desktop = Join-Path $env:USERPROFILE 'Desktop'
    }

    return [System.IO.Path]::GetFullPath($desktop)
}

function Protect-CredentialFileAcl {
    param([Parameter(Mandatory = $true)][string]$Path)

    $currentSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
    $systemSid = New-Object System.Security.Principal.SecurityIdentifier(
        'S-1-5-18'
    )
    $rights = [System.Security.AccessControl.FileSystemRights]::FullControl
    $allow = [System.Security.AccessControl.AccessControlType]::Allow

    $acl = New-Object System.Security.AccessControl.FileSecurity(
        $Path,
        [System.Security.AccessControl.AccessControlSections]::Access
    )
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($existingRule in @($acl.Access)) {
        [void]$acl.RemoveAccessRuleSpecific($existingRule)
    }
    $acl.AddAccessRule(
        (New-Object System.Security.AccessControl.FileSystemAccessRule(
            $currentSid,
            $rights,
            $allow
        ))
    )
    $acl.AddAccessRule(
        (New-Object System.Security.AccessControl.FileSystemAccessRule(
            $systemSid,
            $rights,
            $allow
        ))
    )
    [System.IO.File]::SetAccessControl($Path, $acl)
}

function Save-ProtectedPassword {
    param([Parameter(Mandatory = $true)][string]$Path)

    Add-Type -AssemblyName System.Security
    $securePassword = Read-Host `
        '请输入云服务器 SSH 密码（仅本机 DPAPI 加密保存）' `
        -AsSecureString
    if ($securePassword.Length -eq 0) {
        throw '密码不能为空。'
    }

    $bstr = [IntPtr]::Zero
    $clearBytes = $null
    $encryptedBytes = $null
    try {
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR(
            $securePassword
        )
        $clearText = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        $clearBytes = [System.Text.Encoding]::UTF8.GetBytes($clearText)
        $clearText = $null
        $encryptedBytes = [System.Security.Cryptography.ProtectedData]::Protect(
            $clearBytes,
            $null,
            [System.Security.Cryptography.DataProtectionScope]::CurrentUser
        )
        [System.IO.File]::WriteAllBytes($Path, $encryptedBytes)
    }
    finally {
        if ($null -ne $clearBytes) {
            [Array]::Clear($clearBytes, 0, $clearBytes.Length)
        }
        if ($null -ne $encryptedBytes) {
            [Array]::Clear($encryptedBytes, 0, $encryptedBytes.Length)
        }
        if ($bstr -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
        $securePassword.Dispose()
    }

    Protect-CredentialFileAcl -Path $Path
}

function Stop-OwnedTunnelForUninstall {
    param([Parameter(Mandatory = $true)][string]$StatePath)

    if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf)) {
        return
    }

    try {
        $state = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 |
            ConvertFrom-Json
        $process = Get-Process -Id ([int]$state.pid) -ErrorAction Stop
        if (
            $process.ProcessName -eq 'ssh' -and
            $process.StartTime.ToUniversalTime().Ticks -eq
                [long]$state.process_start_ticks
        ) {
            Stop-Process -Id $process.Id -ErrorAction Stop
            $process.WaitForExit(5000)
        }
    }
    catch {
        Write-Warning (
            '没有终止保存的 SSH 隧道；安装器不会按进程名模糊结束其他进程。'
        )
    }
}

if ($env:OS -ne 'Windows_NT') {
    throw '桌面快捷应用安装器仅支持 Windows。'
}

$projectRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot '..\..')
)
if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
    throw 'LOCALAPPDATA 未定义，拒绝计算本地安装路径。'
}
$localAppDataRoot = [System.IO.Path]::GetFullPath(
    $env:LOCALAPPDATA
).TrimEnd('\')
$installRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $localAppDataRoot 'FruitMergeRL\TrainingDashboard')
)
$requiredSuffix = '\FruitMergeRL\TrainingDashboard'
if (
    -not $installRoot.StartsWith(
        "$localAppDataRoot\",
        [StringComparison]::OrdinalIgnoreCase
    ) -or
    -not $installRoot.EndsWith(
        $requiredSuffix,
        [StringComparison]::OrdinalIgnoreCase
    )
) {
    throw "拒绝使用不安全的安装路径：$installRoot"
}
$desktopPath = Get-CurrentDesktopPath
$shortcutPath = Join-Path $desktopPath '合成大西瓜训练面板.lnk'
$credentialPath = Join-Path $installRoot 'credential.bin'
$statePath = Join-Path $installRoot 'tunnel.json'

if ($Uninstall) {
    Stop-OwnedTunnelForUninstall -StatePath $statePath
    Remove-Item -LiteralPath $shortcutPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $installRoot `
        -Recurse `
        -Force `
        -ErrorAction SilentlyContinue
    Write-Host '训练面板桌面快捷应用已卸载。'
    exit 0
}

$sshExecutable = Join-Path $env:WINDIR 'System32\OpenSSH\ssh.exe'
if (-not (Test-Path -LiteralPath $sshExecutable -PathType Leaf)) {
    throw '找不到 Windows OpenSSH 客户端，请先安装 OpenSSH Client。'
}

$sourceFiles = [ordered]@{
    'open_training_dashboard.ps1' = Join-Path `
        $PSScriptRoot `
        'open_training_dashboard.ps1'
    'launch_training_dashboard.vbs' = Join-Path `
        $PSScriptRoot `
        'launch_training_dashboard.vbs'
    'DashboardAskPass.cs' = Join-Path `
        $PSScriptRoot `
        'DashboardAskPass.cs'
    'training_dashboard.ico' = Join-Path `
        $projectRoot `
        'assets\dashboard\training_dashboard.ico'
}
foreach ($source in $sourceFiles.Values) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "安装源文件不存在：$source"
    }
}

[void](New-Item -ItemType Directory -Path $installRoot -Force)
[void](New-Item -ItemType Directory -Path $desktopPath -Force)

Copy-Item -LiteralPath $sourceFiles['open_training_dashboard.ps1'] `
    -Destination (Join-Path $installRoot 'open_training_dashboard.ps1') `
    -Force
Copy-Item -LiteralPath $sourceFiles['launch_training_dashboard.vbs'] `
    -Destination (Join-Path $installRoot 'launch_training_dashboard.vbs') `
    -Force
Copy-Item -LiteralPath $sourceFiles['training_dashboard.ico'] `
    -Destination (Join-Path $installRoot 'training_dashboard.ico') `
    -Force

$askPassSource = Get-Content `
    -LiteralPath $sourceFiles['DashboardAskPass.cs'] `
    -Raw `
    -Encoding UTF8
$askPassExecutable = Join-Path $installRoot 'DashboardAskPass.exe'
Remove-Item -LiteralPath $askPassExecutable `
    -Force `
    -ErrorAction SilentlyContinue
Add-Type `
    -TypeDefinition $askPassSource `
    -Language CSharp `
    -OutputAssembly $askPassExecutable `
    -OutputType ConsoleApplication `
    -ReferencedAssemblies @('System.dll', 'System.Security.dll')

if ($UpdateCredential -or -not (Test-Path -LiteralPath $credentialPath)) {
    Save-ProtectedPassword -Path $credentialPath
}
else {
    Protect-CredentialFileAcl -Path $credentialPath
}

$config = [ordered]@{
    schema_version = 1
    ssh_executable = $sshExecutable
    ssh_host = $SshHost
    ssh_port = $SshPort
    ssh_user = $SshUser
    remote_port = $RemoteDashboardPort
    local_port_candidates = @($LocalPortCandidates | Select-Object -Unique)
}
$configJson = $config | ConvertTo-Json -Depth 3
[System.IO.File]::WriteAllText(
    (Join-Path $installRoot 'launcher.config.json'),
    $configJson,
    (New-Object System.Text.UTF8Encoding($false))
)

$shell = New-Object -ComObject WScript.Shell
try {
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = Join-Path $env:WINDIR 'System32\wscript.exe'
    $shortcut.Arguments = (
        '"' +
        (Join-Path $installRoot 'launch_training_dashboard.vbs') +
        '"'
    )
    $shortcut.WorkingDirectory = $installRoot
    $shortcut.IconLocation = (
        (Join-Path $installRoot 'training_dashboard.ico') +
        ',0'
    )
    $shortcut.Description = '自动连接云服务器并打开合成大西瓜训练面板'
    $shortcut.Save()
}
finally {
    if ($null -ne $shell) {
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($shell)
    }
}

Write-Host "安装完成：$shortcutPath"
Write-Host '双击快捷方式会自动建立 SSH 隧道并打开训练面板。'
