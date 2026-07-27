[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$NoDialogs,
    [ValidateRange(3, 120)]
    [int]$StartupTimeoutSeconds = 25
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$script:AppDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:ConfigPath = Join-Path $script:AppDirectory 'launcher.config.json'
$script:CredentialPath = Join-Path $script:AppDirectory 'credential.bin'
$script:AskPassPath = Join-Path $script:AppDirectory 'DashboardAskPass.exe'
$script:StatePath = Join-Path $script:AppDirectory 'tunnel.json'
$script:LauncherLogPath = Join-Path $script:AppDirectory 'launcher.log'
$script:SshLogPath = Join-Path $script:AppDirectory 'ssh.log'
$script:StartedProcess = $null
$script:TunnelReady = $false

function Write-LauncherLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    try {
        if (
            (Test-Path -LiteralPath $script:LauncherLogPath) -and
            (Get-Item -LiteralPath $script:LauncherLogPath).Length -gt 1MB
        ) {
            $oldLog = "$($script:LauncherLogPath).old"
            Remove-Item -LiteralPath $oldLog -Force -ErrorAction SilentlyContinue
            Move-Item -LiteralPath $script:LauncherLogPath -Destination $oldLog -Force
        }

        $timestamp = [DateTime]::UtcNow.ToString('o')
        Add-Content -LiteralPath $script:LauncherLogPath `
            -Value "[$timestamp] $Message" `
            -Encoding UTF8
    }
    catch {
        # Logging must not hide the actual launcher outcome.
    }
}

function Show-LauncherError {
    param([Parameter(Mandatory = $true)][string]$Message)

    if ($NoDialogs) {
        Write-Error $Message
        return
    }

    try {
        Add-Type -AssemblyName System.Windows.Forms
        [void][System.Windows.Forms.MessageBox]::Show(
            $Message,
            '合成大西瓜训练面板',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        )
    }
    catch {
        Write-Error $Message
    }
}

function Test-DashboardEndpoint {
    param([Parameter(Mandatory = $true)][int]$Port)

    $response = $null
    $reader = $null

    try {
        $uri = "http://127.0.0.1:${Port}/api/health"
        $request = [System.Net.HttpWebRequest]::Create($uri)
        $request.AllowAutoRedirect = $false
        $request.Proxy = $null
        $request.Timeout = 2000
        $request.ReadWriteTimeout = 2000
        $request.UserAgent = 'FruitMergeDashboardLauncher/1'

        $response = [System.Net.HttpWebResponse]$request.GetResponse()
        if ([int]$response.StatusCode -ne 200) {
            return $false
        }
        if ($response.ContentLength -gt 65536) {
            return $false
        }

        $reader = New-Object System.IO.StreamReader(
            $response.GetResponseStream(),
            [System.Text.Encoding]::UTF8,
            $true,
            1024,
            $false
        )
        $health = $reader.ReadToEnd() | ConvertFrom-Json
        $properties = @($health.PSObject.Properties.Name)
        $required = @(
            'schema_version',
            'status',
            'service_ok',
            'data_fresh',
            'generated_at'
        )
        foreach ($name in $required) {
            if ($properties -notcontains $name) {
                return $false
            }
        }

        return [int]$health.schema_version -eq 1
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $reader) {
            $reader.Dispose()
        }
        if ($null -ne $response) {
            $response.Dispose()
        }
    }
}

function Test-LocalPortAvailable {
    param([Parameter(Mandatory = $true)][int]$Port)

    $listener = $null
    try {
        $address = [System.Net.IPAddress]::Parse('127.0.0.1')
        $listener = New-Object System.Net.Sockets.TcpListener($address, $Port)
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $listener) {
            $listener.Stop()
        }
    }
}

function Get-PortOwnerDescription {
    param([Parameter(Mandatory = $true)][int]$Port)

    try {
        $connection = Get-NetTCPConnection `
            -LocalAddress 127.0.0.1 `
            -LocalPort $Port `
            -State Listen `
            -ErrorAction Stop |
            Select-Object -First 1
        if ($null -eq $connection) {
            return "端口 $Port"
        }

        $owner = Get-Process -Id $connection.OwningProcess -ErrorAction Stop
        return "端口 $Port（PID $($owner.Id)，$($owner.ProcessName)）"
    }
    catch {
        return "端口 $Port"
    }
}

function Read-LauncherConfig {
    if (-not (Test-Path -LiteralPath $script:ConfigPath -PathType Leaf)) {
        throw '缺少 launcher.config.json，请重新运行桌面快捷应用安装脚本。'
    }

    $config = Get-Content -LiteralPath $script:ConfigPath -Raw -Encoding UTF8 |
        ConvertFrom-Json
    $properties = @($config.PSObject.Properties.Name)
    $required = @(
        'schema_version',
        'ssh_executable',
        'ssh_host',
        'ssh_port',
        'ssh_user',
        'remote_port',
        'local_port_candidates'
    )
    foreach ($name in $required) {
        if ($properties -notcontains $name) {
            throw "启动器配置缺少字段：$name"
        }
    }

    if ([int]$config.schema_version -ne 1) {
        throw '启动器配置版本不受支持，请重新安装桌面快捷应用。'
    }
    if ([string]$config.ssh_host -notmatch '^[A-Za-z0-9.-]+$') {
        throw 'SSH 主机配置格式无效。'
    }
    if ([string]$config.ssh_user -notmatch '^[A-Za-z0-9._-]+$') {
        throw 'SSH 用户配置格式无效。'
    }
    if (
        [int]$config.ssh_port -lt 1 -or
        [int]$config.ssh_port -gt 65535 -or
        [int]$config.remote_port -lt 1 -or
        [int]$config.remote_port -gt 65535
    ) {
        throw 'SSH 或远端面板端口配置无效。'
    }
    if (-not (Test-Path -LiteralPath $config.ssh_executable -PathType Leaf)) {
        throw "找不到 OpenSSH：$($config.ssh_executable)"
    }

    $ports = @($config.local_port_candidates)
    if ($ports.Count -eq 0) {
        throw '至少需要配置一个本地端口。'
    }
    foreach ($port in $ports) {
        if ([int]$port -lt 1 -or [int]$port -gt 65535) {
            throw "本地端口配置无效：$port"
        }
    }

    return $config
}

function Read-OwnedTunnel {
    param([Parameter(Mandatory = $true)]$Config)

    if (-not (Test-Path -LiteralPath $script:StatePath -PathType Leaf)) {
        return $null
    }

    try {
        $state = Get-Content -LiteralPath $script:StatePath -Raw -Encoding UTF8 |
            ConvertFrom-Json
        $process = Get-Process -Id ([int]$state.pid) -ErrorAction Stop
        $actualTicks = $process.StartTime.ToUniversalTime().Ticks
        if (
            $process.ProcessName -ne 'ssh' -or
            $actualTicks -ne [long]$state.process_start_ticks
        ) {
            return $null
        }

        $stateProperties = @($state.PSObject.Properties.Name)
        $targetProperties = @(
            'ssh_host',
            'ssh_port',
            'ssh_user',
            'remote_port'
        )
        $targetMatches = $true
        foreach ($name in $targetProperties) {
            if ($stateProperties -notcontains $name) {
                $targetMatches = $false
            }
        }
        if ($targetMatches) {
            $targetMatches = (
                [string]$state.ssh_host -eq [string]$Config.ssh_host -and
                [int]$state.ssh_port -eq [int]$Config.ssh_port -and
                [string]$state.ssh_user -eq [string]$Config.ssh_user -and
                [int]$state.remote_port -eq [int]$Config.remote_port
            )
        }

        return [PSCustomObject]@{
            Process = $process
            Port = [int]$state.local_port
            TargetMatches = $targetMatches
        }
    }
    catch {
        return $null
    }
}

function Remove-StaleTunnelState {
    param([Parameter(Mandatory = $true)]$Config)

    $owned = Read-OwnedTunnel -Config $Config
    if ($null -eq $owned -and (Test-Path -LiteralPath $script:StatePath)) {
        Remove-Item -LiteralPath $script:StatePath -Force -ErrorAction SilentlyContinue
    }
}

function Save-TunnelState {
    param(
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)][int]$Port,
        [Parameter(Mandatory = $true)]$Config
    )

    $state = [ordered]@{
        schema_version = 1
        pid = $Process.Id
        process_start_ticks = $Process.StartTime.ToUniversalTime().Ticks
        local_port = $Port
        ssh_host = [string]$Config.ssh_host
        ssh_port = [int]$Config.ssh_port
        ssh_user = [string]$Config.ssh_user
        remote_port = [int]$Config.remote_port
        created_at = [DateTime]::UtcNow.ToString('o')
    }
    $json = $state | ConvertTo-Json
    $temporaryPath = "$($script:StatePath).tmp"
    [System.IO.File]::WriteAllText(
        $temporaryPath,
        $json,
        (New-Object System.Text.UTF8Encoding($false))
    )
    Move-Item -LiteralPath $temporaryPath `
        -Destination $script:StatePath `
        -Force
}

function Get-SshLogTail {
    if (-not (Test-Path -LiteralPath $script:SshLogPath -PathType Leaf)) {
        return ''
    }

    try {
        $tail = (Get-Content -LiteralPath $script:SshLogPath -Tail 20) -join "`n"
        $tail = $tail -replace '[\x00-\x08\x0B\x0C\x0E-\x1F]', ''
        if ($tail.Length -gt 4000) {
            return $tail.Substring($tail.Length - 4000)
        }
        return $tail
    }
    catch {
        return ''
    }
}

function ConvertTo-NativeArgument {
    param([Parameter(Mandatory = $true)][string]$Value)

    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    if ($Value.Contains('"')) {
        throw '本地路径包含不受支持的双引号。'
    }
    return '"' + $Value + '"'
}

function Start-SshTunnel {
    param(
        [Parameter(Mandatory = $true)]$Config,
        [Parameter(Mandatory = $true)][int]$LocalPort
    )

    if (-not (Test-Path -LiteralPath $script:CredentialPath -PathType Leaf)) {
        throw '缺少本机加密凭据，请重新运行安装脚本并录入密码。'
    }
    if (-not (Test-Path -LiteralPath $script:AskPassPath -PathType Leaf)) {
        throw '缺少 ASKPASS 辅助程序，请重新运行桌面快捷应用安装脚本。'
    }

    if (
        (Test-Path -LiteralPath $script:SshLogPath) -and
        (Get-Item -LiteralPath $script:SshLogPath).Length -gt 1MB
    ) {
        Remove-Item -LiteralPath "$($script:SshLogPath).old" `
            -Force `
            -ErrorAction SilentlyContinue
        Move-Item -LiteralPath $script:SshLogPath `
            -Destination "$($script:SshLogPath).old" `
            -Force
    }

    $forward = "127.0.0.1:${LocalPort}:127.0.0.1:$([int]$Config.remote_port)"
    $target = "$([string]$Config.ssh_user)@$([string]$Config.ssh_host)"
    $arguments = @(
        '-F', 'NUL',
        '-N',
        '-T',
        '-n',
        '-L', $forward,
        '-p', ([int]$Config.ssh_port).ToString(),
        '-E', $script:SshLogPath,
        '-o', 'ExitOnForwardFailure=yes',
        '-o', 'StrictHostKeyChecking=accept-new',
        '-o', 'ConnectTimeout=12',
        '-o', 'ConnectionAttempts=1',
        '-o', 'NumberOfPasswordPrompts=1',
        '-o', 'PreferredAuthentications=password,keyboard-interactive',
        '-o', 'ServerAliveInterval=30',
        '-o', 'ServerAliveCountMax=3',
        '-o', 'LogLevel=ERROR',
        $target
    )

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = [string]$Config.ssh_executable
    $startInfo.Arguments = (
        $arguments |
        ForEach-Object { ConvertTo-NativeArgument ([string]$_) }
    ) -join ' '
    $startInfo.WorkingDirectory = $script:AppDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $startInfo.EnvironmentVariables['SSH_ASKPASS'] = $script:AskPassPath
    $startInfo.EnvironmentVariables['SSH_ASKPASS_REQUIRE'] = 'force'
    $startInfo.EnvironmentVariables['DISPLAY'] = 'fruit-merge-dashboard'

    $process = [System.Diagnostics.Process]::Start($startInfo)
    if ($null -eq $process) {
        throw 'OpenSSH 进程未能启动。'
    }

    return $process
}

function Open-DashboardBrowser {
    param([Parameter(Mandatory = $true)][int]$Port)

    $url = "http://127.0.0.1:${Port}/"
    if ($NoBrowser) {
        Write-Output $url
        return
    }

    try {
        Start-Process $url
    }
    catch {
        throw "隧道已经建立，但浏览器启动失败。请手动打开：$url"
    }
}

$mutex = $null
$hasMutex = $false

try {
    $mutex = New-Object System.Threading.Mutex(
        $false,
        'Local\FruitMergeRL.TrainingDashboard.Launcher'
    )
    try {
        $hasMutex = $mutex.WaitOne(0)
    }
    catch [System.Threading.AbandonedMutexException] {
        $hasMutex = $true
        Write-LauncherLog 'recovered an abandoned launcher mutex'
    }
    if (-not $hasMutex) {
        Write-LauncherLog 'another launcher instance is already running'
        exit 0
    }

    Write-LauncherLog 'launch requested'
    $config = Read-LauncherConfig
    Remove-StaleTunnelState -Config $config

    $ownedTunnel = Read-OwnedTunnel -Config $config
    if ($null -ne $ownedTunnel -and -not $ownedTunnel.TargetMatches) {
        Write-LauncherLog (
            "stopping owned tunnel PID $($ownedTunnel.Process.Id) after target change"
        )
        try {
            $ownedTunnel.Process.Kill()
            [void]$ownedTunnel.Process.WaitForExit(5000)
        }
        finally {
            Remove-Item -LiteralPath $script:StatePath `
                -Force `
                -ErrorAction SilentlyContinue
        }
        $ownedTunnel = $null
    }

    if ($null -ne $ownedTunnel) {
        for ($attempt = 0; $attempt -lt 6; $attempt += 1) {
            if (Test-DashboardEndpoint -Port $ownedTunnel.Port) {
                Write-LauncherLog "reusing owned tunnel PID $($ownedTunnel.Process.Id)"
                $script:TunnelReady = $true
                Open-DashboardBrowser -Port $ownedTunnel.Port
                exit 0
            }
            Start-Sleep -Milliseconds 500
        }

        throw (
            "SSH 隧道进程仍在运行（PID $($ownedTunnel.Process.Id)），" +
            '但远端训练面板没有响应。请确认云服务器上的面板服务仍在运行后重试。'
        )
    }

    $localPort = $null
    $occupied = @()
    foreach ($candidate in @($config.local_port_candidates)) {
        $candidatePort = [int]$candidate
        if (Test-DashboardEndpoint -Port $candidatePort) {
            $occupied += (
                "端口 $candidatePort（存在未由此入口管理的训练面板）"
            )
            continue
        }
        if (Test-LocalPortAvailable -Port $candidatePort) {
            $localPort = $candidatePort
            break
        }
        $occupied += Get-PortOwnerDescription -Port $candidatePort
    }
    if ($null -eq $localPort) {
        throw (
            '面板候选端口均被其他程序占用：' +
            ($occupied -join '；') +
            '。启动器没有终止这些进程。'
        )
    }

    $script:StartedProcess = Start-SshTunnel `
        -Config $config `
        -LocalPort $localPort
    Save-TunnelState `
        -Process $script:StartedProcess `
        -Port $localPort `
        -Config $config
    Write-LauncherLog (
        "started ssh PID $($script:StartedProcess.Id) on port $localPort"
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($script:StartedProcess.HasExited) {
            $detail = Get-SshLogTail
            if ([string]::IsNullOrWhiteSpace($detail)) {
                $detail = 'OpenSSH 没有留下额外错误信息。'
            }
            throw (
                "SSH 隧道建立失败（退出码 $($script:StartedProcess.ExitCode)）。`n`n" +
                $detail
            )
        }

        if (Test-DashboardEndpoint -Port $localPort) {
            $script:TunnelReady = $true
            Write-LauncherLog "dashboard became healthy on port $localPort"
            Open-DashboardBrowser -Port $localPort
            exit 0
        }

        Start-Sleep -Milliseconds 500
    }

    throw (
        "SSH 已连接，但在 $StartupTimeoutSeconds 秒内没有检测到训练面板。" +
        '请确认云服务器上的 dashboard 服务监听 127.0.0.1:8765。'
    )
}
catch {
    $message = $_.Exception.Message
    Write-LauncherLog ("launch failed: " + ($message -replace '[\r\n]+', ' '))

    if (
        $null -ne $script:StartedProcess -and
        -not $script:TunnelReady
    ) {
        try {
            if (-not $script:StartedProcess.HasExited) {
                $script:StartedProcess.Kill()
                [void]$script:StartedProcess.WaitForExit(5000)
            }
        }
        catch {
            Write-LauncherLog 'failed to stop the tunnel created by this launch'
        }
        Remove-Item -LiteralPath $script:StatePath `
            -Force `
            -ErrorAction SilentlyContinue
    }

    Show-LauncherError -Message $message
    exit 1
}
finally {
    if ($hasMutex -and $null -ne $mutex) {
        $mutex.ReleaseMutex()
    }
    if ($null -ne $mutex) {
        $mutex.Dispose()
    }
}
