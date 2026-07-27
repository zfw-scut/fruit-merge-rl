Option Explicit

Dim shell
Dim fileSystem
Dim appDirectory
Dim powerShellPath
Dim launcherPath
Dim command

Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")

appDirectory = fileSystem.GetParentFolderName(WScript.ScriptFullName)
powerShellPath = shell.ExpandEnvironmentStrings( _
    "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" _
)
launcherPath = fileSystem.BuildPath( _
    appDirectory, _
    "open_training_dashboard.ps1" _
)

command = Quote(powerShellPath) _
    & " -NoLogo -NoProfile -NonInteractive" _
    & " -ExecutionPolicy Bypass -WindowStyle Hidden -File " _
    & Quote(launcherPath)

shell.Run command, 0, False

Function Quote(value)
    Quote = Chr(34) & value & Chr(34)
End Function
