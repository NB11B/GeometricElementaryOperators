[CmdletBinding()]
param(
    [string]$Port = "COM5",
    [string]$IdfExport = "C:\Users\nateb\esp\esp-idf\export.ps1",
    [int]$ExpectedRows = 181,
    [switch]$SkipFlash
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Convert-NativeOutputToText {
    process {
        if ($_ -is [System.Management.Automation.ErrorRecord]) {
            $_.Exception.Message
        }
        else {
            $_.ToString()
        }
    }
}

function Invoke-LoggedNativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    $PreviousErrorActionPreference = $ErrorActionPreference
    $PreviousNativePreference = $null
    $HasNativePreference = Test-Path variable:PSNativeCommandUseErrorActionPreference

    if ($HasNativePreference) {
        $PreviousNativePreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }

    try {
        # ESP-IDF tools legitimately write warnings and progress messages to
        # stderr. Windows PowerShell wraps those records as NativeCommandError
        # objects. Temporarily keep native stderr non-terminating, convert every
        # stream record to plain text, and rely on the actual process exit code.
        $ErrorActionPreference = "Continue"
        $global:LASTEXITCODE = 0

        & $Command 2>&1 |
            Convert-NativeOutputToText |
            Tee-Object -FilePath $LogPath

        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
        if ($HasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $PreviousNativePreference
        }
    }

    if ($ExitCode -ne 0) {
        throw "Command failed with exit code $ExitCode. See $LogPath"
    }
}

function Invoke-LoggedMonitor {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SerialPort,
        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    $PreviousErrorActionPreference = $ErrorActionPreference
    $PreviousNativePreference = $null
    $HasNativePreference = Test-Path variable:PSNativeCommandUseErrorActionPreference

    if ($HasNativePreference) {
        $PreviousNativePreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }

    try {
        $ErrorActionPreference = "Continue"

        # Do not treat the monitor's COMx/GDB warning as a failure. The monitor
        # intentionally remains interactive and is terminated with Ctrl+].
        idf.py -p $SerialPort monitor 2>&1 |
            Convert-NativeOutputToText |
            Tee-Object -FilePath $LogPath
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
        if ($HasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $PreviousNativePreference
        }
    }
}

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$EvidenceDirectory = Join-Path $BenchmarkDirectory "evidence\fusion-$Timestamp"

New-Item -ItemType Directory -Force -Path $EvidenceDirectory | Out-Null

if (-not (Test-Path $IdfExport)) {
    throw "ESP-IDF export script not found: $IdfExport"
}

Push-Location $RepositoryRoot
try {
    $Commit = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to resolve the current Git commit"
    }

    $Branch = (git branch --show-current).Trim()
    $Status = git status --short

    $Commit | Set-Content (Join-Path $EvidenceDirectory "commit.txt")
    $Branch | Set-Content (Join-Path $EvidenceDirectory "branch.txt")
    $Status | Set-Content (Join-Path $EvidenceDirectory "git-status.txt")

    Write-Host "Repository: $RepositoryRoot"
    Write-Host "Branch:     $Branch"
    Write-Host "Commit:     $Commit"
    Write-Host "Evidence:   $EvidenceDirectory"

    if ($Branch -ne "benchmark/esp32-imu-fused-a1-b1") {
        Write-Warning "Current branch is '$Branch', not benchmark/esp32-imu-fused-a1-b1"
    }

    . $IdfExport

    Invoke-LoggedNativeCommand `
        -Command { idf.py --version } `
        -LogPath (Join-Path $EvidenceDirectory "idf-version.txt")

    Push-Location $BenchmarkDirectory
    try {
        Invoke-LoggedNativeCommand `
            -Command { idf.py fullclean } `
            -LogPath (Join-Path $EvidenceDirectory "fullclean.log")

        Invoke-LoggedNativeCommand `
            -Command { idf.py set-target esp32c6 } `
            -LogPath (Join-Path $EvidenceDirectory "set-target.log")

        Invoke-LoggedNativeCommand `
            -Command { idf.py build } `
            -LogPath (Join-Path $EvidenceDirectory "build.log")

        Copy-Item ".\sdkconfig" `
            (Join-Path $EvidenceDirectory "sdkconfig") -Force

        Get-ChildItem ".\build" -File |
            Where-Object {
                $_.Extension -in ".bin", ".elf", ".map" -or
                $_.Name -eq "flasher_args.json"
            } |
            Copy-Item -Destination $EvidenceDirectory -Force

        if (-not $SkipFlash) {
            Write-Host ""
            Write-Host "Flashing firmware on $Port."

            Invoke-LoggedNativeCommand `
                -Command { idf.py -p $Port flash } `
                -LogPath (Join-Path $EvidenceDirectory "flash.log")
        }
        else {
            Write-Warning "Skipping flash because -SkipFlash was supplied"
        }

        $MonitorLog = Join-Path $EvidenceDirectory "esp32-imu-fusion.log"

        Write-Host ""
        Write-Host "Starting the monitor on $Port."
        Write-Host "The COMx/GDB warning is expected and will not stop this script."
        Write-Host "Wait for GEO_ESP32_IMU_BENCHMARK,status=complete, then press Ctrl+]."
        Write-Host ""

        Invoke-LoggedMonitor `
            -SerialPort $Port `
            -LogPath $MonitorLog

        $CsvPath = Join-Path $EvidenceDirectory "esp32-imu-fusion.csv"
        Select-String "CSV," $MonitorLog |
            ForEach-Object {
                $_.Line -replace "^.*?CSV,", "CSV,"
            } |
            Set-Content $CsvPath

        if (-not (Test-Path $CsvPath)) {
            throw "No CSV output was captured. See $MonitorLog"
        }

        $RowCount = (Get-Content $CsvPath).Count
        $RowCount | Set-Content (Join-Path $EvidenceDirectory "csv-row-count.txt")

        if ($RowCount -ne $ExpectedRows) {
            throw "Expected $ExpectedRows CSV rows, found $RowCount"
        }

        $SummaryMarkdown = Join-Path $EvidenceDirectory "summary.md"
        $SummaryCsv = Join-Path $EvidenceDirectory "summary.csv"

        Invoke-LoggedNativeCommand `
            -Command {
                python ".\scripts\summarize_results.py" $MonitorLog `
                    --markdown-out $SummaryMarkdown `
                    --csv-out $SummaryCsv
            } `
            -LogPath (Join-Path $EvidenceDirectory "summary-validation.log")

        $CompleteMarker = Select-String `
            "GEO_ESP32_IMU_BENCHMARK,status=complete" `
            $MonitorLog
        if (-not $CompleteMarker) {
            throw "The benchmark completion marker is missing"
        }

        $CheckMarker = Select-String `
            "GEO_AB_FUSION_CHECKS,status=pass" `
            $MonitorLog
        if (-not $CheckMarker) {
            throw "The GEO A/B fusion startup check did not pass"
        }

        Write-Host ""
        Write-Host "Fusion benchmark evidence is complete:"
        Write-Host $EvidenceDirectory
    }
    finally {
        Pop-Location
    }
}
finally {
    Pop-Location
}
