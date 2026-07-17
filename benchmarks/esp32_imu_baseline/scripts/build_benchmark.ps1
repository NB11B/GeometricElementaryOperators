[CmdletBinding()]
param(
    [string]$IdfExport = "C:\Users\nateb\esp\esp-idf\export.ps1",
    [string]$ExpectedBranch = "benchmark/esp32-imu-clean-v1"
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
    $ExitCode = 0

    if ($HasNativePreference) {
        $PreviousNativePreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }

    try {
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

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$EvidenceDirectory = Join-Path $BenchmarkDirectory "evidence\build-$Timestamp"

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

    if ($ExpectedBranch -and $Branch -ne $ExpectedBranch) {
        Write-Warning "Current branch is '$Branch', not expected branch '$ExpectedBranch'"
    }

    if (Test-Path ".\tools\generate_imu_sparse_schedule.py") {
        Invoke-LoggedNativeCommand `
            -Command {
                python .\tools\generate_imu_sparse_schedule.py `
                    --schedule .\benchmarks\esp32_imu_baseline\schedules\imu_orientation_sparse_v1.json `
                    --output .\benchmarks\esp32_imu_baseline\main\geo_imu_generated_schedule.h `
                    --check
            } `
            -LogPath (Join-Path $EvidenceDirectory "schedule-check.log")

        Invoke-LoggedNativeCommand `
            -Command { python -m unittest tests.test_imu_sparse_schedule } `
            -LogPath (Join-Path $EvidenceDirectory "schedule-tests.log")
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
    }
    finally {
        Pop-Location
    }

    Write-Host ""
    Write-Host "Build validation completed successfully:"
    Write-Host $EvidenceDirectory
}
finally {
    Pop-Location
}
