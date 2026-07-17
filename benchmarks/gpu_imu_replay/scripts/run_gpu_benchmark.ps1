[CmdletBinding()]
param(
    [int]$Device = 0,
    [int[]]$Batches = @(1, 32, 256, 1024, 4096),
    [int]$Runs = 30,
    [int]$Warmup = 5,
    [string]$ExpectedBranch = "benchmark/gpu-imu-replay-v1"
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
$EvidenceDirectory = Join-Path $BenchmarkDirectory "evidence\gpu-$Timestamp"
$BuildDirectory = Join-Path $EvidenceDirectory "build"

New-Item -ItemType Directory -Force -Path $EvidenceDirectory | Out-Null

Push-Location $RepositoryRoot
try {
    $Commit = (git rev-parse HEAD).Trim()
    $Branch = (git branch --show-current).Trim()
    $Status = git status --short

    $Commit | Set-Content (Join-Path $EvidenceDirectory "commit.txt")
    $Branch | Set-Content (Join-Path $EvidenceDirectory "branch.txt")
    $Status | Set-Content (Join-Path $EvidenceDirectory "git-status.txt")

    if ($Branch -ne $ExpectedBranch) {
        throw "Current branch is '$Branch'; expected '$ExpectedBranch'"
    }

    Invoke-LoggedNativeCommand `
        -Command { nvcc --version } `
        -LogPath (Join-Path $EvidenceDirectory "nvcc-version.txt")

    Invoke-LoggedNativeCommand `
        -Command {
            nvidia-smi --query-gpu=index,name,uuid,driver_version,compute_cap,clocks.current.graphics,clocks.current.memory,power.limit,memory.total --format=csv
        } `
        -LogPath (Join-Path $EvidenceDirectory "gpu-identity.csv")

    Invoke-LoggedNativeCommand `
        -Command {
            python .\tools\generate_imu_cuda_schedule.py `
                --schedule .\benchmarks\esp32_imu_baseline\schedules\imu_orientation_sparse_v1.json `
                --output .\benchmarks\gpu_imu_replay\generated\geo_imu_generated_schedule.cuh `
                --check
        } `
        -LogPath (Join-Path $EvidenceDirectory "generator-check.log")

    Invoke-LoggedNativeCommand `
        -Command {
            python -m unittest `
                tests.test_imu_sparse_schedule `
                tests.test_imu_cuda_schedule
        } `
        -LogPath (Join-Path $EvidenceDirectory "generator-tests.log")

    Invoke-LoggedNativeCommand `
        -Command {
            cmake -S .\benchmarks\gpu_imu_replay `
                -B $BuildDirectory `
                -DCMAKE_BUILD_TYPE=Release
        } `
        -LogPath (Join-Path $EvidenceDirectory "configure.log")

    Invoke-LoggedNativeCommand `
        -Command {
            cmake --build $BuildDirectory --config Release --parallel
        } `
        -LogPath (Join-Path $EvidenceDirectory "build.log")

    $Executable = Get-ChildItem $BuildDirectory -Recurse -File |
        Where-Object { $_.Name -eq "geo_gpu_imu_replay.exe" } |
        Select-Object -First 1

    if (-not $Executable) {
        throw "Unable to find geo_gpu_imu_replay.exe under $BuildDirectory"
    }

    $AllCsvLines = New-Object System.Collections.Generic.List[string]
    foreach ($Batch in $Batches) {
        if ($Batch -le 0) {
            throw "Batch values must be positive"
        }

        $RunLog = Join-Path $EvidenceDirectory "gpu-batch-$Batch.log"
        $RunCsv = Join-Path $EvidenceDirectory "gpu-batch-$Batch.csv"

        Invoke-LoggedNativeCommand `
            -Command {
                & $Executable.FullName `
                    --device $Device `
                    --batch $Batch `
                    --runs $Runs `
                    --warmup $Warmup `
                    --csv $RunCsv
            } `
            -LogPath $RunLog

        $Lines = Get-Content $RunCsv
        if ($Lines.Count -ne (1 + 2 * $Runs)) {
            throw "Batch $Batch expected $(1 + 2 * $Runs) CSV lines, found $($Lines.Count)"
        }

        if ($AllCsvLines.Count -eq 0) {
            foreach ($Line in $Lines) {
                $AllCsvLines.Add($Line)
            }
        }
        else {
            foreach ($Line in $Lines | Select-Object -Skip 1) {
                $AllCsvLines.Add($Line)
            }
        }
    }

    $AggregateCsv = Join-Path $EvidenceDirectory "gpu-imu-replay-all.csv"
    $Utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($AggregateCsv, $AllCsvLines, $Utf8)

    Get-ChildItem $EvidenceDirectory -Recurse -File |
        Sort-Object FullName |
        Get-FileHash -Algorithm SHA256 |
        Select-Object Path, Algorithm, Hash |
        Export-Csv `
            (Join-Path $EvidenceDirectory "sha256-manifest.csv") `
            -NoTypeInformation `
            -Encoding UTF8

    Write-Host ""
    Write-Host "GPU benchmark evidence is complete:"
    Write-Host $EvidenceDirectory
}
finally {
    Pop-Location
}
