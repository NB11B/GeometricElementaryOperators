[CmdletBinding()]
param(
    [int]$Device = 0,
    [int[]]$Batches = @(1, 32, 256, 1024, 4096),
    [int]$Runs = 30,
    [int]$Warmup = 5,
    [string]$CudaArchitectures = "120",
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

function Resolve-CommandPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $Command) {
        throw "Required command was not found: $Name"
    }
    return $Command.Source
}

function Resolve-NinjaPath {
    $Command = Get-Command ninja.exe -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }

    $EspressifNinjaRoot = Join-Path $env:USERPROFILE ".espressif\tools\ninja"
    if (Test-Path $EspressifNinjaRoot) {
        $Candidate = Get-ChildItem $EspressifNinjaRoot -Filter ninja.exe -Recurse -File |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($Candidate) {
            return $Candidate.FullName
        }
    }

    throw "Ninja was not found in PATH or under $EspressifNinjaRoot"
}

function Resolve-VisualStudio2022Path {
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $VsWhere)) {
        throw "vswhere.exe was not found at $VsWhere"
    }

    $InstallationPath = & $VsWhere `
        -latest `
        -products * `
        -version "[17.0,18.0)" `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath

    if ($LASTEXITCODE -ne 0 -or -not $InstallationPath) {
        throw "A Visual Studio 2022 installation with the x64 C++ toolchain was not found"
    }

    return ($InstallationPath | Select-Object -First 1).Trim()
}

function Import-VisualStudioEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallationPath
    )

    $VsDevCmd = Join-Path $InstallationPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $VsDevCmd)) {
        throw "VsDevCmd.bat was not found at $VsDevCmd"
    }

    $CommandLine = "`"$VsDevCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
    $EnvironmentLines = & $env:ComSpec /d /s /c $CommandLine
    if ($LASTEXITCODE -ne 0) {
        throw "Visual Studio developer environment initialization failed"
    }

    foreach ($Line in $EnvironmentLines) {
        $Separator = $Line.IndexOf("=")
        if ($Separator -gt 0) {
            $Name = $Line.Substring(0, $Separator)
            $Value = $Line.Substring($Separator + 1)
            [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
        }
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

    $NvccPath = Resolve-CommandPath "nvcc.exe"
    $CmakePath = Resolve-CommandPath "cmake.exe"
    $NinjaPath = Resolve-NinjaPath
    $CudaRoot = Split-Path (Split-Path $NvccPath -Parent) -Parent
    $VisualStudioPath = Resolve-VisualStudio2022Path

    Import-VisualStudioEnvironment -InstallationPath $VisualStudioPath
    $ClPath = Resolve-CommandPath "cl.exe"

    @(
        "cmake=$CmakePath"
        "ninja=$NinjaPath"
        "nvcc=$NvccPath"
        "cuda_root=$CudaRoot"
        "cuda_architectures=$CudaArchitectures"
        "visual_studio=$VisualStudioPath"
        "cl=$ClPath"
    ) | Set-Content (Join-Path $EvidenceDirectory "toolchain-paths.txt")

    Invoke-LoggedNativeCommand `
        -Command { & $NvccPath --version } `
        -LogPath (Join-Path $EvidenceDirectory "nvcc-version.txt")

    Invoke-LoggedNativeCommand `
        -Command { & $CmakePath --version } `
        -LogPath (Join-Path $EvidenceDirectory "cmake-version.txt")

    Invoke-LoggedNativeCommand `
        -Command { & $NinjaPath --version } `
        -LogPath (Join-Path $EvidenceDirectory "ninja-version.txt")

    Invoke-LoggedNativeCommand `
        -Command { & $ClPath 2>&1 } `
        -LogPath (Join-Path $EvidenceDirectory "msvc-version.txt")

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
            & $CmakePath `
                -S .\benchmarks\gpu_imu_replay `
                -B $BuildDirectory `
                -G Ninja `
                "-DCMAKE_MAKE_PROGRAM=$NinjaPath" `
                -DCMAKE_BUILD_TYPE=Release `
                "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" `
                "-DCMAKE_CUDA_COMPILER=$NvccPath" `
                "-DCMAKE_CUDA_HOST_COMPILER=$ClPath" `
                "-DCMAKE_CXX_COMPILER=$ClPath" `
                "-DCUDAToolkit_ROOT=$CudaRoot"
        } `
        -LogPath (Join-Path $EvidenceDirectory "configure.log")

    Invoke-LoggedNativeCommand `
        -Command {
            & $CmakePath --build $BuildDirectory --parallel
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
