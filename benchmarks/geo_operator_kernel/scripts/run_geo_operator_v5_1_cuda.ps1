[CmdletBinding()]
param(
    [int]$Device = 0,
    [UInt64]$Assignments = 1024,
    [string]$CudaArchitectures = "120",
    [string]$ExpectedBranch = "research/geometric-operator-kernel-v5-1-acceptance",
    [string]$OutputRoot = ".\local-evidence\v5-1\operator-kernel",
    [switch]$Archive
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-CommandPath {
    param([Parameter(Mandatory = $true)][string]$Name)
    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $Command) { throw "Required command was not found: $Name" }
    return $Command.Source
}

function Invoke-LoggedNative {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Command,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $Previous = $ErrorActionPreference
    $PreviousNative = $null
    $HasNative = Test-Path variable:PSNativeCommandUseErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        if ($HasNative) {
            $PreviousNative = $PSNativeCommandUseErrorActionPreference
            $PSNativeCommandUseErrorActionPreference = $false
        }
        $global:LASTEXITCODE = 0
        & $Command 2>&1 |
            ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message }
                else { $_.ToString() }
            } |
            Tee-Object -FilePath $LogPath
        $Code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $Previous
        if ($HasNative) { $PSNativeCommandUseErrorActionPreference = $PreviousNative }
    }
    if ($Code -ne 0) { throw "Command failed with exit code $Code. See $LogPath" }
}

if ($Assignments -eq 0) { throw "Assignments must be positive" }

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$HostGate = Join-Path $ScriptDirectory "run_geo_operator_v5_1_host_gate.ps1"
$PipelineRoot = Join-Path $RepositoryRoot $OutputRoot
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$EvidenceDirectory = Join-Path $BenchmarkDirectory "evidence\operator-$Timestamp"
$BuildDirectory = Join-Path $EvidenceDirectory "build"
$Git = Resolve-CommandPath "git.exe"
$Cmake = Resolve-CommandPath "cmake.exe"
$Nvcc = Resolve-CommandPath "nvcc.exe"
$NvidiaSmi = Resolve-CommandPath "nvidia-smi.exe"

New-Item -ItemType Directory -Force -Path $EvidenceDirectory | Out-Null

Push-Location $RepositoryRoot
try {
    $Branch = (& $Git branch --show-current).Trim()
    if ($Branch -ne $ExpectedBranch) { throw "Current branch is '$Branch'; expected '$ExpectedBranch'" }

    & $HostGate -ExpectedBranch $ExpectedBranch -OutputRoot $OutputRoot
    if (-not $?) { throw "V5.1 host gate failed" }

    (& $Git rev-parse HEAD).Trim() | Set-Content (Join-Path $EvidenceDirectory "commit.txt")
    $Branch | Set-Content (Join-Path $EvidenceDirectory "branch.txt")
    (& $Git status --short) | Set-Content (Join-Path $EvidenceDirectory "git-status.txt")

    Invoke-LoggedNative -Command { & $Nvcc --version } -LogPath (Join-Path $EvidenceDirectory "nvcc-version.txt")
    Invoke-LoggedNative -Command { & $Cmake --version } -LogPath (Join-Path $EvidenceDirectory "cmake-version.txt")
    Invoke-LoggedNative `
        -Command {
            & $NvidiaSmi `
                --query-gpu=index,name,uuid,driver_version,compute_cap,clocks.current.graphics,clocks.current.memory,memory.total `
                --format=csv
        } `
        -LogPath (Join-Path $EvidenceDirectory "gpu-identity.csv")

    Invoke-LoggedNative `
        -Command {
            & $Cmake `
                -S .\benchmarks\geo_operator_kernel `
                -B $BuildDirectory `
                -G "Visual Studio 17 2022" `
                -A x64 `
                "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
        } `
        -LogPath (Join-Path $EvidenceDirectory "configure.log")

    Invoke-LoggedNative `
        -Command { & $Cmake --build $BuildDirectory --config Release --parallel } `
        -LogPath (Join-Path $EvidenceDirectory "build.log")

    $HostExecutable = Get-ChildItem $BuildDirectory -Recurse -File |
        Where-Object { $_.Name -eq "geo_operator_host_smoke.exe" } |
        Select-Object -First 1
    $CudaExecutable = Get-ChildItem $BuildDirectory -Recurse -File |
        Where-Object { $_.Name -eq "geo_operator_cuda_smoke.exe" } |
        Select-Object -First 1
    if (-not $HostExecutable) { throw "Unable to find geo_operator_host_smoke.exe" }
    if (-not $CudaExecutable) { throw "Unable to find geo_operator_cuda_smoke.exe" }

    Invoke-LoggedNative -Command { & $HostExecutable.FullName } -LogPath (Join-Path $EvidenceDirectory "host-smoke.log")
    Invoke-LoggedNative `
        -Command { & $CudaExecutable.FullName $Assignments $Device } `
        -LogPath (Join-Path $EvidenceDirectory "cuda-smoke.log")

    $PipelineEvidence = Join-Path $EvidenceDirectory "pipeline"
    New-Item -ItemType Directory -Force -Path $PipelineEvidence | Out-Null
    Copy-Item -Path (Join-Path $PipelineRoot "*") -Destination $PipelineEvidence -Recurse -Force

    $Manifest = Join-Path $EvidenceDirectory "sha256-manifest.csv"
    $ManifestHash = Join-Path $EvidenceDirectory "sha256-manifest.sha256.txt"
    Get-ChildItem $EvidenceDirectory -Recurse -File |
        Where-Object { $_.FullName -ne $Manifest -and $_.FullName -ne $ManifestHash } |
        Sort-Object FullName |
        ForEach-Object { Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 } |
        Select-Object Path, Algorithm, Hash |
        Export-Csv -LiteralPath $Manifest -NoTypeInformation -Encoding UTF8
    Get-FileHash -LiteralPath $Manifest -Algorithm SHA256 |
        Format-List |
        Out-String |
        Set-Content -LiteralPath $ManifestHash -Encoding UTF8

    if ($Archive) {
        $ArchivePath = "$EvidenceDirectory.zip"
        $ArchiveHash = "$ArchivePath.sha256.txt"
        Compress-Archive -Path (Join-Path $EvidenceDirectory "*") -DestinationPath $ArchivePath -Force
        Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256 |
            Format-List |
            Out-String |
            Set-Content -LiteralPath $ArchiveHash -Encoding UTF8
    }

    Write-Host "V5.1 operator evidence is complete:"
    Write-Host $EvidenceDirectory
    Write-Host "Manifest: $Manifest"
    Write-Host "Manifest hash: $ManifestHash"
    if ($Archive) {
        Write-Host "Archive: $ArchivePath"
        Write-Host "Archive hash: $ArchiveHash"
    }
    Write-Host "GEO_OPERATOR_V5_1,status=complete"
}
finally {
    Pop-Location
}
