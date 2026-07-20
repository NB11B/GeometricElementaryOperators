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

function Resolve-ShortPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing path: $Path" }
    $CommandLine = "for %I in (`"$Path`") do @echo %~sI"
    $ShortPath = & $env:ComSpec /d /s /c $CommandLine
    if ($LASTEXITCODE -ne 0 -or -not $ShortPath) { throw "Unable to resolve short path for: $Path" }
    $Resolved = ($ShortPath | Select-Object -First 1).Trim()
    if (-not $Resolved -or -not (Test-Path -LiteralPath $Resolved)) {
        throw "Resolved short path is invalid for '$Path': '$Resolved'"
    }
    return $Resolved
}

function Resolve-NinjaPath {
    $Command = Get-Command ninja.exe -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }
    $Root = Join-Path $env:USERPROFILE ".espressif\tools\ninja"
    if (Test-Path -LiteralPath $Root) {
        $Candidate = Get-ChildItem -LiteralPath $Root -Filter ninja.exe -Recurse -File |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($Candidate) { return $Candidate.FullName }
    }
    throw "Ninja was not found in PATH or under $Root"
}

function Resolve-VisualStudio2022Path {
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $VsWhere)) { throw "vswhere.exe was not found at $VsWhere" }
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
    param([Parameter(Mandatory = $true)][string]$InstallationPath)
    $VsDevCmd = Join-Path $InstallationPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path -LiteralPath $VsDevCmd)) { throw "VsDevCmd.bat was not found at $VsDevCmd" }

    $VsDevCmdShort = Resolve-ShortPath -Path $VsDevCmd
    $TempEnvironment = Join-Path $env:TEMP "geo-operator-vs-env-$PID-$([Guid]::NewGuid().ToString('N')).txt"
    $MinimalPath = @(
        (Join-Path $env:SystemRoot "System32")
        $env:SystemRoot
        (Join-Path $env:SystemRoot "System32\Wbem")
    ) -join ";"

    $env:PATH = $MinimalPath
    Remove-Item Env:INCLUDE -ErrorAction SilentlyContinue
    Remove-Item Env:LIB -ErrorAction SilentlyContinue
    Remove-Item Env:LIBPATH -ErrorAction SilentlyContinue
    Remove-Item Env:__VSCMD_PREINIT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:CUDAHOSTCXX -ErrorAction SilentlyContinue

    try {
        $CommandLine = "call $VsDevCmdShort -no_logo -arch=x64 -host_arch=x64 >nul && set > `"$TempEnvironment`""
        & $env:ComSpec /d /s /c $CommandLine
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $TempEnvironment)) {
            throw "Visual Studio developer environment initialization failed with exit code $LASTEXITCODE"
        }
        $Lines = Get-Content -LiteralPath $TempEnvironment
        if (-not $Lines) { throw "Visual Studio developer environment produced no variables" }
        foreach ($Line in $Lines) {
            $Separator = $Line.IndexOf("=")
            if ($Separator -gt 0) {
                [Environment]::SetEnvironmentVariable(
                    $Line.Substring(0, $Separator),
                    $Line.Substring($Separator + 1),
                    "Process"
                )
            }
        }
    }
    finally {
        Remove-Item -LiteralPath $TempEnvironment -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-LoggedNative {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Command,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $Parent = Split-Path -Parent $LogPath
    if ($Parent) { New-Item -ItemType Directory -Force -Path $Parent | Out-Null }
    $Previous = $ErrorActionPreference
    $PreviousNative = $null
    $HasNative = Test-Path variable:PSNativeCommandUseErrorActionPreference
    $Code = 0
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

function Write-Sha256Manifest {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Manifest,
        [Parameter(Mandatory = $true)][string]$ManifestHash
    )
    Remove-Item -LiteralPath $Manifest -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $ManifestHash -Force -ErrorAction SilentlyContinue
    $Files = @(
        Get-ChildItem -LiteralPath $Root -Recurse -File -ErrorAction Stop |
            Where-Object {
                $_.FullName -ne $Manifest -and
                $_.FullName -ne $ManifestHash
            } |
            Sort-Object FullName
    )
    if ($Files.Count -eq 0) { throw "No evidence files were found under $Root" }
    $Hashes = @(
        $Files |
            ForEach-Object {
                Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 -ErrorAction Stop
            } |
            Select-Object Path, Algorithm, Hash
    )
    $Hashes | Export-Csv -LiteralPath $Manifest -NoTypeInformation -Encoding UTF8
    Get-FileHash -LiteralPath $Manifest -Algorithm SHA256 -ErrorAction Stop |
        Format-List |
        Out-String |
        Set-Content -LiteralPath $ManifestHash -Encoding UTF8
}

if ($Assignments -eq 0) { throw "Assignments must be positive" }

$CallerEnvironment = @{}
Get-ChildItem Env: | ForEach-Object { $CallerEnvironment[$_.Name] = $_.Value }
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$HostGate = Join-Path $ScriptDirectory "run_geo_operator_v5_1_host_gate.ps1"
$PipelineRoot = Join-Path $RepositoryRoot $OutputRoot
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$EvidenceDirectory = Join-Path $BenchmarkDirectory "evidence\operator-$Timestamp"
$BuildDirectory = Join-Path $EvidenceDirectory "build"

New-Item -ItemType Directory -Force -Path $EvidenceDirectory | Out-Null

Push-Location $RepositoryRoot
try {
    $Git = Resolve-CommandPath "git.exe"
    $Cmake = Resolve-CommandPath "cmake.exe"
    $Nvcc = Resolve-CommandPath "nvcc.exe"
    $NvidiaSmi = Resolve-CommandPath "nvidia-smi.exe"
    $Ninja = Resolve-NinjaPath
    $VisualStudio = Resolve-VisualStudio2022Path
    $Branch = (& $Git branch --show-current).Trim()
    if ($Branch -ne $ExpectedBranch) { throw "Current branch is '$Branch'; expected '$ExpectedBranch'" }

    & $HostGate -ExpectedBranch $ExpectedBranch -OutputRoot $OutputRoot
    if (-not $?) { throw "V5.1 host gate failed" }

    Import-VisualStudioEnvironment -InstallationPath $VisualStudio
    $Cl = Resolve-CommandPath "cl.exe"
    $NvccShort = Resolve-ShortPath $Nvcc
    $CudaRoot = Split-Path (Split-Path $Nvcc -Parent) -Parent
    $env:CUDAHOSTCXX = $Cl
    $env:PATH = "$(Join-Path $CudaRoot 'bin');$env:PATH"

    (& $Git rev-parse HEAD).Trim() | Set-Content (Join-Path $EvidenceDirectory "commit.txt")
    $Branch | Set-Content (Join-Path $EvidenceDirectory "branch.txt")
    (& $Git status --short) | Set-Content (Join-Path $EvidenceDirectory "git-status.txt")
    @(
        "cmake=$Cmake"
        "ninja=$Ninja"
        "nvcc=$Nvcc"
        "nvcc_short=$NvccShort"
        "nvidia_smi=$NvidiaSmi"
        "visual_studio=$VisualStudio"
        "cl=$Cl"
        "cuda_root=$CudaRoot"
        "cuda_architectures=$CudaArchitectures"
        "device=$Device"
        "assignments=$Assignments"
    ) | Set-Content (Join-Path $EvidenceDirectory "toolchain-paths.txt")

    Invoke-LoggedNative -Command { & $NvccShort --version } -LogPath (Join-Path $EvidenceDirectory "nvcc-version.txt")
    Invoke-LoggedNative -Command { & $Cmake --version } -LogPath (Join-Path $EvidenceDirectory "cmake-version.txt")
    Invoke-LoggedNative -Command { & $Ninja --version } -LogPath (Join-Path $EvidenceDirectory "ninja-version.txt")
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
                -G Ninja `
                "-DCMAKE_MAKE_PROGRAM=$Ninja" `
                -DCMAKE_BUILD_TYPE=Release `
                "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" `
                "-DCMAKE_CUDA_COMPILER=$NvccShort" `
                "-DCMAKE_CUDA_HOST_COMPILER=$Cl" `
                "-DCMAKE_C_COMPILER=$Cl" `
                "-DCMAKE_CXX_COMPILER=$Cl"
        } `
        -LogPath (Join-Path $EvidenceDirectory "configure.log")

    Invoke-LoggedNative `
        -Command { & $Cmake --build $BuildDirectory --parallel } `
        -LogPath (Join-Path $EvidenceDirectory "build.log")

    $HostExecutable = Get-ChildItem -LiteralPath $BuildDirectory -Recurse -File |
        Where-Object { $_.Name -eq "geo_operator_host_smoke.exe" } |
        Select-Object -First 1
    $CudaExecutable = Get-ChildItem -LiteralPath $BuildDirectory -Recurse -File |
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
    Write-Sha256Manifest -Root $EvidenceDirectory -Manifest $Manifest -ManifestHash $ManifestHash

    if ($Archive) {
        $ArchivePath = "$EvidenceDirectory.zip"
        $ArchiveHash = "$ArchivePath.sha256.txt"
        Remove-Item -LiteralPath $ArchivePath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $ArchiveHash -Force -ErrorAction SilentlyContinue
        Compress-Archive -Path (Join-Path $EvidenceDirectory "*") -DestinationPath $ArchivePath -Force
        Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256 -ErrorAction Stop |
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
    $CurrentNames = @(Get-ChildItem Env: | Select-Object -ExpandProperty Name)
    foreach ($Name in $CurrentNames) {
        if (-not $CallerEnvironment.ContainsKey($Name)) {
            [Environment]::SetEnvironmentVariable($Name, $null, "Process")
        }
    }
    foreach ($Name in $CallerEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($Name, $CallerEnvironment[$Name], "Process")
    }
}
