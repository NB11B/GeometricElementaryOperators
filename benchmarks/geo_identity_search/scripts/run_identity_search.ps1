[CmdletBinding()]
param(
    [int]$Device = 0,
    [UInt64]$Assignments = 1048576,
    [UInt64]$CpuChecks = 4096,
    [int]$BlockSize = 256,
    [string]$CudaArchitectures = "120",
    [string]$ExpectedBranch = "research/geometric-identity-engine-v1",
    [switch]$Archive
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

function Resolve-GitPath {
    $Command = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }

    $Candidates = @(
        (Join-Path $env:ProgramFiles "Git\cmd\git.exe"),
        (Join-Path $env:ProgramFiles "Git\bin\git.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Git\cmd\git.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Git\cmd\git.exe")
    )

    foreach ($Candidate in $Candidates) {
        if ($Candidate -and (Test-Path $Candidate)) {
            return $Candidate
        }
    }

    throw "Git for Windows was not found in PATH or standard installation locations"
}

function Resolve-ShortPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        throw "Cannot resolve a short path for missing path: $Path"
    }

    $CommandLine = "for %I in (`"$Path`") do @echo %~sI"
    $ShortPath = & $env:ComSpec /d /s /c $CommandLine
    if ($LASTEXITCODE -ne 0 -or -not $ShortPath) {
        throw "Unable to resolve DOS short path for: $Path"
    }

    $Resolved = ($ShortPath | Select-Object -First 1).Trim()
    if (-not $Resolved -or -not (Test-Path $Resolved)) {
        throw "Resolved short path is invalid for '$Path': '$Resolved'"
    }
    return $Resolved
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

    $VsDevCmdShort = Resolve-ShortPath $VsDevCmd
    $TempEnvironment = Join-Path $env:TEMP "geo-identity-vs-env-$PID-$([Guid]::NewGuid().ToString('N')).txt"

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

    try {
        $CommandLine = "call $VsDevCmdShort -no_logo -arch=x64 -host_arch=x64 >nul && set > `"$TempEnvironment`""
        & $env:ComSpec /d /s /c $CommandLine
        $ExitCode = $LASTEXITCODE

        if ($ExitCode -ne 0 -or -not (Test-Path $TempEnvironment)) {
            throw "Visual Studio developer environment initialization failed with exit code $ExitCode"
        }

        $EnvironmentLines = Get-Content $TempEnvironment
        if (-not $EnvironmentLines) {
            throw "Visual Studio developer environment produced no environment variables"
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
    finally {
        Remove-Item $TempEnvironment -Force -ErrorAction SilentlyContinue
    }
}

if ($Assignments -eq 0) {
    throw "Assignments must be positive"
}
if ($CpuChecks -gt $Assignments) {
    throw "CpuChecks cannot exceed Assignments"
}
if ($BlockSize -lt 1 -or $BlockSize -gt 1024) {
    throw "BlockSize must be in [1,1024]"
}

$CallerEnvironment = @{
    PATH = $env:PATH
    INCLUDE = $env:INCLUDE
    LIB = $env:LIB
    LIBPATH = $env:LIBPATH
    CUDAHOSTCXX = $env:CUDAHOSTCXX
    __VSCMD_PREINIT_PATH = $env:__VSCMD_PREINIT_PATH
}

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$EvidenceDirectory = Join-Path $BenchmarkDirectory "evidence\identity-$Timestamp"
$BuildDirectory = Join-Path $EvidenceDirectory "build"

New-Item -ItemType Directory -Force -Path $EvidenceDirectory | Out-Null

Push-Location $RepositoryRoot
try {
    $GitPath = Resolve-GitPath
    $NvccPath = Resolve-CommandPath "nvcc.exe"
    $CmakePath = Resolve-CommandPath "cmake.exe"
    $NinjaPath = Resolve-NinjaPath
    $PythonPath = Resolve-CommandPath "python.exe"
    $NvidiaSmiPath = Resolve-CommandPath "nvidia-smi.exe"
    $CudaRoot = Split-Path (Split-Path $NvccPath -Parent) -Parent
    $VisualStudioPath = Resolve-VisualStudio2022Path

    $Commit = (& $GitPath rev-parse HEAD).Trim()
    $Branch = (& $GitPath branch --show-current).Trim()
    $Status = & $GitPath status --short

    $Commit | Set-Content (Join-Path $EvidenceDirectory "commit.txt")
    $Branch | Set-Content (Join-Path $EvidenceDirectory "branch.txt")
    $Status | Set-Content (Join-Path $EvidenceDirectory "git-status.txt")

    if ($Branch -ne $ExpectedBranch) {
        throw "Current branch is '$Branch'; expected '$ExpectedBranch'"
    }

    Import-VisualStudioEnvironment -InstallationPath $VisualStudioPath
    $ClPath = Resolve-CommandPath "cl.exe"

    $NvccShortPath = Resolve-ShortPath $NvccPath
    $ClShortPath = Resolve-ShortPath $ClPath
    $env:CUDAHOSTCXX = $ClShortPath
    $env:PATH = "$(Join-Path $CudaRoot 'bin');$env:PATH"

    @(
        "git=$GitPath"
        "cmake=$CmakePath"
        "ninja=$NinjaPath"
        "python=$PythonPath"
        "nvidia_smi=$NvidiaSmiPath"
        "nvcc=$NvccPath"
        "nvcc_short=$NvccShortPath"
        "cuda_root=$CudaRoot"
        "cuda_architectures=$CudaArchitectures"
        "visual_studio=$VisualStudioPath"
        "cl=$ClPath"
        "cl_short=$ClShortPath"
        "CUDAHOSTCXX=$env:CUDAHOSTCXX"
    ) | Set-Content (Join-Path $EvidenceDirectory "toolchain-paths.txt")

    Invoke-LoggedNativeCommand `
        -Command { & $NvccShortPath --version } `
        -LogPath (Join-Path $EvidenceDirectory "nvcc-version.txt")

    Invoke-LoggedNativeCommand `
        -Command { & $CmakePath --version } `
        -LogPath (Join-Path $EvidenceDirectory "cmake-version.txt")

    Invoke-LoggedNativeCommand `
        -Command { & $NinjaPath --version } `
        -LogPath (Join-Path $EvidenceDirectory "ninja-version.txt")

    Invoke-LoggedNativeCommand `
        -Command {
            & $NvidiaSmiPath --query-gpu=index,name,uuid,driver_version,compute_cap,clocks.current.graphics,clocks.current.memory,memory.total --format=csv
        } `
        -LogPath (Join-Path $EvidenceDirectory "gpu-identity.csv")

    $IdentityArguments = @(
        "--identity", ".\experiments\geometric_identity_engine\corpus\01_vector_square_scalar.json",
        "--identity", ".\experiments\geometric_identity_engine\corpus\02_reverse_product_order.json",
        "--identity", ".\experiments\geometric_identity_engine\corpus\03_vector_wedge_antisymmetry.json",
        "--identity", ".\experiments\geometric_identity_engine\corpus\04_vector_product_commutativity_false.json",
        "--identity", ".\experiments\geometric_identity_engine\corpus\05_mutated_reverse_order_false.json",
        "--output", ".\benchmarks\geo_identity_search\generated\geo_identity_corpus.cuh",
        "--python-checks", "256"
    )

    Invoke-LoggedNativeCommand `
        -Command {
            & $PythonPath .\tools\geo_identity_compiler.py @IdentityArguments
        } `
        -LogPath (Join-Path $EvidenceDirectory "generator-check.log")

    Invoke-LoggedNativeCommand `
        -Command {
            & $PythonPath -m unittest tests.test_geo_identity_compiler
        } `
        -LogPath (Join-Path $EvidenceDirectory "generator-tests.log")

    Invoke-LoggedNativeCommand `
        -Command {
            & $CmakePath `
                -S .\benchmarks\geo_identity_search `
                -B $BuildDirectory `
                -G Ninja `
                "-DCMAKE_MAKE_PROGRAM=$NinjaPath" `
                -DCMAKE_BUILD_TYPE=Release `
                "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" `
                "-DCMAKE_CUDA_COMPILER=$NvccShortPath" `
                "-DCMAKE_CUDA_HOST_COMPILER=$ClShortPath" `
                "-DCMAKE_CXX_COMPILER=$ClShortPath"
        } `
        -LogPath (Join-Path $EvidenceDirectory "configure.log")

    Invoke-LoggedNativeCommand `
        -Command {
            & $CmakePath --build $BuildDirectory --parallel
        } `
        -LogPath (Join-Path $EvidenceDirectory "build.log")

    $HostSmoke = Get-ChildItem $BuildDirectory -Recurse -File |
        Where-Object { $_.Name -eq "geo_identity_host_smoke.exe" } |
        Select-Object -First 1
    $GpuExecutable = Get-ChildItem $BuildDirectory -Recurse -File |
        Where-Object { $_.Name -eq "geo_identity_search.exe" } |
        Select-Object -First 1

    if (-not $HostSmoke) {
        throw "Unable to find geo_identity_host_smoke.exe under $BuildDirectory"
    }
    if (-not $GpuExecutable) {
        throw "Unable to find geo_identity_search.exe under $BuildDirectory"
    }

    Invoke-LoggedNativeCommand `
        -Command { & $HostSmoke.FullName } `
        -LogPath (Join-Path $EvidenceDirectory "host-smoke.log")

    $RawCsv = Join-Path $EvidenceDirectory "identity-search.csv"
    Invoke-LoggedNativeCommand `
        -Command {
            & $GpuExecutable.FullName `
                --device $Device `
                --assignments $Assignments `
                --block-size $BlockSize `
                --cpu-checks $CpuChecks `
                --csv $RawCsv
        } `
        -LogPath (Join-Path $EvidenceDirectory "identity-search.log")

    $SummaryCsv = Join-Path $EvidenceDirectory "identity-summary.csv"
    $SummaryMarkdown = Join-Path $EvidenceDirectory "summary.md"
    Invoke-LoggedNativeCommand `
        -Command {
            & $PythonPath .\benchmarks\geo_identity_search\scripts\summarize_identity_results.py `
                $RawCsv `
                --summary-csv $SummaryCsv `
                --markdown-out $SummaryMarkdown
        } `
        -LogPath (Join-Path $EvidenceDirectory "validation.log")

    $Manifest = Join-Path $EvidenceDirectory "sha256-manifest.csv"
    $ManifestHash = Join-Path $EvidenceDirectory "sha256-manifest.sha256.txt"
    Remove-Item $Manifest -Force -ErrorAction SilentlyContinue
    Remove-Item $ManifestHash -Force -ErrorAction SilentlyContinue

    $FilesToHash = @(
        Get-ChildItem $EvidenceDirectory -Recurse -File |
            Where-Object {
                $_.FullName -ne $Manifest -and
                $_.FullName -ne $ManifestHash
            } |
            Sort-Object FullName
    )

    $Hashes = @(
        $FilesToHash |
            ForEach-Object {
                Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 -ErrorAction Stop
            } |
            Select-Object Path, Algorithm, Hash
    )

    $Hashes | Export-Csv `
        -LiteralPath $Manifest `
        -NoTypeInformation `
        -Encoding UTF8

    Get-FileHash -LiteralPath $Manifest -Algorithm SHA256 |
        Format-List |
        Out-String |
        Set-Content -LiteralPath $ManifestHash -Encoding UTF8

    if ($Archive) {
        $ArchivePath = "$EvidenceDirectory.zip"
        $ArchiveHash = "$ArchivePath.sha256.txt"
        Remove-Item $ArchivePath -Force -ErrorAction SilentlyContinue
        Remove-Item $ArchiveHash -Force -ErrorAction SilentlyContinue

        Compress-Archive `
            -Path (Join-Path $EvidenceDirectory "*") `
            -DestinationPath $ArchivePath `
            -Force

        Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256 |
            Format-List |
            Out-String |
            Set-Content -LiteralPath $ArchiveHash -Encoding UTF8
    }

    Write-Host ""
    Write-Host "Geometric identity evidence is complete:"
    Write-Host $EvidenceDirectory
    Write-Host "Manifest: $Manifest"
    Write-Host "Manifest hash: $ManifestHash"
    if ($Archive) {
        Write-Host "Archive: $ArchivePath"
        Write-Host "Archive hash: $ArchiveHash"
    }
}
finally {
    Pop-Location
    foreach ($Name in $CallerEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable(
            $Name,
            $CallerEnvironment[$Name],
            "Process"
        )
    }
}
