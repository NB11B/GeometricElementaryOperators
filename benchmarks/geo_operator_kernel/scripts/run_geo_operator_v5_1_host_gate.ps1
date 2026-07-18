[CmdletBinding()]
param(
    [string]$ExpectedBranch = "research/geometric-operator-kernel-v5-1-acceptance",
    [string]$OutputRoot = ".\local-evidence\v5-1\operator-kernel",
    [string]$Config = ".\experiments\geometric_operator_kernel_v5_1\config.json",
    [int]$Seed = 501
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-CommandPath {
    param([Parameter(Mandatory = $true)][string]$Name)
    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $Command) { throw "Required command was not found: $Name" }
    return $Command.Source
}

function Invoke-VisibleNative {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Command,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $Parent = Split-Path -Parent $LogPath
    if ($Parent) { New-Item -ItemType Directory -Force -Path $Parent | Out-Null }
    $PreviousErrorActionPreference = $ErrorActionPreference
    $PreviousNativePreference = $null
    $HasNativePreference = Test-Path variable:PSNativeCommandUseErrorActionPreference
    $ExitCode = 0
    try {
        $ErrorActionPreference = "Continue"
        if ($HasNativePreference) {
            $PreviousNativePreference = $PSNativeCommandUseErrorActionPreference
            $PSNativeCommandUseErrorActionPreference = $false
        }
        $global:LASTEXITCODE = 0
        & $Command 2>&1 |
            ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message }
                else { $_.ToString() }
            } |
            Tee-Object -FilePath $LogPath
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
        if ($HasNativePreference) { $PSNativeCommandUseErrorActionPreference = $PreviousNativePreference }
    }
    if ($ExitCode -ne 0) { throw "Command failed with exit code $ExitCode. See $LogPath" }
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
    if (-not (Test-Path -LiteralPath $Manifest)) { throw "Manifest was not created: $Manifest" }

    Get-FileHash -LiteralPath $Manifest -Algorithm SHA256 -ErrorAction Stop |
        Format-List |
        Out-String |
        Set-Content -LiteralPath $ManifestHash -Encoding UTF8
    if (-not (Test-Path -LiteralPath $ManifestHash)) { throw "Manifest hash was not created: $ManifestHash" }
}

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$Python = Resolve-CommandPath "python.exe"
$Cmake = Resolve-CommandPath "cmake.exe"
$Git = Resolve-CommandPath "git.exe"
$ResolvedOutput = Join-Path $RepositoryRoot $OutputRoot
$ResolvedConfig = Join-Path $RepositoryRoot $Config
$BuildDirectory = Join-Path $ResolvedOutput "host-build"
$LogDirectory = Join-Path $ResolvedOutput "host-gate-logs"

if (-not (Test-Path -LiteralPath $ResolvedConfig)) { throw "Missing V5.1 config: $ResolvedConfig" }

Push-Location $RepositoryRoot
try {
    $Branch = (& $Git branch --show-current).Trim()
    if ($Branch -ne $ExpectedBranch) { throw "Current branch is '$Branch'; expected '$ExpectedBranch'" }

    Remove-Item -LiteralPath $ResolvedOutput -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $ResolvedOutput | Out-Null
    (& $Git rev-parse HEAD).Trim() | Set-Content (Join-Path $ResolvedOutput "commit.txt")
    $Branch | Set-Content (Join-Path $ResolvedOutput "branch.txt")
    (& $Git status --short) | Set-Content (Join-Path $ResolvedOutput "git-status.txt")

    Invoke-VisibleNative `
        -Command {
            & $Python -m py_compile `
                .\tools\geo_operator_v5_1.py `
                .\tools\geo_operator_v5_1_pipeline.py `
                .\tools\verify_geo_operator_certificate.py
        } `
        -LogPath (Join-Path $LogDirectory "py-compile.log")

    Invoke-VisibleNative `
        -Command {
            & $Python -m unittest `
                tests.test_geo_identity_compiler `
                tests.test_geo_identity_discovery `
                tests.test_geo_identity_grammar_discovery `
                tests.test_identity_result_summarizer `
                tests.test_geo_identity_v4_1_fixed_blade `
                tests.test_geo_identity_v4_2_native `
                tests.test_geo_operator_v5_1
        } `
        -LogPath (Join-Path $LogDirectory "python-tests.log")

    Invoke-VisibleNative `
        -Command {
            & $Python .\tools\geo_operator_v5_1_pipeline.py `
                --config $ResolvedConfig `
                --output-root $ResolvedOutput `
                --seed $Seed
        } `
        -LogPath (Join-Path $LogDirectory "pipeline.log")

    Invoke-VisibleNative `
        -Command {
            & $Python .\tools\verify_geo_operator_certificate.py `
                (Join-Path $ResolvedOutput "certificates")
        } `
        -LogPath (Join-Path $LogDirectory "certificate-verify.log")

    Invoke-VisibleNative `
        -Command {
            & $Cmake `
                -S . `
                -B $BuildDirectory `
                -G "Visual Studio 17 2022" `
                -A x64 `
                -DGEO_BUILD_TESTS=ON `
                -DGEO_BUILD_BENCHMARKS=OFF `
                -DGEO_USE_DOUBLE=ON
        } `
        -LogPath (Join-Path $LogDirectory "cmake-configure.log")

    Invoke-VisibleNative `
        -Command { & $Cmake --build $BuildDirectory --config Release --target test_operator_kernel --parallel } `
        -LogPath (Join-Path $LogDirectory "cmake-build.log")

    $HostExecutable = Get-ChildItem -LiteralPath $BuildDirectory -Recurse -File |
        Where-Object { $_.Name -eq "test_operator_kernel.exe" } |
        Select-Object -First 1
    if (-not $HostExecutable) { throw "Unable to find test_operator_kernel.exe under $BuildDirectory" }

    Invoke-VisibleNative `
        -Command { & $HostExecutable.FullName } `
        -LogPath (Join-Path $LogDirectory "c-kernel-test.log")

    $Manifest = Join-Path $ResolvedOutput "sha256-manifest.csv"
    $ManifestHash = Join-Path $ResolvedOutput "sha256-manifest.sha256.txt"
    Write-Sha256Manifest -Root $ResolvedOutput -Manifest $Manifest -ManifestHash $ManifestHash

    Write-Host "GEO_OPERATOR_V5_1_HOST_GATE: PASS"
    Write-Host "Output: $ResolvedOutput"
    Write-Host "Pipeline report: $(Join-Path $ResolvedOutput 'pipeline-report.md')"
    Write-Host "Manifest: $Manifest"
    Write-Host "Manifest hash: $ManifestHash"
}
finally {
    Pop-Location
}
