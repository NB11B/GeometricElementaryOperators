[CmdletBinding()]
param(
    [string]$ExpectedBranch = "research/geometric-operator-kernel-v5-1-acceptance",
    [string]$OutputRoot = ".\local-evidence\v5-1\operator-kernel",
    [string]$BuildRoot = ".\local-evidence\v5-1\build",
    [int]$Seed = 501
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-CommandPath {
    param([Parameter(Mandatory = $true)][string]$Name)
    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $Command) { throw "$Name was not found in PATH" }
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

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$Python = Resolve-CommandPath "python.exe"
$Git = Resolve-CommandPath "git.exe"
$Cmake = Resolve-CommandPath "cmake.exe"
$ResolvedOutput = Join-Path $RepositoryRoot $OutputRoot
$ResolvedBuild = Join-Path $RepositoryRoot $BuildRoot
$Config = Join-Path $RepositoryRoot "experiments\geometric_operator_kernel_v5_1\config.json"
$Logs = Join-Path $ResolvedOutput "logs"

Push-Location $RepositoryRoot
try {
    $Branch = (& $Git branch --show-current).Trim()
    if ($Branch -ne $ExpectedBranch) { throw "Current branch is '$Branch'; expected '$ExpectedBranch'" }

    Remove-Item -LiteralPath $ResolvedOutput -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $ResolvedBuild -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $ResolvedOutput | Out-Null

    Invoke-VisibleNative `
        -Command { & $Python -m py_compile .\tools\geo_operator_v5_1.py .\tools\verify_geo_operator_certificate.py } `
        -LogPath (Join-Path $Logs "py-compile.log")

    Invoke-VisibleNative `
        -Command { & $Python -m unittest tests.test_geo_operator_v5_1 } `
        -LogPath (Join-Path $Logs "python-tests.log")

    Invoke-VisibleNative `
        -Command {
            & $Python .\tools\geo_operator_v5_1.py `
                --config $Config `
                --output-root $ResolvedOutput `
                --seed $Seed
        } `
        -LogPath (Join-Path $Logs "pipeline.log")

    Invoke-VisibleNative `
        -Command { & $Python .\tools\verify_geo_operator_certificate.py (Join-Path $ResolvedOutput "certificates") } `
        -LogPath (Join-Path $Logs "certificate-verify.log")

    Invoke-VisibleNative `
        -Command {
            & $Cmake `
                -S . `
                -B $ResolvedBuild `
                -DGEO_BUILD_TESTS=ON `
                -DGEO_BUILD_BENCHMARKS=OFF `
                -DGEO_USE_DOUBLE=ON
        } `
        -LogPath (Join-Path $Logs "configure.log")

    Invoke-VisibleNative `
        -Command { & $Cmake --build $ResolvedBuild --config Release --parallel } `
        -LogPath (Join-Path $Logs "build.log")

    Invoke-VisibleNative `
        -Command { & $Cmake --build $ResolvedBuild --target test_operator_kernel --config Release } `
        -LogPath (Join-Path $Logs "operator-target.log")

    $CTest = Get-Command ctest.exe -ErrorAction SilentlyContinue
    if ($CTest) {
        Invoke-VisibleNative `
            -Command { & $CTest.Source --test-dir $ResolvedBuild -C Release --output-on-failure -R "^(operator_kernel|abi_consumer|release_hardening)$" } `
            -LogPath (Join-Path $Logs "ctest.log")
    }

    $Commit = (& $Git rev-parse HEAD).Trim()
    $Status = & $Git status --short
    $Commit | Set-Content (Join-Path $ResolvedOutput "commit.txt")
    $Branch | Set-Content (Join-Path $ResolvedOutput "branch.txt")
    $Status | Set-Content (Join-Path $ResolvedOutput "git-status.txt")

    $Manifest = Join-Path $ResolvedOutput "sha256-manifest.csv"
    $ManifestHash = Join-Path $ResolvedOutput "sha256-manifest.sha256.txt"
    Get-ChildItem $ResolvedOutput -Recurse -File |
        Where-Object { $_.FullName -ne $Manifest -and $_.FullName -ne $ManifestHash } |
        Sort-Object FullName |
        ForEach-Object { Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 } |
        Select-Object Path, Algorithm, Hash |
        Export-Csv -LiteralPath $Manifest -NoTypeInformation -Encoding UTF8
    Get-FileHash -LiteralPath $Manifest -Algorithm SHA256 |
        Format-List |
        Out-String |
        Set-Content -LiteralPath $ManifestHash -Encoding UTF8

    Write-Host "GEO_OPERATOR_V5_1_HOST_GATE: PASS"
    Write-Host "Evidence: $ResolvedOutput"
    Write-Host "Manifest: $Manifest"
}
finally {
    Pop-Location
}
