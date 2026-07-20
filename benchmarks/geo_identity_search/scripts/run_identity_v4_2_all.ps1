[CmdletBinding()]
param(
    [string]$ExpectedBranch = "research/geometric-identity-engine-v4-2",
    [string]$OutputRoot = ".\local-evidence\v4-2\signature-matrix",
    [int]$PythonChecks = 128,
    [int]$LoweringIterations = 2000,
    [int]$MaxRelationsPerSignature = 0,
    [switch]$Cuda,
    [int]$Device = 0,
    [UInt64]$Assignments = 65536,
    [int]$BlockSize = 256,
    [string]$CudaArchitectures = "120",
    [switch]$Archive
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$Config = Join-Path $RepositoryRoot "experiments\geometric_identity_engine_v4_2\config.json"
$GrammarJson = Join-Path $RepositoryRoot "local-evidence\v4-2\grammar-preflight.json"
$GrammarMarkdown = Join-Path $RepositoryRoot "local-evidence\v4-2\grammar-preflight.md"
$HostGate = Join-Path $ScriptDirectory "run_identity_v4_2_host_gate.ps1"
$CudaGate = Join-Path $ScriptDirectory "run_identity_v4_2_cuda_clean.ps1"
$Python = (Get-Command python.exe -ErrorAction Stop).Source
$Git = (Get-Command git.exe -ErrorAction Stop).Source

Push-Location $RepositoryRoot
try {
    $Branch = (& $Git branch --show-current).Trim()
    if ($Branch -ne $ExpectedBranch) { throw "Current branch is '$Branch'; expected '$ExpectedBranch'" }

    & $Python .\tools\geo_identity_v4_2_grammar_preflight.py `
        --config $Config `
        --output-json $GrammarJson `
        --markdown-out $GrammarMarkdown
    if ($LASTEXITCODE -ne 0) { throw "V4.2 grammar preflight failed" }

    & $HostGate `
        -ExpectedBranch $ExpectedBranch `
        -OutputRoot $OutputRoot `
        -PythonChecks $PythonChecks `
        -LoweringIterations $LoweringIterations `
        -MaxRelationsPerSignature $MaxRelationsPerSignature
    if (-not $?) { throw "V4.2 host gate failed" }

    if ($Cuda) {
        & $CudaGate `
            -Device $Device `
            -Assignments $Assignments `
            -CpuChecks ([UInt64]$PythonChecks) `
            -BlockSize $BlockSize `
            -CudaArchitectures $CudaArchitectures `
            -ExpectedBranch $ExpectedBranch `
            -CorpusRoot $OutputRoot `
            -MaxRelationsPerSignature $MaxRelationsPerSignature `
            -Archive:$Archive
        if (-not $?) { throw "V4.2 CUDA gate failed" }
    }

    Write-Host "V4_2_ALL: PASS"
}
finally {
    Pop-Location
}
