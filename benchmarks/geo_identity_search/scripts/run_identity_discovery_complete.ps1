[CmdletBinding()]
param(
    [int]$Device = 0,
    [UInt64]$Assignments = 262144,
    [UInt64]$CpuChecks = 1024,
    [int]$BlockSize = 256,
    [int[]]$Primes = @(65521, 65519, 65497, 32749),
    [int]$MaxMutations = 8,
    [int]$PrecheckAssignments = 4096,
    [int]$PolynomialTermLimit = 200000,
    [string]$CudaArchitectures = "120",
    [string]$ExpectedBranch = "research/geometric-identity-engine-v2",
    [switch]$Archive
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$Runner = Join-Path $ScriptDirectory "run_identity_discovery.ps1"
$Finalizer = Join-Path $ScriptDirectory "finalize_identity_discovery.ps1"
$Started = Get-Date
$NeedsFinalize = $false

try {
    & $Runner `
        -Device $Device `
        -Assignments $Assignments `
        -CpuChecks $CpuChecks `
        -BlockSize $BlockSize `
        -Primes $Primes `
        -MaxMutations $MaxMutations `
        -PrecheckAssignments $PrecheckAssignments `
        -PolynomialTermLimit $PolynomialTermLimit `
        -CudaArchitectures $CudaArchitectures `
        -ExpectedBranch $ExpectedBranch `
        -Archive:$false
}
catch {
    if ($_.Exception.Message -notlike "*Unable to resolve the base runner evidence directory*") {
        throw
    }
    Write-Warning "Base CUDA evidence passed, but the discovery handoff marker was not captured. Recovering from the generated evidence directory."
    $NeedsFinalize = $true
}

if (-not $NeedsFinalize) {
    return
}

$EvidenceDirectory = Get-ChildItem (Join-Path $BenchmarkDirectory "evidence") -Directory |
    Where-Object {
        $_.LastWriteTime -ge $Started.AddMinutes(-1) -and
        (Test-Path (Join-Path $_.FullName "identity-search.csv"))
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $EvidenceDirectory) {
    throw "Unable to locate the completed CUDA evidence directory for recovery"
}

& $Finalizer `
    -EvidenceDirectory $EvidenceDirectory.FullName `
    -Primes $Primes `
    -MaxMutations $MaxMutations `
    -PrecheckAssignments $PrecheckAssignments `
    -PolynomialTermLimit $PolynomialTermLimit `
    -Archive:$Archive
