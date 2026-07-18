[CmdletBinding()]
param(
    [int[]]$Primes = @(65521, 65519),
    [int]$PythonChecks = 512,
    [string]$ExpectedBranch = "research/geometric-identity-engine-v4-1",
    [string]$CorpusRoot = ".\local-evidence\v4-1\duality-corpus"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$Python = (Get-Command python.exe -ErrorAction Stop).Source
$Git = (Get-Command git.exe -ErrorAction Stop).Source
$Corpus = Join-Path $RepositoryRoot $CorpusRoot
$Manifest = Join-Path $Corpus "corpus-manifest.json"
$GeneratedHeader = Join-Path $BenchmarkDirectory "generated\geo_identity_corpus.cuh"
$ValidationJson = Join-Path $RepositoryRoot "local-evidence\v4-1\duality-validation.json"
$ValidationMd = Join-Path $RepositoryRoot "local-evidence\v4-1\duality-validation.md"

Push-Location $RepositoryRoot
try {
    $Branch = (& $Git branch --show-current).Trim()
    if ($Branch -ne $ExpectedBranch) {
        throw "Current branch is '$Branch'; expected '$ExpectedBranch'"
    }

    Remove-Item $Corpus -Recurse -Force -ErrorAction SilentlyContinue
    $CorpusArguments = @(
        ".\tools\geo_identity_v4_1_duality_corpus.py",
        "--output-root", $Corpus,
        "--primes"
    )
    foreach ($Prime in $Primes) {
        $CorpusArguments += $Prime.ToString()
    }
    & $Python @CorpusArguments
    if ($LASTEXITCODE -ne 0) { throw "V4.1 corpus generation failed" }

    & $Python .\tools\geo_identity_v4_1_duality_validate.py `
        --corpus-root $Corpus `
        --output-json $ValidationJson `
        --markdown-out $ValidationMd
    if ($LASTEXITCODE -ne 0) { throw "V4.1 corpus validation failed" }

    & $Python -m unittest `
        tests.test_geo_identity_compiler `
        tests.test_geo_identity_discovery `
        tests.test_geo_identity_grammar_discovery `
        tests.test_geo_identity_v4_1_fixed_blade
    if ($LASTEXITCODE -ne 0) { throw "V4.1 regression tests failed" }

    & $Python .\tools\geo_identity_v4_1_manifest_compiler.py `
        --manifest $Manifest `
        --output $GeneratedHeader `
        --python-checks $PythonChecks
    if ($LASTEXITCODE -ne 0) { throw "V4.1 generated-header compilation failed" }

    Write-Host "V4_1_HOST_GATE: PASS"
    Write-Host "Manifest: $Manifest"
    Write-Host "Validation: $ValidationMd"
}
finally {
    if (Test-Path -LiteralPath $GeneratedHeader) {
        & $Git restore -- "benchmarks/geo_identity_search/generated/geo_identity_corpus.cuh" 2>$null
    }
    Pop-Location
}
