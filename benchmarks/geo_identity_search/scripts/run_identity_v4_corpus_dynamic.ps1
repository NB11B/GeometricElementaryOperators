[CmdletBinding()]
param(
    [int]$Device = 0,
    [UInt64]$Assignments = 131072,
    [UInt64]$CpuChecks = 512,
    [int]$BlockSize = 256,
    [string]$CudaArchitectures = "120",
    [string]$ExpectedBranch = "research/geometric-identity-engine-v4",
    [string]$CorpusRoot = ".\local-evidence\v4\corpus",
    [switch]$Archive
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$BaseRunner = Join-Path $ScriptDirectory "run_identity_search.ps1"
$Python = (Get-Command python.exe).Source
$Git = (Get-Command git.exe).Source
$Corpus = Resolve-Path (Join-Path $RepositoryRoot $CorpusRoot)
$Manifest = Join-Path $Corpus "corpus-manifest.json"
$Validator = Join-Path $RepositoryRoot "tools\geo_identity_v4_corpus_validate.py"
$GeneratedHeader = Join-Path $BenchmarkDirectory "generated\geo_identity_corpus.cuh"
$TemporaryRoot = Join-Path $env:TEMP "geo-v4-dynamic-$PID-$([Guid]::NewGuid().ToString('N'))"
$TemporaryRunner = Join-Path $TemporaryRoot "run.ps1"
$ValidationJson = Join-Path $TemporaryRoot "validation.json"
$ValidationMd = Join-Path $TemporaryRoot "validation.md"
New-Item -ItemType Directory -Force -Path $TemporaryRoot | Out-Null

if ((& $Git -C $RepositoryRoot branch --show-current).Trim() -ne $ExpectedBranch) {
    throw "Run from $ExpectedBranch"
}
if (-not (Test-Path $Manifest)) { throw "Missing V4 corpus manifest: $Manifest" }

Push-Location $RepositoryRoot
try {
    & $Python $Validator --corpus-root $Corpus --output-json $ValidationJson --markdown-out $ValidationMd
    if ($LASTEXITCODE -ne 0) { throw "V4 corpus validation failed" }

    $Text = Get-Content $BaseRunner -Raw
    $OriginalDirectory = '$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path'
    $DirectoryLiteral = $ScriptDirectory.Replace("'", "''")
    $Text = $Text.Replace($OriginalDirectory, '$ScriptDirectory = ''' + $DirectoryLiteral + '''')
    $Text = $Text.Replace('$env:CUDAHOSTCXX = $ClShortPath', '$env:CUDAHOSTCXX = $ClPath')
    $Text = $Text.Replace('"-DCMAKE_CUDA_HOST_COMPILER=$ClShortPath"', '"-DCMAKE_CUDA_HOST_COMPILER=$ClPath"')
    $Text = $Text.Replace('"-DCMAKE_CXX_COMPILER=$ClShortPath"', '"-DCMAKE_CXX_COMPILER=$ClPath"')

    $Start = $Text.IndexOf('    $IdentityArguments = @(')
    $Invoke = '    Invoke-LoggedNativeCommand `'
    $First = $Text.IndexOf($Invoke, $Start)
    $Second = $Text.IndexOf($Invoke, $First + 1)
    if ($Start -lt 0 -or $First -lt 0 -or $Second -lt 0) { throw "Base runner layout changed" }

    $ManifestLiteral = $Manifest.Replace("'", "''")
    $Generator = @"
    Invoke-LoggedNativeCommand ``
        -Command {
            & `$PythonPath .\tools\geo_identity_manifest_compiler.py ``
                --manifest '$ManifestLiteral' ``
                --output .\benchmarks\geo_identity_search\generated\geo_identity_corpus.cuh ``
                --python-checks $CpuChecks
        } ``
        -LogPath (Join-Path `$EvidenceDirectory "generator-check.log")

"@
    $Text = $Text.Substring(0, $Start) + $Generator + $Text.Substring($Second)

    $SummaryNeedle = '                --markdown-out $SummaryMarkdown'
    $SummaryReplacement = "                --markdown-out `$SummaryMarkdown ```r`n                --allow-dynamic-corpus"
    if (-not $Text.Contains($SummaryNeedle)) { throw "Summary invocation marker not found" }
    $Text = $Text.Replace($SummaryNeedle, $SummaryReplacement)

    Set-Content $TemporaryRunner $Text -Encoding UTF8

    & $TemporaryRunner -Device $Device -Assignments $Assignments -CpuChecks $CpuChecks -BlockSize $BlockSize -CudaArchitectures $CudaArchitectures -ExpectedBranch $ExpectedBranch -Archive:$Archive
    if (-not $?) { throw "V4 CUDA run failed" }

    Write-Host "GEO_IDENTITY_V4,status=complete"
}
finally {
    Pop-Location
    Remove-Item $GeneratedHeader -Force -ErrorAction SilentlyContinue
    Remove-Item $TemporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
}
