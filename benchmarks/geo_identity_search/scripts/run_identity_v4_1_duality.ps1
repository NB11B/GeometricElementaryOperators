[CmdletBinding()]
param(
    [int]$Device = 0,
    [UInt64]$Assignments = 131072,
    [UInt64]$CpuChecks = 512,
    [int]$BlockSize = 256,
    [string]$CudaArchitectures = "120",
    [string]$ExpectedBranch = "research/geometric-identity-engine-v4-1",
    [string]$CorpusRoot = ".\local-evidence\v4-1\duality-corpus",
    [switch]$Archive
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$BaseRunner = Join-Path $ScriptDirectory "run_identity_search.ps1"
$HostGate = Join-Path $ScriptDirectory "run_identity_v4_1_host_gate.ps1"
$Python = (Get-Command python.exe -ErrorAction Stop).Source
$Git = (Get-Command git.exe -ErrorAction Stop).Source
$Corpus = Resolve-Path (Join-Path $RepositoryRoot $CorpusRoot)
$Manifest = Join-Path $Corpus "corpus-manifest.json"
$GeneratedHeader = Join-Path $BenchmarkDirectory "generated\geo_identity_corpus.cuh"
$TemporaryRoot = Join-Path $env:TEMP "geo-v4-1-$PID-$([Guid]::NewGuid().ToString('N'))"
$TemporaryRunner = Join-Path $TemporaryRoot "run.ps1"
New-Item -ItemType Directory -Force -Path $TemporaryRoot | Out-Null

if ((& $Git -C $RepositoryRoot branch --show-current).Trim() -ne $ExpectedBranch) {
    throw "Run from $ExpectedBranch"
}
if (-not (Test-Path -LiteralPath $Manifest)) {
    throw "Missing V4.1 corpus manifest. Run run_identity_v4_1_host_gate.ps1 first."
}

Push-Location $RepositoryRoot
try {
    & $HostGate -PythonChecks $CpuChecks -ExpectedBranch $ExpectedBranch -CorpusRoot $CorpusRoot
    if ($LASTEXITCODE -ne 0) { throw "V4.1 host gate failed" }

    $Text = Get-Content -LiteralPath $BaseRunner -Raw
    $OriginalDirectory = '$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path'
    $DirectoryLiteral = $ScriptDirectory.Replace("'", "''")
    if (-not $Text.Contains($OriginalDirectory)) { throw "Base runner directory marker changed" }
    $Text = $Text.Replace($OriginalDirectory, '$ScriptDirectory = ''' + $DirectoryLiteral + '''')

    # Preserve the canonical long cl.exe path. This avoids the V3/V4 short-path mismatch.
    $Text = $Text.Replace('$env:CUDAHOSTCXX = $ClShortPath', '$env:CUDAHOSTCXX = $ClPath')
    $Text = $Text.Replace('"-DCMAKE_CUDA_HOST_COMPILER=$ClShortPath"', '"-DCMAKE_CUDA_HOST_COMPILER=$ClPath"')
    $Text = $Text.Replace('"-DCMAKE_CXX_COMPILER=$ClShortPath"', '"-DCMAKE_CXX_COMPILER=$ClPath"')

    $Start = $Text.IndexOf('    $IdentityArguments = @(')
    $Invoke = '    Invoke-LoggedNativeCommand `'
    $First = $Text.IndexOf($Invoke, $Start)
    $Second = $Text.IndexOf($Invoke, $First + 1)
    if ($Start -lt 0 -or $First -lt 0 -or $Second -lt 0) {
        throw "Base runner generator layout changed"
    }

    $ManifestLiteral = $Manifest.Replace("'", "''")
    $Generator = @"
    Invoke-LoggedNativeCommand ``
        -Command {
            & `$PythonPath .\tools\geo_identity_v4_1_manifest_compiler.py ``
                --manifest '$ManifestLiteral' ``
                --output .\benchmarks\geo_identity_search\generated\geo_identity_corpus.cuh ``
                --python-checks $CpuChecks
        } ``
        -LogPath (Join-Path `$EvidenceDirectory "generator-check.log")

"@
    $Text = $Text.Substring(0, $Start) + $Generator + $Text.Substring($Second)

    $SummaryNeedle = '                --markdown-out $SummaryMarkdown'
    $SummaryReplacement = "                --markdown-out `$SummaryMarkdown ```r`n                --allow-dynamic-corpus"
    if (-not $Text.Contains($SummaryNeedle)) { throw "Summary invocation marker changed" }
    $Text = $Text.Replace($SummaryNeedle, $SummaryReplacement)

    Set-Content -LiteralPath $TemporaryRunner -Value $Text -Encoding UTF8
    & $TemporaryRunner `
        -Device $Device `
        -Assignments $Assignments `
        -CpuChecks $CpuChecks `
        -BlockSize $BlockSize `
        -CudaArchitectures $CudaArchitectures `
        -ExpectedBranch $ExpectedBranch `
        -Archive:$Archive
    if (-not $?) { throw "V4.1 CUDA run failed" }
    Write-Host "GEO_IDENTITY_V4_1,status=complete"
}
finally {
    Pop-Location
    if (Test-Path -LiteralPath $GeneratedHeader) {
        & $Git -C $RepositoryRoot restore -- "benchmarks/geo_identity_search/generated/geo_identity_corpus.cuh" 2>$null
    }
    Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
}
