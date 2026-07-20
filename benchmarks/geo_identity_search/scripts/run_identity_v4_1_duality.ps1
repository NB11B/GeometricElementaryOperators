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

function Resolve-CommandPath {
    param([Parameter(Mandatory = $true)][string]$Name)
    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $Command) { throw "$Name was not found in PATH" }
    return $Command.Source
}

if ($Assignments -eq 0) { throw "Assignments must be positive" }
if ($CpuChecks -gt $Assignments) { throw "CpuChecks cannot exceed Assignments" }
if ($BlockSize -lt 1 -or $BlockSize -gt 1024) { throw "BlockSize must be in [1,1024]" }

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$BaseRunner = Join-Path $ScriptDirectory "run_identity_search.ps1"
$HostGate = Join-Path $ScriptDirectory "run_identity_v4_1_host_gate.ps1"
$Git = Resolve-CommandPath "git.exe"
$Corpus = Join-Path $RepositoryRoot $CorpusRoot
$Manifest = Join-Path $Corpus "corpus-manifest.json"
$GeneratedHeader = Join-Path $BenchmarkDirectory "generated\geo_identity_corpus.cuh"
$GeneratedHeaderRelative = "benchmarks/geo_identity_search/generated/geo_identity_corpus.cuh"
$EvidenceRoot = Join-Path $BenchmarkDirectory "evidence"
$TemporaryRoot = Join-Path $env:TEMP "geo-v4-1-duality-$PID-$([Guid]::NewGuid().ToString('N'))"
$TemporaryRunner = Join-Path $TemporaryRoot "run.ps1"

if (-not (Test-Path -LiteralPath $BaseRunner)) { throw "Missing base runner: $BaseRunner" }
if (-not (Test-Path -LiteralPath $HostGate)) { throw "Missing host gate: $HostGate" }
New-Item -ItemType Directory -Force -Path $TemporaryRoot | Out-Null
New-Item -ItemType Directory -Force -Path $EvidenceRoot | Out-Null

Push-Location $RepositoryRoot
try {
    $Branch = (& $Git branch --show-current).Trim()
    if ($Branch -ne $ExpectedBranch) {
        throw "Current branch is '$Branch'; expected '$ExpectedBranch'"
    }

    & $HostGate `
        -ExpectedBranch $ExpectedBranch `
        -OutputRoot $CorpusRoot `
        -Prechecks ([int]$CpuChecks) `
        -PythonChecks ([int]$CpuChecks)
    if (-not $?) { throw "V4.1 host gate failed" }
    if (-not (Test-Path -LiteralPath $Manifest)) {
        throw "Missing V4.1 duality manifest after host gate: $Manifest"
    }

    $Text = Get-Content -LiteralPath $BaseRunner -Raw
    $OriginalDirectory = '$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path'
    $DirectoryLiteral = $ScriptDirectory.Replace("'", "''")
    if (-not $Text.Contains($OriginalDirectory)) { throw "Base runner directory marker not found" }
    $Text = $Text.Replace($OriginalDirectory, '$ScriptDirectory = ''' + $DirectoryLiteral + '''')

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
    if (-not $Text.Contains($SummaryNeedle)) { throw "Summary invocation marker not found" }
    $Text = $Text.Replace($SummaryNeedle, $SummaryReplacement)
    Set-Content -LiteralPath $TemporaryRunner -Value $Text -Encoding UTF8

    $Before = @(
        Get-ChildItem -LiteralPath $EvidenceRoot -Directory -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName
    )

    & $TemporaryRunner `
        -Device $Device `
        -Assignments $Assignments `
        -CpuChecks $CpuChecks `
        -BlockSize $BlockSize `
        -CudaArchitectures $CudaArchitectures `
        -ExpectedBranch $ExpectedBranch
    if (-not $?) { throw "V4.1 CUDA run failed" }

    $EvidenceDirectory = Get-ChildItem -LiteralPath $EvidenceRoot -Directory |
        Where-Object {
            $_.FullName -notin $Before -and
            (Test-Path (Join-Path $_.FullName "identity-search.csv"))
        } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $EvidenceDirectory) { throw "Unable to locate V4.1 CUDA evidence directory" }

    $CorpusEvidence = Join-Path $EvidenceDirectory.FullName "v4-1-duality-corpus"
    New-Item -ItemType Directory -Force -Path $CorpusEvidence | Out-Null
    Copy-Item -Path (Join-Path $Corpus "*") -Destination $CorpusEvidence -Recurse -Force

    $ManifestFile = Join-Path $EvidenceDirectory.FullName "sha256-manifest.csv"
    $ManifestHash = Join-Path $EvidenceDirectory.FullName "sha256-manifest.sha256.txt"
    Remove-Item $ManifestFile -Force -ErrorAction SilentlyContinue
    Remove-Item $ManifestHash -Force -ErrorAction SilentlyContinue
    Get-ChildItem $EvidenceDirectory.FullName -Recurse -File |
        Where-Object { $_.FullName -ne $ManifestFile -and $_.FullName -ne $ManifestHash } |
        Sort-Object FullName |
        ForEach-Object { Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 } |
        Select-Object Path, Algorithm, Hash |
        Export-Csv -LiteralPath $ManifestFile -NoTypeInformation -Encoding UTF8
    Get-FileHash -LiteralPath $ManifestFile -Algorithm SHA256 |
        Format-List |
        Out-String |
        Set-Content -LiteralPath $ManifestHash -Encoding UTF8

    if ($Archive) {
        $ArchivePath = "$($EvidenceDirectory.FullName).zip"
        $ArchiveHash = "$ArchivePath.sha256.txt"
        Remove-Item $ArchivePath -Force -ErrorAction SilentlyContinue
        Remove-Item $ArchiveHash -Force -ErrorAction SilentlyContinue
        Compress-Archive -Path (Join-Path $EvidenceDirectory.FullName "*") -DestinationPath $ArchivePath -Force
        Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256 |
            Format-List |
            Out-String |
            Set-Content -LiteralPath $ArchiveHash -Encoding UTF8
    }

    Write-Host ""
    Write-Host "V4.1 duality evidence is complete:"
    Write-Host $EvidenceDirectory.FullName
    Write-Host "Manifest: $ManifestFile"
    Write-Host "Manifest hash: $ManifestHash"
    if ($Archive) {
        Write-Host "Archive: $ArchivePath"
        Write-Host "Archive hash: $ArchiveHash"
    }
    Write-Host "GEO_IDENTITY_V4_1_DUALITY,status=complete"
}
finally {
    Pop-Location
    Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $GeneratedHeader) {
        $TrackedHeader = @(& $Git -C $RepositoryRoot ls-files -- $GeneratedHeaderRelative)
        if ($LASTEXITCODE -eq 0 -and $TrackedHeader.Count -gt 0) {
            & $Git -C $RepositoryRoot restore -- $GeneratedHeaderRelative 2>$null
            if ($LASTEXITCODE -ne 0) {
                throw "Unable to restore tracked generated header: $GeneratedHeaderRelative"
            }
        }
        else {
            Remove-Item -LiteralPath $GeneratedHeader -Force -ErrorAction SilentlyContinue
        }
    }
}
