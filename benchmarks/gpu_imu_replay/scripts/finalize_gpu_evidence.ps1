[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EvidenceDirectory,
    [switch]$Archive
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Evidence = Get-Item -LiteralPath $EvidenceDirectory
if (-not $Evidence.PSIsContainer) {
    throw "EvidenceDirectory is not a directory: $EvidenceDirectory"
}

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")

$AggregateCsv = Join-Path $Evidence.FullName "gpu-imu-replay-all.csv"
if (-not (Test-Path $AggregateCsv)) {
    throw "Aggregate CSV not found: $AggregateCsv"
}

$SummaryCsv = Join-Path $Evidence.FullName "gpu-summary.csv"
$ParityCsv = Join-Path $Evidence.FullName "gpu-parity-summary.csv"
$SummaryMarkdown = Join-Path $Evidence.FullName "summary.md"
$ValidationLog = Join-Path $Evidence.FullName "summary-validation.log"

Push-Location $RepositoryRoot
try {
    python .\benchmarks\gpu_imu_replay\scripts\summarize_gpu_results.py `
        $AggregateCsv `
        --summary-csv $SummaryCsv `
        --parity-csv $ParityCsv `
        --markdown-out $SummaryMarkdown 2>&1 |
        Tee-Object -FilePath $ValidationLog

    if ($LASTEXITCODE -ne 0) {
        throw "GPU result validation failed. See $ValidationLog"
    }
}
finally {
    Pop-Location
}

$Manifest = Join-Path $Evidence.FullName "sha256-manifest.csv"
$ManifestHash = Join-Path $Evidence.FullName "sha256-manifest.sha256.txt"
$ArchivePath = "$($Evidence.FullName).zip"
$ArchiveHash = "$ArchivePath.sha256.txt"

Remove-Item $Manifest -Force -ErrorAction SilentlyContinue
Remove-Item $ManifestHash -Force -ErrorAction SilentlyContinue
Remove-Item $ArchiveHash -Force -ErrorAction SilentlyContinue

$FilesToHash = @(
    Get-ChildItem $Evidence.FullName -Recurse -File |
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
    Remove-Item $ArchivePath -Force -ErrorAction SilentlyContinue
    Compress-Archive `
        -Path (Join-Path $Evidence.FullName "*") `
        -DestinationPath $ArchivePath `
        -Force

    Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256 |
        Format-List |
        Out-String |
        Set-Content -LiteralPath $ArchiveHash -Encoding UTF8
}

Write-Host ""
Write-Host "GPU evidence finalized:"
Write-Host $Evidence.FullName
Write-Host "Manifest: $Manifest"
Write-Host "Manifest hash: $ManifestHash"
if ($Archive) {
    Write-Host "Archive: $ArchivePath"
    Write-Host "Archive hash: $ArchiveHash"
}
