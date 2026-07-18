[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$EvidenceDirectory,
    [int[]]$Primes = @(65521, 65519, 65497, 32749),
    [int]$MaxMutations = 8,
    [int]$PrecheckAssignments = 4096,
    [int]$PolynomialTermLimit = 200000,
    [switch]$Archive
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Resolve-Path (Join-Path $ScriptDirectory "..\..\..")
$EvidenceDirectory = (Resolve-Path -LiteralPath $EvidenceDirectory).Path
$Python = (Get-Command python.exe -ErrorAction Stop).Source
$Tool = Join-Path $RepositoryRoot "tools\geo_identity_discovery.py"
$RawCsv = Join-Path $EvidenceDirectory "identity-search.csv"
if (-not (Test-Path $RawCsv)) { throw "Missing $RawCsv" }

$Temp = Join-Path $env:TEMP "geo-discovery-finalize-$PID"
$Generated = Join-Path $Temp "generated"
Remove-Item $Temp -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Temp | Out-Null

$Sources = @(
    "experiments\geometric_identity_engine\corpus\01_vector_square_scalar.json",
    "experiments\geometric_identity_engine\corpus\02_reverse_product_order.json",
    "experiments\geometric_identity_engine\corpus\03_vector_wedge_antisymmetry.json",
    "experiments\geometric_identity_engine\corpus\04_vector_product_commutativity_false.json",
    "experiments\geometric_identity_engine\corpus\05_mutated_reverse_order_false.json"
)

Push-Location $RepositoryRoot
try {
    $Args = @("build-corpus")
    foreach ($Source in $Sources) { $Args += @("--source", $Source) }
    $Args += @("--output-dir", $Generated)
    foreach ($Prime in $Primes) { $Args += @("--prime", $Prime.ToString()) }
    $Args += @(
        "--precheck-assignments", $PrecheckAssignments.ToString(),
        "--max-mutations", $MaxMutations.ToString(),
        "--term-limit", $PolynomialTermLimit.ToString(),
        "--clean"
    )
    & $Python $Tool @Args | Tee-Object -FilePath (Join-Path $EvidenceDirectory "discovery-corpus.log")
    if ($LASTEXITCODE -ne 0) { throw "Discovery corpus generation failed" }

    $Discovery = Join-Path $EvidenceDirectory "discovery"
    Remove-Item $Discovery -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $Discovery | Out-Null
    Copy-Item (Join-Path $Generated "*") $Discovery -Recurse -Force

    & $Python $Tool reduce-results `
        --manifest (Join-Path $Discovery "corpus-manifest.json") `
        --csv $RawCsv `
        --output-json (Join-Path $EvidenceDirectory "discovery-report.json") `
        --markdown-out (Join-Path $EvidenceDirectory "discovery-summary.md") |
        Tee-Object -FilePath (Join-Path $EvidenceDirectory "discovery-validation.log")
    if ($LASTEXITCODE -ne 0) { throw "Discovery result reduction failed" }

    $Manifest = Join-Path $EvidenceDirectory "sha256-manifest.csv"
    $ManifestHash = Join-Path $EvidenceDirectory "sha256-manifest.sha256.txt"
    Remove-Item $Manifest, $ManifestHash -Force -ErrorAction SilentlyContinue
    Get-ChildItem $EvidenceDirectory -Recurse -File |
        Where-Object { $_.FullName -ne $Manifest -and $_.FullName -ne $ManifestHash } |
        Sort-Object FullName |
        Get-FileHash -Algorithm SHA256 |
        Select-Object Path, Algorithm, Hash |
        Export-Csv $Manifest -NoTypeInformation -Encoding UTF8
    Get-FileHash $Manifest -Algorithm SHA256 | Format-List | Out-String | Set-Content $ManifestHash

    if ($Archive) {
        $Zip = "$EvidenceDirectory.zip"
        $ZipHash = "$Zip.sha256.txt"
        Remove-Item $Zip, $ZipHash -Force -ErrorAction SilentlyContinue
        Compress-Archive (Join-Path $EvidenceDirectory "*") $Zip -Force
        Get-FileHash $Zip -Algorithm SHA256 | Format-List | Out-String | Set-Content $ZipHash
    }

    Write-Host "GEO_IDENTITY_DISCOVERY_FINALIZE,status=complete"
    Write-Host $EvidenceDirectory
}
finally {
    Pop-Location
    Remove-Item $Temp -Recurse -Force -ErrorAction SilentlyContinue
}
