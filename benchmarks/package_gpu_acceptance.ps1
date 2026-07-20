param(
    [string]$ResultsDir = ".\gpu-results-nine-trial-hand",
    [string]$ProfilerDir = ".\gpu-profiler-evidence",
    [string]$EvidenceDir = ".\gpu-evidence",
    [string]$ReportPath = ".\gpu-results-nine-trial-hand\GEO_V8_CUDA_ACCEPTANCE_REPORT.md",
    [string]$ArchiveStage = ".\geo-v8-gpu-final-evidence",
    [string]$ArchivePath = ".\geo-v8-gpu-final-evidence.zip"
)

$ErrorActionPreference = "Stop"

foreach ($Required in @($ResultsDir, $ProfilerDir, $EvidenceDir, $ReportPath)) {
    if (-not (Test-Path $Required)) {
        throw "Required evidence path is missing: $Required"
    }
}

$SourceSha = (git rev-parse HEAD).Trim()
$TrackedChanges = git status --porcelain --untracked-files=no
if ($TrackedChanges) {
    throw "Tracked source tree is dirty. Commit or reset tracked changes before packaging.`n$TrackedChanges"
}

Remove-Item $ArchiveStage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $ArchiveStage | Out-Null

Copy-Item $ResultsDir (Join-Path $ArchiveStage "gpu-results") -Recurse
Copy-Item $ProfilerDir (Join-Path $ArchiveStage "gpu-profiler-evidence") -Recurse
Copy-Item $EvidenceDir (Join-Path $ArchiveStage "gpu-environment-and-correctness") -Recurse
Copy-Item $ReportPath (Join-Path $ArchiveStage "GEO_V8_CUDA_ACCEPTANCE_REPORT.md")

@(
    "source_sha=$SourceSha",
    "tracked_tree_clean=true",
    "packaged_at_utc=$([DateTime]::UtcNow.ToString('o'))",
    "results_dir=$ResultsDir",
    "profiler_dir=$ProfilerDir"
) | Set-Content (Join-Path $ArchiveStage "acceptance-record.txt") -Encoding UTF8

$ManifestPath = Join-Path $ArchiveStage "SHA256SUMS.csv"
$ArchiveRoot = (Resolve-Path $ArchiveStage).Path

$Rows = Get-ChildItem $ArchiveStage -Recurse -File |
    Where-Object { $_.FullName -ne $ManifestPath } |
    ForEach-Object {
        $Hash = Get-FileHash $_.FullName -Algorithm SHA256
        [PSCustomObject]@{
            SHA256 = $Hash.Hash
            Path = $_.FullName.Substring($ArchiveRoot.Length + 1)
            Bytes = $_.Length
        }
    } |
    Sort-Object Path

$Rows | Export-Csv $ManifestPath -NoTypeInformation -Encoding UTF8

$Failures = @()
Import-Csv $ManifestPath | ForEach-Object {
    $Path = Join-Path $ArchiveStage $_.Path
    if (-not (Test-Path $Path)) {
        $Failures += "Missing: $($_.Path)"
        return
    }
    $Actual = (Get-FileHash $Path -Algorithm SHA256).Hash
    if ($Actual -ne $_.SHA256) {
        $Failures += "Hash mismatch: $($_.Path)"
    }
}
if ($Failures.Count -ne 0) {
    $Failures | ForEach-Object { Write-Error $_ }
    throw "Manifest verification failed"
}

Remove-Item $ArchivePath -Force -ErrorAction SilentlyContinue
Compress-Archive -Path "$ArchiveStage\*" -DestinationPath $ArchivePath -CompressionLevel Optimal

$ArchiveHash = Get-FileHash $ArchivePath -Algorithm SHA256
$HashPath = [System.IO.Path]::ChangeExtension($ArchivePath, ".sha256.txt")
@(
    "SHA256=$($ArchiveHash.Hash)",
    "File=$([System.IO.Path]::GetFileName($ArchivePath))",
    "Bytes=$((Get-Item $ArchivePath).Length)",
    "SourceSHA=$SourceSha"
) | Set-Content $HashPath -Encoding ASCII

Write-Host "MANIFEST_VERIFICATION: PASS files=$($Rows.Count)"
Write-Host "GEO_GPU_EVIDENCE_ARCHIVE: PASS archive=$ArchivePath sha256=$($ArchiveHash.Hash)"
