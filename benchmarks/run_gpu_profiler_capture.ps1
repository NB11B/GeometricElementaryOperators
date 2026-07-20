param(
    [string]$ProfileExe = ".\build\geo-v8-cuda\Release\profile_v8_cuda.exe",
    [string]$OutDir = ".\gpu-profiler-evidence",
    [int]$Iterations = 100
)

$ErrorActionPreference = "Stop"

function Require-Command([string]$Name) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Required command not found: $Name"
    }
    return $command.Source
}

if (-not (Test-Path $ProfileExe)) {
    throw "Profiler executable not found: $ProfileExe"
}

$Nsys = Require-Command "nsys"
$Ncu = Require-Command "ncu"

New-Item -ItemType Directory -Force $OutDir | Out-Null

@(
    "profile_exe=$((Resolve-Path $ProfileExe).Path)",
    "iterations=$Iterations",
    "nsys=$Nsys",
    "ncu=$Ncu",
    "source_sha=$(git rev-parse HEAD)"
) | Set-Content (Join-Path $OutDir "profile-environment.txt")

$TraceSet = if ($IsWindows -or $env:OS -eq "Windows_NT") { "cuda,nvtx" } else { "cuda,nvtx,osrt" }

$Cases = @(
    @{ Name = "d2_b1_planned_inference"; Dimension = 2; Batch = 1; Backend = "planned"; Mode = "inference" },
    @{ Name = "d6_b64_planned_training"; Dimension = 6; Batch = 64; Backend = "planned"; Mode = "training" },
    @{ Name = "d6_b1024_planned_inference"; Dimension = 6; Batch = 1024; Backend = "planned"; Mode = "inference" },
    @{ Name = "d6_b64_hand_training"; Dimension = 6; Batch = 64; Backend = "hand"; Mode = "training" }
)

foreach ($Case in $Cases) {
    $Prefix = Join-Path $OutDir $Case.Name
    $NsysReport = "$Prefix.nsys-rep"
    $NcuReport = "$Prefix-ncu.ncu-rep"

    if ((Test-Path $NsysReport) -and (Get-Item $NsysReport).Length -gt 0) {
        Write-Host "Skipping existing Nsight Systems report for $($Case.Name)"
    } else {
        Write-Host "Profiling $($Case.Name) with Nsight Systems"
        & $Nsys profile `
            --force-overwrite=true `
            --trace=$TraceSet `
            --sample=none `
            --stats=false `
            --output=$Prefix `
            $ProfileExe `
            $Case.Dimension `
            $Case.Batch `
            $Case.Backend `
            $Case.Mode `
            $Iterations `
            2>&1 | Tee-Object "$Prefix-nsys.log"

        if ($LASTEXITCODE -ne 0) {
            throw "Nsight Systems failed for $($Case.Name)"
        }
    }

    if ((Test-Path $NcuReport) -and (Get-Item $NcuReport).Length -gt 0) {
        Write-Host "Skipping existing Nsight Compute report for $($Case.Name)"
    } else {
        Write-Host "Profiling $($Case.Name) with Nsight Compute"
        & $Ncu `
            --force-overwrite `
            --set full `
            --target-processes all `
            --launch-skip 20 `
            --launch-count 1 `
            --export "$Prefix-ncu" `
            $ProfileExe `
            $Case.Dimension `
            $Case.Batch `
            $Case.Backend `
            $Case.Mode `
            $Iterations `
            2>&1 | Tee-Object "$Prefix-ncu.log"

        if ($LASTEXITCODE -ne 0) {
            throw "Nsight Compute failed for $($Case.Name)"
        }
    }
}

foreach ($Case in $Cases) {
    $Prefix = Join-Path $OutDir $Case.Name
    $NsysReport = "$Prefix.nsys-rep"
    $NcuReport = "$Prefix-ncu.ncu-rep"
    if (-not (Test-Path $NsysReport) -or (Get-Item $NsysReport).Length -le 0) {
        throw "Missing or empty Nsight Systems report: $NsysReport"
    }
    if (-not (Test-Path $NcuReport) -or (Get-Item $NcuReport).Length -le 0) {
        throw "Missing or empty Nsight Compute report: $NcuReport"
    }
}

Get-ChildItem $OutDir -Recurse -File |
    Select-Object Name, Length, LastWriteTime |
    Export-Csv (Join-Path $OutDir "profile-files.csv") -NoTypeInformation

Write-Host "GEO_GPU_PROFILER_CAPTURE: PASS cases=$($Cases.Count) out=$OutDir"
