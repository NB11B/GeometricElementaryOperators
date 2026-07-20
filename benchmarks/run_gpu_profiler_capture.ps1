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
$IsWindowsHost = $env:OS -eq "Windows_NT"
$NsysTrace = if ($IsWindowsHost) { "cuda,nvtx" } else { "cuda,nvtx,osrt" }

New-Item -ItemType Directory -Force $OutDir | Out-Null

@(
    "profile_exe=$((Resolve-Path $ProfileExe).Path)",
    "iterations=$Iterations",
    "nsys=$Nsys",
    "ncu=$Ncu",
    "nsys_trace=$NsysTrace",
    "source_sha=$(git rev-parse HEAD)"
) | Set-Content (Join-Path $OutDir "profile-environment.txt")

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

    if (Test-Path $NsysReport) {
        Write-Host "Skipping existing Nsight Systems report for $($Case.Name)"
    } else {
        Write-Host "Profiling $($Case.Name) with Nsight Systems"

        & $Nsys profile `
            --force-overwrite=true `
            --trace=$NsysTrace `
            --sample=none `
            --stats=true `
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

    if (Test-Path $NcuReport) {
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

$Missing = @()
foreach ($Case in $Cases) {
    $Prefix = Join-Path $OutDir $Case.Name
    foreach ($Required in @("$Prefix.nsys-rep", "$Prefix-ncu.ncu-rep")) {
        if (-not (Test-Path $Required)) {
            $Missing += $Required
        }
    }
}

if ($Missing.Count -gt 0) {
    $Missing | ForEach-Object { Write-Error "Missing profiler report: $_" }
    throw "Profiler capture incomplete."
}

Get-ChildItem $OutDir -Recurse -File |
    Select-Object Name, Length, LastWriteTime |
    Export-Csv (Join-Path $OutDir "profile-files.csv") -NoTypeInformation

Write-Host "GEO_GPU_PROFILER_CAPTURE: PASS cases=$($Cases.Count) out=$OutDir"
