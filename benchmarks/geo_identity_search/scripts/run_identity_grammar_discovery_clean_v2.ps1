[CmdletBinding()]
param(
    [int]$Device = 0,
    [UInt64]$Assignments = 262144,
    [UInt64]$CpuChecks = 1024,
    [int]$BlockSize = 256,
    [int[]]$Primes = @(65521, 65519, 65497, 32749),
    [int]$PrecheckAssignments = 2048,
    [int]$MaxRelations = 12,
    [int]$MaxControls = 4,
    [int]$PolynomialTermLimit = 200000,
    [string]$CudaArchitectures = "120",
    [string]$ExpectedBranch = "research/geometric-identity-engine-v3",
    [switch]$Archive
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceRunner = Join-Path $ScriptDirectory "run_identity_grammar_discovery.ps1"
if (-not (Test-Path -LiteralPath $SourceRunner)) {
    throw "Missing grammar discovery runner: $SourceRunner"
}

$CallerEnvironment = @{}
Get-ChildItem Env: | ForEach-Object {
    $CallerEnvironment[$_.Name] = $_.Value
}

$TemporaryRoot = Join-Path $env:TEMP "geo-identity-grammar-wrapper-$PID-$([Guid]::NewGuid().ToString('N'))"
$TemporaryRunner = Join-Path $TemporaryRoot "run_identity_grammar_discovery.normalized.ps1"
New-Item -ItemType Directory -Force -Path $TemporaryRoot | Out-Null

$VisualStudioEnvironmentPatterns = @(
    '^__VSCMD', '^VSCMD', '^VCINSTALLDIR$', '^VCTools', '^VSINSTALLDIR$',
    '^VisualStudioVersion$', '^DevEnvDir$', '^CommandPromptType$',
    '^PreferredToolArchitecture$', '^WindowsSdkDir$', '^WindowsSDKVersion$',
    '^WindowsSdkBinPath$', '^WindowsLibPath$', '^UniversalCRTSdkDir$',
    '^UCRTVersion$', '^NETFXSDKDir$', '^FrameworkDir', '^FrameworkVersion',
    '^ExtensionSdkDir$', '^INCLUDE$', '^LIB$', '^LIBPATH$', '^CUDAHOSTCXX$'
)

function Test-VisualStudioEnvironmentName {
    param([Parameter(Mandatory = $true)][string]$Name)
    foreach ($Pattern in $VisualStudioEnvironmentPatterns) {
        if ($Name -match $Pattern) { return $true }
    }
    return $false
}

try {
    Get-ChildItem Env: |
        Where-Object { Test-VisualStudioEnvironmentName -Name $_.Name } |
        ForEach-Object {
            [Environment]::SetEnvironmentVariable($_.Name, $null, "Process")
        }

    $RunnerText = Get-Content -LiteralPath $SourceRunner -Raw

    # The source runner is copied into a temporary directory. Preserve the
    # original scripts directory so its relative repository-root calculation
    # still resolves against the checkout rather than %TEMP%.
    $OriginalScriptDirectoryLine = '$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path'
    $EscapedScriptDirectory = $ScriptDirectory.Replace("'", "''")
    $ReplacementScriptDirectoryLine = '$ScriptDirectory = ''' + $EscapedScriptDirectory + ''''
    if (-not $RunnerText.Contains($OriginalScriptDirectoryLine)) {
        throw "Unable to find script-directory marker in $SourceRunner"
    }
    $RunnerText = $RunnerText.Replace(
        $OriginalScriptDirectoryLine,
        $ReplacementScriptDirectoryLine
    )

    $Marker = '$RunnerText = Get-Content -LiteralPath $BaseRunner -Raw'
    if (-not $RunnerText.Contains($Marker)) {
        throw "Unable to find base-runner load marker in $SourceRunner"
    }

    $Injection = @'
$RunnerText = Get-Content -LiteralPath $BaseRunner -Raw

    # CUDA 13.1 --use-local-env requires the compiler found in PATH to be the
    # exact same path string passed with -ccbin. The base runner historically
    # passed an 8.3 short path while VsDevCmd placed the canonical long path in
    # PATH. Normalize both CMake host-compiler settings and CUDAHOSTCXX to the
    # canonical path returned by Get-Command cl.exe.
    $RunnerText = $RunnerText.Replace(
        '$env:CUDAHOSTCXX = $ClShortPath',
        '$env:CUDAHOSTCXX = $ClPath'
    )
    $RunnerText = $RunnerText.Replace(
        '"-DCMAKE_CUDA_HOST_COMPILER=$ClShortPath"',
        '"-DCMAKE_CUDA_HOST_COMPILER=$ClPath"'
    )
    $RunnerText = $RunnerText.Replace(
        '"-DCMAKE_CXX_COMPILER=$ClShortPath"',
        '"-DCMAKE_CXX_COMPILER=$ClPath"'
    )
'@
    $RunnerText = $RunnerText.Replace($Marker, $Injection)

    # The generated header is intentionally ignored. Remove it directly rather
    # than asking Git to restore an untracked path in the source runner's cleanup.
    $OldCleanup = @'
    if (Test-Path -LiteralPath $GeneratedHeader) {
        & $GitPath -C $RepositoryRoot restore -- "benchmarks/geo_identity_search/generated/geo_identity_corpus.cuh" 2>$null
    }
'@
    $NewCleanup = @'
    if (Test-Path -LiteralPath $GeneratedHeader) {
        Remove-Item -LiteralPath $GeneratedHeader -Force -ErrorAction SilentlyContinue
    }
'@
    $RunnerText = $RunnerText.Replace($OldCleanup, $NewCleanup)

    Set-Content -LiteralPath $TemporaryRunner -Value $RunnerText -Encoding UTF8

    & $TemporaryRunner `
        -Device $Device `
        -Assignments $Assignments `
        -CpuChecks $CpuChecks `
        -BlockSize $BlockSize `
        -Primes $Primes `
        -PrecheckAssignments $PrecheckAssignments `
        -MaxRelations $MaxRelations `
        -MaxControls $MaxControls `
        -PolynomialTermLimit $PolynomialTermLimit `
        -CudaArchitectures $CudaArchitectures `
        -ExpectedBranch $ExpectedBranch `
        -Archive:$Archive

    if ($LASTEXITCODE -ne 0) {
        throw "Normalized grammar discovery runner failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force -ErrorAction SilentlyContinue

    $CurrentNames = @(Get-ChildItem Env: | Select-Object -ExpandProperty Name)
    foreach ($Name in $CurrentNames) {
        if (-not $CallerEnvironment.ContainsKey($Name)) {
            [Environment]::SetEnvironmentVariable($Name, $null, "Process")
        }
    }
    foreach ($Name in $CallerEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable(
            $Name,
            $CallerEnvironment[$Name],
            "Process"
        )
    }
}
