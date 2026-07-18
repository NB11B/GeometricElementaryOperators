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
$Runner = Join-Path $ScriptDirectory "run_identity_v4_1_duality.ps1"
if (-not (Test-Path -LiteralPath $Runner)) { throw "Missing V4.1 duality runner: $Runner" }

$CallerEnvironment = @{}
Get-ChildItem Env: | ForEach-Object { $CallerEnvironment[$_.Name] = $_.Value }

$Patterns = @(
    '^__VSCMD', '^VSCMD', '^VCINSTALLDIR$', '^VCTools', '^VSINSTALLDIR$',
    '^VisualStudioVersion$', '^DevEnvDir$', '^CommandPromptType$',
    '^PreferredToolArchitecture$', '^WindowsSdkDir$', '^WindowsSDKVersion$',
    '^WindowsSdkBinPath$', '^WindowsLibPath$', '^UniversalCRTSdkDir$',
    '^UCRTVersion$', '^NETFXSDKDir$', '^FrameworkDir', '^FrameworkVersion',
    '^ExtensionSdkDir$', '^INCLUDE$', '^LIB$', '^LIBPATH$', '^CUDAHOSTCXX$'
)

function Test-StaleToolchainVariable {
    param([Parameter(Mandatory = $true)][string]$Name)
    foreach ($Pattern in $Patterns) {
        if ($Name -match $Pattern) { return $true }
    }
    return $false
}

try {
    Get-ChildItem Env: |
        Where-Object { Test-StaleToolchainVariable -Name $_.Name } |
        ForEach-Object {
            [Environment]::SetEnvironmentVariable($_.Name, $null, "Process")
        }

    & $Runner `
        -Device $Device `
        -Assignments $Assignments `
        -CpuChecks $CpuChecks `
        -BlockSize $BlockSize `
        -CudaArchitectures $CudaArchitectures `
        -ExpectedBranch $ExpectedBranch `
        -CorpusRoot $CorpusRoot `
        -Archive:$Archive

    if ($LASTEXITCODE -ne 0) {
        throw "V4.1 duality runner failed with exit code $LASTEXITCODE"
    }
}
finally {
    $CurrentNames = @(Get-ChildItem Env: | Select-Object -ExpandProperty Name)
    foreach ($Name in $CurrentNames) {
        if (-not $CallerEnvironment.ContainsKey($Name)) {
            [Environment]::SetEnvironmentVariable($Name, $null, "Process")
        }
    }
    foreach ($Name in $CallerEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($Name, $CallerEnvironment[$Name], "Process")
    }
}
