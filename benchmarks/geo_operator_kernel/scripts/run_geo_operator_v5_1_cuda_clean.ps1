[CmdletBinding()]
param(
    [int]$Device = 0,
    [UInt64]$Assignments = 1024,
    [string]$CudaArchitectures = "120",
    [string]$ExpectedBranch = "research/geometric-operator-kernel-v5-1-acceptance",
    [string]$OutputRoot = ".\local-evidence\v5-1\operator-kernel",
    [switch]$Archive
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $ScriptDirectory "run_geo_operator_v5_1_cuda.ps1"
if (-not (Test-Path -LiteralPath $Runner)) { throw "Missing V5.1 runner: $Runner" }

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
        ForEach-Object { [Environment]::SetEnvironmentVariable($_.Name, $null, "Process") }

    & $Runner `
        -Device $Device `
        -Assignments $Assignments `
        -CudaArchitectures $CudaArchitectures `
        -ExpectedBranch $ExpectedBranch `
        -OutputRoot $OutputRoot `
        -Archive:$Archive
    if ($LASTEXITCODE -ne 0) { throw "V5.1 runner failed with exit code $LASTEXITCODE" }
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
