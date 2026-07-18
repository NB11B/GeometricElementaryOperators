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
$Runner = Join-Path $ScriptDirectory "run_identity_grammar_discovery.ps1"

if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Missing grammar discovery runner: $Runner"
}

# Repeated Visual Studio developer-environment imports can leave enough state in
# a long-lived PowerShell process to make VsDevCmd.bat exceed cmd.exe's command
# line/environment limits. Preserve the complete caller environment, remove only
# Visual Studio/SDK state before the real runner starts, and restore everything
# afterward.
$CallerEnvironment = @{}
Get-ChildItem Env: | ForEach-Object {
    $CallerEnvironment[$_.Name] = $_.Value
}

$VisualStudioEnvironmentPatterns = @(
    '^__VSCMD',
    '^VSCMD',
    '^VCINSTALLDIR$',
    '^VCTools',
    '^VSINSTALLDIR$',
    '^VisualStudioVersion$',
    '^DevEnvDir$',
    '^CommandPromptType$',
    '^PreferredToolArchitecture$',
    '^WindowsSdkDir$',
    '^WindowsSDKVersion$',
    '^WindowsSdkBinPath$',
    '^WindowsLibPath$',
    '^UniversalCRTSdkDir$',
    '^UCRTVersion$',
    '^NETFXSDKDir$',
    '^FrameworkDir',
    '^FrameworkVersion',
    '^ExtensionSdkDir$',
    '^INCLUDE$',
    '^LIB$',
    '^LIBPATH$',
    '^CUDAHOSTCXX$'
)

function Test-VisualStudioEnvironmentName {
    param([Parameter(Mandatory = $true)][string]$Name)
    foreach ($Pattern in $VisualStudioEnvironmentPatterns) {
        if ($Name -match $Pattern) {
            return $true
        }
    }
    return $false
}

try {
    Get-ChildItem Env: |
        Where-Object { Test-VisualStudioEnvironmentName -Name $_.Name } |
        ForEach-Object {
            [Environment]::SetEnvironmentVariable($_.Name, $null, "Process")
        }

    & $Runner `
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
        throw "Grammar discovery runner failed with exit code $LASTEXITCODE"
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
        [Environment]::SetEnvironmentVariable(
            $Name,
            $CallerEnvironment[$Name],
            "Process"
        )
    }
}
