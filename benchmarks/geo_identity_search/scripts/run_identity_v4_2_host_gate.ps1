[CmdletBinding()]
param(
    [string]$ExpectedBranch = "research/geometric-identity-engine-v4-2",
    [string]$OutputRoot = ".\local-evidence\v4-2\signature-matrix",
    [int]$PythonChecks = 128,
    [int]$LoweringIterations = 2000,
    [int]$MaxRelationsPerSignature = 0
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-CommandPath {
    param([Parameter(Mandatory = $true)][string]$Name)
    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $Command) { throw "$Name was not found in PATH" }
    return $Command.Source
}

function Invoke-VisibleNative {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Command,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $Parent = Split-Path -Parent $LogPath
    if ($Parent) { New-Item -ItemType Directory -Force -Path $Parent | Out-Null }
    $PreviousErrorActionPreference = $ErrorActionPreference
    $PreviousNativePreference = $null
    $HasNativePreference = Test-Path variable:PSNativeCommandUseErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        if ($HasNativePreference) {
            $PreviousNativePreference = $PSNativeCommandUseErrorActionPreference
            $PSNativeCommandUseErrorActionPreference = $false
        }
        $global:LASTEXITCODE = 0
        & $Command 2>&1 |
            ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message }
                else { $_.ToString() }
            } |
            Tee-Object -FilePath $LogPath
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
        if ($HasNativePreference) { $PSNativeCommandUseErrorActionPreference = $PreviousNativePreference }
    }
    if ($ExitCode -ne 0) { throw "Command failed with exit code $ExitCode. See $LogPath" }
}

if ($PythonChecks -lt 0) { throw "PythonChecks must be non-negative" }
if ($LoweringIterations -lt 1) { throw "LoweringIterations must be positive" }
if ($MaxRelationsPerSignature -lt 0) { throw "MaxRelationsPerSignature must be non-negative" }

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$PythonPath = Resolve-CommandPath "python.exe"
$GitPath = Resolve-CommandPath "git.exe"
$ResolvedOutputRoot = Join-Path $RepositoryRoot $OutputRoot
$Config = Join-Path $RepositoryRoot "experiments\geometric_identity_engine_v4_2\config.json"
$Manifest = Join-Path $ResolvedOutputRoot "corpus-manifest.json"
$ValidationJson = Join-Path $ResolvedOutputRoot "validation.json"
$ValidationMarkdown = Join-Path $ResolvedOutputRoot "validation.md"
$AuditJson = Join-Path $ResolvedOutputRoot "relation-audit.json"
$AuditMarkdown = Join-Path $ResolvedOutputRoot "relation-audit.md"
$LoweringJson = Join-Path $ResolvedOutputRoot "lowering.json"
$LoweringMarkdown = Join-Path $ResolvedOutputRoot "lowering.md"
$LoweringHeader = Join-Path $ResolvedOutputRoot "geo_fixed_blade_lowering_v4_2.hpp"
$GeneratedHeader = Join-Path $ResolvedOutputRoot "geo_identity_corpus.cuh"
$LogDirectory = Join-Path $ResolvedOutputRoot "host-gate-logs"

if (-not (Test-Path -LiteralPath $Config)) { throw "Missing V4.2 config: $Config" }

Push-Location $RepositoryRoot
try {
    $Branch = (& $GitPath branch --show-current).Trim()
    if ($Branch -ne $ExpectedBranch) { throw "Current branch is '$Branch'; expected '$ExpectedBranch'" }

    Remove-Item -LiteralPath $ResolvedOutputRoot -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $ResolvedOutputRoot | Out-Null

    Invoke-VisibleNative `
        -Command {
            & $PythonPath -m py_compile `
                .\tools\geo_identity_v4_2_exact.py `
                .\tools\geo_identity_v4_2_compiler.py `
                .\tools\geo_identity_v4_2_engine.py `
                .\tools\geo_identity_v4_2_corpus.py `
                .\tools\geo_identity_v4_2_validate.py `
                .\tools\geo_identity_v4_2_manifest_compiler.py `
                .\tools\geo_identity_v4_2_relation_audit.py `
                .\tools\geo_identity_v4_2_lowering.py
        } `
        -LogPath (Join-Path $LogDirectory "syntax.log")

    Invoke-VisibleNative `
        -Command {
            & $PythonPath -m unittest `
                tests.test_geo_identity_compiler `
                tests.test_geo_identity_discovery `
                tests.test_geo_identity_grammar_discovery `
                tests.test_identity_result_summarizer `
                tests.test_geo_identity_v4_1_fixed_blade `
                tests.test_geo_identity_v4_2_native
        } `
        -LogPath (Join-Path $LogDirectory "unit-tests.log")

    $CorpusArguments = New-Object System.Collections.Generic.List[string]
    $CorpusArguments.Add(".\tools\geo_identity_v4_2_corpus.py")
    $CorpusArguments.Add("--config")
    $CorpusArguments.Add($Config)
    $CorpusArguments.Add("--output-root")
    $CorpusArguments.Add($ResolvedOutputRoot)
    if ($MaxRelationsPerSignature -gt 0) {
        $CorpusArguments.Add("--max-relations-per-signature")
        $CorpusArguments.Add($MaxRelationsPerSignature.ToString())
    }
    Invoke-VisibleNative `
        -Command { & $PythonPath @CorpusArguments } `
        -LogPath (Join-Path $LogDirectory "corpus-build.log")

    Invoke-VisibleNative `
        -Command {
            & $PythonPath .\tools\geo_identity_v4_2_validate.py `
                --corpus-root $ResolvedOutputRoot `
                --output-json $ValidationJson `
                --markdown-out $ValidationMarkdown `
                --python-checks $PythonChecks
        } `
        -LogPath (Join-Path $LogDirectory "validation.log")

    Invoke-VisibleNative `
        -Command {
            & $PythonPath .\tools\geo_identity_v4_2_relation_audit.py `
                --manifest $Manifest `
                --config $Config `
                --output-json $AuditJson `
                --markdown-out $AuditMarkdown
        } `
        -LogPath (Join-Path $LogDirectory "relation-audit.log")

    Invoke-VisibleNative `
        -Command {
            & $PythonPath .\tools\geo_identity_v4_2_lowering.py `
                --config $Config `
                --output-json $LoweringJson `
                --markdown-out $LoweringMarkdown `
                --header-out $LoweringHeader `
                --iterations $LoweringIterations
        } `
        -LogPath (Join-Path $LogDirectory "lowering.log")

    Invoke-VisibleNative `
        -Command {
            & $PythonPath .\tools\geo_identity_v4_2_manifest_compiler.py `
                --manifest $Manifest `
                --output $GeneratedHeader `
                --python-checks $PythonChecks
        } `
        -LogPath (Join-Path $LogDirectory "manifest-compile.log")

    Write-Host "V4_2_HOST_GATE: PASS"
    Write-Host "Corpus: $ResolvedOutputRoot"
    Write-Host "Manifest: $Manifest"
    Write-Host "Validation: $ValidationMarkdown"
    Write-Host "Audit: $AuditMarkdown"
    Write-Host "Lowering: $LoweringMarkdown"
    Write-Host "Generated header: $GeneratedHeader"
}
finally {
    Pop-Location
}
