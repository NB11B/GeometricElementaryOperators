[CmdletBinding()]
param(
    [string]$ExpectedBranch = "research/geometric-identity-engine-v4-1",
    [string]$OutputRoot = ".\local-evidence\v4-1\duality-corpus",
    [int[]]$Primes = @(65521, 65519),
    [int]$Prechecks = 512,
    [int]$PythonChecks = 512
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
    $ExitCode = 0
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
                if ($_ -is [System.Management.Automation.ErrorRecord]) {
                    $_.Exception.Message
                }
                else {
                    $_.ToString()
                }
            } |
            Tee-Object -FilePath $LogPath
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
        if ($HasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $PreviousNativePreference
        }
    }
    if ($ExitCode -ne 0) {
        throw "Command failed with exit code $ExitCode. See $LogPath"
    }
}

if ($Prechecks -lt 0) { throw "Prechecks must be non-negative" }
if ($PythonChecks -lt 0) { throw "PythonChecks must be non-negative" }
if (-not $Primes -or $Primes.Count -eq 0) { throw "At least one prime is required" }

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$PythonPath = Resolve-CommandPath "python.exe"
$GitPath = Resolve-CommandPath "git.exe"
$ResolvedOutputRoot = Join-Path $RepositoryRoot $OutputRoot
$Grammar = Join-Path $RepositoryRoot "experiments\geometric_identity_engine_v4_1\grammars\01_contraction_duality_contract.json"
$ValidationJson = Join-Path $ResolvedOutputRoot "corpus-validation.json"
$ValidationMarkdown = Join-Path $ResolvedOutputRoot "corpus-validation.md"
$GeneratedHeader = Join-Path $ResolvedOutputRoot "geo_identity_corpus.cuh"
$LogDirectory = Join-Path $ResolvedOutputRoot "host-gate-logs"

if (-not (Test-Path -LiteralPath $Grammar)) { throw "Missing grammar: $Grammar" }

Push-Location $RepositoryRoot
try {
    $Branch = (& $GitPath branch --show-current).Trim()
    if ($Branch -ne $ExpectedBranch) {
        throw "Current branch is '$Branch'; expected '$ExpectedBranch'"
    }

    Remove-Item -LiteralPath $ResolvedOutputRoot -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $ResolvedOutputRoot | Out-Null

    Invoke-VisibleNative `
        -Command {
            & $PythonPath -m unittest `
                tests.test_geo_identity_compiler `
                tests.test_geo_identity_discovery `
                tests.test_geo_identity_grammar_discovery `
                tests.test_identity_result_summarizer `
                tests.test_geo_identity_v4_1_fixed_blade
        } `
        -LogPath (Join-Path $LogDirectory "unit-tests.log")

    $CorpusArguments = New-Object System.Collections.Generic.List[string]
    $CorpusArguments.Add(".\tools\geo_identity_v4_1_duality_corpus.py")
    $CorpusArguments.Add("--grammar")
    $CorpusArguments.Add($Grammar)
    $CorpusArguments.Add("--output-root")
    $CorpusArguments.Add($ResolvedOutputRoot)
    $CorpusArguments.Add("--prechecks")
    $CorpusArguments.Add($Prechecks.ToString())
    $CorpusArguments.Add("--primes")
    foreach ($Prime in $Primes) { $CorpusArguments.Add($Prime.ToString()) }

    Invoke-VisibleNative `
        -Command { & $PythonPath @CorpusArguments } `
        -LogPath (Join-Path $LogDirectory "corpus-build.log")

    Invoke-VisibleNative `
        -Command {
            & $PythonPath .\tools\geo_identity_v4_1_duality_corpus_validate.py `
                --corpus-root $ResolvedOutputRoot `
                --output-json $ValidationJson `
                --markdown-out $ValidationMarkdown
        } `
        -LogPath (Join-Path $LogDirectory "corpus-validation.log")

    Invoke-VisibleNative `
        -Command {
            & $PythonPath .\tools\geo_identity_v4_1_manifest_compiler.py `
                --manifest (Join-Path $ResolvedOutputRoot "corpus-manifest.json") `
                --output $GeneratedHeader `
                --python-checks $PythonChecks
        } `
        -LogPath (Join-Path $LogDirectory "manifest-compile.log")

    Write-Host "V4_1_HOST_GATE: PASS"
    Write-Host "Corpus: $ResolvedOutputRoot"
    Write-Host "Manifest: $(Join-Path $ResolvedOutputRoot 'corpus-manifest.json')"
    Write-Host "Generated header: $GeneratedHeader"
}
finally {
    Pop-Location
}
