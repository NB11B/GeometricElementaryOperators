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

function Resolve-PythonPath {
    $Command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }
    $Command = Get-Command python -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }
    throw "Python was not found in PATH"
}

function Resolve-GitPath {
    $Command = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }
    $Candidates = @(
        (Join-Path $env:ProgramFiles "Git\cmd\git.exe"),
        (Join-Path $env:ProgramFiles "Git\bin\git.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Git\cmd\git.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Git\cmd\git.exe")
    )
    foreach ($Candidate in $Candidates) {
        if ($Candidate -and (Test-Path -LiteralPath $Candidate)) {
            return $Candidate
        }
    }
    throw "Git for Windows was not found"
}

function Invoke-VisibleCommand {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    $PreviousErrorActionPreference = $ErrorActionPreference
    $PreviousNativePreference = $null
    $HasNativePreference = Test-Path variable:PSNativeCommandUseErrorActionPreference
    $ExitCode = 0
    if ($HasNativePreference) {
        $PreviousNativePreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    try {
        $ErrorActionPreference = "Continue"
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

if ($Assignments -eq 0) { throw "Assignments must be positive" }
if ($CpuChecks -gt $Assignments) { throw "CpuChecks cannot exceed Assignments" }
if ($BlockSize -lt 1 -or $BlockSize -gt 1024) { throw "BlockSize must be in [1,1024]" }
if (-not $Primes -or $Primes.Count -eq 0) { throw "At least one prime is required" }
if ($PrecheckAssignments -lt 0) { throw "PrecheckAssignments cannot be negative" }
if ($MaxRelations -lt 1) { throw "MaxRelations must be positive" }
if ($MaxControls -lt 0) { throw "MaxControls cannot be negative" }
if ($PolynomialTermLimit -lt 1) { throw "PolynomialTermLimit must be positive" }

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$BaseRunner = Join-Path $ScriptDirectory "run_identity_search.ps1"
$GrammarTool = Join-Path $RepositoryRoot "tools\geo_identity_grammar_discovery.py"
$PythonPath = Resolve-PythonPath
$GitPath = Resolve-GitPath
$TemporaryRoot = Join-Path $env:TEMP "geo-identity-grammar-$PID-$([Guid]::NewGuid().ToString('N'))"
$GeneratedRoot = Join-Path $TemporaryRoot "generated"
$TemporaryRunner = Join-Path $TemporaryRoot "run_identity_search.generated.ps1"
$TemporaryLog = Join-Path $TemporaryRoot "grammar-generation.log"
$TestLog = Join-Path $TemporaryRoot "grammar-tests.log"
$EvidenceRoot = Join-Path $BenchmarkDirectory "evidence"
$GeneratedHeader = Join-Path $BenchmarkDirectory "generated\geo_identity_corpus.cuh"

$GrammarPaths = @(
    Join-Path $RepositoryRoot "experiments\geometric_identity_engine_v3\grammars\01_vector_product_relations.json"
    Join-Path $RepositoryRoot "experiments\geometric_identity_engine_v3\grammars\02_general_reversion_relations.json"
    Join-Path $RepositoryRoot "experiments\geometric_identity_engine_v3\grammars\03_commutator_jacobi.json"
)
foreach ($Path in $GrammarPaths) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing grammar: $Path" }
}
if (-not (Test-Path -LiteralPath $BaseRunner)) { throw "Missing base CUDA runner: $BaseRunner" }
if (-not (Test-Path -LiteralPath $GrammarTool)) { throw "Missing grammar tool: $GrammarTool" }

New-Item -ItemType Directory -Force -Path $TemporaryRoot | Out-Null
New-Item -ItemType Directory -Force -Path $EvidenceRoot | Out-Null

Push-Location $RepositoryRoot
try {
    $Branch = (& $GitPath branch --show-current).Trim()
    if ($Branch -ne $ExpectedBranch) {
        throw "Current branch is '$Branch'; expected '$ExpectedBranch'"
    }

    $BuildArguments = New-Object System.Collections.Generic.List[string]
    $BuildArguments.Add("build-corpus")
    foreach ($GrammarPath in $GrammarPaths) {
        $BuildArguments.Add("--grammar")
        $BuildArguments.Add($GrammarPath)
    }
    $BuildArguments.Add("--output-dir")
    $BuildArguments.Add($GeneratedRoot)
    foreach ($Prime in $Primes) {
        $BuildArguments.Add("--prime")
        $BuildArguments.Add($Prime.ToString())
    }
    $BuildArguments.Add("--precheck-assignments")
    $BuildArguments.Add($PrecheckAssignments.ToString())
    $BuildArguments.Add("--term-limit")
    $BuildArguments.Add($PolynomialTermLimit.ToString())
    $BuildArguments.Add("--max-relations")
    $BuildArguments.Add($MaxRelations.ToString())
    $BuildArguments.Add("--max-controls")
    $BuildArguments.Add($MaxControls.ToString())
    $BuildArguments.Add("--clean")

    Invoke-VisibleCommand `
        -Command { & $PythonPath $GrammarTool @BuildArguments } `
        -LogPath $TemporaryLog

    Invoke-VisibleCommand `
        -Command {
            & $PythonPath -m unittest `
                tests.test_geo_identity_compiler `
                tests.test_geo_identity_discovery `
                tests.test_identity_result_summarizer `
                tests.test_geo_identity_grammar_discovery
        } `
        -LogPath $TestLog

    $ManifestPath = Join-Path $GeneratedRoot "corpus-manifest.json"
    if (-not (Test-Path -LiteralPath $ManifestPath)) {
        throw "Grammar corpus manifest was not generated"
    }

    $RunnerText = Get-Content -LiteralPath $BaseRunner -Raw
    $ScriptDirectoryLiteral = $ScriptDirectory.Replace("'", "''")
    $OriginalDirectoryLine = '$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path'
    $ReplacementDirectoryLine = '$ScriptDirectory = ''' + $ScriptDirectoryLiteral + ''''
    if (-not $RunnerText.Contains($OriginalDirectoryLine)) {
        throw "Base runner script-directory marker was not found"
    }
    $RunnerText = $RunnerText.Replace($OriginalDirectoryLine, $ReplacementDirectoryLine)

    $StartMarker = '    $IdentityArguments = @('
    $InvocationMarker = '    Invoke-LoggedNativeCommand `'
    $StartIndex = $RunnerText.IndexOf($StartMarker, [StringComparison]::Ordinal)
    if ($StartIndex -lt 0) { throw "Base runner generator block was not found" }
    $FirstInvocation = $RunnerText.IndexOf($InvocationMarker, $StartIndex, [StringComparison]::Ordinal)
    $SecondInvocation = $RunnerText.IndexOf($InvocationMarker, $FirstInvocation + 1, [StringComparison]::Ordinal)
    if ($FirstInvocation -lt 0 -or $SecondInvocation -lt 0) {
        throw "Base runner generator invocation boundaries were not found"
    }

    $ManifestLiteral = $ManifestPath.Replace("'", "''")
    $ReplacementGenerator = @'
    Invoke-LoggedNativeCommand `
        -Command {
            & $PythonPath .\tools\geo_identity_manifest_compiler.py `
                --manifest '__MANIFEST__' `
                --output .\benchmarks\geo_identity_search\generated\geo_identity_corpus.cuh `
                --python-checks __PYTHON_CHECKS__
        } `
        -LogPath (Join-Path $EvidenceDirectory "generator-check.log")

'@
    $ReplacementGenerator = $ReplacementGenerator.Replace(
        "__MANIFEST__",
        $ManifestLiteral
    ).Replace(
        "__PYTHON_CHECKS__",
        $PrecheckAssignments.ToString()
    )
    $RunnerText =
        $RunnerText.Substring(0, $StartIndex) +
        $ReplacementGenerator +
        $RunnerText.Substring($SecondInvocation)
    Set-Content -LiteralPath $TemporaryRunner -Value $RunnerText -Encoding UTF8

    $Before = @(
        Get-ChildItem -LiteralPath $EvidenceRoot -Directory -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName
    )
    & $TemporaryRunner `
        -Device $Device `
        -Assignments $Assignments `
        -CpuChecks $CpuChecks `
        -BlockSize $BlockSize `
        -CudaArchitectures $CudaArchitectures `
        -ExpectedBranch $ExpectedBranch
    if (-not $?) {
        throw "Generated CUDA runner failed"
    }

    $EvidenceDirectory = Get-ChildItem -LiteralPath $EvidenceRoot -Directory |
        Where-Object {
            $_.FullName -notin $Before -and
            (Test-Path (Join-Path $_.FullName "identity-search.csv"))
        } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $EvidenceDirectory) {
        $EvidenceDirectory = Get-ChildItem -LiteralPath $EvidenceRoot -Directory |
            Where-Object { Test-Path (Join-Path $_.FullName "identity-search.csv") } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
    }
    if (-not $EvidenceDirectory) { throw "Unable to locate completed CUDA evidence" }

    $DiscoveryDirectory = Join-Path $EvidenceDirectory.FullName "grammar-discovery"
    New-Item -ItemType Directory -Force -Path $DiscoveryDirectory | Out-Null
    Copy-Item -Path (Join-Path $GeneratedRoot "*") -Destination $DiscoveryDirectory -Recurse -Force
    Copy-Item -LiteralPath $TemporaryLog -Destination $DiscoveryDirectory -Force
    Copy-Item -LiteralPath $TestLog -Destination $DiscoveryDirectory -Force

    $RawCsv = Join-Path $EvidenceDirectory.FullName "identity-search.csv"
    $DiscoveryManifest = Join-Path $DiscoveryDirectory "corpus-manifest.json"
    Invoke-VisibleCommand `
        -Command {
            & $PythonPath .\tools\geo_identity_discovery.py reduce-results `
                --manifest $DiscoveryManifest `
                --csv $RawCsv `
                --output-json (Join-Path $EvidenceDirectory.FullName "grammar-discovery-report.json") `
                --markdown-out (Join-Path $EvidenceDirectory.FullName "grammar-discovery-summary.md")
        } `
        -LogPath (Join-Path $EvidenceDirectory.FullName "grammar-discovery-validation.log")

    $Manifest = Join-Path $EvidenceDirectory.FullName "sha256-manifest.csv"
    $ManifestHash = Join-Path $EvidenceDirectory.FullName "sha256-manifest.sha256.txt"
    Remove-Item $Manifest -Force -ErrorAction SilentlyContinue
    Remove-Item $ManifestHash -Force -ErrorAction SilentlyContinue
    $Files = @(
        Get-ChildItem $EvidenceDirectory.FullName -Recurse -File |
            Where-Object { $_.FullName -ne $Manifest -and $_.FullName -ne $ManifestHash } |
            Sort-Object FullName
    )
    $Files |
        ForEach-Object { Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 } |
        Select-Object Path, Algorithm, Hash |
        Export-Csv -LiteralPath $Manifest -NoTypeInformation -Encoding UTF8
    Get-FileHash -LiteralPath $Manifest -Algorithm SHA256 |
        Format-List |
        Out-String |
        Set-Content -LiteralPath $ManifestHash -Encoding UTF8

    if ($Archive) {
        $ArchivePath = "$($EvidenceDirectory.FullName).zip"
        $ArchiveHash = "$ArchivePath.sha256.txt"
        Remove-Item $ArchivePath -Force -ErrorAction SilentlyContinue
        Remove-Item $ArchiveHash -Force -ErrorAction SilentlyContinue
        Compress-Archive -Path (Join-Path $EvidenceDirectory.FullName "*") -DestinationPath $ArchivePath -Force
        Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256 |
            Format-List |
            Out-String |
            Set-Content -LiteralPath $ArchiveHash -Encoding UTF8
    }

    Write-Host ""
    Write-Host "GEO_IDENTITY_GRAMMAR_DISCOVERY,status=complete"
    Write-Host $EvidenceDirectory.FullName
    Write-Host "Manifest: $Manifest"
    Write-Host "Manifest hash: $ManifestHash"
    if ($Archive) {
        Write-Host "Archive: $ArchivePath"
        Write-Host "Archive hash: $ArchiveHash"
    }
}
finally {
    Pop-Location
    if (Test-Path -LiteralPath $GeneratedHeader) {
        & $GitPath -C $RepositoryRoot restore -- "benchmarks/geo_identity_search/generated/geo_identity_corpus.cuh" 2>$null
    }
    Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
}
