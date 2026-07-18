[CmdletBinding()]
param(
    [int]$Device = 0,
    [UInt64]$Assignments = 262144,
    [UInt64]$CpuChecks = 1024,
    [int]$BlockSize = 256,
    [int[]]$Primes = @(65521, 65519, 65497, 32749),
    [int]$MaxMutations = 8,
    [int]$PrecheckAssignments = 4096,
    [int]$PolynomialTermLimit = 200000,
    [string]$CudaArchitectures = "120",
    [string]$ExpectedBranch = "research/geometric-identity-engine-v2",
    [switch]$Archive
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-PythonPath {
    $Command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }
    $Command = Get-Command python -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }
    throw "Python was not found in PATH"
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
                $Text = if ($_ -is [System.Management.Automation.ErrorRecord]) {
                    $_.Exception.Message
                }
                else {
                    $_.ToString()
                }
                $Text
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

if ($Assignments -eq 0) {
    throw "Assignments must be positive"
}
if ($CpuChecks -gt $Assignments) {
    throw "CpuChecks cannot exceed Assignments"
}
if ($BlockSize -lt 1 -or $BlockSize -gt 1024) {
    throw "BlockSize must be in [1,1024]"
}
if ($MaxMutations -lt 0) {
    throw "MaxMutations cannot be negative"
}
if ($PrecheckAssignments -lt 0) {
    throw "PrecheckAssignments cannot be negative"
}
if ($PolynomialTermLimit -lt 1) {
    throw "PolynomialTermLimit must be positive"
}
if (-not $Primes -or $Primes.Count -eq 0) {
    throw "At least one prime is required"
}

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BenchmarkDirectory = Resolve-Path (Join-Path $ScriptDirectory "..")
$RepositoryRoot = Resolve-Path (Join-Path $BenchmarkDirectory "..\..")
$BaseRunner = Join-Path $ScriptDirectory "run_identity_search.ps1"
$DiscoveryTool = Join-Path $RepositoryRoot "tools\geo_identity_discovery.py"
$PythonPath = Resolve-PythonPath
$TemporaryRoot = Join-Path $env:TEMP "geo-identity-discovery-$PID-$([Guid]::NewGuid().ToString('N'))"
$GeneratedRoot = Join-Path $TemporaryRoot "generated"
$TemporaryRunner = Join-Path $TemporaryRoot "run_identity_search.generated.ps1"

$SourcePaths = @(
    Join-Path $RepositoryRoot "experiments\geometric_identity_engine\corpus\01_vector_square_scalar.json"
    Join-Path $RepositoryRoot "experiments\geometric_identity_engine\corpus\02_reverse_product_order.json"
    Join-Path $RepositoryRoot "experiments\geometric_identity_engine\corpus\03_vector_wedge_antisymmetry.json"
    Join-Path $RepositoryRoot "experiments\geometric_identity_engine\corpus\04_vector_product_commutativity_false.json"
    Join-Path $RepositoryRoot "experiments\geometric_identity_engine\corpus\05_mutated_reverse_order_false.json"
)

foreach ($SourcePath in $SourcePaths) {
    if (-not (Test-Path -LiteralPath $SourcePath)) {
        throw "Missing source identity: $SourcePath"
    }
}
if (-not (Test-Path -LiteralPath $BaseRunner)) {
    throw "Missing base CUDA runner: $BaseRunner"
}
if (-not (Test-Path -LiteralPath $DiscoveryTool)) {
    throw "Missing discovery tool: $DiscoveryTool"
}

New-Item -ItemType Directory -Force -Path $TemporaryRoot | Out-Null

try {
    Push-Location $RepositoryRoot
    try {
        $BuildArguments = New-Object System.Collections.Generic.List[string]
        $BuildArguments.Add("build-corpus")
        foreach ($SourcePath in $SourcePaths) {
            $BuildArguments.Add("--source")
            $BuildArguments.Add($SourcePath)
        }
        $BuildArguments.Add("--output-dir")
        $BuildArguments.Add($GeneratedRoot)
        foreach ($Prime in $Primes) {
            $BuildArguments.Add("--prime")
            $BuildArguments.Add($Prime.ToString())
        }
        $BuildArguments.Add("--precheck-assignments")
        $BuildArguments.Add($PrecheckAssignments.ToString())
        $BuildArguments.Add("--max-mutations")
        $BuildArguments.Add($MaxMutations.ToString())
        $BuildArguments.Add("--term-limit")
        $BuildArguments.Add($PolynomialTermLimit.ToString())
        $BuildArguments.Add("--clean")

        Invoke-VisibleCommand `
            -Command { & $PythonPath $DiscoveryTool @BuildArguments } `
            -LogPath (Join-Path $TemporaryRoot "discovery-corpus.log")

        Invoke-VisibleCommand `
            -Command {
                & $PythonPath -m unittest `
                    tests.test_geo_identity_compiler `
                    tests.test_geo_identity_discovery
            } `
            -LogPath (Join-Path $TemporaryRoot "discovery-tests.log")

        $GeneratedCorpus = Join-Path $GeneratedRoot "corpus"
        $GeneratedIdentities = @(
            Get-ChildItem -LiteralPath $GeneratedCorpus -Filter "*.json" -File |
                Sort-Object Name
        )
        if (-not $GeneratedIdentities -or $GeneratedIdentities.Count -eq 0) {
            throw "Discovery corpus generation produced no identity files"
        }

        $RunnerText = Get-Content -LiteralPath $BaseRunner -Raw
        $ScriptDirectoryLiteral = $ScriptDirectory.Replace("'", "''")
        $OriginalDirectoryLine = '$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path'
        $ReplacementDirectoryLine = '$ScriptDirectory = ''' + $ScriptDirectoryLiteral + ''''
        if (-not $RunnerText.Contains($OriginalDirectoryLine)) {
            throw "Base runner script-directory marker was not found"
        }
        $RunnerText = $RunnerText.Replace(
            $OriginalDirectoryLine,
            $ReplacementDirectoryLine
        )

        $StartMarker = '    $IdentityArguments = @('
        $EndMarker = '    Invoke-LoggedNativeCommand `'
        $StartIndex = $RunnerText.IndexOf($StartMarker, [StringComparison]::Ordinal)
        if ($StartIndex -lt 0) {
            throw "Base runner identity-argument start marker was not found"
        }
        $EndIndex = $RunnerText.IndexOf(
            $EndMarker,
            $StartIndex,
            [StringComparison]::Ordinal
        )
        if ($EndIndex -lt 0) {
            throw "Base runner identity-argument end marker was not found"
        }

        $GeneratedCorpusLiteral = $GeneratedCorpus.Replace("'", "''")
        $DynamicIdentityBlock = @"
    `$IdentityArguments = @()
    `$GeneratedIdentityPaths = @(
        Get-ChildItem -LiteralPath '$GeneratedCorpusLiteral' -Filter '*.json' -File |
            Sort-Object Name
    )
    if (-not `$GeneratedIdentityPaths -or `$GeneratedIdentityPaths.Count -eq 0) {
        throw 'Generated discovery corpus is empty'
    }
    foreach (`$IdentityPath in `$GeneratedIdentityPaths) {
        `$IdentityArguments += @('--identity', `$IdentityPath.FullName)
    }
    `$IdentityArguments += @(
        '--output', '.\benchmarks\geo_identity_search\generated\geo_identity_corpus.cuh',
        '--python-checks', '256'
    )

"@
        $RunnerText =
            $RunnerText.Substring(0, $StartIndex) +
            $DynamicIdentityBlock +
            $RunnerText.Substring($EndIndex)
        Set-Content -LiteralPath $TemporaryRunner -Value $RunnerText -Encoding UTF8

        $CapturedOutput = New-Object System.Collections.Generic.List[string]
        & $TemporaryRunner `
            -Device $Device `
            -Assignments $Assignments `
            -CpuChecks $CpuChecks `
            -BlockSize $BlockSize `
            -CudaArchitectures $CudaArchitectures `
            -ExpectedBranch $ExpectedBranch 2>&1 |
            ForEach-Object {
                $Text = if ($_ -is [System.Management.Automation.ErrorRecord]) {
                    $_.Exception.Message
                }
                else {
                    $_.ToString()
                }
                $CapturedOutput.Add($Text)
                Write-Host $Text
            }

        $EvidenceDirectory = $null
        for ($Index = 0; $Index -lt $CapturedOutput.Count - 1; ++$Index) {
            if ($CapturedOutput[$Index] -eq "Geometric identity evidence is complete:") {
                $EvidenceDirectory = $CapturedOutput[$Index + 1].Trim()
            }
        }
        if (-not $EvidenceDirectory -or -not (Test-Path -LiteralPath $EvidenceDirectory)) {
            throw "Unable to resolve the base runner evidence directory"
        }

        $DiscoveryEvidence = Join-Path $EvidenceDirectory "discovery"
        New-Item -ItemType Directory -Force -Path $DiscoveryEvidence | Out-Null
        Copy-Item -Path (Join-Path $GeneratedRoot "*") -Destination $DiscoveryEvidence -Recurse -Force
        Copy-Item -LiteralPath (Join-Path $TemporaryRoot "discovery-corpus.log") -Destination $DiscoveryEvidence -Force
        Copy-Item -LiteralPath (Join-Path $TemporaryRoot "discovery-tests.log") -Destination $DiscoveryEvidence -Force

        $RawCsv = Join-Path $EvidenceDirectory "identity-search.csv"
        $DiscoveryManifest = Join-Path $DiscoveryEvidence "corpus-manifest.json"
        $DiscoveryJson = Join-Path $EvidenceDirectory "discovery-report.json"
        $DiscoveryMarkdown = Join-Path $EvidenceDirectory "discovery-summary.md"

        Invoke-VisibleCommand `
            -Command {
                & $PythonPath $DiscoveryTool reduce-results `
                    --manifest $DiscoveryManifest `
                    --csv $RawCsv `
                    --output-json $DiscoveryJson `
                    --markdown-out $DiscoveryMarkdown
            } `
            -LogPath (Join-Path $EvidenceDirectory "discovery-validation.log")

        $Manifest = Join-Path $EvidenceDirectory "sha256-manifest.csv"
        $ManifestHash = Join-Path $EvidenceDirectory "sha256-manifest.sha256.txt"
        Remove-Item $Manifest -Force -ErrorAction SilentlyContinue
        Remove-Item $ManifestHash -Force -ErrorAction SilentlyContinue

        $FilesToHash = @(
            Get-ChildItem $EvidenceDirectory -Recurse -File |
                Where-Object {
                    $_.FullName -ne $Manifest -and
                    $_.FullName -ne $ManifestHash
                } |
                Sort-Object FullName
        )
        $Hashes = @(
            $FilesToHash |
                ForEach-Object {
                    Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 -ErrorAction Stop
                } |
                Select-Object Path, Algorithm, Hash
        )
        $Hashes | Export-Csv `
            -LiteralPath $Manifest `
            -NoTypeInformation `
            -Encoding UTF8
        Get-FileHash -LiteralPath $Manifest -Algorithm SHA256 |
            Format-List |
            Out-String |
            Set-Content -LiteralPath $ManifestHash -Encoding UTF8

        if ($Archive) {
            $ArchivePath = "$EvidenceDirectory.zip"
            $ArchiveHash = "$ArchivePath.sha256.txt"
            Remove-Item $ArchivePath -Force -ErrorAction SilentlyContinue
            Remove-Item $ArchiveHash -Force -ErrorAction SilentlyContinue
            Compress-Archive `
                -Path (Join-Path $EvidenceDirectory "*") `
                -DestinationPath $ArchivePath `
                -Force
            Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256 |
                Format-List |
                Out-String |
                Set-Content -LiteralPath $ArchiveHash -Encoding UTF8
        }

        Write-Host ""
        Write-Host "Geometric identity discovery evidence is complete:"
        Write-Host $EvidenceDirectory
        Write-Host "Generated statements: $($GeneratedIdentities.Count)"
        Write-Host "Manifest: $Manifest"
        Write-Host "Manifest hash: $ManifestHash"
        if ($Archive) {
            Write-Host "Archive: $ArchivePath"
            Write-Host "Archive hash: $ArchiveHash"
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
}
