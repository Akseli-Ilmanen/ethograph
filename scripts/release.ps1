param(
    [string]$Message = "fix: linting"
)

$ErrorActionPreference = "Stop"

# Native commands (git, pre-commit) don't honour $ErrorActionPreference; check exit codes.
function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Failed (exit $LASTEXITCODE): $Command" }
}

# Everything CI gates the PyPI upload on (lint + check-manifest + a buildable sdist/wheel)
# runs here first, so a broken release fails locally before a tag is pushed.

$branch = git rev-parse --abbrev-ref HEAD
if ($branch -ne "main") { throw "Release from main, not '$branch'" }

# The check-manifest executable is blocked by Application Control on this machine,
# so the hook is skipped and the module is run through python instead.
$env:SKIP = "check-manifest"
pre-commit run --all-files  # first pass may fail after fixing files
Invoke-Checked { pre-commit run --all-files }  # second pass must be clean
$env:SKIP = $null

Invoke-Checked { git add -A }
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { Invoke-Checked { git commit -m $Message } }

Write-Host "Checking MANIFEST.in against git..."
Invoke-Checked {
    uv run --no-project --with check-manifest --with "setuptools>=77" --with wheel `
        --with "setuptools-scm[toml]>=8" python -m check_manifest --no-build-isolation
}

# Mirrors tox: a fresh env with core + [dev] only. Collection fails (pytest exit 2/4)
# when a test or plugin imports something those extras don't install.
Write-Host "Collecting tests in a CI-like environment (core + dev extras)..."
Invoke-Checked {
    uv run --isolated --no-project -p 3.11 --with ".[dev]" python -m pytest --collect-only -q -p no:cacheprovider
}

Write-Host "Building sdist + wheel..."
$dist = Join-Path ([System.IO.Path]::GetTempPath()) "ethograph-release-dist"
if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
Invoke-Checked { uv build --out-dir $dist }

$wheel = Get-ChildItem "$dist/*.whl" | Select-Object -First 1
$required = @(
    "ethograph/segment/dlc2action/config/",
    "ethograph/segment/models/config/",
    "ethograph/defaults/"
)
$names = uv run --no-project python -c "import sys, zipfile; print('\n'.join(zipfile.ZipFile(sys.argv[1]).namelist()))" $wheel.FullName
foreach ($prefix in $required) {
    if (-not ($names | Where-Object { $_.StartsWith($prefix) })) {
        throw "Wheel is missing package data under $prefix (check [tool.setuptools.package-data])"
    }
}

Invoke-Checked { git push origin main }

$latest = git ls-remote --tags --refs --sort=-v:refname origin "v*" |
    Select-Object -First 1 |
    ForEach-Object { ($_ -split 'refs/tags/')[1] }
if (-not $latest) { throw "No v* tags found on origin" }
$parts = $latest.Split('.')
$parts[-1] = [int]$parts[-1] + 1
$tag = $parts -join '.'

$version = $tag.TrimStart('v')
$pypi = Invoke-RestMethod "https://pypi.org/pypi/ethograph/json" -ErrorAction SilentlyContinue
if ($pypi -and $pypi.releases.PSObject.Properties.Name -contains $version) {
    throw "$version is already on PyPI; PyPI never accepts the same version twice"
}

Invoke-Checked { git tag $tag }
Invoke-Checked { git push origin $tag }

Write-Host "Released $tag - watch: https://github.com/Akseli-Ilmanen/ethograph/actions"
