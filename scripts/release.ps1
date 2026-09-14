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

pre-commit run --all-files  # may fail after fixing files; the fixes are committed below
Invoke-Checked { git add -A }
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { Invoke-Checked { git commit -m $Message } }
Invoke-Checked { git push origin main }

$latest = git ls-remote --tags --refs --sort=-v:refname origin "v*" |
    Select-Object -First 1 |
    ForEach-Object { ($_ -split 'refs/tags/')[1] }
if (-not $latest) { throw "No v* tags found on origin" }
$parts = $latest.Split('.')
$parts[-1] = [int]$parts[-1] + 1
$tag = $parts -join '.'
Invoke-Checked { git tag $tag }
Invoke-Checked { git push origin $tag }

Write-Host "Released $tag"
