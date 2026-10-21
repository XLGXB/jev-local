[CmdletBinding()]
param(
    [string] $Python,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

$ErrorActionPreference = "Stop"
$project = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $Python) {
    $Python = Join-Path $project ".venv/Scripts/python.exe"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Supply -Python with an authorized isolated interpreter, or prepare this project's .venv. No automatic install or system fallback."
}
$previous = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $project "src"
    & $Python -B -X utf8 -m jevlocal.integration.cli @Arguments
    $code = $LASTEXITCODE
}
finally {
    $env:PYTHONPATH = $previous
}
exit $code
