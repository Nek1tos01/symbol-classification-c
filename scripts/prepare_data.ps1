$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $root "data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

$trainCsv = Join-Path $dataDir "mnist_train.csv"
$testCsv = Join-Path $dataDir "mnist_test.csv"

if (-not (Test-Path -LiteralPath $trainCsv)) {
    Expand-Archive -Force -LiteralPath (Join-Path $root "mnist_train.csv.zip") -DestinationPath $dataDir
}

if (-not (Test-Path -LiteralPath $testCsv)) {
    Expand-Archive -Force -LiteralPath (Join-Path $root "mnist_test.csv.zip") -DestinationPath $dataDir
}

Write-Host "MNIST CSV files are ready in $dataDir"
