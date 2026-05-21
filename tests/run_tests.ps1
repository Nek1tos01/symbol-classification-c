$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

& ".\scripts\prepare_data.ps1"

Write-Host "[build] gcc"
gcc -std=c11 -O2 -Wall -Wextra -pedantic -Iinclude src/*.c -o nn_classifier.exe -lm
if ($LASTEXITCODE -ne 0) {
    throw "Build failed"
}

function Invoke-Case($name, $config) {
    Write-Host "[run] $name"
    Remove-Item -Force -ErrorAction SilentlyContinue heatmap.txt, heatmap.pgm, loss_history.txt, metrics_history.txt
    $output = & ".\nn_classifier.exe" $config 2>&1
    $output | Set-Content -Encoding UTF8 "tests\last_$name.log"
    if ($LASTEXITCODE -ne 0) {
        $output
        throw "$name failed"
    }
    if (-not (Test-Path -LiteralPath "heatmap.txt")) {
        throw "$name did not create heatmap.txt"
    }
    if (-not (Test-Path -LiteralPath "heatmap.pgm")) {
        throw "$name did not create heatmap.pgm"
    }
    if (-not (Test-Path -LiteralPath "loss_history.txt")) {
        throw "$name did not create loss_history.txt"
    }
    if (-not (Test-Path -LiteralPath "metrics_history.txt")) {
        throw "$name did not create metrics_history.txt"
    }
    Write-Host "[pass] $name"
}

Invoke-Case "synthetic" "tests\config_synthetic.txt"
Invoke-Case "mnist" "tests\config_mnist_quick.txt"

Write-Host "[run] invalid-config"
$badOutput = cmd /c ".\nn_classifier.exe tests\config_invalid.txt 2>&1"
$badCode = $LASTEXITCODE
if ($badCode -eq 0) {
    $badOutput
    throw "invalid-config unexpectedly succeeded"
}
Write-Host "[pass] invalid-config"

Write-Host "All tests passed"
