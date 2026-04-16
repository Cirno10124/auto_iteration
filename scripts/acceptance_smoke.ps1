param(
    [string]$ConfigPath = "config_template/orchestrator_config.test.json",
    [switch]$SkipPipInstall,
    [switch]$SkipPytest,
    [switch]$SkipGpuCheck
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Name,
        [string]$Command
    )
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    Write-Host $Command -ForegroundColor DarkGray
    Invoke-Expression $Command
}

Write-Host "验收冒烟脚本（PowerShell）" -ForegroundColor Green
Write-Host "ConfigPath=$ConfigPath"

if (-not (Test-Path $ConfigPath)) {
    throw "配置文件不存在: $ConfigPath"
}

Run-Step -Name "Python 版本检查" -Command "python --version"

if (-not $SkipPipInstall) {
    Run-Step -Name "安装依赖" -Command "pip install -r requirements.txt"
}

if (-not $SkipGpuCheck) {
    Run-Step -Name "GPU 检查（失败不终止）" -Command "nvidia-smi"
}

Run-Step -Name "输出当前版本" -Command "git rev-parse --short HEAD"
Run-Step -Name "执行单次全流程冒烟" -Command "python orchestrator.py --config $ConfigPath --override iteration.once=true"

if (-not $SkipPytest) {
    Run-Step -Name "执行轻量测试集" -Command "pytest -q"
}

Write-Host ""
Write-Host "=== 冒烟执行完成 ===" -ForegroundColor Green
Write-Host "请按 docs/验收部署与测试清单.md 核对产物目录、日志和评估报告。"
