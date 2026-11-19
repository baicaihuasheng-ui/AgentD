param(
    [ValidateSet('verify','qna','pipeline')][string]$Mode='verify',
    [switch]$AllowLive
)
$ErrorActionPreference='Stop'
$env:TEMP='D:\CodexData\Temp'
$env:TMP=$env:TEMP
$env:PYTHONNOUSERSITE='1'
$env:HF_HOME='D:\CodexData\Temp\hf-cache-agentd-local'
$env:MPLCONFIGDIR='D:\CodexData\Temp\mpl-agentd-local'
$taskRoot=Split-Path $PSScriptRoot -Parent
Push-Location $taskRoot
try {
    if ($Mode -eq 'verify') {
        & 'D:\anaconda\Scripts\conda.exe' run --no-capture-output -n agentd-deepseek python scripts/verify_local.py
    } else {
        if (-not $AllowLive) {throw 'Live API calls require -AllowLive after confirming test costs.'}
        $taskEnvFile=Join-Path $taskRoot '.env.deepseek'
        if (Test-Path -LiteralPath $taskEnvFile) {$env:AGENTD_ENV_FILE=$taskEnvFile}
        if ($Mode -eq 'qna') {
            & 'D:\anaconda\Scripts\conda.exe' run --no-capture-output -n agentd-deepseek python run_agentd.py --qna --config pipeline_config.deepseek.yaml
        } else {
            & 'D:\anaconda\Scripts\conda.exe' run --no-capture-output -n agentd-deepseek python run_agentd.py --config pipeline_config.deepseek.yaml
        }
    }
    if ($LASTEXITCODE -ne 0) {throw "AgentD process exited with code $LASTEXITCODE"}
} finally {Pop-Location}
