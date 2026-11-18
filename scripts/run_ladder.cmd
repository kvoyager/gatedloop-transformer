@echo off
REM The five-rung ladder on the mixture, in the order that gets the core result
REM first (looped_6 vs plain_6), then the upper bound, then the depth-reduction pair.
REM Each run lands in runs_ladder\<model>\version_N. Prerequisites:
REM   python scripts\train_tokenizer.py --mix gatedloop\configs\data\mix_fwe_code_math.yaml
REM   python scripts\prepare_mix.py     --data gatedloop\configs\data\mix_fwe_code_math.yaml
cd /d %~dp0\..
for %%M in (plain_6 looped_6 plain_12 looped_3 plain_3) do (
    echo.
    echo ===== %%M =====
    python -m gatedloop.train --config-name train_mix model=%%M %*
    if errorlevel 1 (
        echo run %%M failed with exit code %errorlevel%
        exit /b %errorlevel%
    )
)
