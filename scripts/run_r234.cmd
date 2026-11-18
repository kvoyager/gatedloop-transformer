@echo off
REM The R ~ U{2,3,4} pair: does sampling R without ever forcing R = 1 keep the
REM loop gates high? Fresh K/V first (compare with looped_6_rs / looped_6),
REM then static K/V (compare with looped_6_skv_rs / looped_6_skv).
REM Each run lands in runs_ladder\<model>\version_N.
cd /d %~dp0\..
for %%M in (looped_6_r234 looped_6_skv_r234) do (
    echo.
    echo ===== %%M =====
    python -m gatedloop.train --config-name train_mix model=%%M %*
    if errorlevel 1 (
        echo run %%M failed with exit code %errorlevel%
        exit /b %errorlevel%
    )
)
