@echo off
schtasks /delete /tn "PhaseBreak_Monitor" /f >nul 2>&1
schtasks /create /tn "PhaseBreak_Monitor" /tr "C:\Users\serge\miniconda3\envs\ape311\python.exe \"D:\ВСЕ ПРОЕКТЫ ОПТИМЕЗАЦИЯ\PhaseBreak 2026\monitor_cron.py\" --dry-run" /sc daily /st 18:00 /f
if %errorlevel%==0 (
    echo SUCCESS: PhaseBreak_Monitor updated — daily at 18:00
    echo Path: D:\ВСЕ ПРОЕКТЫ ОПТИМЕЗАЦИЯ\PhaseBreak 2026\monitor_cron.py
) else (
    echo FAILED: Run as Administrator
)
pause
