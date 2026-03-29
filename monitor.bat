@echo off
echo ========================================
echo PhaseBreak Weekly Scorecard Check
echo %date% %time%
echo ========================================
cd /d "D:\ВСЕ ПРОЕКТЫ ОПТИМЕЗАЦИЯ\PhaseBreak 2026"
python -m src.benchmark.verify_scorecard
echo.
echo Done. Press any key to close.
pause
