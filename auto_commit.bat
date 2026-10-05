@echo off
title ExpressEval Auto-Commit Suite
echo ================================================================
echo  ExpressEval: Running Benchmark Evaluation & Auto-Commit
echo ================================================================
cd /d "%~dp0"
python scripts\auto_eval_commit.py
echo.
pause
