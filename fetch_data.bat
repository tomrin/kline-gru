@echo off
REM K线训练营 - 采集股票数据（使用 conda ai_env 环境）
REM 双击此文件下载股票数据

cd /d F:\project\kline-studio
C:\Users\Lenovo\.conda\envs\ai_env\python.exe data_fetcher.py
pause
