@echo off
title DNN_GD - Deep Neural Network from Scratch
cls
echo ======================================================================
echo         DNN + GD: Deep Neural Network Trained by Gradient Descent
echo                  Created from Scratch by Ilhan Kocaslan
echo ======================================================================
echo.

py -3.11 example.py
if %errorlevel% neq 0 (
    python example.py
)

pause
