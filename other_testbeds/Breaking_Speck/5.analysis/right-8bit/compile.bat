@echo off

nvcc kernel.cu helpers.cu -arch=sm_21
pause
