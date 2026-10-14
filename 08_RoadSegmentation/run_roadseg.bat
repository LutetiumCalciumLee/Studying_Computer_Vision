@echo off
setlocal
set "ROADSEG_PYTHON=C:\Users\ljhlu\anaconda3\envs\vision_ai\python.exe"
if not exist "%ROADSEG_PYTHON%" (
    echo Road segmentation Python environment was not found: %ROADSEG_PYTHON%
    exit /b 1
)
"%ROADSEG_PYTHON%" "%~dp0Roads_Object_Detection.py" %*
exit /b %errorlevel%
