@echo off
setlocal enabledelayedexpansion

:: ==============================================================================
:: GroundingDINO Portable Environment Setup Script (Conda / Windows)
:: ==============================================================================
:: Run from project root: environments\conda_setup\setup.bat

echo ==============================================================================
echo   🚀 Initializing GroundingDINO Portable Setup (Conda / Windows)
echo ==============================================================================
echo.

:: 1. Check for conda
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Conda is not installed or not in your PATH. Please install Miniconda or Anaconda first.
    exit /b 1
)
echo ✓ Conda detected.

:: 2. Setup paths
set "SCRIPT_DIR=%~dp0"
:: Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_ROOT=%SCRIPT_DIR%\..\.."

set "ENV_NAME=groundingdino_env"
set "ENV_FILE=%SCRIPT_DIR%\envs\environment_windows.yml"

:: 3. Create or update conda environment
echo.
echo ⏳ Setting up the Conda environment '%ENV_NAME%' from environment_windows.yml...
call conda env list | findstr /b /c:"%ENV_NAME% " >nul
if %ERRORLEVEL% equ 0 (
    echo Environment already exists. Updating it...
    call conda env update -f "%ENV_FILE%" --prune
) else (
    call conda env create -f "%ENV_FILE%"
)
if %ERRORLEVEL% neq 0 (
    echo ❌ Environment setup failed.
    exit /b 1
)
echo ✓ Environment setup complete.

:: 4. Activate environment
echo.
echo ⏳ Activating '%ENV_NAME%'...
call conda activate %ENV_NAME%
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to activate environment.
    exit /b 1
)
echo ✓ Activated conda environment.

:: 5. Verify nvcc availability and export CUDA_HOME
echo.
echo ⏳ Verifying CUDA compiler (nvcc)...
where nvcc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ nvcc not found in the environment. CUDA Toolkit installation may have failed.
    echo Ensure you have MSVC (Visual Studio Build Tools) installed on Windows for C++ compilation.
    exit /b 1
)
set "CUDA_HOME=%CONDA_PREFIX%"
echo ✓ nvcc verified.
echo ✓ CUDA_HOME exported as: %CUDA_HOME%

:: 6. Build the C++ extensions
echo.
echo ⏳ Building GroundingDINO C++ CUDA Extensions...
cd /d "%PROJECT_ROOT%\groundingdino"
call pip install -e . --no-build-isolation
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to build C++ CUDA Extensions.
    exit /b 1
)
cd /d "%PROJECT_ROOT%"
echo ✓ GroundingDINO C++ Extension Built Successfully!

:: 7. Download model weights
echo.
echo ⏳ Downloading GroundingDINO Model Weights...
python src\download_weights.py
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to download model weights.
    exit /b 1
)
echo ✓ Weights downloaded!

:: 8. Success message
echo.
echo ==============================================================================
echo   🎉 Setup Complete! The environment is ready.
echo ==============================================================================
echo You can now run the scripts in the src\ directory.
echo Don't forget to activate the environment before running your code:
echo.
echo     conda activate %ENV_NAME%
echo.
