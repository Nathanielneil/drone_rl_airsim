@echo off
REM Windows 10 + CUDA 12.1 专用安装脚本
REM 适用于 AirSim 1.8.1 + UE4.7.2 环境

echo ========================================
echo Drone RL AirSim - CUDA 12.1 安装脚本
echo 适用于 Windows 10 + AirSim 1.8.1 + UE4.7.2
echo ========================================
echo.

REM 检查Python版本
echo 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python未安装或未添加到PATH
    echo 请从 https://python.org 下载并安装Python 3.8-3.11
    pause
    exit /b 1
)

python -c "import sys; major, minor = sys.version_info[:2]; exit(0 if 3.8 <= (major*10 + minor/10) <= 3.11 else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python版本需要3.8-3.11
    python -c "import sys; print(f'当前版本: {sys.version}')"
    pause
    exit /b 1
)

echo ✓ Python版本检查通过

REM 检查CUDA
echo.
echo 检查CUDA环境...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: 未检测到NVIDIA GPU或驱动未安装
    echo 将安装CPU版本，性能会降低
    set USE_CUDA=false
) else (
    echo ✓ 检测到NVIDIA GPU
    nvidia-smi | findstr "CUDA Version: 12.1" >nul 2>&1
    if errorlevel 1 (
        echo WARNING: CUDA版本可能不是12.1
        echo 当前CUDA信息:
        nvidia-smi | findstr "CUDA Version"
        echo 建议安装CUDA 12.1以获得最佳兼容性
    ) else (
        echo ✓ CUDA 12.1检测通过
    )
    set USE_CUDA=true
)

REM 创建虚拟环境
echo.
echo 创建虚拟环境...
if exist venv (
    echo WARNING: 虚拟环境已存在，将删除并重新创建
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ERROR: 虚拟环境创建失败
    pause
    exit /b 1
)

echo ✓ 虚拟环境创建成功

REM 激活虚拟环境
echo.
echo 激活虚拟环境...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: 虚拟环境激活失败
    pause
    exit /b 1
)

echo ✓ 虚拟环境已激活

REM 升级pip
echo.
echo 升级pip工具...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: pip升级失败
    pause
    exit /b 1
)

echo ✓ pip工具升级完成

REM 安装PyTorch
echo.
if "%USE_CUDA%"=="true" (
    echo 安装CUDA 12.1版本PyTorch...
    echo 这可能需要几分钟，请耐心等待...
    
    REM 卸载可能存在的CPU版本
    pip uninstall torch torchvision torchaudio -y >nul 2>&1
    
    REM 安装CUDA版本
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo ERROR: CUDA版本PyTorch安装失败，尝试安装CPU版本
        pip install torch torchvision torchaudio
    ) else (
        echo ✓ CUDA版本PyTorch安装成功
    )
) else (
    echo 安装CPU版本PyTorch...
    pip install torch torchvision torchaudio
    if errorlevel 1 (
        echo ERROR: PyTorch安装失败
        pause
        exit /b 1
    )
    echo ✓ CPU版本PyTorch安装成功
)

REM 验证PyTorch
echo.
echo 验证PyTorch安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
if errorlevel 1 (
    echo ERROR: PyTorch验证失败
    pause
    exit /b 1
)

REM 安装基础依赖
echo.
echo 安装基础依赖包...
pip install -r requirements/base.txt
if errorlevel 1 (
    echo ERROR: 基础依赖安装失败
    pause
    exit /b 1
)

echo ✓ 基础依赖安装完成

REM 安装AirSim特定版本
echo.
echo 安装AirSim 1.8.1...
pip install airsim==1.8.1
if errorlevel 1 (
    echo WARNING: AirSim 1.8.1安装失败，尝试最新版本
    pip install airsim>=1.8.0
)

echo ✓ AirSim安装完成

REM 安装CUDA特定优化包（如果使用GPU）
if "%USE_CUDA%"=="true" (
    echo.
    echo 安装GPU优化包...
    pip install gpustat pynvml
    echo ✓ GPU优化包安装完成
)

REM 创建必要目录
echo.
echo 创建项目目录...
if not exist data mkdir data
if not exist data\training mkdir data\training  
if not exist data\evaluation mkdir data\evaluation
if not exist models mkdir models
if not exist models\checkpoints mkdir models\checkpoints
if not exist experiments\logs mkdir experiments\logs
if not exist experiments\results mkdir experiments\results

echo ✓ 项目目录创建完成

REM 配置AirSim
echo.
echo 配置AirSim设置...
if not exist "%USERPROFILE%\Documents\AirSim" mkdir "%USERPROFILE%\Documents\AirSim"
if exist config\airsim_settings.json (
    copy config\airsim_settings.json "%USERPROFILE%\Documents\AirSim\settings.json" >nul
    echo ✓ AirSim配置文件已复制
) else (
    echo WARNING: 未找到AirSim配置文件，请手动配置
)

REM 运行安装测试
echo.
echo 运行安装验证测试...
python scripts\test_installation.py
if errorlevel 1 (
    echo WARNING: 某些测试失败，请检查安装日志
) else (
    echo ✓ 所有测试通过
)

REM 完成安装
echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 环境信息:
python -c "import torch, airsim; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'AirSim: 已安装')"
echo.
echo 下一步操作:
echo 1. 下载并启动AirSim环境 (如Blocks.exe)
echo 2. 运行快速测试: python scripts\train_sac_gpu.py
echo 3. 查看文档: docs\user_guide\windows_deployment.md
echo.
echo 激活环境命令: venv\Scripts\activate.bat
echo.
pause