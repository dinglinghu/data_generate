@echo off
echo ========================================
echo STK卫星星座数据采集系统启动脚本
echo ========================================
echo.

echo 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo 错误: Python未安装或未添加到PATH
    pause
    exit /b 1
)

echo.
echo 检查依赖包...
pip show pywin32 >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: pywin32未安装，正在安装...
    pip install pywin32
)

pip show PyYAML >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: PyYAML未安装，正在安装...
    pip install PyYAML
)

echo.
echo 创建输出目录...
if not exist "output" mkdir output
if not exist "output\data" mkdir output\data
if not exist "output\logs" mkdir output\logs
if not exist "output\visualization" mkdir output\visualization

echo.
echo 启动STK数据采集系统...
echo ========================================
python main.py

echo.
echo ========================================
echo 程序执行完成
echo ========================================
pause
