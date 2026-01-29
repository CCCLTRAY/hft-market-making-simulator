@echo off
echo 编译OBI增强做市策略模拟器...

:: 编译所有文件
g++ -std=c++11 -o market_maker_obi.exe ^
    market_maker_simulator.cpp ^
    price_simulator.cpp ^
    market_maker.cpp ^
    order_book.cpp ^
    -I. -O2

if %errorlevel% equ 0 (
    echo 编译成功！
    echo.
    echo 运行程序：
    echo   market_maker_obi.exe
    echo.
    echo 然后运行分析脚本：
    echo   python analyzer_enhanced.py
) else (
    echo 编译失败！
    echo 请检查是否安装了g++编译器。
)