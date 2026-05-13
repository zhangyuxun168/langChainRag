@echo off
echo ========================================
echo IIS 部署环境配置脚本
echo ========================================
echo.

echo [1/3] 安装 wfastcgi 模块...
pip install wfastcgi

echo.
echo [2/3] 启用 wfastcgi...
wfastcgi-enable

echo.
echo [3/3] 配置完成！
echo.
echo ========================================
echo 接下来请按以下步骤操作：
echo ========================================
echo.
echo 1. 打开 IIS 管理器
echo 2. 创建新网站，物理路径指向项目目录
echo 3. 应用程序池选择 "无托管代码"
echo 4. 将生成的 web.config 文件放入项目根目录
echo.
echo 详细步骤请查看 IIS-DEPLOY.md 文档
echo.
pause
