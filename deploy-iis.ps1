# IIS 快速部署脚本
# 使用方法：以管理员身份运行 PowerShell，然后执行此脚本

param(
    [string]$SiteName = "LangChainRAG",
    [string]$SitePath = "C:\inetpub\wwwroot\langChainRag",
    [int]$Port = 80
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IIS 快速部署脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查管理员权限
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "错误：请以管理员身份运行此脚本！" -ForegroundColor Red
    Write-Host "右键点击 PowerShell -> 以管理员身份运行" -ForegroundColor Yellow
    pause
    exit 1
}

# 1. 检查 IIS 是否安装
Write-Host "[1/7] 检查 IIS 安装状态..." -ForegroundColor Yellow
$iisFeature = Get-WindowsFeature -Name Web-Server -ErrorAction SilentlyContinue
if ($iisFeature -and $iisFeature.Installed) {
    Write-Host "  ✓ IIS 已安装" -ForegroundColor Green
} else {
    Write-Host "  ! IIS 未安装，正在安装..." -ForegroundColor Yellow
    Enable-WindowsOptionalFeature -Online -FeatureName IIS-WebServerRole -All
    Write-Host "  ✓ IIS 安装完成" -ForegroundColor Green
}

# 2. 检查 CGI 功能
Write-Host "[2/7] 检查 CGI 功能..." -ForegroundColor Yellow
$cgiFeature = Get-WindowsOptionalFeature -Online -FeatureName IIS-CGI -ErrorAction SilentlyContinue
if ($cgiFeature -and $cgiFeature.State -eq "Enabled") {
    Write-Host "  ✓ CGI 功能已启用" -ForegroundColor Green
} else {
    Write-Host "  ! CGI 功能未启用，正在启用..." -ForegroundColor Yellow
    Enable-WindowsOptionalFeature -Online -FeatureName IIS-CGI -All
    Write-Host "  ✓ CGI 功能已启用" -ForegroundColor Green
}

# 3. 安装 Python 依赖
Write-Host "[3/7] 安装 Python 依赖..." -ForegroundColor Yellow
Set-Location $SitePath
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
Write-Host "  ✓ Python 依赖安装完成" -ForegroundColor Green

# 4. 启用 wfastcgi
Write-Host "[4/7] 配置 wfastcgi..." -ForegroundColor Yellow
pip install wfastcgi
$output = wfastcgi-enable 2>&1
Write-Host "  ✓ wfastcgi 已启用" -ForegroundColor Green

# 提取 Python 路径
if ($output -match "C:\\[^|]+\\python\.exe\|[^|]+\\wfastcgi\.py") {
    $scriptProcessor = $matches[0]
    Write-Host "  FastCGI 路径: $scriptProcessor" -ForegroundColor Gray
}

# 5. 创建应用程序池
Write-Host "[5/7] 创建应用程序池..." -ForegroundColor Yellow
$appPoolName = "${SiteName}Pool"
try {
    $existingPool = Get-WebAppPool -Name $appPoolName -ErrorAction Stop
    Write-Host "  ! 应用程序池已存在，跳过创建" -ForegroundColor Yellow
} catch {
    New-WebAppPool -Name $appPoolName
    Set-ItemProperty IIS:\AppPools\$appPoolName -name managedRuntimeVersion -value ""
    Write-Host "  ✓ 应用程序池创建完成" -ForegroundColor Green
}

# 6. 创建网站
Write-Host "[6/7] 创建网站..." -ForegroundColor Yellow
try {
    $existingSite = Get-Website -Name $SiteName -ErrorAction Stop
    Write-Host "  ! 网站已存在，跳过创建" -ForegroundColor Yellow
} catch {
    New-Website -Name $SiteName `
                -PhysicalPath $SitePath `
                -ApplicationPool $appPoolName `
                -Port $Port
    Write-Host "  ✓ 网站创建完成" -ForegroundColor Green
}

# 7. 配置文件夹权限
Write-Host "[7/7] 配置文件夹权限..." -ForegroundColor Yellow
$identity = "IIS AppPool\$appPoolName"

# 授予项目目录读取权限
icacls $SitePath /grant "${identity}:(OI)(CI)RX" /T /C /Q

# 授予 uploads 目录修改权限
$uploadsPath = Join-Path $SitePath "uploads"
if (Test-Path $uploadsPath) {
    icacls $uploadsPath /grant "${identity}:(OI)(CI)M" /T /C /Q
}

# 授予 chroma_db 目录修改权限
$chromaPath = Join-Path $SitePath "chroma_db"
if (Test-Path $chromaPath) {
    icacls $chromaPath /grant "${identity}:(OI)(CI)M" /T /C /Q
}

Write-Host "  ✓ 文件夹权限配置完成" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "部署完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "访问地址：" -ForegroundColor Yellow
Write-Host "  用户端: http://localhost:$Port/" -ForegroundColor White
Write-Host "  管理端: http://localhost:$Port/admin/" -ForegroundColor White
Write-Host ""
Write-Host "注意事项：" -ForegroundColor Yellow
Write-Host "  1. 请检查 web.config 中的 Python 路径是否正确" -ForegroundColor White
Write-Host "  2. 请检查 web.config 中的项目路径是否正确" -ForegroundColor White
Write-Host "  3. 请创建 .env 文件并配置大模型参数" -ForegroundColor White
Write-Host ""
Write-Host "详细文档请查看: IIS-DEPLOY.md" -ForegroundColor Gray
Write-Host ""
pause
