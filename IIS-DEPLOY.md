# IIS 部署指南

本文档详细说明如何在 Windows Server / Windows 10/11 上使用 IIS 部署 LangChain RAG 系统。

## 📋 前置要求

- Windows Server 2012+ 或 Windows 10/11 专业版/企业版
- Python 3.8+ 已安装
- IIS 已安装并启用 CGI 功能
- 管理员权限

## 🔧 第一步：安装 IIS 和 CGI 功能

### Windows 10/11

1. 打开 **控制面板** → **程序** → **启用或关闭 Windows 功能**
2. 勾选以下功能：
   - ✅ Internet Information Services
   - ✅ Web 管理工具 → IIS 管理控制台
   - ✅ 万维网服务 → 应用程序开发功能 → CGI
3. 点击确定，等待安装完成

### Windows Server

1. 打开 **服务器管理器**
2. 点击 **添加角色和功能**
3. 选择 **Web 服务器 (IIS)**
4. 在角色服务中勾选 **CGI**
5. 完成安装

## 📦 第二步：安装 Python 依赖

```powershell
# 以管理员身份运行 PowerShell
cd C:\inetpub\wwwroot\langChainRag

# 安装依赖
pip install -r requirements.txt

# 安装 wfastcgi
pip install wfastcgi

# 启用 wfastcgi（会显示 FastCGI 配置信息）
wfastcgi-enable
```

**重要：** 记录 `wfastcgi-enable` 输出的路径，例如：
```
C:\Python310\python.exe|C:\Python310\Lib\site-packages\wfastcgi.py
```

## ⚙️ 第三步：配置 web.config

编辑项目根目录下的 `web.config` 文件：

```xml
<add name="PythonFastCGI" 
     path="*" 
     verb="*" 
     modules="FastCgiModule" 
     scriptProcessor="C:\Python310\python.exe|C:\Python310\Lib\site-packages\wfastcgi.py" 
     resourceType="Unspecified" 
     requireAccess="Script" />
```

**修改 `scriptProcessor` 为您的 Python 实际安装路径！**

同时修改 `PYTHONPATH` 为您的项目实际路径：
```xml
<add key="PYTHONPATH" value="C:\inetpub\wwwroot\langChainRag" />
```

## 🌐 第四步：创建 IIS 网站

### 方法1：使用 IIS 管理器（图形界面）

1. 打开 **IIS 管理器**（Win+R 输入 `inetmgr`）
2. 右键点击 **网站** → **添加网站**
3. 配置如下：
   - **网站名称**：LangChainRAG
   - **物理路径**：`C:\inetpub\wwwroot\langChainRag`
   - **绑定类型**：http
   - **端口**：80（或其他可用端口）
4. 点击确定

### 方法2：使用 PowerShell（命令行）

```powershell
# 创建应用程序池（无托管代码）
New-WebAppPool -Name "LangChainRAGPool"
Set-ItemProperty IIS:\AppPools\LangChainRAGPool -name managedRuntimeVersion -value ""

# 创建网站
New-Website -Name "LangChainRAG" `
            -PhysicalPath "C:\inetpub\wwwroot\langChainRag" `
            -ApplicationPool "LangChainRAGPool" `
            -Port 80

# 启动网站
Start-Website -Name "LangChainRAG"
```

## 🔐 第五步：配置文件夹权限

IIS 应用程序池标识需要读取/写入权限：

```powershell
# 获取 IIS 应用程序池标识（默认为 IIS AppPool\LangChainRAGPool）
$identity = "IIS AppPool\LangChainRAGPool"

# 授予项目目录权限
$path = "C:\inetpub\wwwroot\langChainRag"

# 读取/执行权限
icacls $path /grant "${identity}:(OI)(CI)RX"

# uploads 目录需要写入权限
icacls "$path\uploads" /grant "${identity}:(OI)(CI)M"

# chroma_db 目录需要写入权限
icacls "$path\chroma_db" /grant "${identity}:(OI)(CI)M"
```

或者在文件资源管理器中：
1. 右键点击项目文件夹 → **属性** → **安全**
2. 点击 **编辑** → **添加**
3. 输入 `IIS AppPool\LangChainRAGPool` → **检查名称** → **确定**
4. 勾选 **读取和执行**、**列出文件夹内容**、**读取**
5. 对 `uploads` 和 `chroma_db` 文件夹额外授予 **修改** 权限

## 🚀 第六步：配置环境变量

### 方法1：使用 .env 文件（推荐）

在项目根目录创建 `.env` 文件：

```env
APP_PORT=8000
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=sk-xxx
LLM_MODEL_NAME=qwen2.5:7b-instruct
CHROMA_DB_PATH=./chroma_db
OLLAMA_API_BASE=http://localhost:11434
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
TEMPERATURE=0.7
```

### 方法2：在 web.config 中配置

```xml
<appSettings>
  <add key="LLM_API_BASE" value="http://localhost:11434/v1" />
  <add key="LLM_MODEL_NAME" value="qwen2.5:7b-instruct" />
  <!-- 其他配置... -->
</appSettings>
```

## ✅ 第七步：测试访问

打开浏览器访问：
- 用户端：http://localhost/
- 管理端：http://localhost/admin/

如果看到页面，说明部署成功！

## 🔧 常见问题

### 1. 500.19 错误 - 配置文件无法读取

**原因**：web.config 格式错误或权限不足

**解决**：
- 检查 web.config XML 格式是否正确
- 确保 IIS 应用程序池有读取权限

### 2. 500.0 错误 - FastCGI 模块问题

**原因**：Python 路径配置错误

**解决**：
- 检查 `scriptProcessor` 中的 Python 路径是否正确
- 确认 wfastcgi.py 文件存在

### 3. 404 错误 - 静态文件无法访问

**原因**：静态文件处理配置问题

**解决**：
- 在 web.config 中添加静态文件处理规则
- 确保 frontend 目录有读取权限

### 4. 文件上传失败

**原因**：uploads 目录权限不足

**解决**：
```powershell
icacls "C:\inetpub\wwwroot\langChainRag\uploads" /grant "IIS AppPool\LangChainRAGPool:(OI)(CI)M"
```

### 5. 向量数据库写入失败

**原因**：chroma_db 目录权限不足

**解决**：
```powershell
icacls "C:\inetpub\wwwroot\langChainRag\chroma_db" /grant "IIS AppPool\LangChainRAGPool:(OI)(CI)M"
```

### 6. Ollama 连接失败

**原因**：Docker 容器或远程 Ollama 服务

**解决**：
- 如果 Ollama 在宿主机：使用 `http://host.docker.internal:11434`
- 如果 Ollama 在远程服务器：使用远程服务器 IP

## 📊 性能优化

### 1. 应用程序池设置

在 IIS 管理器中：
1. 选择应用程序池 → 高级设置
2. 调整以下参数：
   - **常规 → 启动模式**：AlwaysRunning（始终运行）
   - **进程模型 → 空闲超时**：0（不超时）
   - **进程模型 → 最大工作进程**：1（单进程，避免数据库冲突）

### 2. 回收设置

- **回收 → 固定时间间隔**：0（禁用定期回收）
- **回收 → 虚拟内存**：0（禁用虚拟内存回收）

### 3. FastCGI 设置

在 IIS 管理器中：
1. 选择服务器 → FastCGI 设置
2. 编辑 Python 应用程序
3. 调整：
   - **实例最大请求数**：10000
   - **请求超时**：300（秒）
   - **活动超时**：300（秒）

## 🔄 更新部署

当代码更新后：

```powershell
# 停止网站
Stop-Website -Name "LangChainRAG"

# 更新代码（git pull 或复制新文件）
cd C:\inetpub\wwwroot\langChainRag
git pull

# 安装新依赖（如有）
pip install -r requirements.txt

# 启动网站
Start-Website -Name "LangChainRAG"
```

## 🗑️ 卸载部署

```powershell
# 停止并删除网站
Stop-Website -Name "LangChainRAG"
Remove-Website -Name "LangChainRAG"

# 删除应用程序池
Remove-WebAppPool -Name "LangChainRAGPool"

# 禁用 wfastcgi
wfastcgi-disable

# 删除项目文件
Remove-Item -Path "C:\inetpub\wwwroot\langChainRag" -Recurse -Force
```

## 📝 检查清单

部署前请确认：

- [ ] IIS 已安装并启用 CGI 功能
- [ ] Python 3.8+ 已安装
- [ ] 所有依赖已安装（`pip install -r requirements.txt`）
- [ ] wfastcgi 已启用（`wfastcgi-enable`）
- [ ] web.config 中的 Python 路径正确
- [ ] web.config 中的项目路径正确
- [ ] IIS 应用程序池已创建并设置为"无托管代码"
- [ ] 项目文件夹权限已配置
- [ ] uploads 和 chroma_db 目录有写入权限
- [ ] .env 文件已创建并配置

## 🆘 获取帮助

如遇到问题：
1. 查看 IIS 日志：`C:\inetpub\logs\LogFiles`
2. 查看 Windows 事件查看器：应用程序日志
3. 检查 Python 版本：`python --version`
4. 检查 wfastcgi 路径：`wfastcgi-enable`
