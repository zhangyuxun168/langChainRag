from modelscope import snapshot_download

# 下载模型到你指定的文件夹，改成你自己的路径
model_dir = snapshot_download(
    model_id="AI-ModelScope/bge-small-zh-v1.5",
    local_dir=r"G:\langChainRag\backend\models\bge-small-zh-v1.5",  # 你的目标文件夹
    revision="master"
)

print(f"✅ 模型下载完成！文件夹路径：{model_dir}")