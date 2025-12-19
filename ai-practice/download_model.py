from modelscope.hub.snapshot_download import snapshot_download

# 模型名称（DeepSeek-R1-Distill-Qwen-7B）
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# 模型保存路径（自定义，比如D:\models\DeepSeek-R1-Distill-Qwen-7B）
save_dir = r"D:\deepseek-r1\DeepSeek-R1-Distill-Qwen-1.5B"

# 下载模型
snapshot_download(
    model_id=model_id,
    cache_dir=save_dir,
    revision="master"
)
print(f"模型已下载到：{save_dir}")