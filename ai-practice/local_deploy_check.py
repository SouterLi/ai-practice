# 先验证NumPy版本
import numpy
import transformers
import importlib.metadata
import os
import sys
import torch

print("NumPy版本：", numpy.__version__)  # 应显示1.26.4

# 验证PyTorch
import torch
print("PyTorch版本：", torch.__version__)
print("CUDA是否可用：", torch.cuda.is_available())  # 应返回True（有GPU）

# 验证transformers

from transformers import AutoTokenizer, AutoModelForCausalLM
print("transformers版本：", transformers.__version__)  # 应显示4.36.2
print("库加载成功！")

# 检查Python包安装路径
print("Python包安装路径：", sys.path)

# 手动查找bitsandbytes
try:
    # 检查是否能导入
    import bitsandbytes

    print("bitsandbytes导入成功，路径：", bitsandbytes.__file__)

    # 检查元数据（模拟报错逻辑）
    version = importlib.metadata.version("bitsandbytes")
    print("bitsandbytes版本：", version)
except importlib.metadata.PackageNotFoundError:
    print("元数据缺失！手动修复...")
    # 手动生成元数据（兜底）
    pkg_path = os.path.dirname(os.path.dirname(bitsandbytes.__file__))
    print(f"尝试在 {pkg_path} 生成元数据")
except ImportError:
    print("bitsandbytes未安装！请重新执行方案1")

# 检查CUDA是否可用
print("CUDA是否可用：", torch.cuda.is_available())

# 检查显卡信息和算力
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("pytorch版本：", torch.__version__)
    print("显卡名称：", torch.cuda.get_device_name(0))
    print("显卡算力：", torch.cuda.get_device_capability(0))  # 应输出(12, 0)即sm_120
    print("PyTorch支持的算力：", torch.cuda.get_arch_list())  # 需包含'compute_120'
else:
    print("仍未识别显卡，请检查PyTorch版本和CUDA驱动")