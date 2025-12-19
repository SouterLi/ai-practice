# 保留你已调好的GPU适配环境变量
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "sm_120;sm_90"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["USE_FBGEMM"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型路径（替换为你的本地路径）
model_path = r"D:\deepseek-r1\deepseek-ai\DeepSeek-R1-Distill-Qwen-1___5B"
device = "cuda"  # 强制使用GPU
torch_dtype = torch.float16

# 1. 加载tokenizer（保留你的配置）
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    padding_side="right",
    eos_token_id=151643,
    pad_token_id=151643
)

# 2. 加载模型（移除gradient_checkpointing参数）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
    # ✅ 已删除：gradient_checkpointing=True（Qwen2不支持该构造参数）
    # 保留你已配置的4bit量化
    # load_in_4bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.float16,
    # bnb_4bit_use_double_quant=True
)

# ✅ 正确启用梯度检查点（模型加载后单独调用方法）
model.gradient_checkpointing_enable()

print(f"✅ GPU模式已启用，显卡：{torch.cuda.get_device_name(0)}")
print("===== DeepSeek-R1-Distill-Qwen-7B 对话开始（输入exit退出）=====")


# 3. 对话函数（无任何无效参数）
def chat():
    while True:
        user_input = input("你：")
        if user_input.lower() == "exit":
            print("对话结束")
            break

        messages = [{"role": "user", "content": user_input}]
        encodeds = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        attention_mask = torch.ones_like(encodeds)

        with torch.no_grad():
            outputs = model.generate(
                encodeds,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
                # 无任何无效参数
            )

        response = tokenizer.decode(outputs[0][encodeds.shape[-1]:], skip_special_tokens=True)
        print(f"模型：{response.strip()}\n")


if __name__ == "__main__":
    chat()