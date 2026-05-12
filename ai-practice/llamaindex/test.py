# full_diagnostic.py
import os
import sys
from pathlib import Path

print("=" * 60)
print("LlamaParse 诊断工具")
print("=" * 60)

# 1. 检查Python环境
print(f"\n1. Python版本: {sys.version}")
print(f"   当前目录: {os.getcwd()}")

# 2. 检查已安装的包
print("\n2. 检查安装的包:")
try:
    import llama_parse
except ImportError:
    print("   ❌ llama-parse 未安装，请运行: pip install llama-parse")

# 3. 检查API密钥
print("\n3. 检查API密钥:")
api_key = os.getenv("LLAMA_CLOUD_API_KEY")
if api_key:
    print(f"   ✅ 环境变量已设置: {api_key[:10]}...")
    print(f"   密钥长度: {len(api_key)}")
    print(f"   格式正确: {'✓' if api_key.startswith('llx-') else '✗'}")
else:
    print("   ❌ 环境变量 LLAMA_CLOUD_API_KEY 未设置")

# 4. 检查文件
print("\n4. 检查数据文件:")
data_dir = Path("F:\\文件\\微信备份\\xwechat_files\\liyongping5933_05bf\\msg\\file\\2024-04")
if data_dir.exists():
    pdf_files = list(data_dir.glob("*.pdf"))
    print(f"   📁 data目录存在，找到 {len(pdf_files)} 个PDF文件")
    for pdf in pdf_files[:3]:
        print(f"      - {pdf.name} ({pdf.stat().st_size / 1024:.1f}KB)")
else:
    print("   ❌ ./data 目录不存在")

# 5. 测试API连接
print("\n5. 测试API连接:")
if api_key:
    try:
        from llama_parse import LlamaParse
        import httpx

        # 测试API健康检查
        headers = {"Authorization": f"Bearer {api_key}"}
        response = httpx.get(
            "https://api.cloud.llamaindex.ai/api/v1/health",
            headers=headers,
            timeout=10
        )
        print(f"   健康检查状态: {response.status_code}")

        if response.status_code == 200:
            print("   ✅ API服务正常")

            # 尝试简单的解析
            print("\n6. 尝试解析PDF:")
            parser = LlamaParse(
                api_key=api_key,
                result_type="markdown",
                verbose=True
            )

            if pdf_files:
                doc = parser.load_data(str(pdf_files[0]))
                print(f"   ✅ 解析成功，生成 {len(doc)} 个文档")
            else:
                print("   ⚠️ 没有PDF文件可解析")
        else:
            print(f"   ❌ API返回错误: {response.text}")

    except Exception as e:
        print(f"   ❌ 连接失败: {e}")
else:
    print("   ⚠️ 跳过API测试（无API密钥）")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)