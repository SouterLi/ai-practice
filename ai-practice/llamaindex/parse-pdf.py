# llama_parse_demo.py
import os
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope

# 配置embedding
Settings.llm = DashScope(
    model_name="qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v2",  # Qwen的embedding模型
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从阿里云获取
)

# 1. 使用LlamaParse解析PDF
parser = LlamaParse(
    result_type="markdown",  # 返回格式：markdown, text, json
    num_workers=4,  # 并行处理
    verbose=True,
    language="ch_sim",  # 文档语言
    parsing_instruction="""  
    这是一个pdf格式的文档，请特别注意：
    1. 保持表格结构完整
    2. 提取所有图片中的文字
    3. 保留标题层级关系
    """
)

# 2. 使用解析器加载文档
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    input_dir="./files/",
    file_extractor=file_extractor
).load_data()

print(f"解析完成，生成 {len(documents)} 个文档对象")

# 3. 查看解析效果
for i, doc in enumerate(documents[:2]):  # 只看前两个
    print(f"\n--- 文档 {i+1} ---")
    print(f"元数据: {doc.metadata}")
    print(f"内容预览: {doc.text[:500]}...")

# 4. 创建索引
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
#
# # 5. 测试查询
# response = query_engine.query("表格3中的数据说明了什么趋势？")
# print(f"\n回答: {response}")