# simple_rag.py
import os

# 1. 导入必要的模块
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope

api_key = os.getenv("DASHSCOPE_API_KEY")

Settings.llm = DashScope(
    model_name="qwen-plus",
    api_key=api_key,
)
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v2",  # Qwen的embedding模型
    api_key=api_key,  # 从阿里云获取
)

def main():
    # 2. 加载文档 (从本地的 './data' 文件夹)
    # 确保你已经在当前目录下创建了一个名为 'data' 的文件夹，
    # 并往里放了一些 .txt, .pdf 或 .md 文件。
    print("正在从 './data' 文件夹加载文档...")
    reader = SimpleDirectoryReader(input_dir="F:\\文件\\Obsidian仓库\\test")
    documents = reader.load_data()
    print(f"成功加载 {len(documents)} 个文档文件。")

    # 3. (可选但推荐) 将文档切分为更小的节点 (Node)
    # 这一步是为了提高检索的精准度 [citation:1]
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"文档被切分为 {len(nodes)} 个节点。")
    
    # 简便方法：直接在创建索引时进行切分，LlamaIndex会使用默认的切分器
    print("正在创建索引（这会将文档切分并生成向量）...")
    # 这里传入的是documents，LlamaIndex内部会自动处理切分
    index = VectorStoreIndex(nodes)
    print("索引创建完成！")

    # 4. 创建查询引擎
    # 查询引擎是我们进行问答的接口 [citation:4]
    query_engine = index.as_query_engine(similarity_top_k=3) # 检索时返回最相关的3个文本块作为上下文

    # 5. 开始问答循环
    print("\n" + "="*50)
    print(" RAG 问答系统已启动 (输入 'q' 退出)")
    print("="*50)
    while True:
        user_question = input("\n请输入你的问题: ")
        if user_question.lower() == 'q':
            break
        
        print("正在思考...")
        # 执行查询
        response = query_engine.query(user_question)
        
        # 打印答案
        print(f"\n🤖 答案: {response}")
        
        # 打印参考来源 (可选，帮助你理解答案是从哪里来的)
        print("\n📚 参考来源:")
        for i, source_node in enumerate(response.source_nodes):
            print(f"   {i+1}. 来自: {source_node.node.metadata.get('file_name', '未知文件')}")
            # 打印部分原文以作验证
            # print(f"     片段: {source_node.node.text[:150]}...") 

if __name__ == "__main__":
    main()