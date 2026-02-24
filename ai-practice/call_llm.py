import os

from openai import OpenAI
import dashscope
from dashscope import Generation
from langchain_community.llms import Tongyi

# 几种常见的调用大模型的方法

api_key = os.getenv("DASHSCOPE_API_KEY")
review = '这款产品真的非常好用，性价比高，客服态度也很棒！'
conversation = [
    {"role": "system", "content": "你是一名舆情分析师，帮我判断产品口碑的正负向，回复请用一个词语，正向 或者 负向"},
    {"role": "user", "content": review}
]

# 写法一：使用OpenAI提供的api
def call_llm_by_openai():
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=conversation,
        stream=False
    )

    print(response)
    print(response.choices[0].message.content)

# 写法二：使用dashscope提供的api
def call_llm_by_dashscope():
    dashscope.api_key = api_key
    response = Generation.call(
        model='qwen-turbo',  # 模型名
        messages=conversation
    )
    print(response)
    print(response.output.text)

# 写法三：使用langchain
def call_llm_by_langchain():
    llm = Tongyi(
        model_name="qwen-turbo",
        dashscope_api_key=api_key
    )
    response = llm.invoke(conversation)
    print(response)

if __name__ == "__main__":
    print("调用方法一，使用openai的方式调用大模型")
    call_llm_by_openai()
    print("调用方法二，使用dashscope的方式调用大模型")
    call_llm_by_dashscope()
    print("调用方法三，使用langchain的方式调用大模型")
    call_llm_by_langchain()
