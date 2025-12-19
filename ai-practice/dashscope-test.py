import dashscope
from dashscope.api_entities.dashscope_response import Role
dashscope.api_key = "sk-c565af51a657466fae8d90b69a598f5f"

# 封装模型响应函数
def get_response(messages):
    response = dashscope.Generation.call(
        model='deepseek-r1',  # 使用 deepseek-r1 模型
        messages=messages,
        result_format='message'  # 将输出设置为message形式
    )
    return response

# 测试对话
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "你好，你是什么大模型？"}
]
response = get_response(messages)
print(response.output.choices[0].message.content)
