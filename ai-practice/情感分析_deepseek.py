# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

review= '这款产品真的非常好用，性价比高，客服态度也很棒！'

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一名舆情分析师，帮我判断产品口碑的正负向，回复请用一个词语，正向 或者 负向"},
        {"role": "user", "content": review},
    ],
    stream=False
)


print(response.choices[0].message.content)

