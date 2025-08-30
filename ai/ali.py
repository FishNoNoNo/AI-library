import os
from typing import AsyncGenerator
from openai import AsyncOpenAI


class AliClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_API_BASE"),
        )

        self.content = """Your name is 百晓喵, and you are a cat. Your identity is a student's learning and life assistant.
        Based on the document content provided below, answer the user's questions briefly and professionally.
        If there is no relevant information in the document, please first state that there is no relevant information, and then answer the question based on your known knowledge.
        Please answer in Chinese, in a tone befitting a cat."""

    async def event_generator(self, query: str) -> AsyncGenerator[str, None]:
        """
        异步生成器：流式返回来自 Qwen 的响应
        输出格式为 SSE 兼容的字符串：data: <content>\n\n
        """
        try:
            response = await self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": self.content},
                    {"role": "user", "content": query},
                ],
                stream=True,
            )

            # 注意：即使使用 async client，stream 仍需用 async for
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    # 清理内容
                    content = content.replace("\n", "").replace("*", "")
                    # 构造 SSE 格式
                    yield f"data: {content}\n\n"

            # 可选：发送结束标记（前端可用）
            # yield f"data: [DONE]\n\n"

        except Exception as e:
            # 错误也通过流返回
            yield f'data: {{"error": "{str(e)}"}}\n\n'
