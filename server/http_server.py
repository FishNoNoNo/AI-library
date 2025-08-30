# server/api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncio

# 你的模块（根据实际路径调整）
from ai.ali import AliClient
from rag.embed import EmbeddingClient


def create_app():
    """
    创建并返回 FastAPI 应用实例
    """
    app = FastAPI(title="AI 助手 API")

    # 在应用创建时初始化客户端
    ali_client = AliClient()
    embedding_client = EmbeddingClient()

    class ChatRequest(BaseModel):
        query: str
        chat_history: Optional[List[dict]] = None

    class AddLibraryRequest(BaseModel):
        sources: List[str]

    @app.post("/chat")
    async def chat(request: ChatRequest):
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        async def event_stream():
            try:
                # 使用线程池执行同步方法（RAG 查询）
                loop = asyncio.get_event_loop()
                docs = await loop.run_in_executor(
                    None, embedding_client.query_db, query
                )
                prompt = await loop.run_in_executor(
                    None,
                    embedding_client.create_prompt,
                    query,
                    docs or None,
                    request.chat_history or [],
                )

                # 流式输出 AI 响应
                async for chunk in ali_client.event_generator(prompt):
                    yield chunk  # 已经是 "data: ...\n\n" 格式

            except Exception as e:
                yield f'data: {{"error": "{str(e)}"}}\n\n'

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/add_library")
    async def add_library(request: AddLibraryRequest):
        sources = request.sources
        if not sources:
            raise HTTPException(status_code=400, detail="Sources are required")

        def run_db():
            embedding_client.create_db(sources)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, run_db)

        return {"message": "知识库构建任务已启动"}

    @app.get("/login")
    async def login():
        """
        模拟登录接口，实际应用中应实现具体的登录逻辑
        """
        return {"message": "Login successful"}

    return app
