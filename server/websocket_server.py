# websocket_server.py
import asyncio
import json
import time
import websockets
from websockets.server import WebSocketServerProtocol

# 你的模块（根据实际路径调整）
from ai.ali import AliClient
from rag.embed import EmbeddingClient


class WebSocketServer:

    def __init__(self, idle_timeout=60 * 5):
        self.ali_client = AliClient()
        self.embedding_client = EmbeddingClient()
        self.idle_timeout = idle_timeout

    async def handler(self, websocket: WebSocketServerProtocol):
        print(f"[WebSocket] 客户端连接: {websocket.remote_address}")
        last_activity = time.time()

        def update_activity():
            nonlocal last_activity
            last_activity = time.time()

        async def monitor_timeout():
            while True:
                await asyncio.sleep(10)  # 每 10 秒检查一次
                if time.time() - last_activity > self.idle_timeout:
                    print(f"[WebSocket] 连接超时关闭: {websocket.remote_address}")
                    await websocket.close(code=1000, reason="Idle timeout")
                    break

        # 异步启动监控任务
        monitor_task = asyncio.create_task(monitor_timeout())
        try:
            async for message in websocket:
                update_activity()
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            print(f"[WebSocket] 客户端断开: {websocket.remote_address}")
        except Exception as e:
            print(f"[WebSocket] 未知错误: {e}")
        finally:
            monitor_task.cancel()
            print(f"[WebSocket] 连接已关闭: {websocket.remote_address}")

    async def handle_message(self, websocket: WebSocketServerProtocol, message: str):
        try:
            data = json.loads(message)
            methods = data.get("methods", "").strip()
            if methods != "chat":
                await websocket.send(
                    json.dumps({"type": "error", "error": "Invalid method"})
                )
                await websocket.close(code=1002, reason="Invalid method")
                return
            query = data.get("query", "").strip()

            if not query:
                await websocket.send(
                    json.dumps({"type": "error", "error": "Query is required"})
                )
                return
            chat_history = data.get("chat_history", None)
            # 使用线程池执行同步方法
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                None, self.embedding_client.query_db, query
            )
            prompt = await loop.run_in_executor(
                None,
                self.embedding_client.create_prompt,
                query,
                docs or None,
                chat_history,
            )

            # 流式返回 token
            async for chunk in self.ali_client.event_generator(prompt):
                if chunk.startswith("data:"):
                    content = chunk[5:].strip()
                    try:
                        parsed = json.loads(content)
                        await websocket.send(json.dumps({"type": "token", **parsed}))
                    except:
                        await websocket.send(
                            json.dumps({"type": "token", "text": content})
                        )
                else:
                    await websocket.send(
                        json.dumps({"type": "token", "text": chunk.strip()})
                    )

            # 发送完成信号
            await websocket.send(json.dumps({"type": "done"}))

        except json.JSONDecodeError:
            await websocket.send(json.dumps({"type": "error", "error": "Invalid JSON"}))
        except Exception as e:
            await websocket.send(json.dumps({"type": "error", "error": str(e)}))
            print(f"[WebSocket Error] {e}")

    async def start(self, host="0.0.0.0", port=5001):
        server = await websockets.serve(
            self.handler,
            host,
            port,
            # 可选：增加心跳
            ping_interval=20,
            ping_timeout=30,
        )
        print(f"WebSocket 服务启动: ws://{host}:{port}")
        return server
