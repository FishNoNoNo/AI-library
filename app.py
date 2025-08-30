# app.py
import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve

from server.http_server import create_app as create_fastapi_app
from server.websocket_server import WebSocketServer
from config.appConfig import AppConfig


async def main():
    # =============== 启动 FastAPI (HTTP) 服务 ===============
    AppConfig()
    fastapi_app = create_fastapi_app()
    http_config = Config()
    http_config.bind = ["0.0.0.0:5000"]

    # =============== 启动 WebSocket 服务 ===============
    ws_server = WebSocketServer()

    # 并行启动两个服务
    http_task = serve(fastapi_app, http_config)
    ws_task = ws_server.start(host="0.0.0.0", port=5001)

    print("🚀 服务启动中...")
    print("🌐 HTTP 服务: http://0.0.0.0:5000")
    print("🔗 WebSocket 服务: ws://0.0.0.0:5001")

    # 等待两个服务运行
    await asyncio.gather(http_task, ws_task)


if __name__ == "__main__":
    asyncio.run(main())
