import os


class AppConfig:
    def __init__(self):
        os.environ["DASHSCOPE_API_KEY"] = (
            ""  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        )
        os.environ["DASHSCOPE_API_BASE"] = ""  # 百炼服务的base_url
        pass
