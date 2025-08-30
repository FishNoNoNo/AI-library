from .text_chunk import ChunkClient
import chromadb
from openai import OpenAI
import os

# 获取当前脚本所在目录
current_script_directory = os.path.dirname(os.path.abspath(__file__))

parent_directory = os.path.dirname(current_script_directory)

db_path = os.path.join(parent_directory, "chroma_db")


class EmbeddingClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_API_BASE"),
        )

        self.chromadb_client = chromadb.PersistentClient(path=str(db_path))
        self.chromadb_collection = self.chromadb_client.get_or_create_collection(
            name="langchain_docs"
        )

    def embed_text(self, text):
        """Embed text using OpenAI."""
        completion = self.client.embeddings.create(
            model="text-embedding-v4",
            input=text,
            dimensions=1024,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
            encoding_format="float",
        )
        return completion.data[0].embedding

    def create_db(self, sources):
        chunk = ChunkClient(sources=sources)
        for idx, c in enumerate(chunk.get_chunk()):
            print(c.page_content)
            embedding = self.embed_text(c.page_content)
            # print(embedding)
            self.chromadb_collection.upsert(
                ids=[str(idx)], documents=[c.page_content], embeddings=[embedding]
            )

    def query_db(self, query):
        results = self.chromadb_collection.query(
            query_embeddings=self.embed_text(query), n_results=5
        )
        return results["documents"][0]

    def create_prompt(self, query, docs, chat_history=None):
        if not docs:
            context = f"No relevant documents found"
        else:
            context = "\n".join([doc for doc in docs if doc.strip() != ""])
        prompt = f"""
        ## context:
        {context}
        ## chat history:
        {chat_history if chat_history else ""}
        ## question:
        {query}
        """
        return prompt


# if __name__ == "__main__":
#     os.environ["DASHSCOPE_API_KEY"] = (
#         "sk-cbd4f422094d4717bf2aaab2218dc51f"  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
#     )
#     os.environ["DASHSCOPE_API_BASE"] = (
#         "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
#     )
#     embedding_client = EmbeddingClient()
#     embedding_client.create_db(
#         sources=[
#             "./library/接口文档.md",
#             "https://www.runoob.com/python3/python3-tutorial.html",
#             "./library/02-第十六届中国大学生服务外包创新创业大赛企业命题类赛题手册（A类）.pdf",
#             "./library/4_信息工程学院《蜕变》(1).docx",
#         ],
#     )
# pass
# print("请输入查询内容：")
# query = input()
# embedding_client = EmbeddingClient()
# results = embedding_client.query_db(query)
# for i, doc in enumerate(results["documents"][0]):
#     print(f"结果 {i + 1}: {doc}")
#     print("-" * 40)

# print(embedding_client.create_prompt(query, results["documents"][0]))
