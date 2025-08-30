import os
import re
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)

# 或者更底层的方式（推荐）
from unstructured.partition.md import partition_md
from langchain_core.documents import Document

# from langchain_community.embeddings import DashScopeEmbeddings
# from langchain_openai import OpenAIEmbeddings


class MarkdownLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        elements = partition_md(filename=self.file_path)
        text = "\n\n".join([str(el) for el in elements])
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]


class ChunkClient:
    def __init__(self, sources: List[str] = None):
        self.loaders = {
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
        }
        self.sources = sources if sources else []

    def _load_file(self, file_path: str) -> List[Document]:
        """根据文件扩展名选择合适的加载器"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext not in self.loaders:
            raise ValueError(f"不支持的文件格式: {ext}")

        loader_class = self.loaders[ext]
        loader = loader_class(file_path)
        documents = loader.load()

        # 添加元数据：source
        for doc in documents:
            doc.metadata["source"] = file_path

        return documents

    def _load_web(self, url: str) -> List[Document]:
        """加载网页内容"""
        loader = WebBaseLoader(url)
        documents = loader.load()
        # 添加元数据：source
        for doc in documents:
            doc.metadata["source"] = url  # 原始 URL

        return documents

    def preprocess_text(self, docs: List[Document]) -> List[Document]:
        """清理多余空白、换行等"""
        for doc in docs:
            content = re.sub(r"[ \t]+", " ", doc.page_content)  # 压缩水平空白
            content = re.sub(r"\n{3,}", "\n\n", content)  # 压缩垂直空白
            doc.page_content = content.strip()
        return docs

    def read_data(self) -> List[Document]:
        docs = []
        for source in self.sources:
            if source.startswith("http"):
                docs.extend(self._load_web(source))
            else:
                docs.extend(self._load_file(source))
        return docs

    def get_chunk(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # from langchain_experimental.text_splitter import SemanticChunker

        # embeddings = OpenAIEmbeddings(
        #     model="text-embedding-v4",
        #     api_key=os.getenv("DASHSCOPE_API_KEY"),
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     # temperature=0.7  # 其他参数也可设置
        # )
        # text_splitter = SemanticChunker(
        #     embeddings,
        #     breakpoint_threshold_type="percentile",
        #     buffer_size=2,
        #     sentence_split_regex=r"(?<=[。!?!.?])\s*",  # 中文句子分隔符
        #     min_chunk_size=100,
        # )
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        # docs = self.preprocess_text(self.read_data())
        docs = self.read_data()
        # print(len(docs))
        doc_splits = text_splitter.split_documents(docs)
        return doc_splits


# if __name__ == "__main__":
#     os.environ["DASHSCOPE_API_KEY"] = (
#         "sk-cbd4f422094d4717bf2aaab2218dc51f"  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
#     )
#     os.environ["DASHSCOPE_API_BASE"] = (
#         "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
#     )
#     sources = [
#         # "./library/接口文档.md",
#         # "https://www.runoob.com/python3/python3-tutorial.html",
#         # "./library/02-第十六届中国大学生服务外包创新创业大赛企业命题类赛题手册（A类）.pdf",
#         "./library/4_信息工程学院《蜕变》(1).docx",
#     ]
#     chunk = ChunkClient(sources=sources)

#     doc_splits = chunk.get_chunk()
#     print(f"总共切分成 {len(doc_splits)} 个文本块")
#     with open("output_chunks.txt", "w", encoding="utf-8") as f:
#         for i, doc in enumerate(doc_splits):
#             f.write(f"第 {i + 1} 个文本块：\n")
#             f.write(doc.page_content + "\n")
#             f.write("-" * 40 + "\n")
