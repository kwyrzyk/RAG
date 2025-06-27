from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import os
import shutil
from langchain_community.vectorstores import Chroma
from dataclasses import dataclass

@dataclass
class SplitterConfig:
    chunk_size: int
    chunk_overlap: int


class DocumentsLoader:

    @staticmethod
    def load_files(dir_path):
        pdf_loader = DirectoryLoader(
            path=dir_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        pdf_docs = pdf_loader.load()
        print(len(pdf_docs))

        txt_loader = DirectoryLoader(
            path=dir_path,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}  
        )
        txt_docs = txt_loader.load()

        concat_docs = pdf_docs + txt_docs

        print(f"{len(concat_docs)} page(s) read")

        return concat_docs
    
    @staticmethod
    def split_text(docs, config):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            add_start_index=True
        )

        chunks = text_splitter.split_documents(docs)

        print(f"{len(docs)} splitted into {len(chunks)} chunks")

        return chunks
    
    @staticmethod
    def save_to_db(chunks, db_path):

        # Clear db if exists
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        embedder = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

        db = Chroma.from_documents(
            chunks,
            embedder,
            persist_directory=db_path
        )

        print(f"Saved {len(chunks)} to {db_path}")
