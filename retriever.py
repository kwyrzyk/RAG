from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class Retriever:
    @staticmethod
    def search_db(query_text, db_path, k=3):
        embedder = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

        db = Chroma(
            persist_directory=db_path,
            embedding_function=embedder
        )

        result = db.similarity_search_with_relevance_scores(query_text, k)

        return result