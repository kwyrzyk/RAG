from documents_loader import DocumentsLoader, SplitterConfig
from retriever import Retriever
from response_generator import ResponseGenerator
from dataclasses import dataclass
from hf_token_initializer import init_hf_token
import argparse


DOCS_PATH = "./documents"
SPLITTER_CONFIG = SplitterConfig(
    chunk_size=300,
    chunk_overlap=100
)
CHROMA_PATH = "./chroma"
HF_TOKEN_PATH = "./hf_token.txt"


def main():
    init_hf_token(HF_TOKEN_PATH)

    loaded_docs = DocumentsLoader.load_files(DOCS_PATH)

    chunks = DocumentsLoader.split_text(loaded_docs, SPLITTER_CONFIG)

    DocumentsLoader.save_to_db(chunks, CHROMA_PATH)

    generator = ResponseGenerator()

    while True:
        print("Ask question about your documents:")
        query_text = input()
        result = Retriever.search_db(query_text, db_path=CHROMA_PATH, k=3)

        prompt = generator.prepare_prompt(query_text, result)
        response = generator.predict(prompt)
        formatted_response = generator.format_response(response, result)
        print(formatted_response)

if __name__ == "__main__":
    main()