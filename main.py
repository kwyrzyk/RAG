from documents_loader import DocumentsLoader, SplitterConfig
from retriever import Retriever
from response_generator import ResponseGenerator
from dataclasses import dataclass
import argparse


DOCS_PATH = "./documents"
SPLITTER_CONFIG = SplitterConfig(
    chunk_size=300,
    chunk_overlap=100
)
CHROMA_PATH = "./chroma"




def main():
    loaded_docs = DocumentsLoader.load_files(DOCS_PATH)

    chunks = DocumentsLoader.split_text(loaded_docs, SPLITTER_CONFIG)

    DocumentsLoader.save_to_db(chunks, CHROMA_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text. Ask about your documents")
    args = parser.parse_args()
    query_text = args.query_text

    result = Retriever.search_db(query_text, db_path=CHROMA_PATH, k=3)
    for doc, _score in result:
        print(_score, doc.page_content)

    prompt = ResponseGenerator.prepare_prompt(query_text, result)
    response = ResponseGenerator.predict(prompt)
    formatted_response = ResponseGenerator.format_response(response, result)
    print(formatted_response)

if __name__ == "__main__":
    main()