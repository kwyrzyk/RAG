from documents_loader import DocumentsLoader, SplitterConfig
from dataclasses import dataclass

DOCS_PATH = "./test_docs"
SPLITTER_CONFIG = SplitterConfig(
    chunk_size=300,
    chunk_overlap=100
)
CHROMA_PATH = "./chroma"




def main():
    loaded_docs = DocumentsLoader.load_files(DOCS_PATH)

    chunks = DocumentsLoader.split_text(loaded_docs, SPLITTER_CONFIG)

    DocumentsLoader.save_to_db(chunks, CHROMA_PATH)






if __name__ == "__main__":
    main()