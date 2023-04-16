from pathlib import Path

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader


PERSIST_DIRECTORY = "db"
COLLECTION_NAME = "chatgpt_ret_plugin"


def create_db():
    data_files = list(Path("data/").rglob("*.md"))
    docs = []
    for f in data_files:
        docs.extend(TextLoader(str(f)).load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    texts = text_splitter.split_documents(documents=docs)

    embedding = OpenAIEmbeddings()
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=PERSIST_DIRECTORY,
    )
    vectordb.add_documents(
        documents=texts,
        embedding=embedding,
    )
    vectordb.persist()


def load_db() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=OpenAIEmbeddings(),
    )


def main():
    # create_db()
    vectordb = load_db()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        vectorstore=vectordb,
    )
    query = "what vector databases can I use with the retrievers?"
    print(vectordb.similarity_search(query))
    res = vectordb.similarity_search(query)
    print(vectordb.similarity_search(query))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
