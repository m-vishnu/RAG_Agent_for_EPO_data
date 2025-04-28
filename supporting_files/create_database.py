import getpass
import os
import time
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from uuid import uuid4
from tqdm import tqdm

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

boa_df_proc = pd.read_pickle("all_en_df.pkl")

print("Data loaded successfully")

def embedding_model(model_name: str):
    """
    Initialize the OpenAI embeddings model.
    """
    embeddings = OpenAIEmbeddings(model=model_name)

    return embeddings

def vector_store_initialize(embeddings: OpenAIEmbeddings, persist_directory, collection_name) -> Chroma:
    """
    Initialize the Chroma vector store.
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,  # Where to save data locally
    )

    return vector_store

def create_documents_list(boa_df_proc: pd.DataFrame) -> list:
    """
    Create a list of Document objects from the DataFrame.
    """
    documents = []

    # Set up the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=30000,
        chunk_overlap=3000,  # To preserve some context
        separators=["\n\n", "\n", ".", " "]  # Try to split at logical places
    )

    for index, row in tqdm(boa_df_proc.iterrows(), total=boa_df_proc.shape[0]):
        relevant_fields = [
            f"Summary: {row['Summary']}",
            f"Decision reasons: {row.get('Decision reasons', '')}",
            f"Order: {row.get('Order', '')}",
        ]
        enriched_content = "\n\n".join(relevant_fields)

        # Split the long enriched_content into chunks
        content_chunks = text_splitter.split_text(enriched_content)

        metadata={"Decision date": row["Decision date"],
                      "Case number": row["Case number"],
                      "Application number": row["Application number"],
                      "Publication number": row["Publication number"],
                      "IPC pharma": row["IPC pharma"],
                      "IPC biosimilar": row["IPC biosimilar"],
                      "IPCs": row["IPCs"],
                      "Language": row["Language"],
                      "Title of Invention": row["Title of Invention"],
                      "Patent Proprietor": row["Patent Proprietor"],
                      "Headword": row["Headword"],
                      "Provisions": row["Provisions"],
                      "Keywords": row["Keywords"],
                      "Decisions cited": row["Decisions cited"],
                      "Order status": row["Order status"],
                      "Order status web": row["Order status web"],
                      "Order status manual": row["Order status manual"],
                      "Opponents": row["Opponents"]}

        for i, chunk in enumerate(content_chunks):
            # Create a Document object for each chunk
            document = Document(
                page_content=chunk,
                metadata={**metadata, "chunk_index": i, "original_index": index},
                id=f"{index}_{i}"  # Unique ID for each chunk
            )
            documents.append(document)

    return documents

def add_documents_to_vector_store(model: str, persist_directory: str, collection_name: str, documents: list) -> None:
    """
    Add documents to the vector store.
    """
    embeddings = embedding_model(model_name=model)

    vector_store=vector_store_initialize(embeddings, persist_directory, collection_name)

    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Loop that takes 10 documents at a time and sleeps for 1 sec after adding them
    for i in tqdm(range(0, len(documents), 10)):
        # Get the next batch of documents
        batch_documents = documents[i:i + 10]
        batch_uuids = uuids[i:i + 10]

        # Add the batch to the vector store
        vector_store.add_documents(documents=batch_documents, ids=batch_uuids)

        # Sleep for a bit to avoid overwhelming the API
        time.sleep(1)

if __name__ == "__main__":
    MODEL = "text-embedding-3-small"
    PERSIST_DIRECTORY = "./embedded_all_en_db"
    COLLECTION_NAME = "summary_reason_order_collection"


    # Create a list of Document objects
    documents = create_documents_list(boa_df_proc)
    print("Documents created successfully")

    # Add documents to the vector store
    add_documents_to_vector_store(MODEL, PERSIST_DIRECTORY, COLLECTION_NAME, documents)
    print("Documents added to vector store successfully")
