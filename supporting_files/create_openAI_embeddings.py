import os
import time
from dotenv import load_dotenv
load_dotenv()
import getpass

import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

#Imports for the embedding model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
from tqdm import tqdm


def get_embedding_model(model_name: str):
    """
    Initialize the OpenAI embeddings model.
    """
    embeddings = OpenAIEmbeddings(model=model_name)

    return embeddings


def vector_store_initialize(embeddings: OpenAIEmbeddings, persist_directory, collection_name) -> Chroma:
    """
    Init Chroma vector store.

    params:
    - embeddings: The OpenAI embeddings model.
    - persist_directory: Directory to save the vector store.
    - collection_name: Name of the vector store collection
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    return vector_store


def create_documents_list_with_title_and_summary(boa_df: pd.DataFrame) -> list:
    """
    Create a list of Document objects from the DataFrame.

    params:
    - boa_df: DataFrame containing the data to be embedded.

    returns:
    - List of Document objects.
    """
    documents = []

    # Set up the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=30000,
        chunk_overlap=3000,  # To preserve some context
        separators=["\n\n", "\n", ".", " "]  # Try to split at logical places
    )

    # For each BoA decision, create a Document with Title + Summary
    for index, row in tqdm(boa_df.iterrows(), total=boa_df.shape[0]):
        relevant_fields = [
            f"Title: {row['Title of Invention']}",
            f"Summary: {row['Summary']}",
        ]

        text_for_embedding = "\n\n".join(relevant_fields)

        # Split the long enriched_content into chunks
        content_chunks = text_splitter.split_text(text_for_embedding)

        metadata={"Decision date": row["Decision date"],
                      "Case number": row["Case number"],
                      "Application number": row["Application number"],
                      "Publication number": row["Publication number"],
                      "IPC pharma": row["IPC pharma"],
                      "IPC biosimilar": row["IPC biosimilar"],
                      "IPCs": row["IPCs"],
                      "Language": row["Language"],
                      "Title of Invention": row["Title of Invention"],
                      "Summary": row["Summary"],
                      "Patent Proprietor": row["Patent Proprietor"],
                      "Headword": row["Headword"],
                      "Provisions": row["Provisions"],
                      "Keywords": row["Keywords"],
                      "Decisions cited": row["Decisions cited"],
                      "Order": row["Order"],
                      "Decision reasons": row["Decision reasons"],
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


def embed_and_persist_documents(model: str, persist_directory: str, collection_name: str, documents: list) -> None:
    """
    Add documents to the vector store.

    Args:
    - model: The OpenAI model to use for embeddings.
    - persist_directory: Directory to save the vector store.
    - collection_name: Name of the vector store collection.
    - documents: List of Document objects to be embedded and added to the vector store.

    Returns:
    - None
    """
    embeddings = get_embedding_model(model_name=model)

    vector_store=vector_store_initialize(embeddings, persist_directory, collection_name)

    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Loop that takes 10 documents at a time and sleeps for 1 sec after adding them
    for i in tqdm(range(0, len(documents), 10)):
        # Get the next batch of documents
        batch_documents = documents[i:i + 10]
        batch_uuids = uuids[i:i + 10]

        # Add the batch to the vector store
        vector_store.add_documents(documents=batch_documents, ids=batch_uuids)

        # Sleep to avoid API rate limiter
        time.sleep(1)


def undo_document_chunking(embedding_model_name, vector_store_collection_name, vector_store_persist_directory):
    embedding_model = get_embedding_model(model_name=embedding_model_name)

    vector_store = vector_store_initialize(
        embedding_model,
        persist_directory=vector_store_persist_directory,
        collection_name=vector_store_collection_name
        )

    # 3. Get all data (texts, metadatas, embeddings)
    result = vector_store.get(include=["embeddings", "metadatas", "documents"])

    # 4. Extract separately
    embeddings = result['embeddings']
    metadatas = result['metadatas']

    # 5. Reassemble original documents
    # Create a dictionary to group chunks by their original document
    embeddings_by_document = defaultdict(list)

    for embedding, metadata in zip(embeddings, metadatas):
        original_index = metadata['original_index']
        embeddings_by_document[original_index].append(embedding)

    # 6. Sort the chunks for each document and concatenate
    document_embeddings = {}
    for original_index, chunk_embeddings in embeddings_by_document.items():
        chunk_embeddings_array = np.array(chunk_embeddings)
        mean_embedding = np.mean(chunk_embeddings_array, axis=0)
        document_embeddings[original_index] = mean_embedding

    print(f"Generated {len(document_embeddings)} document-level embeddings.")
    return document_embeddings



def load_raw_data(filepath: str = "all_en_df.pkl") -> pd.DataFrame:
    return pd.read_pickle(filepath)


if __name__ == "__main__":
    MODEL = "text-embedding-3-small"
    PERSIST_DIRECTORY = "./title_summary_embedding_db"
    COLLECTION_NAME = "title_summary_embedding"

    NON_CHUNKED_EMBEDDING_WRITE_PATH = "raw_data_with_openai_embeddings.pkl"

    # Get the OpenAI API key if necessary
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    raw_data = load_raw_data()

    # Check if vector store does not exist
    if not os.path.exists(PERSIST_DIRECTORY):

        print(f"Directory {PERSIST_DIRECTORY} does not exist. Proceeding with embedding.")
        # Create a list of Document objects
        documents = create_documents_list_with_title_and_summary(raw_data)

        # Add documents to the vector store
        embed_and_persist_documents(MODEL, PERSIST_DIRECTORY, COLLECTION_NAME, documents)

    non_chunked_embeddings = undo_document_chunking(MODEL, COLLECTION_NAME, PERSIST_DIRECTORY)

    embedded_df = raw_data.copy()
    embedded_df["title_summary_embedding"] = None
    for i in non_chunked_embeddings.keys():
        #embedded_df.loc[i, "title_summary_embedding"] = non_chunked_embeddings[i].tolist()
        embedded_df.at[i, "title_summary_embedding"] = non_chunked_embeddings[i].tolist()

    if os.path.exists(NON_CHUNKED_EMBEDDING_WRITE_PATH):
        print(f"File {NON_CHUNKED_EMBEDDING_WRITE_PATH} already exists. Skipping save.")
    else:
        embedded_df.to_pickle(NON_CHUNKED_EMBEDDING_WRITE_PATH)
        embedded_df.to_csv(NON_CHUNKED_EMBEDDING_WRITE_PATH.replace(".pkl", ".csv"), index=False)
        print(f"Saved the DataFrame with embeddings to {NON_CHUNKED_EMBEDDING_WRITE_PATH}.")
