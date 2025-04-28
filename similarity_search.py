import getpass
import os
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Initialize the OpenAI embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize the Chroma vector store
vector_store = Chroma(
    collection_name="summary_reason_order_collection",
    embedding_function=embeddings,
    persist_directory="./embedded_boa_db",  # Where to save data locally
)

def similarity_search(query, k=5):
    """
    Perform a similarity search in the vector store.

    Args:
        query (str): The query string to search for.
        k (int): The number of results to return.

    Returns:
        list: A list of tuples containing the result and its score.
    """
    results = vector_store.similarity_search_with_score(query, k=k)

    # for res, score in results:
    #     print(f"* [SIM={score:.3f}] [{res.metadata['Title of Invention']}], Status: {res.metadata['Order status']}")

    return results

if __name__ == "__main__":
    # Example usage
    query = ""
    while query != "exit":
        query = input("Enter your query: ")
        results = similarity_search(query, k=5)
