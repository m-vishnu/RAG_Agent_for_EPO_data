import getpass
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Tuple
import chainlit as cl

import similarity_search

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = init_chat_model("gpt-4o-mini", model_provider="openai", streaming=True)

# Initialize the OpenAI embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize the Chroma vector store
vector_store = Chroma(
    collection_name="summary_reason_order_collection",
    embedding_function=embeddings,
    persist_directory="./embedded_all_en_db",  # Where to save data locally
)

# Define prompt for question-answering
prompt_template = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:
""")

prompt_template2 = ChatPromptTemplate.from_template("""
You are a question-answering assistant that is an expert on patent law.
Use the following pieces of retrieved context from Board of Appeals decisions to answer the question.
Please provide a short summary of each context (2 sentences),
the topics/drugs/inventions involved, th proprietor and the order status for each context.
Separate each context with a new line and a dashed separator. Always give the case number of the context you are summarising
If you don't know the answer, just say that you don't know.
If the question is not related to patent law, please answer that you are designed to answer questions related to patent law only.
Be helpful and friendly. You can use words like Hello, please, etc.

Give a short summary at the end aggregating the information from all contexts. 
(Order status for each context, and overview of repeating and distinct topics in the summary).

At the end ask if you can help further.

Question: {question}
Context: {content}
Metadata: {metadata}
Answer:
""")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Tuple[Document, float]]
    answer: str


# Define application steps
def retrieve(state: State):
    user_query = state["question"].strip().lower()

    # Simple rule-based check for greetings
    if user_query in ["hello", "hi", "hey"]:
        return {"context": [], "answer": f"{user_query.capitalize()}, How can I help?"}

    retrieved_docs = similarity_search.similarity_search(user_query)
    return {"context": retrieved_docs}


async def generate(state: State):
    docs_content = "\n\n".join(doc[0].page_content for doc in state["context"])
    docs_metadata = "\n\n".join(str(doc[0].metadata) for doc in state["context"])
    messages = prompt_template2.invoke({"question": state["question"],
                                        "content": docs_content,
                                        "metadata": docs_metadata})

    response_content = ""

    # Create a Chainlit message object to update progressively
    msg = cl.Message(content="")
    await msg.send()  # Send an empty message first

    async for chunk in llm.astream(messages):
        if chunk.content:
            response_content += chunk.content
            await msg.stream_token(chunk.content)

    await msg.update()  # Finalize the message
    return {"answer": response_content}


# Compile application and test
graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")

graph = graph_builder.compile()


@cl.on_message
async def main(message: cl.Message):
    # Get the question from the user
    question = message.content

    # Create the initial state
    state = {"question": question}

    # Run the graph with the initial state
    response = await graph.ainvoke(state)

    # Print the answer
    cl.info(f"Answer: \n {response['answer']}")
    cl.info(f"Context: \n {response['context']}")

    await cl.Message(content=f"**Answer:**\n{response['answer']}").send()
