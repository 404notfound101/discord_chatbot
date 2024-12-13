from typing import List, Dict
from typing_extensions import TypedDict
import os
import logging
import openai
import qdrant_client
from qdrant_client.models import PointStruct
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import boto3
from langgraph.graph import END, StateGraph, START

EMBEDDING_MODEL = "text-embedding-3-small"
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

prompt = hub.pull("rlm/rag-prompt")
rag_chain = prompt | llm | StrOutputParser()

find_user_query = """You are trying to find any user name information from the User question if it's possible. Your answer 
should be a single word with only common characters. There should be any space or quotation mark in your answer, e.g. 
john_smith. If you can't find any user name information, please answer with no_user_name."""

search_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", find_user_query),
        ("human", "User question: \n\n {question}"),
    ]
)
search_chain = search_prompt | rag_chain | StrOutputParser()


def add_docs(texts: List[str]):
    """
    Add documents to the vector database

    Args:
        txt (str): The text to be added to the vector database

    """
    openai_client = openai.Client(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    result = openai_client.embeddings.create(input=texts, model=EMBEDDING_MODEL)

    points = [
        PointStruct(
            id=idx,
            vector=data.embedding,
            payload={"text": text},
        )
        for idx, (data, text) in enumerate(zip(result.data, texts))
    ]
    qdrant_client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    qdrant_client.upsert("discord", points)



class GraphState(TypedDict):
    """
    Represents the state of RAG graph.

    Attributes:
        question: question
        memory: memory in form of {question: answer}
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    memory: List[str, str]
    generation: str
    web_search: str
    documents: List[str]

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    logging.info("---Retrieving documents---")
    openai_client = openai.Client(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    qdrant_client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    results = qdrant_client.search(
        collection_name="discord",
        query_vector=openai_client.embeddings.create(
            input=[state["question"]],
            model=EMBEDDING_MODEL,
        ).data[0].embedding,
        limit=2
    )
    docs = [result.payload["text"] for result in results if result.score > 0.7]
    # TODO: some other reranking
    docs += state["documents"]
    return {"documents": docs, "question": state["question"]}

def search_db(state):
    """
    Search the database for documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    logging.info("---Searching database---")
    user = search_chain.invoke({"question": state["question"]})
    try:
        db = boto3.client("dynamodb")
        response = db.get_item(
            TableName=os.getenv("DYNAMODB_TABLE"),
            Key={
                "user": {"S": user},
            }
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"documents": [], "question": state["question"]}
    docs = []
    if "Item" in response:
        data = f"Information from database for user state['user']: "
        for key, typed_value in response["Item"].items():
            values = [v for k, v in typed_value.items()]
            data += f"{key}: {values} | "
        docs.append(data)

    return {"documents": docs, "question": state["question"]}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

workflow = StateGraph(GraphState)

workflow.add_node("database_search", search_db)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "database_search")
workflow.add_edge("database_search", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()