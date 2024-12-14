from typing import List
from typing_extensions import TypedDict
import os
import logging
import openai
import qdrant_client
from qdrant_client.models import PointStruct
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import boto3
from langgraph.graph import END, StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from functools import partial

EMBEDDING_MODEL = "text-embedding-3-small"
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

prompt = hub.pull("rlm/rag-prompt")
rag_chain = prompt | llm | StrOutputParser()

find_user_query = """You are attempting to extract a username from the user’s question, if any. Your answer should
be a single word containing only common characters (e.g. crazy_TAFAWF or Dove Smith as dove_smith) and must not
include spaces, quotation marks, or additional formatting. \n\nUser question: {question}"""

search_prompt = PromptTemplate(
    template=find_user_query,
    input_variables=["question"],
)
search_chain = search_prompt | llm | StrOutputParser()


def add_docs(
    texts: List[str], openai_client: openai.Client, q_client: qdrant_client.QdrantClient
):
    """
    Add documents to the vector database

    Args:
        txt (str): The text to be added to the vector database

    """
    result = openai_client.embeddings.create(input=texts, model=EMBEDDING_MODEL)

    points = [
        PointStruct(
            id=idx,
            vector=data.embedding,
            payload={"text": text},
        )
        for idx, (data, text) in enumerate(zip(result.data, texts))
    ]
    q_client.upsert("discord", points)


class GraphState(TypedDict):
    """
    Represents the state of RAG graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


def retrieve(state, openai_client: openai.Client, q_client: qdrant_client.QdrantClient):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    logging.info("---Retrieving documents---")
    results = q_client.search(
        collection_name="discord",
        query_vector=openai_client.embeddings.create(
            input=[state["question"]],
            model=EMBEDDING_MODEL,
        )
        .data[0]
        .embedding,
        limit=3,
    )
    docs = [result.payload["text"] for result in results if result.score > 0.7]
    # TODO: some other reranking
    docs += state["documents"]
    return {"documents": docs, "question": state["question"]}


def search_db(state, db: boto3.client):
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
        response = db.get_item(
            TableName=os.getenv("DYNAMODB_TABLE"),
            Key={
                "user": {"S": user},
            },
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"documents": [], "question": state["question"]}
    docs = []
    if "Item" in response:
        data = "Information from database for user state['user']: "
        for key, typed_value in response["Item"].items():
            values = [
                v for _, v in typed_value.items()
            ]  # the format of value is {'S': 'value'}
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


def generate_graph(
    openai_client: openai.Client, q_client: qdrant_client.QdrantClient, db: boto3.client
) -> CompiledStateGraph:
    """
    Generate the RAG graph

    Returns:
        CompiledStateGraph: The compiled RAG graph
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("database_search", partial(search_db, db=db))
    workflow.add_node(
        "retrieve", partial(retrieve, openai_client=openai_client, q_client=q_client)
    )
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "database_search")
    workflow.add_edge("database_search", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
