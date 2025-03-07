import os
import logging
import warnings
from typing import List, Tuple

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LangChain Core
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangChain Community
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

# LangChain OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# LangChain Experimental
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv(override=True)
NEO4J_URI="neo4j+s://6f619797.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="loVyer5cvr7MO2MXwob-k7GFq18Bu2iYSoTzxHCR_2A"

# NEO4J_URI="neo4j+s://771eef14.databases.neo4j.io"
# NEO4J_USERNAME="neo4j"
# NEO4J_PASSWORD="XB0t7KZlTx56J1AM2nL6zI4Pkx_HIlgZ2tXy3k69qUc"
# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv(override=True)

# Flask Setup
app = Flask(__name__)
CORS(app)

# Replace sqlite3 with psycopg2
import psycopg2
from psycopg2.extras import RealDictCursor

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL")  # Get from environment variables

def init_db():
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS faq (
                        id SERIAL PRIMARY KEY,
                        question TEXT NOT NULL,
                        answer TEXT,
                        status TEXT DEFAULT 'pending'
                    )
                """)
                conn.commit()
    except Exception as e:
        logging.error(f"Error initializing database: {e}")

init_db()

### changes made starting from here



llm = AzureChatOpenAI(
    openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key=os.environ['AZURE_OPENAI_APIKEY'],
    azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    temperature=0
)

graph = Neo4jGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

# Setting up text embeddings
text_embedding =  AzureOpenAIEmbeddings(
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key=os.environ['AZURE_OPENAI_APIKEY'],
    azure_deployment=os.environ["AZURE_EMBEDDING_DEPLOYMENT_NAME"],
    model='text-embedding-ada-002'
)

vector_index = Neo4jVector.from_existing_graph(
    text_embedding,  
    search_type="hybrid", #i.e search is done on keywords as well as the embedding
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
    url=NEO4J_URI,  
    username=NEO4J_USERNAME,  
    password=NEO4J_PASSWORD
)

graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text.",
    )

# Define HR-specific assistant behavior
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an HR assistant for Hoya Electronics. Your job is to provide "
            "clear and concise answers based on data stored "
            "in a knowledge graph and a document database. If there is no relevant information, "
            "always respond with: 'No relevant information available. Escalating to HR.' Do not guess."
            "Do not try and format the response when there is no relevant information. Directly respond with : 'No relevant information available. Escalating to HR.' "
        ),
        (
            "human",
            "Employee question: {question}",
        ),
    ]
)


entity_chain = prompt | llm.with_structured_output(Entities)

     
def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Fulltext index query
def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    
    # Ensure that entities.names is not empty or None
    if not hasattr(entities, 'names') or not entities.names:
        return "No entities found in the question."

    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node, score
            WITH node, score  // Ensure that node and score are available to the subquery
            CALL {
            WITH node  // Pass the node variable into the subquery
            MATCH (node)-[r:!MENTIONS]->(neighbor)
            RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
            UNION ALL
            WITH node  // Pass the node variable into the next part of the subquery
            MATCH (node)<-[r:!MENTIONS]-(neighbor)
            RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output
            LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )

        # Handle case if response is empty or doesn't contain 'output'
        if not response:
            result += f"\nNo relevant information found for entity: {entity}."
        else:
            for el in response:
                if 'output' in el and el['output']:
                    result += "\n" + el['output']
                else:
                    result += "\nNo output found for this entity."

    return result


SIMILARITY_THRESHOLD = 0.7  # Adjust based on experimentation

def retriever(question: str):
    print(f"Search query: {question}")
    
    # Handle structured retrieval
    structured_data = structured_retriever(question)
    print(f"Structured data retrieved: {structured_data}")
    
    # Handle unstructured data retrieval (from vector index)
    retrieved_docs = vector_index.similarity_search_with_score(question, k=5)  # Retrieve top 5 similar chunks
    
    # Extract content and similarity scores
    unstructured_data = []
    for doc, score in retrieved_docs:
        print(f"Retrieved: {doc.page_content} with score {score}")
        if score >= SIMILARITY_THRESHOLD:  # Keep only relevant chunks
            unstructured_data.append(doc.page_content)

    # Check if both structured and unstructured data are empty
    if (
        (not structured_data.strip() or "No relevant information" in structured_data)
        and not unstructured_data
    ):
        return "No relevant information available."

    # Combine and format the final response
    final_data = f"""Structured data:
    {structured_data}
    Unstructured data:
    {"#Document ".join(unstructured_data)}
    """
    return final_data


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)


## Azure OpenAI chat model
chat_model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_APIKEY"),
    model_kwargs={"api_base": os.getenv("AZURE_OPENAI_ENDPOINT")},
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
    temperature=0
)

api_key = os.getenv("AZURE_OPENAI_APIKEY")
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

# Format chat history for context retention
def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

# Define the condense question prompt
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an assistant that refines follow-up questions based on chat history."),
        ("human", "Given the conversation history:\n\n{chat_history}\n\nHow would you rewrite this question?"),
    ]
)

# Search query pipeline
_search_query = RunnableBranch(
    # If input includes chat_history, condense follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | chat_model
        | StrOutputParser(),
    ),
    # Else, just pass through the question
    RunnableLambda(lambda x: x["question"]),
)

template = """Answer the question based only on the following context:
{context}

Question: {question}

Use natural language and be concise.

If the context does not contain relevant information or is empty, always respond with: "No relevant information found."

Answer:"""



prompt = ChatPromptTemplate.from_template(template)    

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)  

# Route for checking if the API is running
@app.route('/', methods=['GET'])
def home():
    return "Flask API is running!", 200

# Handle OPTIONS requests for CORS preflight
@app.route('/', methods=['OPTIONS'])
def options():
    response = jsonify({})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return response, 200

# Update the /chatbot route
@app.route("/", methods=["POST"])
def chatbot():
    try:
        raw_data = request.data.decode("utf-8").strip()
        if not raw_data:
            return jsonify({"message": "No input provided"}), 400
        
        # Ensure the input is passed as a dictionary
        result = chain.invoke({"question": raw_data})

        if result and result.strip() and "relevant information" not in result:
            return jsonify({"answer": result}), 200  
        
        # If no relevant answer is found (retriever returned None), log to HR FAQ DB
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO faq (question) VALUES (%s)", (raw_data,))
                conn.commit()

        return jsonify({"message": "No answer found. Escalating to HR."}), 202  

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
