import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import openai
from langchain.llms import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(override=True)

# Flask App Setup
app = Flask(__name__)
CORS(app)

# API Key & Environment Checks
api_key = os.getenv("AZURE_OPENAI_APIKEY")
if not api_key:
    raise ValueError("Missing AZURE_OPENAI_APIKEY. Ensure it's set in your environment.")

# Load PDFs
pdf_folder = "downloaded_files/"
pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith(".pdf")]
all_documents = []

for pdf in pdf_files:
    try:
        loader = PyPDFLoader(pdf)
        all_documents.extend(loader.load())
        print(f"Loaded PDF: {pdf}")
    except Exception as e:
        print(f"Error loading {pdf}: {e}")

# Split Documents
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunked_documents = text_splitter.split_documents(all_documents)

# Azure OpenAI Client
client = openai.AzureOpenAI(
    api_key=api_key,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
)

# Initialize LLM and Neo4j Graph
llm = AzureChatOpenAI(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_APIKEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0
)

NEO4J_URI="neo4j+s://6f619797.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="loVyer5cvr7MO2MXwob-k7GFq18Bu2iYSoTzxHCR_2A"

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Ensure index exists in Neo4j
graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:Entity) ON EACH [e.id]")

# Set up text embeddings
text_embedding = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key=os.environ['AZURE_OPENAI_APIKEY'],
    azure_deployment=os.environ["AZURE_EMBEDDING_DEPLOYMENT_NAME"],
    model='text-embedding-ada-002'
)

vector_index = Neo4jVector.from_existing_graph(
    text_embedding,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Define entity extraction class
class Entities(BaseModel):
    names: List[str] = Field(..., description="All person, organization, or business entities in the text.")

# Define HR-specific assistant behavior
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an HR assistant for Hoya Electronics. Provide concise answers using knowledge graphs and documents. Escalate if unsure."),
    ("human", "Employee question: {question}"),
])

entity_chain = prompt | llm.with_structured_output(Entities)

def generate_full_text_query(input: str) -> str:
    words = [word for word in input.split() if word]
    return " AND ".join([f"{word}~2" for word in words])

def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    if not hasattr(entities, 'names') or not entities.names:
        return "No entities found in the question."

    for entity in entities.names:
        response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node, score
            WITH node
            CALL {
                WITH node
                MATCH (node)-[r:MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output
            LIMIT 50
            """,
            {"query": generate_full_text_query(entity)}
        )
        result += "\n" + (
            "\n".join([el["output"] for el in response if el["output"] is not None]) 
            if response 
            else f"No relevant information found for entity: {entity}."
        )

    return result

def retriever(question: str):
    structured_data = structured_retriever(question)
    return f"Structured data:\n{structured_data}\nUnstructured data:\n{'#Document'.join(unstructured_data)}"


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    return [msg for pair in chat_history for msg in (HumanMessage(content=pair[0]), AIMessage(content=pair[1]))]

# Define search query pipeline
_search_query = RunnableBranch(
    (RunnableLambda(lambda x: bool(x.get("chat_history"))), 
     RunnablePassthrough.assign(chat_history=lambda x: _format_chat_history(x["chat_history"])) 
     | ChatPromptTemplate.from_messages([
         ("system", "You refine follow-up questions based on chat history."),
         ("human", "Given the conversation history:\n\n{chat_history}\n\nHow would you rewrite this question?")
     ]) | llm | StrOutputParser()),
    RunnableLambda(lambda x: x["question"])
)

# Define answer retrieval chain
prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}\n\nQuestion: {question}\nUse natural language and be concise.\nAnswer:""")

chain = RunnableParallel({"context": _search_query | retriever, "question": RunnablePassthrough()}) | prompt | llm | StrOutputParser()

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

# Flask API Route for handling both JSON and raw string input
@app.route("/", methods=["POST"])
def chatbot():
    try:
        # Get raw data as text
        raw_data = request.data.decode("utf-8").strip()

        if not raw_data:
            return jsonify({"error": "No input provided"}), 400

        print(f"User input: {raw_data}")

        # Placeholder for LangChain response
        response = chain.invoke({"question": raw_data})

        return jsonify({"user_input": raw_data, "response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return an error if something goes wrong

if name == "__main__":
    print("Flask app is starting...")  # Health check message
    app.run (https://app.run/)(debug=True, host="0.0.0.0 (https://0.0.0.0/)", port=3000)
