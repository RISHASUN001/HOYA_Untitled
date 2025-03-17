import os
import warnings
from dotenv import load_dotenv
from flask import Flask, json, request, jsonify
from flask_cors import CORS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
import openai
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import AzureOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from flask import Flask, request, jsonify, session
import psycopg2
from psycopg2.extras import RealDictCursor
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import AzureOpenAIEmbeddings
import threading
import pytesseract
from PIL import Image

# Load environment variables
load_dotenv(override=True)

# Neo4j configurations


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")  # Needed for session
CORS(app)

SECONDARY_NEO4J_URI = "neo4j+s://771eef14.databases.neo4j.io"
PRIMARY_NEO4J_URI = "neo4j+s://6f619797.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
SECONDARY_NEO4J_PASSWORD="XB0t7KZlTx56J1AM2nL6zI4Pkx_HIlgZ2tXy3k69qUc"
PRIMARY_NEO4J_PASSWORD="loVyer5cvr7MO2MXwob-k7GFq18Bu2iYSoTzxHCR_2A"

# Initialize primary and secondary graphs
primary_graph = Neo4jGraph(url=PRIMARY_NEO4J_URI, username=NEO4J_USERNAME, password=PRIMARY_NEO4J_PASSWORD)
secondary_graph = Neo4jGraph(url=SECONDARY_NEO4J_URI, username=NEO4J_USERNAME, password=SECONDARY_NEO4J_PASSWORD)

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

chat_history = {}

llm = AzureChatOpenAI(
    openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key=os.environ['AZURE_OPENAI_APIKEY'],
    azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    temperature=0
)



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
    url=PRIMARY_NEO4J_URI,  
    username=NEO4J_USERNAME,  
    password=PRIMARY_NEO4J_PASSWORD
)




class Entities(BaseModel):
    """Identifying information about entities in the query."""

    names: List[str] = Field(
        ...,
        description="Extract all key entities from the text, including software, applications, services, "
                    "departments, systems, and business entities. Do not filter out any useful keywords.",
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

# from langchain.chat_models import AzureChatOpenAI
# from langchain.schema import HumanMessage

# llm_synonym = AzureChatOpenAI(
#     openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
#     azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
#     api_key=os.environ['AZURE_OPENAI_APIKEY'],
#     azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
#     temperature=0.3
# )


# def expand_entity_with_llm(entity: str) -> List[str]:
#     """Use Azure OpenAI to find alternative names for an entity."""
#     prompt = f"""
#     Provide a list of alternative names, synonyms, and variations for the entity: "{entity}". 
#     Keep responses short and return just a list of words or phrases.
#     """
    
#     response = llm_synonym([HumanMessage(content=prompt)]).content
#     return response.split(", ")

     
# def generate_full_text_query(input: str) -> str:
#     """Generate a Neo4j full-text search query with LLM-generated synonyms."""
#     words = [el for el in remove_lucene_chars(input).split() if el]

#     # Get expanded terms from Azure OpenAI
#     expanded_words = set(words)  # Start with original words
#     for word in words:
#         expanded_terms = expand_entity_with_llm(word)
#         expanded_words.update(expanded_terms)

#     # Format for Neo4j full-text search
#     return " OR ".join([f"{word}~2" for word in expanded_words])

from neo4j import GraphDatabase
import os
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars


# Initialize Neo4j driver
driver = GraphDatabase.driver(
    uri= PRIMARY_NEO4J_URI,
    auth=(NEO4J_USERNAME, PRIMARY_NEO4J_PASSWORD)
)

#Function to create the full-text index
def create_fulltext_index(tx):
    query = '''
    CREATE FULLTEXT INDEX `fulltext_entity_id` 
    FOR (n:__Entity__) 
    ON EACH [n.id];
    '''
    tx.run(query)

# Run the function
def create_index():
    with driver.session() as session:
        session.execute_write(create_fulltext_index)
        print("Fulltext index created successfully.")

# Try creating the index
try:
    create_index()
except Exception as e:
    print("Error creating index:", e)

# Close connection
driver.close()

# Function to clean input and generate a full-text search query for Neo4j
def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print(f"Generated Query: {full_text_query}")
    return full_text_query.strip()


def structured_retriever(question: str, chat_history: List[Tuple[str, str]] = None) -> str:
    result = ""

    # Merge chat history and question
    if chat_history:
        history_context = " ".join([f"Q: {q} A: {a}" for q, a in chat_history[-3:]])
        context_input = f"Chat history:\n{history_context}\n\nFollow-up question: {question}"
    else:
        context_input = question

    entities = entity_chain.invoke({"question": context_input})
    
    if not hasattr(entities, 'names') or not entities.names:
        return "No entities found in the question."

    for entity in entities.names:
        # Generate expanded query using LLM
        query_text = generate_full_text_query(entity)
        
        response = primary_graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": query_text},  # Use expanded query here
        )
        
        if response:
            result += "\n".join([el['output'] for el in response]) + "\n"
        else:
            result += f"\nNo relationships found for: {entity}\n"

    return result.strip()

# # Fulltext index query
# def structured_retriever(question: str, chat_history: List[Tuple[str, str]] = None) -> str:
#     result = ""

#     # Merge chat history and question
#     if chat_history:
#         history_context = " ".join([f"Q: {q} A: {a}" for q, a in chat_history[-3:]])  
#         context_input = f"Chat history:\n{history_context}\n\nFollow-up question: {question}"
#     else:
#         context_input = question

#     entities = entity_chain.invoke({"question": context_input})  
    
#     if not hasattr(entities, 'names') or not entities.names:
#         return "No entities found in the question."

#     for entity in entities.names:
#         query_text = generate_full_text_query(entity)  # Now uses LLM-based expansion
        
#         response = primary_graph.query(
#             """CALL db.index.fulltext.queryNodes('entity', $query, {limit:5})
#             YIELD node, score
#             WHERE score > 0.5  
#             RETURN node.name AS entity_name, score
#             ORDER BY score DESC
#             LIMIT 3
#             """,
#             {"query": query_text},
#         )

#         if not response:
#             result += f"\nNo relevant information found for entity: {entity}."
#         else:
#             for el in response:
#                 result += f"\nPossible match: {el['entity_name']} (Score: {el['score']:.2f})"

#     return result



SIMILARITY_THRESHOLD = 0.7  # Adjust based on experimentation

def clean_text(text: str) -> str:
    """Remove extra spaces and normalize formatting"""
    # Replace newlines with spaces and collapse multiple spaces
    return ' '.join(text.replace('\n', ' ').split()).strip()

def retriever(question: str):
    print(f"Search query: {question}")

    # Retrieve chat history
    chat_history = session.get("chat_history", [])

    # Structured retrieval with cleaning
    structured_data = clean_text(structured_retriever(question, chat_history))
    print(f"Cleaned structured data: {structured_data}")

    # Vector search retrieval with cleaning
    retrieved_docs = vector_index.similarity_search_with_score(question, k=5)
    
    unstructured_data = []
    for doc, score in retrieved_docs:
        cleaned_content = clean_text(doc.page_content)
        print(f"Cleaned retrieved: {cleaned_content} with score {score}")
        if score >= SIMILARITY_THRESHOLD:
            unstructured_data.append(cleaned_content)

    # Check with cleaned data
    if (
        (not structured_data.strip() 
         or "No relevant information" in structured_data)
        and not unstructured_data
    ):
        return "No relevant information available."

    # Format clean final response
    final_data = f"""Structured data:
    {structured_data}
    
    Unstructured data:
    {" ".join([f"#Document {i+1}: {text}" 
              for i, text in enumerate(unstructured_data)])}
    """
    return final_data


# _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
# in its original language.
# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""

# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

chat_model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_APIKEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # Use `azure_endpoint` instead of `api_base`
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
    temperature=0
)


# ## Azure OpenAI chat model
# chat_model = AzureChatOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_APIKEY"),
#     model_kwargs={"api_base": os.getenv("AZURE_OPENAI_ENDPOINT")},
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-01-15-preview"),
#     deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
#     temperature=0
# )

# api_key = os.getenv("AZURE_OPENAI_APIKEY")
# api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
# deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

# Store chat history in session
CHAT_HISTORY_LIMIT = 10  # Limit stored messages

def is_follow_up_question(question: str) -> bool:
    """Check if the question is a follow-up based on connector words."""
    connector_words = ["so", "then", "therefore", "also", "and", "but", "however", "thus"]
    return any(word in question.lower() for word in connector_words)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List[dict]:
    """Format chat history for conversation retention."""
    if not chat_history:
        return []
    
    # Only keep the last few messages to avoid overwhelming the context
    recent_history = chat_history[-3:]  # Adjust the number as needed
    buffer = []
    for human, ai in recent_history:
        buffer.append({"role": "user", "content": human})
        buffer.append({"role": "assistant", "content": ai})
    return buffer
    


# Define the condense question prompt
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant that refines follow-up questions based on chat history. "
            "If the user explicitly references the chat history, include relevant context. "
            "Otherwise, focus only on the most recent question."
        ),
        (
            "human",
            "Given the conversation history:\n\n{chat_history}\n\n"
            "How would you rewrite this question to focus on the most relevant information?\n"
            "Question: {question}"
        ),
    ]
)

from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
import re

def validate_fuzzy_query(query: str) -> str:
    """Allow fuzzy operators but escape problematic characters"""
    # Preserve valid ~2 fuzzy syntax but escape other special chars
    return re.sub(r'([{}[\]()^"*?:\\/])(?![0-9~])', r'\\\1', query)

# Combined search query pipeline
_search_query = RunnableBranch(
    # Priority 1: Poster context with chat history
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="PosterContextCheck"
        ),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | chat_model
        | StrOutputParser()
        | RunnableLambda(validate_fuzzy_query)
    ),
    # Priority 2: Follow-up questions without poster context
    (
        RunnableLambda(lambda x: is_follow_up_question(x["question"])).with_config(
            run_name="FollowUpCheck"
        ),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x.get("chat_history", []))
        )
        | CONDENSE_QUESTION_PROMPT
        | chat_model
        | StrOutputParser()
        | RunnableLambda(validate_fuzzy_query)
    ),
    # Default: Raw question with validation
    RunnableLambda(lambda x: validate_fuzzy_query(x["question"])).with_config(
        run_name="RawQuestion"
    )
)



template = """Answer the question based only on the following context:
{context}

Question: {question}

Use natural language and be concise.

If the context does not contain relevant information or is empty, always respond with: "No relevant information available. Escalating to HR"

Answer:"""



prompt = ChatPromptTemplate.from_template(template)    

from langchain_core.runnables import Runnable

class RetrieverWrapper(Runnable):
    def __init__(self, retriever_func):
        self.retriever_func = retriever_func
        
    def invoke(self, input, config=None, **kwargs):
        # Extract question from LangChain's input format
        question = input.get("question") if isinstance(input, dict) else str(input)
        return self.retriever_func(question)

# Wrap your existing retriever
chain = (
    RunnableParallel({
        "context": _search_query | RetrieverWrapper(retriever),
        "question": RunnablePassthrough(),
    })
    | prompt
    | llm
    | StrOutputParser()
)

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']

credentials = service_account.Credentials.from_service_account_info(
    {
        "type": os.getenv("TYPE"),
        "project_id": os.getenv("PROJECT_ID"),
        "private_key_id": os.getenv("PRIVATE_KEY_ID"),
        "private_key": os.getenv("PRIVATE_KEY").replace('\\n', '\n'),
        "client_email": os.getenv("CLIENT_EMAIL"),
        "client_id": os.getenv("CLIENT_ID"),
        "auth_uri": os.getenv("AUTH_URI"),
        "token_uri": os.getenv("TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
        "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
        "universe_domain": os.getenv("UNIVERSE_DOMAIN"),
    },
    scopes=SCOPES
)

drive_service = build('drive', 'v3', credentials=credentials)

# Folder ID to monitor
FOLDER_ID = os.getenv("FOLDER_ID")
if not FOLDER_ID:
    raise ValueError("GOOGLE_DRIVE_FOLDER_ID environment variable is not set.")

# Local directory to store downloaded files
DOWNLOAD_FOLDER = "downloaded_files"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


# Function to clear the graph
def clear_graph():
    driver = secondary_graph._driver  # Define the driver variable
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("✅ Graph cleared.")
    
def get_folder_state():
    """Retrieve the current state of the Google Drive folder from Google Drive API, including file names and last modified times."""
    state = {}

    try:
        results = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and trashed=false",
            fields="files(id, name, modifiedTime)"
        ).execute()

        files = results.get("files", [])
        for file in files:
            state[file["name"]] = file["modifiedTime"]  # Store last modified time as a string

    except Exception as e:
        print(f"Error fetching folder state: {e}")

    return state  # Dictionary of {filename: last_modified_time}


def download_files():
    """Download all files from the Google Drive folder and remove deleted files."""
    query = f"'{FOLDER_ID}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name, modifiedTime)").execute()
    files = results.get('files', [])

    # Get the list of current files in Google Drive (name, ID, and modifiedTime)
    current_files = {file['name']: {'id': file['id'], 'modifiedTime': file['modifiedTime']} for file in files}

    # Get the list of files currently in the DOWNLOAD_FOLDER
    local_files = set(os.listdir(DOWNLOAD_FOLDER))

    # Remove files from DOWNLOAD_FOLDER that are no longer in Google Drive
    for local_file in local_files:
        if local_file not in current_files:
            local_file_path = os.path.join(DOWNLOAD_FOLDER, local_file)
            os.remove(local_file_path)
            print(f"Removed: {local_file}")

    # Download new or updated files
    for file_name, file_info in current_files.items():
        local_path = os.path.join(DOWNLOAD_FOLDER, file_name)
        file_id = file_info['id']

        # Always re-download hr_faq.txt, regardless of modifiedTime
        if file_name == "hr_faq.txt":
            print(f"Force re-downloading: {file_name}")
            # Download the file
            request = drive_service.files().get_media(fileId=file_id)
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                print(f"Downloaded: {file_name}")
        else:
            # For other files, check if they are up-to-date
            if os.path.exists(local_path):
                # Get the local file's modification time
                local_modified_time = os.path.getmtime(local_path)

                # Parse remote modifiedTime into a timestamp
                remote_modified_time = datetime.datetime.strptime(
                    file_info['modifiedTime'], "%Y-%m-%dT%H:%M:%S.%fZ"
                ).timestamp()

                # Skip download if the local file is up-to-date
                if local_modified_time >= remote_modified_time:
                    print(f"Skipping up-to-date file: {file_name}")
                    continue

            # Download the file (either new or updated)
            request = drive_service.files().get_media(fileId=file_id)
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                print(f"Downloaded: {file_name}")

def rebuild_graph():
    """Rebuild the secondary Neo4j graph with the latest files."""
    print("Clearing existing graph...")
    clear_graph()  # Clear existing graph data
    print("Downloading and processing new files...")
    all_documents = []
    for file_name in os.listdir(DOWNLOAD_FOLDER):
        file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Skipping unsupported file: {file_name}")
            continue

        try:
            documents = loader.load()
            # Add metadata (including 'source') to each document
            for doc in documents:
                doc.metadata["source"] = file_name  # Add the file name as the source
                all_documents.append(doc)
        except Exception as e:
            print(f"Error loading file {file_name}: {e}")

    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    chunked_documents = text_splitter.split_documents(all_documents)

    try:
        print("Converting documents to graph documents...")
        llm_transformer = LLMGraphTransformer(llm=llm)
        graph_documents = llm_transformer.convert_to_graph_documents(chunked_documents)
        print("Graph documents created successfully.")
    except Exception as e:
        print(f"Error converting documents to graph documents: {e}")
        return

    print("Adding graph documents to Neo4j...")
    try:
        secondary_graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
        print("✅ Secondary Neo4j graph rebuilt successfully.")
    except Exception as e:
        print(f"Error adding graph documents to Neo4j: {e}")

def switch_graphs():
    """Switch primary and secondary graphs."""
    global primary_graph, secondary_graph, vector_index, PRIMARY_NEO4J_URI, SECONDARY_NEO4J_URI, PRIMARY_NEO4J_PASSWORD, SECONDARY_NEO4J_PASSWORD
    primary_graph, secondary_graph = secondary_graph, primary_graph
    PRIMARY_NEO4J_URI, SECONDARY_NEO4J_URI = SECONDARY_NEO4J_URI, PRIMARY_NEO4J_URI
    PRIMARY_NEO4J_PASSWORD, SECONDARY_NEO4J_PASSWORD = SECONDARY_NEO4J_PASSWORD, PRIMARY_NEO4J_PASSWORD

    # Update vector index to point to the new primary graph
    vector_index = Neo4jVector.from_existing_graph(
        text_embedding,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
        url=PRIMARY_NEO4J_URI,
        username=NEO4J_USERNAME,
        password=PRIMARY_NEO4J_PASSWORD
    )
    print("✅ Switched primary and secondary graphs.")


import time
import threading
import datetime

stop_event = threading.Event()

def monitor_folder():
    """Monitor the Google Drive folder for changes (file additions, deletions, and modifications)."""
    last_state = get_folder_state()

    while not stop_event.is_set():  # Allow stopping
        current_state = get_folder_state()
        
        if current_state != last_state:  # Detects any change (added, deleted, or modified)
            print("Change detected in Google Drive folder. Rebuilding graph...")
            try:
                download_files()
                rebuild_graph()
                switch_graphs()
                print("✅ Graph switched successfully.")
            except Exception as e:
                print(f"Error during graph rebuilding and switching: {e}")
            last_state = current_state  # Update state after handling changes
        else:
            print("No changes detected in Google Drive folder.")

        stop_event.wait(5)  # Allows other tasks to run


# Set Tesseract OCR path (only needed for Windows)
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# **New API: Extract and Enhance Poster Text**
from PIL import Image, ImageEnhance, ImageFilter

@app.route("/api/extract-text", methods=["POST"])
def extract_text_from_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files["image"]
        image = Image.open(image_file)

        # Log image details
        logger.info(f"Image format: {image.format}")
        logger.info(f"Image size: {image.size}")
        logger.info(f"Image mode: {image.mode}")

        # Preprocess the image
        image = image.convert("L")  # Convert to grayscale
        image = ImageEnhance.Contrast(image).enhance(2.0)  # Increase contrast
        image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image

        # Extract raw text using Tesseract OCR
        raw_text = pytesseract.image_to_string(image).strip()
        logger.info(f"Extracted raw text: {raw_text}")

        if not raw_text:
            logger.warning("No text extracted. Trying alternative preprocessing.")
            image = image.convert("1")  # Convert to black and white
            raw_text = pytesseract.image_to_string(image).strip()

        if not raw_text:
            return jsonify({"error": "No readable text found in the image"}), 400

        # Send the extracted text to Azure OpenAI for refinement
        refined_description = generate_poster_description(raw_text)
        
        # Return refined description
        return jsonify({"poster_description": refined_description}), 200

    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return jsonify({"error": str(e)}), 500



def generate_poster_description(raw_text):
    try:
        system_message = (
            "You are a text extraction tool. Your ONLY tasks are:\n"
            "1. Identify EXACT phrases from the raw text that describe a poster\n"
            "2. Preserve ORIGINAL wording without additions\n"
            "3. Return 'No relevant poster description found' if no clear description exists\n"
            "4. Never invent information\n\n"
            "Examples:\n"
            "Raw: 'HR policies document 2023'\n"
            "Output: 'HR policies document 2023'\n\n"
            "Raw: 'Contact: hr@company.com'\n"
            "Output: 'No relevant poster description found'"
        )

        user_message = (
            f"RAW IMAGE TEXT:\n"
            f"{raw_text}\n\n"
            "EXTRACT EXACT POSTER DESCRIPTION OR RETURN 'No relevant poster description found':"
        )

        logger.info(f"Sending extracted text for literal extraction...\n{user_message}")

        # Use the chat model to generate the poster description
        response = chat_model.invoke(
            [

                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature = 0
        )


        # Log the full response for debugging
        logger.info(f"Azure OpenAI API Response: {response}")

        # Extract the refined description from the response
        refined_text = response.content.strip()

        logger.info(f"Generated Poster Description: {refined_text}")
        return refined_text

    except Exception as e:
        logger.error(f"Error generating poster description: {e}")
        return "Failed to generate a coherent poster description."

def mix_poster_and_question(poster_description, question):
    # Define the no-poster placeholder text
    NO_POSTER_TEXT = "No relevant poster description found"
    
    try:
        # Immediately return raw question if no valid poster
        if poster_description.strip().lower() == NO_POSTER_TEXT.lower():
            logger.info("Skipping combination - no relevant poster")
            return question

        # System message with strict constraints
        system_message = """You are a query combiner. Follow these rules:
1. Use ONLY the EXACT wording from inputs
2. NO new information, interpretations, or assumptions
3. Combine into ONE grammatical sentence
4. Format: "How does [poster content] relate to [question]?"

Example:
Poster: "Safety Helmets Policy"
Question: "What are the requirements?"
Combined: "What are the safety helmet policy requirements?"
"""

        # User message with clear separation
        user_message = f"""POSTER CONTENT (use verbatim):
{poster_description}

QUESTION (use verbatim):
{question}

COMBINED QUERY:"""

        # Get deterministic response
        response = chat_model.invoke([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ], temperature=0)

        # Validate and clean response
        combined = response.content.strip()
        
        # Fallback checks
        if any([
            NO_POSTER_TEXT.lower() in combined.lower(),
            not any(word in combined for word in poster_description.split()[:3]),
            "?" not in combined  # Ensure it's still a question
        ]):
            combined = question  # Fall back to raw question

        logger.info(f"Final combined query: {combined}")
        return combined

    except Exception as e:
        logger.error(f"Combination error: {e}")
        return question  # Return raw question on failure
    


@app.route('/api/escalate', methods=['POST'])
def escalate_question():
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        logger.info("Logging escalated question to HR FAQ DB.")
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO faq (question) VALUES (%s)", (question,))
                conn.commit()
        return jsonify({"success": True}), 200
    except Exception as e:
        logger.error(f"Database error: {e}")
        return jsonify({"error": "Failed to log question to database"}), 500

@app.route("/api/hr-query", methods=["POST"])
def chatbot():
    try:
        # Get raw request data and decode it
        raw_data = request.data.decode("utf-8").strip()
        
        # Log the raw input for debugging
        logger.info(f"Raw input: {raw_data}")

        # Check if the input is empty
        if not raw_data:
            logger.error("No input provided.")
            return jsonify({"error": "No input provided"}), 400

        # Parse input data (handle both JSON and plain string)
        try:
            # Try to parse the request data as JSON
            data = json.loads(raw_data)
            question = data.get("question", "").strip()
            poster_description = data.get("poster_description", "").strip()
            logger.info(f"Parsed JSON input - Question: {question}, Poster Description: {poster_description}")
        except json.JSONDecodeError:
            # If parsing as JSON fails, treat the request data as a plain string (the question itself)
            question = raw_data
            poster_description = ""
            logger.info(f"Plain string input - Question: {question}")

        # Validate the question
        if not question:
            logger.error("Question is required.")
            return jsonify({"error": "Question is required"}), 400

        # Get session state
        current_poster = session.get("active_poster", "")
        poster_chat_history = session.get("poster_chat_history", [])
        general_chat_history = session.get("general_chat_history", [])

        # Determine question type
        is_poster_follow_up = current_poster and is_follow_up_question(question)
        is_general_follow_up = not current_poster and is_follow_up_question(question)

        # Handle poster context
        if poster_description:
            # New poster context - reset all histories
            session["active_poster"] = poster_description
            session["poster_chat_history"] = []
            session["general_chat_history"] = []
            logger.info("New poster context detected. Resetting all histories.")
        elif not is_poster_follow_up and not is_general_follow_up:
            # Clear poster context for standalone questions
            session["active_poster"] = ""
            session["poster_chat_history"] = []
            logger.info("Standalone question detected. Clearing poster context.")

        # Prepare query based on context
        active_poster = session.get("active_poster", "")
        combined_input = question  # Default to raw question

        if active_poster and "No relevant poster" not in active_poster:
            if is_poster_follow_up:
                # Combine with poster context for follow-ups
                combined_input = mix_poster_and_question(active_poster, question)
                logger.info(f"Poster follow-up: {combined_input}")
            else:
                # Clear poster context for non-follow-up questions
                session["active_poster"] = ""
                session["poster_chat_history"] = []
                logger.info("Non-follow-up question - cleared poster context")

        # Manage conversation histories
        if active_poster:
            # Use poster-specific history
            chat_history = poster_chat_history
            history_key = "poster_chat_history"
        else:
            # Use general conversation history
            chat_history = general_chat_history
            history_key = "general_chat_history"

        if not (is_poster_follow_up or is_general_follow_up):
            session[history_key] = []
            chat_history = []
            logger.info("Cleared chat history for new conversation")

        # Process query
        try:
            logger.info("Invoking processing chain")
            result = chain.invoke({
                "question": combined_input,
                "chat_history": chat_history
            })
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return jsonify({"error": "Failed to process question"}), 500

        # Handle response
        if result and "relevant information" not in result:
            # Update appropriate history
            session[history_key] = chat_history + [(question, result)]
            return jsonify({"answer": result}), 200
        else: 
            logger.info("No answer found. Escalating to HR.")
            return jsonify({"answer": result}), 202

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


        
if __name__ == "__main__":
    print("Starting Google Drive folder monitor...")
    
    monitor_thread = threading.Thread(target=monitor_folder, daemon=True)
    monitor_thread.start()
    
    # Start the Flask app
    app.run(debug=True, host="0.0.0.0", port=5001)
