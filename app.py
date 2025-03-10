import os
import warnings
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
import openai
import warnings
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

# Load environment variables
load_dotenv(override=True)

# Neo4j configurations


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")  # Needed for session
CORS(app)

SECONDARY_NEO4J_URI = "neo4j+s://495232c8.databases.neo4j.io"
PRIMARY_NEO4J_URI = "neo4j+s://9eb697a7.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
SECONDARY_NEO4J_PASSWORD="fwVyXeBgxH_vnFyQz0t9zx1srgTFELQz2_Szf1dyuGA"
PRIMARY_NEO4J_PASSWORD="xtQyfqUiVRnGFcQgofngfX-g0LKnIs2X67SBmO1Qm7M"

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

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

llm_synonym = AzureChatOpenAI(
    openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key=os.environ['AZURE_OPENAI_APIKEY'],
    azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    temperature=0.3
)


def expand_entity_with_llm(entity: str) -> List[str]:
    """Use Azure OpenAI to find alternative names for an entity."""
    prompt = f"""
    Provide a list of alternative names, synonyms, and variations for the entity: "{entity}". 
    Keep responses short and return just a list of words or phrases.
    """
    
    response = llm_synonym([HumanMessage(content=prompt)]).content
    return response.split(", ")

     
def generate_full_text_query(input: str) -> str:
    """Generate a Neo4j full-text search query with LLM-generated synonyms."""
    words = [el for el in remove_lucene_chars(input).split() if el]

    # Get expanded terms from Azure OpenAI
    expanded_words = set(words)  # Start with original words
    for word in words:
        expanded_terms = expand_entity_with_llm(word)
        expanded_words.update(expanded_terms)

    # Format for Neo4j full-text search
    return " OR ".join([f"{word}~2" for word in expanded_words])



# Fulltext index query
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
        query_text = generate_full_text_query(entity)  # Now uses LLM-based expansion
        
        response = primary_graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:5})
            YIELD node, score
            WHERE score > 0.5  
            RETURN node.name AS entity_name, score
            ORDER BY score DESC
            LIMIT 3
            """,
            {"query": query_text},
        )

        if not response:
            result += f"\nNo relevant information found for entity: {entity}."
        else:
            for el in response:
                result += f"\nPossible match: {el['entity_name']} (Score: {el['score']:.2f})"

    return result



SIMILARITY_THRESHOLD = 0.7  # Adjust based on experimentation

def retriever(question: str):
    print(f"Search query: {question}")

    # Retrieve chat history
    chat_history = session.get("chat_history", [])

    # Structured retrieval using chat history
    structured_data = structured_retriever(question, chat_history)
    print(f"Structured data retrieved: {structured_data}")

    # Vector search retrieval (unchanged)
    retrieved_docs = vector_index.similarity_search_with_score(question, k=5)
    
    unstructured_data = []
    for doc, score in retrieved_docs:
        print(f"Retrieved: {doc.page_content} with score {score}")
        if score >= SIMILARITY_THRESHOLD:
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

# Store chat history in session
CHAT_HISTORY_LIMIT = 10  # Limit stored messages

def _format_chat_history():
    """Format chat history for conversation retention."""
    history = session.get("chat_history", [])
    buffer = []
    for human, ai in history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer
    


# Define the condense question prompt
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an assistant that refines follow-up questions based on chat history. Focus only on the most recent question and ignore the rest of the chat history unless explicitly referenced."),
        ("human", "Given the conversation history:\n\n{chat_history}\n\nHow would you rewrite this question to focus only on the most recent question?"),
    ]
)

# Search query pipeline
_search_query = RunnableBranch(
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



@app.route("/", methods=["POST"])
def chatbot():
    try:
        # Parse the JSON payload
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400

        # Extract poster description and question from the JSON payload
        poster_description = data.get("poster_description", "")
        question = data.get("question", "")

        # Combine poster description and question (if poster description is provided)
        if poster_description:
            combined_input = f"Poster Description: {poster_description}\nQuestion: {question}"
        else:
            combined_input = question

        # Retrieve chat history
        chat_history = _format_chat_history()

        # Pass the combined input to the chatbot
        result = chain.invoke({"question": combined_input, "chat_history": chat_history})

        if result and result.strip() and "relevant information" not in result:
            # Update session history
            history = session.get("chat_history", [])
            history.append((combined_input, result))
            session["chat_history"] = history[-CHAT_HISTORY_LIMIT:]  # Keep only the last few messages

            return jsonify({"answer": result}), 200

        # If no relevant answer is found, log to HR FAQ DB
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO faq (question) VALUES (%s)", (combined_input,))
                conn.commit()

        return jsonify({"message": "No answer found. Escalating to HR."}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    print("Starting Google Drive folder monitor...")
    
    monitor_thread = threading.Thread(target=monitor_folder, daemon=True)
    monitor_thread.start()
    
    # Start the Flask app
    app.run(debug=True, host="0.0.0.0", port=5001)
