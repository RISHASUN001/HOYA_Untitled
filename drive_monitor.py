import os
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from app import primary_graph, secondary_graph, text_embedding, vector_index, PRIMARY_NEO4J_URI, SECONDARY_NEO4J_URI, NEO4J_USERNAME, PRIMARY_NEO4J_PASSWORD, SECONDARY_NEO4J_PASSWORD
from langchain.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  

# Load environment variables
load_dotenv(override=True)

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account.json'  # Replace with your service account file

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

# Folder ID to monitor
FOLDER_ID = os.getenv("FOLDER_ID")
if not FOLDER_ID:
    raise ValueError("GOOGLE_DRIVE_FOLDER_ID environment variable is not set.")

# Local directory to store downloaded files
DOWNLOAD_FOLDER = "downloaded_files"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

def get_folder_state():
    """Get the current state of the folder (list of file IDs)."""
    query = f"'{FOLDER_ID}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id)").execute()
    return set(file['id'] for file in results.get('files', []))

def download_files():
    """Download all files from the Google Drive folder."""
    query = f"'{FOLDER_ID}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])

    for file in files:
        file_id = file['id']
        file_name = file['name']
        local_path = os.path.join(DOWNLOAD_FOLDER, file_name)

        # Download file
        request = drive_service.files().get_media(fileId=file_id)
        with open(local_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Downloaded: {file_name}")

def rebuild_graph():
    """Rebuild the secondary Neo4j graph with the latest files."""
    # Clear existing graph data
    secondary_graph.query("MATCH (n) DETACH DELETE n")

    # Load and process files
    all_documents = []
    for file_name in os.listdir(DOWNLOAD_FOLDER):
        file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            print(f"Skipping unsupported file: {file_name}")
            continue

        documents = loader.load()
        # Add metadata (including 'source') to each document
        for doc in documents:
            doc.metadata["source"] = file_name  # Add the file name as the source
            all_documents.append(doc)

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    chunked_documents = text_splitter.split_documents(all_documents)

    # Initialize LLM and Graph Transformer
    llm = AzureChatOpenAI(
        openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_APIKEY'],
        azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
        temperature=0
    )


    llm_transformer = LLMGraphTransformer(llm=llm)
    # Convert documents to graph documents
    try:
        graph_documents = llm_transformer.convert_to_graph_documents(chunked_documents)
        print("Graph documents created successfully.")
    except Exception as e:
        print(f"Error converting documents to graph documents: {e}")
        return

    # Add graph documents to Neo4j
    secondary_graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

    print("✅ Secondary Neo4j graph rebuilt successfully.")

def monitor_folder():
    """Monitor the Google Drive folder for changes."""
    last_state = get_folder_state()  # Initialize last_state with the current state
    while True:
        current_state = get_folder_state()
        if current_state != last_state:
            print("Change detected in Google Drive folder. Rebuilding graph...")
            download_files()
            rebuild_graph()
            switch_graphs()  # Switch to the rebuilt graph            
            last_state = current_state
        time.sleep(60)  # Check every 60 seconds

def switch_graphs():
    """Switch primary and secondary graphs."""
    global primary_graph, secondary_graph, vector_index
    primary_graph, secondary_graph = secondary_graph, primary_graph

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

if __name__ == "__main__":
    print("Starting Google Drive folder monitor...")
    monitor_folder()