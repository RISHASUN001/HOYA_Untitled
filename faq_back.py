

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from google.oauth2 import service_account
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from neo4j import GraphDatabase
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import sqlite3
import io
import os
import logging
from io import BytesIO
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph


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



# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
CORS(app)
FAQ_FILENAME = "hr_faq.txt"
### 🔹 Neo4j Setup
# SECONDARY_NEO4J_URI = "neo4j+s://495232c8.databases.neo4j.io"
# PRIMARY_NEO4J_URI = "neo4j+s://9eb697a7.databases.neo4j.io"
# NEO4J_USERNAME="neo4j"
# SECONDARY_NEO4J_PASSWORD="fwVyXeBgxH_vnFyQz0t9zx1srgTFELQz2_Szf1dyuGA"
# PRIMARY_NEO4J_PASSWORD="xtQyfqUiVRnGFcQgofngfX-g0LKnIs2X67SBmO1Qm7M"

# # Initialize primary and secondary graphs
# primary_graph = Neo4jGraph(url=PRIMARY_NEO4J_URI, username=NEO4J_USERNAME, password=PRIMARY_NEO4J_PASSWORD)
# secondary_graph = Neo4jGraph(url=SECONDARY_NEO4J_URI, username=NEO4J_USERNAME, password=SECONDARY_NEO4J_PASSWORD)

# # Clears Neo4j before reinserting new data
# def clear_graph():
#     with driver.session() as session:
#         session.run("MATCH (n) DETACH DELETE n")
#     logging.info("✅ Cleared Neo4j database.")

### 🔹 Google Drive API Setup
SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_ID = os.getenv("FOLDER_ID")
# Authenticate Google Drive
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

drive_service = build("drive", "v3", credentials=credentials)

from googleapiclient.http import MediaFileUpload
import io

FAQ_FILENAME = "hr_faq.txt"

def get_or_create_faq_file():
    response = drive_service.files().list(
        q=f"name='{FAQ_FILENAME}' and '{FOLDER_ID}' in parents",
        fields="files(id, mimeType)"
    ).execute()
    
    files = response.get("files", [])
    if files:
        file_id = files[0]["id"]
        logging.info(f"📄 Found existing FAQ file: {file_id}")
        return file_id
    
    logging.info("❌ FAQ file not found, creating a new one...")

    file_metadata = {
        "name": FAQ_FILENAME,
        "parents": [FOLDER_ID]
    }
    
    # Create a local empty file (ensures it's .txt format)
    with open(FAQ_FILENAME, "w") as f:
        f.write("")

    media = MediaFileUpload(FAQ_FILENAME, mimetype="text/plain")

    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()

    logging.info(f"✅ Successfully created FAQ file: {file['id']}")
    return file["id"]

FAQ_FILE_ID = get_or_create_faq_file()

# # Function to list files from Google Drive
# def list_drive_files(drive_service):
#     try:
#         results = drive_service.files().list(q=f"'{FOLDER_ID}' in parents and trashed=false", fields="nextPageToken, files(id, name, mimeType)").execute()
#         items = results.get('files', [])
#         logging.info(f"Found {len(items)} files in Google Drive.")
#         return items
#     except Exception as e:
#         logging.error(f"Error listing files: {e}")
#         return []

# # Function to download file from Google Drive
# def download_file_from_gdrive(file_id, drive_service):
#     try:
#         request = drive_service.files().get_media(fileId=file_id)
#         fh = BytesIO()
#         downloader = MediaIoBaseDownload(fh, request)
#         done = False
#         while done is False:
#             status, done = downloader.next_chunk()
#         fh.seek(0)  # Reset file pointer
#         return fh
#     except Exception as e:
#         logging.error(f"Error downloading file {file_id}: {e}")
#         return None

# Function to load documents from Google Drive
# def load_documents_from_gdrive(drive_service):
#     try:
#         documents = []
#         files = list_drive_files(drive_service)
#         logging.info(f"Found {len(files)} files in Google Drive.")
        
#         for file in files:
#             file_name = file["name"]
#             file_id = file["id"]
#             file_type = file["mimeType"]
#             logging.info(f"Processing file: {file_name} (ID: {file_id})")
            
#             if file_name.endswith(".txt"):
#                 file_content = download_file_from_gdrive(file_id, drive_service)
#                 if file_content:
#                     content = file_content.read().decode("utf-8")
#                     documents.append(Document(page_content=content, metadata={"source": file_name}))
#                     logging.info(f"Loaded text file: {file_name}")
#                 else:
#                     logging.error(f"Failed to load text file: {file_name}")
            
#             elif file_name.endswith(".pdf"):
#                 file_content = download_file_from_gdrive(file_id, drive_service)
#                 if file_content:
#                     # Save the PDF content to a temporary file for PyPDFLoader
#                     with open("temp.pdf", "wb") as temp_file:
#                         temp_file.write(file_content.read())
                    
#                     # Load the PDF using PyPDFLoader
#                     pdf_loader = PyPDFLoader("temp.pdf")
#                     pdf_documents = pdf_loader.load()
                    
#                     # Append each page as a separate document
#                     for doc in pdf_documents:
#                         documents.append(doc)
                    
#                     logging.info(f"Loaded PDF file: {file_name} (Pages: {len(pdf_documents)})")
                    
#                     # Clean up the temporary file
#                     os.remove("temp.pdf")
#                 else:
#                     logging.error(f"Failed to load PDF file: {file_name}")
        
#         logging.info(f"✅ Total Documents Loaded: {len(documents)}")
#         return documents
#     except Exception as e:
#         logging.error(f"Error in load_documents_from_gdrive: {e}")
#         return []

# # Function to chunk documents
# def chunk_documents(all_documents):
#     try:
#         chunk_size = 1000  # Adjust based on your needs
#         chunk_overlap = 200  # Adjust based on your needs
        
#         # Create the text splitter
#         text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
#         # Split all documents into chunks
#         chunked_documents = text_splitter.split_documents(all_documents)
        
#         logging.info(f"✅ Total Chunks: {len(chunked_documents)}")
#         return chunked_documents
#     except Exception as e:
#         logging.error(f"Error in chunk_documents: {e}")
#         return []
# # Update the update_faq_document function

def update_faq_document():
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT question, answer FROM faq WHERE answer IS NOT NULL")
                faqs = cursor.fetchall()

        faq_content = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in faqs])

        media = MediaIoBaseUpload(io.BytesIO(faq_content.encode()), mimetype="text/plain")
        drive_service.files().update(fileId=FAQ_FILE_ID, media_body=media).execute()
    except Exception as e:
        logging.error(f"Error updating FAQ document: {e}")

# # Initialize Azure OpenAI LLM
# llm = AzureChatOpenAI(
#     openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
#     azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
#     api_key=os.environ['AZURE_OPENAI_APIKEY'],
#     azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
#     temperature=0
# )

# # Function to transform documents to graph
# def transform_to_graph(chunked_documents, llm):
#     try:
#         llm_transformer = LLMGraphTransformer(llm=llm)
#         graph_documents = llm_transformer.convert_to_graph_documents(chunked_documents)
#         logging.info(f"✅ Successfully converted to graph documents: {len(graph_documents)}")
#         return graph_documents
#     except Exception as e:
#         logging.error(f"Error during graph transformation: {e}")
#         return None

# Function to update Neo4j vector database
# def update_vector_db(graph_documents):
#     try:
#         # Setup embeddings
#         text_embedding = AzureOpenAIEmbeddings(
#             azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
#             api_key=os.environ['AZURE_OPENAI_APIKEY'],
#             azure_deployment=os.environ["AZURE_EMBEDDING_DEPLOYMENT_NAME"],
#             model='text-embedding-ada-002'
#         )

#         # Load the vector store (Neo4j vector database)
#         vector_index = Neo4jVector.from_existing_graph(
#             text_embedding,  
#             search_type="hybrid",  # i.e search on both keywords and embeddings
#             node_label="Document",
#             text_node_properties=["text"],
#             embedding_node_property="embedding",
#             url=NEO4J_URI,  
#             username=NEO4J_USERNAME,  
#             password=NEO4J_PASSWORD
#         )
#         logging.info(f"✅ Vector DB updated for hybrid search")
#     except Exception as e:
#         logging.error(f"Error in update_vector_db: {e}")

# Main function to process everything
# def process_drive_files():
#     try:
#         # clear_graph()  # Clear Neo4j before inserting new data
        
#         # Load documents from Google Drive
#         documents = load_documents_from_gdrive(drive_service)
#         if not documents:
#             logging.error("No documents loaded. Exiting.")
#             return
        
#         # Chunk documents
#         chunks = chunk_documents(documents)
#         if not chunks:
#             logging.error("No chunks created. Exiting.")
#             return
        
#         # Transform to graph documents
#         graph_documents = transform_to_graph(chunks, llm)
#         graph = Neo4jGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
#         ## here is where the graph has been stored
#         graph.add_graph_documents(
#             graph_documents,
#             baseEntityLabel=True,
#             include_source=True
#         )
#         if not graph_documents:
#             logging.error("No graph documents created. Exiting.")
#             return
        
#         # Update Neo4j vector DB
#         update_vector_db(graph_documents)
#         logging.info("✅ Knowledge graph updated successfully.")
#     except Exception as e:
#         logging.error(f"Error in process_drive_files: {e}")

@app.route("/unanswered", methods=["GET"])
def get_unanswered_questions():
    """Fetch unanswered questions from the database"""
    try:
        with psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, question FROM faq WHERE status = 'pending'")
                questions = cursor.fetchall()
        return jsonify(questions)
    except Exception as e:
        logging.error(f"Error fetching unanswered questions: {e}")
        return jsonify({"error": "Failed to fetch unanswered questions"}), 500

@app.route("/delete", methods=["POST"])
def delete_question():
    data = request.json  # Use JSON format instead of form-data
    question_id = data.get("id")  # Extract the question ID

    if not question_id:
        return jsonify({"error": "Missing question ID"}), 400

    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM faq WHERE id = %s", (question_id,))
                conn.commit()

        return jsonify({"message": "Deleted successfully"}), 200
    except Exception as e:
        logging.error(f"Error deleting question: {e}")
        return jsonify({"error": "Database error"}), 500



# Update the /answer route
@app.route("/answer", methods=["POST"])
def answer_question():
    """HR submits an answer."""
    data = request.json
    question_id = data.get("id")
    answer = data.get("answer")

    if not question_id or not answer:
        return jsonify({"error": "Missing question ID or answer"}), 400

    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                cursor.execute("UPDATE faq SET answer=%s, status='answered' WHERE id=%s", (answer, question_id))
                conn.commit()
        
        update_faq_document()
        # process_drive_files()

        return jsonify({"message": "Answer saved and FAQ document updated"}), 200
    except Exception as e:
        logging.error(f"Error updating answer: {e}")
        return jsonify({"error": "Failed to update answer"}), 500

# @app.route("/undo-delete", methods=["POST"])
# def undo_delete():
#     """HR undoes a question deletion."""
#     try:
#         with psycopg2.connect(DATABASE_URL) as conn:
#             with conn.cursor() as cursor:
#                 # Move the most recent deleted question back to the main table
#                 cursor.execute("""
#                     INSERT INTO faq SELECT * FROM faq_deleted 
#                     WHERE id = (SELECT id FROM faq_deleted ORDER BY deleted_at DESC LIMIT 1)
#                     RETURNING id;
#                 """)
#                 deleted_id = cursor.fetchone()

#                 if deleted_id:
#                     cursor.execute("DELETE FROM faq_deleted WHERE id=%s", (deleted_id,))
#                     conn.commit()
#                     return jsonify({"message": "Deletion undone successfully."}), 200
#                 else:
#                     return jsonify({"error": "No recent deletions to undo."}), 400
#     except Exception as e:
#         logging.error(f"Error undoing deletion: {e}")
#         return jsonify({"error": "Failed to undo deletion"}), 500


@app.route("/hr", methods=["GET"])
def hr_dashboard():
    return render_template("hr_dashboard.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)


    
