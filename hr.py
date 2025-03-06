import streamlit as st
import sqlite3
import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chat_models import AzureChatOpenAI
from langchain.graphs import Neo4jGraph
from langchain.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.embeddings import AzureOpenAIEmbeddings
from app import primary_graph, secondary_graph, text_embedding, vector_index, PRIMARY_NEO4J_URI, SECONDARY_NEO4J_URI, NEO4J_USERNAME, PRIMARY_NEO4J_PASSWORD, SECONDARY_NEO4J_PASSWORD

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

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect("hr_questions.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS unanswered_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT
        )
    """)
    conn.commit()
    conn.close()
    precreate_questions()  # Pre-create questions

# Pre-create questions
def precreate_questions():
    """Pre-populate the SQLite database with unanswered questions."""
    questions = [
        "What is the company's policy on remote work?",
        "How do I apply for a leave of absence?",
        "What are the benefits of joining the company's wellness program?",
        "How does the performance review process work?",
        "What is the process for reporting workplace harassment?"
    ]

    conn = sqlite3.connect("hr_questions.db")
    c = conn.cursor()
    for question in questions:
        # Check if the question already exists to avoid duplicates
        c.execute("SELECT id FROM unanswered_questions WHERE question = ?", (question,))
        if not c.fetchone():
            c.execute("INSERT INTO unanswered_questions (question) VALUES (?)", (question,))
    conn.commit()
    conn.close()

# Save unanswered question to SQLite
def save_unanswered_question(question):
    conn = sqlite3.connect("hr_questions.db")
    c = conn.cursor()
    c.execute("INSERT INTO unanswered_questions (question) VALUES (?)", (question,))
    conn.commit()
    conn.close()

# Fetch all unanswered questions from SQLite
def fetch_unanswered_questions():
    conn = sqlite3.connect("hr_questions.db")
    c = conn.cursor()
    c.execute("SELECT id, question FROM unanswered_questions WHERE answer IS NULL")
    data = c.fetchall()
    conn.close()
    return data

# Update question with an answer in SQLite
def update_question_answer(question_id, answer):
    conn = sqlite3.connect("hr_questions.db")
    c = conn.cursor()
    c.execute("UPDATE unanswered_questions SET answer = ? WHERE id = ?", (answer, question_id))
    conn.commit()
    conn.close()

# Rebuild the secondary Neo4j graph
def rebuild_graph(new_answer=None):
    """Rebuild the secondary Neo4j graph with the latest files and optionally a new answer."""
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

    # If a new answer is provided, add it as a document
    if new_answer:
        new_doc = Document(
            page_content=new_answer,
            metadata={"source": "HR_Answer"}  # Add metadata for the new answer
        )
        all_documents.append(new_doc)

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

# Switch primary and secondary graphs
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

# Streamlit App
def hr_dashboard():
    """HR Dashboard to answer unanswered questions."""
    st.header("HR Dashboard")

    # Fetch and display unanswered questions
    questions = fetch_unanswered_questions()
    if questions:
        for q in questions:
            st.write(f"**Question ID {q[0]}:** {q[1]}")
            answer = st.text_area(f"Answer for Question ID {q[0]}", key=f"answer_{q[0]}")
            if st.button(f"Submit Answer for Question ID {q[0]}"):
                if answer:
                    # Update the SQLite database with the answer
                    update_question_answer(q[0], answer)

                    # Rebuild the secondary Neo4j graph with the new answer
                    rebuild_graph(new_answer=answer)

                    # Switch the primary and secondary graphs
                    switch_graphs()

                    st.success("Answer submitted and graph updated!")
                else:
                    st.error("Please enter an answer.")
    else:
        st.info("No unanswered questions.")

# Run the Streamlit app
if __name__ == "__main__":
    init_db()  # Initialize database and pre-create questions
    hr_dashboard()