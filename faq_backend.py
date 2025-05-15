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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import smtplib
from dotenv import load_dotenv
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText



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

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")  # Hardcoded receiver email
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT"))

def send_email(question, answer):
    try:
        # Create email message
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER  # Send to yourself
        msg["Subject"] = "New Answer Submitted"

        # Email body
        email_body = f"""
        <html>
        <body>
            <p><strong>Question:</strong></p>
            <p>{question}</p>
            <hr>
            <p><strong>Answer:</strong></p>
            <p>{answer}</p>
            <br>
            <p>This is an automated email from your prototype system.</p>
        </body>
        </html>
        """

        msg.attach(MIMEText(email_body, "html"))

        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure connection
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

        print("Email sent successfully!")
    except Exception as e:
        logging.error(f"Error sending email: {e}")

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

def get_or_create_faq_file():
    response = drive_service.files().list(q=f"name='{FAQ_FILENAME}' and '{FOLDER_ID}' in parents", fields="files(id)").execute()
    files = response.get("files", [])
    if files:
        return files[0]["id"]
    
    file_metadata = {"name": FAQ_FILENAME, "mimeType": "text/plain", "parents": [FOLDER_ID]}
    media = MediaIoBaseUpload(io.BytesIO(b""), mimetype="text/plain")
    file = drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return file["id"]

FAQ_FILE_ID = get_or_create_faq_file()

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
    data = request.json
    question_ids = data.get("ids", [])  # Expecting an array of IDs

    if not question_ids:
        return jsonify({"error": "No question IDs provided"}), 400

    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                # Use WHERE id = ANY(%s) for multiple deletions
                cursor.execute("DELETE FROM faq WHERE id = ANY(%s)", (question_ids,))
                conn.commit()

        return jsonify({"message": "Deleted successfully"}), 200
    except Exception as e:
        logging.error(f"Error deleting questions: {e}")
        return jsonify({"error": "Database error"}), 500

@app.route("/answer", methods=["POST"])
def answer_question():
    """Submit an answer to a question."""
    data = request.json
    question_id = data.get("id")
    answer = data.get("answer")

    if not question_id or not answer:
        return jsonify({"error": "Missing question ID or answer"}), 400

    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT question FROM faq WHERE id = %s", (question_id,))
                result = cursor.fetchone()

                if not result:
                    return jsonify({"error": "Question not found"}), 404
                
                question = result[0]

                cursor.execute("UPDATE faq SET answer=%s, status='answered' WHERE id=%s RETURNING answer", (answer, question_id))
                updated_answer = cursor.fetchone()[0]  # Fetch the newly saved answer
                conn.commit()

        update_faq_document()
        send_email(question, updated_answer) 

        return jsonify({"message": "Answer saved, email sent"}), 200
    except Exception as e:
        logging.error(f"Error updating answer: {e}")
        return jsonify({"error": "Failed to update answer"}), 500
from flask import Flask, render_template, request, redirect, url_for
# Route for the login page
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # Validate credentials
        if email == "abc@gmail.com" and password == "123456":
            return redirect(url_for("hr_dashboard"))  # Redirect to HR dashboard
        else:
            return render_template("login.html", error="Invalid email or password.")

    return render_template("login.html")

@app.route("/hr", methods=["GET"])
def hr_dashboard():
    return render_template("hr_dashboard.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)


    