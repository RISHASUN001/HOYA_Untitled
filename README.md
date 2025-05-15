

# 🏢 Team\_Untitled – ChatHera: The HR Chatbot Revolution 💬

🏆 **1st Runner-Up** at **NTU CampCode x HOYA Chatbot Hackathon 2025**

**ChatHera** – “Her-Era” is a next-gen HR chatbot powered by Retrieval-Augmented Generation (RAG), Neo4j Graph Databases, and Computer Vision. Built for HOYA Electronics, ChatHera reduces HR workloads by automating policy-related queries, learning from unknown questions, and offering seamless, context-aware conversations.

> 💡 **UI inspired by HOYA Electronics' official website**

---

## 📚 Table of Contents

1. [Overview](#1-overview)
2. [Features](#2-features)
3. [How It Works](#3-how-it-works)
4. [Installation](#4-installation)
5. [Usage](#5-usage)
6. [Future Plans](#6-future-plans)
7. [Contributors](#7-contributors)

---

## 1. Overview

Modern HR departments face a flood of repetitive queries. ChatHera helps by:

* Parsing and understanding HR policy PDFs and images.
* Providing context-aware, natural conversations.
* Learning from unanswered questions through a self-updating FAQ system.
* Ensuring **zero downtime** through a **dual-graph database design**.

---

## 2. Features

### 💬 Conversational AI

* Remembers conversation history
* Handles follow-up queries and topic switching

### 🧠 Self-Updating Knowledge Base

* Neo4j dual-graph architecture (`active_graph` and `update_graph`)
* Updates automatically after HR responds to new questions

### 📷 Document Image Processing

* OCR with **Tesseract** for scanned policies and screenshots

### 🧾 HR FAQ Management Dashboard

* Routes unanswered questions to HR
* After response:

  * Updates the knowledge base
  * Notifies the user via email
  * Updates the chatbot

### 🔄 Dual-Graph with Zero Downtime

* `active_graph`: Serves users
* `update_graph`: Updated in the background and swapped in

### 📧 Automated Notifications

* Users receive email alerts when their queries are answered

---

## 3. How It Works

### 🧩 Architecture Overview

1. **Frontend Chatbot**

   * Web UI inspired by Microsoft Teams Copilot
   * Supports real-time chat with context retention

### 2. **Backend: RAG Engine**

- Analyzes the incoming user query to extract **key entities** and **semantic meaning**.
- **Hybrid Search Pipeline**:
  - **Neo4j Graph Database** is used to identify relevant policy nodes and their relationships based on extracted entities.
  - **Vector Database** retrieves semantically similar document chunks using Azure OpenAI embeddings.
- **Combined Results**:
  - Responses from Neo4j (entity-based search) and the Vector DB (semantic search) are merged.
  - Ensures highly relevant and context-aware answers.
- **Optimized Search Time**:
  - Neo4j acts as a pre-filter, reducing the load on the vector DB and speeding up the overall response time.


3. **Neo4j Dual-Graph Design**

   * `active_graph`: Used in production
   * `update_graph`: Updated version activated post-sync

4. **Tesseract OCR**

   * Extracts text from uploaded images

5. **Admin Dashboard (FAQ System)**

   * HR views and responds to unanswered queries
   * Triggers graph updates and user notifications

---

## 4. Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RISHASUN001/HOYA_Untitled.git
cd HOYA_Untitled
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

#### macOS (Homebrew):

```bash
brew install tesseract
```

#### Windows:

* Download from: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
* Add the install path (e.g., `C:\Program Files\Tesseract-OCR`) to the system `PATH`.

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
### Azure OpenAI Configuration
OPENAI_API_TYPE="azure"
AZURE_OPENAI_API_VERSION="2024-08-01"                 
AZURE_OPENAI_ENDPOINT="your_endpoint_here"
AZURE_OPENAI_APIKEY="your_api_key_here"
AZURE_OPENAI_DEPLOYMENT_NAME="your_deployment_name"
AZURE_OPENAI_MODEL_NAME="gpt-4"
AZURE_EMBEDDING_DEPLOYMENT_NAME="embedding_deployment_name"
AZURE_TEXT_EMBEDDING_MODEL="text-embedding-ada-002"

### Document Storage Folder
FOLDER_ID="folder_id_where_documents_are_stored"

### Database URL
DATABASE_URL="your_unanswered_questions_db_url"
```

> ⚠️ Use **2024-08 API version or newer** to ensure compatibility with Neo4j’s LLMGraphTransformer.

### 5. Run the Application

Start the servers:

```bash
# Backend server
python app.py

# FAQ Backend server
python faq_backend.py

# Frontend
npm run dev
```

---

## 5. Usage

1. Access the chatbot via your browser.
2. Ask HR-related questions or upload a screenshot of HR policies.
3. If a question isn't answered:

   * It gets routed to HR via the dashboard.
   * Once HR replies, you'll receive an email notification.
   * The chatbot updates itself automatically.

---

## 6. Future Plans

🌐 **Multi-language support**
💬 **Microsoft Teams integration**
⚖️ **Legal Department**

* Contract clause detection and alarm phrasing

💰 **Finance Department**

* FAQs on payroll, reimbursement, SOPs

🔗 **Microsoft Graph API Integration**

* D365 and HRMS sync

---

## 7. Contributors

- 👩‍💻 **Risha Sunil Shetty** – [@RISHASUN001](https://github.com/RISHASUN001)
- 👩‍💻 **Janhavee Singh** – [@JanhaveeSingh](https://github.com/JanhaveeSingh)
- 👩‍💻 **Shi Ying Wang** – [@yiihsuenn](https://github.com/yiihsuenn)
- 👩‍💻 **Yi Hsuen Cheng** – [@cjkejw](https://github.com/cjkejw)


