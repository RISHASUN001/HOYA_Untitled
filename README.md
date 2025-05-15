# Team_Untitled – ChatHera: The HR Chatbot Revolution 💬

🏆 **1st Runner-Up** at the **NTU CampCode x HOYA Chatbot Hackathon 2025!**

ChatHera – “Her-Era” is a next-generation HR chatbot powered by Retrieval-Augmented Generation (RAG) and Neo4j Graph Databases. Built for HOYA Electronics, ChatHera reduces HR workloads by answering policy-related queries, learning from unknown questions, and offering seamless, context-aware conversations.

> 💡 **UI inspired by HOYA Electronics' official website**

---

## 📚 Table of Contents

- [Overview](#1-overview)
- [Features](#2-features)
- [How It Works](#3-how-it-works)
- [Installation](#4-installation)
- [Usage](#5-usage)
- [Future Plans](#6-future-plans)
- [Contributors List](#7-contributors-list)

---

## 1. Overview

Modern HR departments are often overwhelmed with repetitive queries. ChatHera steps in to automate this process by:

- Parsing HR policy PDFs.
- Handling context-aware, natural conversations.
- Updating itself via an HR-facing dashboard.
- Ensuring **zero service downtime** using a **dual-graph system**.

---

## 2. Features

💬 **Conversational AI Chatbot**  
Maintains conversation history, supports follow-up queries, and enables dynamic topic switching.

🧠 **Self-Updating Knowledge Base**  
Neo4j dual-graph setup enables continuous learning and updates with zero service interruption.

📷 **Computer Vision Integration**  
Processes uploaded document images using **Tesseract OCR**.

📤 **HR FAQ Management Platform**  
Routes unanswered questions to HR. Once addressed:
- Adds the answer to the knowledge base.
- Notifies the user via email.
- Updates chatbot knowledge instantly.

🔄 **Zero Downtime via Dual Graphs**  
Live queries are served via `active_graph`, while `update_graph` handles background updates.

📧 **Automated Notifications**  
Users receive emails when their pending queries are answered.

---

## 3. How It Works

### 🧩 Architecture Layers:

1. **Frontend Chatbot (UI Inspired by MS Teams Copilot)**  
   Conversational interface supporting follow-up and topic switches.

2. **RAG Backend Engine**  
   Embeds documents, performs similarity search, and returns top-ranked answers.

3. **Neo4j Graph Database (Dual-Graph Design)**  
   - `active_graph`: Serves user queries.  
   - `update_graph`: Receives new data and becomes active on sync.

4. **Computer Vision with Tesseract OCR**  
   Processes screenshots or scanned documents submitted by users.

5. **HR Admin Dashboard**  
   - Displays unanswered questions.  
   - Allows HR staff to respond.  
   - Triggers knowledge base and email updates.

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

### 3. Setup Tesseract OCR

** macOS (via Homebrew):

```bash
brew install tesseract
```

* Windows:

** Download from https://github.com/tesseract-ocr/tesseract

** Add the install path (e.g., C:\Program Files\Tesseract-OCR) to your system's PATH.

### 4. Configure Environment Variables

- Create a .env file in the root directory:
```bash
### Azure OpenAI Services 
OPENAI_API_TYPE="azure"
AZURE_OPENAI_API_VERSION="___"                 
AZURE_OPENAI_ENDPOINT="___"
AZURE_OPENAI_APIKEY="___" 
AZURE_OPENAI_DEPLOYMENT_NAME="___"
AZURE_OPENAI_MODEL_NAME="___"
AZURE_EMBEDDING_DEPLOYMENT_NAME="___"
AZURE_TEXT_EMBEDDING_MODEL="___"
FOLDER_ID = "folder_id of where the documents are stored"
DATABASE_URL="url for the databse with all unanswered questions and related users(email)"

```

Note: If utilising Azure OpenAI API Key, utilise 2024 Aug Version and higher. Versions before 2024 Aug are incompatible with Neo4j's LLMGraphTransformer

### 5. Run the following commands to run the project

* Run the backend server
```bash
python app.py
```

* Run the faq_backend server
```bash
python faq_backend.py
```

* Run the applicaton
```bash
npm run dev
```

## 5. Usage
* Launch the chatbot via the web interface.

** Ask HR-related questions or upload policy documents as images.

** If a query is unanswered:

*** It is routed to HR via the admin dashboard.

*** Once answered, you receive an email.

** The chatbot is updated automatically.

---

7. Future Plans
🌐 Multi-language support.

📲 Deploy on Microsoft Teams.

⚖️ Expand to Legal Department:

- Contract clause scanning (via image uploads) using the chatbot.

- Review for alarming phrases.

💰 Expand to Finance Department:

- FAQs on payroll, reimbursement, and finance SOPs.

🔗 Integration with Microsoft Graph API for syncing with D365.

---

## 7. Contributors List
👩‍💻 Risha Sunil Shetty – GitHub: @RISHASUN001

👩‍💻 Janhavee Singh - GitHub: @JanhaveeSingh

👩‍💻 Shi Ying Wang - GitHub: @yiihsuenn

👩‍💻 Yi Hsuen Cheng - GitHub: @cjkejw

