Team_Untitled – ChatHera: The HR Chatbot Revolution
ChatHera – “Her-Era” 💬
Built for HOYA Electronics as part of the NTU CampCode x HOYA Chatbot Hackathon 2025, ChatHera is a next-generation HR chatbot powered by RAG (Retrieval-Augmented Generation) and Neo4j Graph Databases. Designed to reduce redundant HR workloads and improve employee interaction, ChatHera intelligently answers policy-related FAQs, learns from new queries, and maintains context-aware conversations.

🏆 1st Runner-Up at the NTU CampCode x HOYA Chatbot Hackathon 2025!

Table of Contents
Overview

Features

How It Works

Demo

Installation

Usage

Future Plans

Contributing

Contributors List

1. Overview
Modern HR departments face repeated queries on policies that can be automated, yet they require a chatbot smart enough to understand, learn, and engage. That’s where ChatHera steps in.

ChatHera processes company policy PDFs and delivers human-like responses with contextual memory and follow-up awareness. The system also adapts by learning from unanswered queries via a dedicated HR backend platform. To eliminate service downtime, a Dual Graph System ensures seamless transitions during knowledge base updates.

2. Features
💬 Conversational AI Chatbot
Maintains conversation context, handles follow-up questions, and supports dynamic topic-switching.

🧠 Self-Updating Knowledge Base
Neo4j-powered dual-graph system allows continuous updates with no downtime.

📷 Computer Vision Integration
Users can snap photos of HR-related documents, which are processed using Tesseract OCR.

📤 HR FAQ Management Platform
Unanswered queries are routed to an HR-only dashboard where HR can respond. Once answered:

The answer is added to the knowledge base.

The original user is notified via email.

The chatbot is updated immediately.

🔄 Zero Downtime via Dual Graphs
One graph remains active while the other is updated—then the roles switch.

📧 Automated Notifications
Users receive updates when their queries (previously unanswered) have been addressed.

3. How It Works
1. Chatbot Layer (Frontend)
Built with a conversational UI inspired by Microsoft Teams Copilot

Supports follow-up, topic switching, and historical context.

2. RAG Backend Pipeline
Extracts embeddings from policy documents.

Queries the vector database to return the most relevant answer.

3. Neo4j Graph System
Dual-graph architecture: active_graph and update_graph.

Updates are written to one graph while the other remains live.

4. Computer Vision (Tesseract OCR)
Users upload or capture document images.

Text is extracted and passed to the chatbot as input.

5. HR FAQ Dashboard
Stores unknown or unanswered questions.

Allows HR personnel to respond and update the system.

Sends auto-email alerts to users who asked.

4. Demo
🎥 Demo video link: (Insert link here)
📸 Screenshot Highlights: (Insert images if available)

5. Installation

Tesseract OCR

Neo4j Aura DB Instance

Azure OpenAI API Key


Steps
bash
Copy
Edit
# Clone the repository
git clone https://github.com/your-repo/ChatHera.git
cd ChatHera

# Install dependencies
pip install -r requirements.txt

# Setup Tesseract
# (Ensure it's installed and added to PATH)

# Start backend
python app.py
6. Usage
Launch the chatbot via the web interface.

Ask policy-related questions.

Upload or capture policy-related images.

For unknown questions, wait for email response post HR review.

HR uses a secure portal to respond and update knowledge base.

7. Future Plans
🌐 Integrate multi-language support

📲 Deploy on Microsoft Teams/Slack

📚 Add document summarization for long policies

🔐 HR Role-Based Access Control for the admin dashboard

🔁 Sync with SharePoint or Google Drive for auto-policy updates

8. Contributing
We welcome contributions to improve ChatHera!
Fork the repo and submit a pull request for new features or improvements.

bash
Copy
Edit
git checkout -b feature-branch
# Make your changes
git commit -am "Add feature"
git push origin feature-branch
9. Contributors List
Risha Sunil Shetty
Singh Janhavee
Wang Shi Ying
Cheng Yi Hsuen


