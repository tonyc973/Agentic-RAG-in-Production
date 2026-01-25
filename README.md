This is a multi-user **Agentic RAG (Retrieval-Augmented Generation)** application. This platform allows users to interact with PDFs and with the internet, all the being powered by autonomous AI agents. 
The project is fully dockerized with persistent database storage, cookie-based session management, and secure vector namespaces.

---

## Key Features

* Multi-User State: Utilizes 'streamlit-cookies-controller' and PostgreSQL to maintain isolated chat histories for simultaneous users.
* Secure Vector Namespaces: Embeddings are stored in Pinecone vector databases and isolated by `session_id` so users cannot query each other's private data.
* Agentic Workflow (CrewAI): Autonomously searches PDFs, browses the live web (Browserless), and queries Google (Serper).
* Query Contextualization: Automatically rewrites ambiguous voice queries (e.g., "tell me more about that") into standalone vector-searchable prompts.

---

##  Architecture

The application is orchestrated via **Docker Compose**, running three isolated services within a private network:

1. **Streamlit App (Port 8501):** The user interface and LangChain/CrewAI controller.
2. **PostgreSQL DB:** Persistent storage for LangChain `SQLChatMessageHistory`.


---

## 🚀 Quick Start

### 1. Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* API Keys for: [OpenAI](https://platform.openai.com/), [Pinecone](https://www.pinecone.io/), [Serper](https://serper.dev/), and [Browserless](https://www.browserless.io/).

### 2. Clone the Repository

```bash
git clone agentic-voice-rag-prod.git
cd agentic-voice-rag-prod

# create a .env file like this:

OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
SERPER_API_KEY=your_serper_api_key
BROWSERLESS_API_KEY=your_browserless_api_key

POSTGRES_USER=raguser
POSTGRES_PASSWORD=supersecretpassword
POSTGRES_DB=ragdb
DB_CONNECTION=postgresql://raguser:supersecretpassword@postgres:5432/ragdb

WHISPER_URL=http://whisper-api:9000/asr

# Create a new index in Pinecone named agentic-rag-prod with dimensions set to 1536 and the metric set to cosine


docker-compose up --build -d
