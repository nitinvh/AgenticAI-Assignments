# 🧠 Corrective RAG with LangGraph

This project implements a **Corrective Retrieval-Augmented Generation (RAG)** system using **LangGraph**. The goal of Corrective RAG is to enhance the quality of generated answers by introducing feedback loops that identify and rectify issues such as irrelevant document retrieval or the need for external information.

---

## 📌 Project Overview

While traditional RAG systems follow a two-step process — **document retrieval** followed by **LLM-based answer generation** — this project adds an intelligent layer that:

- Decides if tools like retrieval or web search are necessary
- Grades the relevance of retrieved documents
- Rewrites queries when needed
- Performs web searches if knowledge bases are insufficient

This **dynamic and adaptive system** leads to more **accurate, robust, and context-aware responses**.

---

## 🔧 Technologies Used

| Component            | Technology |
|---------------------|------------|
| **LLM**             | [Groq (DeepSeek)](https://groq.com) |
| **Embeddings**      | [GoogleEmbeddings](https://ai.google.dev/gemini-api/docs/models/embedding) |
| **Web Search**      | [Tavily Search API](https://app.tavily.com/) |
| **Orchestration**   | [LangGraph](https://github.com/langchain-ai/langgraph) |
| **RAG Framework**   | [LangChain](https://github.com/langchain-ai/langchain) |

---

## 🧭 Workflow Overview

The corrective RAG system is defined as a **LangGraph DAG (Directed Acyclic Graph)** that connects intelligent agents and tools.

![alt text](image-1.png)


# 🔄 Node Explanations
START: Accepts the user query.

LLM DECIDER: Uses the LLM to decide if external tools are needed.

If not needed, directly goes to END.

If needed, proceeds to RETRIEVER.

RETRIEVER: Uses a vector database + embeddings to retrieve documents.

GRADER FUNCTION (internal logic):

If docs are relevant, flow continues to GENERATOR.

If irrelevant, query is passed to RE-WRITER.

RE-WRITER: Reformulates the query using the LLM.

WEB SEARCH: Uses Tavily API to perform live search.

GENERATOR: Synthesizes the final response from relevant docs.

END: Outputs the final generated answer.

⚙️ Setup & Installation
📁 Clone the repository
bash
Copy
Edit
git clone https://github.com/nitinvh/AgenticAI-Assignments.git
cd AgenticAI-Assignments/Assignment6
🧪 Create and activate a virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
📦 Install dependencies
