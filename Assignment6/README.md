Corrective RAG with LangGraph
This project implements a Corrective Retrieval-Augmented Generation (RAG) system using LangGraph. The goal of Corrective RAG is to enhance the quality of generated answers by introducing feedback loops that identify and rectify issues such as irrelevant document retrieval or the need for external information.

Project Description
Traditional RAG systems retrieve documents and then generate answers. Corrective RAG extends this by introducing intelligent agents (nodes in LangGraph) that can:

Decide if external tools (like retrieval or web search) are necessary.

Grade the relevance of retrieved documents.

Rewrite queries if initial retrieval is insufficient.

Perform Web Search if initial knowledge bases don't suffice.

This workflow ensures that the system dynamically adapts to the complexity and information requirements of each query, leading to more accurate and robust responses.

Technologies Used
Large Language Model (LLM): Groq (specifically, the Deepseek model) for text generation, decision-making, and query rewriting.

Embeddings: GoogleEmbeddings for converting text into vector representations, crucial for efficient similarity search in retrieval.

Web Search: Tavily Search API for dynamic information retrieval from the web when an internal knowledge base is insufficient.

Orchestration Framework: LangGraph for building stateful, multi-actor applications with LLMs, allowing for complex decision flows and feedback loops.

Workflow Explanation
The corrective RAG system is orchestrated using LangGraph, defining a directed acyclic graph (DAG) of agents and tools. Here's a breakdown of the flow:

graph TD
    A[START] --> B{LLM DECIDER}
    B -->|tools| C[RETRIEVER]
    B -->|END (No Tools Needed)| G[END]
    C -->|generator (Relevant Docs)| D[GENERATOR]
    C -->|rewriter (Irrelevant Docs)| E[RE-WRITER]
    D --> G
    E --> F[WEB SEARCH]
    F --> G

START Node: The entry point of the graph, receiving the initial user query.

LLM DECIDER Node:

This agent (an LLM) analyzes the user's question to determine if external tools (like retrieval or web search) are necessary to answer it.

Conditional Edge (tools_condition):

If the LLM decides that tools are needed, the flow transitions to the RETRIEVER node.

If the LLM determines it can answer the question directly (or doesn't require further information), the flow transitions to END.

RETRIEVER Node:

If tools are deemed necessary, this node is activated. It performs a document retrieval operation (e.g., using a vector database with GoogleEmbeddings).

Conditional Edge (grader_func):

After retrieval, a grader_func (another LLM or a rule-based system) assesses the relevance of the retrieved documents to the original question.

If the documents are deemed relevant (generator), the flow transitions to the GENERATOR node.

If the documents are deemed irrelevant or insufficient (rewriter), the flow transitions to the RE-WRITER node.

GENERATOR Node:

If relevant documents are found, this node uses the LLM to synthesize an answer based on the retrieved information and the original question.

Edge: After generating the answer, the flow proceeds to END.

RE-WRITER Node:

If the grader_func indicates that the initial retrieval was poor, this node is activated. It uses the LLM to re-formulate or expand the original query, aiming for better retrieval results.

Edge: The rewritten query then leads to the WEB SEARCH node to fetch new information.

WEB SEARCH Node:

This node uses the Tavily Search API to perform a web search with the (potentially rewritten) query.

Edge: The results of the web search are then likely used to formulate a final answer, and the flow proceeds to END.

END Node: The final state of the graph, where the generated answer is outputted.

Setup and Installation
(Placeholder: Instructions for setting up your environment, installing dependencies, and configuring API keys will go here once the code is available.)

Clone the repository:

git clone https://github.com/nitinvh/AgenticAI-Assignments/tree/main/Assignment6
cd corrective-rag-assignment6

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

(You will need to create a requirements.txt file containing langchain, langgraph, langchain-groq, langchain-google-genai, langchain-tavily, etc.)

Set up your API keys as environment variables:

export GROQ_API_KEY="your_groq_api_key"
export GOOGLE_API_KEY="your_google_api_key"
export TAVILY_API_KEY="your_tavily_api_key"

(Note: For production, consider more secure methods for managing API keys.)

Usage
(Placeholder: Instructions on how to run the application and interact with the Corrective RAG system will go here.)

# Example of how to invoke the graph (once defined in your main script)
# from your_main_script import app  # Assuming your LangGraph app is named 'app'

# inputs = {"question": "What is the capital of France?"}
# for s in app.stream(inputs):
#     print(s)
