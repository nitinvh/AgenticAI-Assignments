# %% [markdown]
# # Agentic Workflow using Langgraph
# 
# # QA bot on solar system

# %% [markdown]
# ### RAG Creation

# %%
import os
from dotenv import load_dotenv

load_dotenv()

# %%
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

# %%
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools import TavilySearchResults

# %%
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# %%
# Creating Embedding model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

# %% [markdown]
# #### Loading Documents

# %%
# Document Loader
loader = DirectoryLoader('./data',glob='./*.pdf',loader_cls=PyPDFLoader)

# %%
docs = loader.load()


# %%
# Splitting into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
splitted_docs = splitter.split_documents(documents=loader.load())


# %% [markdown]
# #### VectorDB creation

# %%
# Let's create a pinecone vector db

pinecone_api_key = os.getenv('PINECONE_API_KEY') 

# %%
from pinecone import Pinecone

pc = Pinecone(api_key=pinecone_api_key)

# %%
index_name = 'solarsytemvdb'

# %%
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        spec=ServerlessSpec(cloud='aws',region='us-east-1'),
        dimension=768,
        metric='cosine'
    )

# %%
# Load index
index = pc.Index(index_name)

# %%
vectorStore = PineconeVectorStore(index=index,embedding=embeddings)

# %%
vectorStore.add_documents(documents=splitted_docs)


# %% [markdown]
# #### Creating Retriever pipeline

# %%
retriever = vectorStore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={"score_threshold":0.7}
)

# %%
retriever.invoke('What is Sun?')

# %% [markdown]
# ## Web crawler Creation

# %%
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

# %%
web_crawler = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)



# %% [markdown]
# ## Workflow creation

# %%
from pydantic import BaseModel, Field
from typing import Annotated,Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from IPython.display import Image, display


# %%
# Agent state creation
import operator
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],operator.add]

# %%
# Pydantic class craetion
class TopicSelectionParser(BaseModel):
    Topic: str = Field(description="Selected Topic")
    Reasoning: str = Field(description="Reasoning behind topic selected")


# %%
parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)


# %%
# Supervisor node
def supervisor(state:AgentState):
    print("--> SUPERVISOR -->")
    question = state["messages"][-1]
    print("Question:", question)

    template="""
    Your task is to classify the given user query into one of the following categories: [Solar System, Not Related, Real Time]. 
    Only respond with the category name and nothing else.
    Rule for classyfying:
    If the question talks about solar system or planets or similar info then classify it as 'Solar System',
    If question talks about sonme recent or latest information, then classify it to 'Real Time Info'.
    Otherwise, classify it to 'Not Related'.

    User query: {question}
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions":parser.get_format_instructions()}
    )

    chain = prompt | model | parser

    res = chain.invoke({"question":question})

    print("Response: ", res)

    return {"messages": [res.Topic]}


# %%
# Router node
def router(state:AgentState):
    print("--> ROUTER -->")
    last_message = state["messages"][-1].lower()
    print("Last Message:",last_message)

    if "solar system" in last_message:
        return "RAG call"
    elif "real time" in last_message:
        return "Web Crawler call"
    else:
        return "LLM call"

# %%
def format_docs(docs):
    return "/n/n".join([doc.page_content for doc in docs])

# %%
# Rag Node
def ragOnSS(state:AgentState):
    print("--> RAG -->")

    question= state["messages"][0]

    prompt=PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:""",
        
        input_variables=['context', 'question']
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    result = rag_chain.invoke(question)
    return {"messages": [result]}

# %%
# LLM node
def llm(state:AgentState):
    print("--> LLM call -->")
    question = state["messages"][0]
    
    # Normal LLM call
    complete_query = "Anwer the follow question with you knowledge of the real world. Following is the user question: " + question
    response = model.invoke(complete_query)
    return {"messages": [response.content]}
    

# %%
# Web crawler node
def webCrawler(state: AgentState):
    print("--> Web Crawler call -->")
    question = state["messages"][0]

    res = web_crawler.invoke({"query":question})

    return {"messages":[res[0].get("content")]}
    

# %%
# validation node

def validator(state:AgentState):
    print("--> Validator -->")

    ans = state["messages"][-1].lower()

    if "i don't know" in ans or len(ans.strip()) < 10:
        return {"messages":["retry"]}
    
    return {"messages":[ans]}

# %%
# retry roter

def retryRouter(state: AgentState):
    print("--> IN RETRY ROUTER -->")

    msg = state["messages"][-1]
    if msg == "retry":
        return "Supervisor"
    return "Final result"

# %%
# final node
def finalNode(state:AgentState):
    print("--> IN Final Node -->")
    return {"messages": [f"✅ Answer :  {state['messages'][-1]}"]}

# %%
from langgraph.graph import StateGraph,END

# %%
workflow = StateGraph(AgentState)

# %%
workflow.add_node("Supervisor",supervisor)
workflow.add_node("RAG",ragOnSS)
workflow.add_node("LLM",llm)
workflow.add_node("Web Crawler",webCrawler)
workflow.add_node("Validator",validator)
workflow.add_node("Final",finalNode)

# %%
workflow.set_entry_point("Supervisor")

# %%
workflow.add_conditional_edges(
    "Supervisor",
    router,
    {
        "RAG call": "RAG",
        "LLM call": "LLM",
        "Web Crawler call": "Web Crawler"
    }
)

# %%
workflow.add_edge("RAG","Validator")
workflow.add_edge("LLM","Validator")
workflow.add_edge("Web Crawler","Validator")

# %%
workflow.add_conditional_edges(
    "Validator",
    retryRouter,
    {
        "Supervisor": "Supervisor",
        "Final result": "Final"
    }
)

# %%
workflow.add_edge("Final",END)

# %%
app = workflow.compile()

import streamlit as st
# Streamlit UI
st.title("LangGraph-powered QA Assistant on Solar System")

question = st.text_input("Enter your question:")


if st.button('Ask'):
    state = {"messages":[question]}
    result = app.invoke(state)

    st.write(result["messages"][-1])

