import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Research Agent - Groq Powered",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Research Agent with AI & RAG")
st.markdown("**Powered by Groq Qwen3-32B | AI Agents + Vector Database**")

# Sidebar Configuration
st.sidebar.header("⚙️ Settings")
groq_key = st.sidebar.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))

st.sidebar.subheader("🤖 Model Configuration")
model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ],
    index=0
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 512, 8192, 4096, 256)
max_iterations = st.sidebar.slider("Max Iterations", 3, 15, 10, 1)

st.sidebar.subheader("🛠️ AI Agent Tools")
enable_search = st.sidebar.toggle("🌐 DuckDuckGo Search", True)
enable_wiki = st.sidebar.toggle("📚 Wikipedia", True)
enable_arxiv = st.sidebar.toggle("📄 arXiv Papers", True)

st.sidebar.subheader("📑 Research Papers (Optional)")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Research Papers",
    type="pdf",
    accept_multiple_files=True,
    help="Upload research papers to create a vector database for querying"
)

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hello! I'm your Research Agent. I can search the web, query academic databases, and analyze uploaded PDFs. How can I help you today?"}
    ]

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Process uploaded PDFs
if uploaded_files:
    with st.sidebar.status("📥 Processing PDFs...", expanded=True) as status:
        try:
            documents = []
            for uploaded_file in uploaded_files:
                st.write(f"Loading {uploaded_file.name}...")
                
                # Save temporarily
                temp_path = f"./temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Load PDF
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                documents.extend(docs)
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            st.write(f"✅ Loaded {len(documents)} pages")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)
            st.write(f"✅ Created {len(splits)} chunks")
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            st.session_state["vectorstore"] = vectorstore
            
            status.update(label=f"✅ Processed {len(uploaded_files)} PDFs successfully!", state="complete")
            
        except Exception as e:
            status.update(label="❌ Error processing PDFs", state="error")
            st.error(f"Error: {str(e)}")

# Function to build tools list
def build_tools():
    tools = []
    
    # Add external search tools
    if enable_search:
        search_tool = DuckDuckGoSearchRun()
        tools.append(
            Tool(
                name="Search",
                func=search_tool.run,
                description="Useful for searching current information on the web. Use this for recent events, news, or general web searches. Input should be a search query string."
            )
        )
    
    if enable_arxiv:
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
        arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        tools.append(
            Tool(
                name="ArxivSearch",
                func=arxiv_tool.run,
                description="Useful for searching academic papers on arXiv. Use this for scientific research, machine learning papers, physics, mathematics, etc. Input should be a search query string."
            )
        )
    
    if enable_wiki:
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
        tools.append(
            Tool(
                name="Wikipedia",
                func=wiki_tool.run,
                description="Useful for getting encyclopedic information about topics, people, places, concepts, and historical facts. Input should be a search query string."
            )
        )
    
    # Add PDF retriever tool if PDFs are uploaded
    if st.session_state["vectorstore"] is not None:
        def search_pdfs(query: str) -> str:
            """Search through uploaded research papers"""
            try:
                retriever = st.session_state["vectorstore"].as_retriever(
                    search_kwargs={"k": 4}
                )
                docs = retriever.invoke(query)
                
                if not docs:
                    return "No relevant information found in the uploaded papers."
                
                # Format results
                results = []
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content[:300]  # Limit content length
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'Unknown')
                    results.append(f"[Result {i}] (Source: {source}, Page: {page})\n{content}...")
                
                return "\n\n".join(results)
            except Exception as e:
                return f"Error searching PDFs: {str(e)}"
        
        tools.append(
            Tool(
                name="ResearchPaperRetriever",
                func=search_pdfs,
                description="Use this tool to search through the uploaded research papers. This is useful for answering questions specific to the PDFs the user has uploaded. Input should be a specific question or topic from the papers."
            )
        )
    
    return tools

# Custom ReAct Prompt Template
CUSTOM_REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT RULES:
- Always follow the exact format above
- Be concise and efficient - use minimal iterations
- If you have enough information after 1-2 tool calls, provide the Final Answer
- Never use <think> tags or any other custom formatting
- Always start with "Thought:" when thinking
- Always use "Action:" and "Action Input:" when using tools
- Use "Final Answer:" only when you're ready to give the final response
- If a tool gives you good information, use it immediately in your Final Answer
- Don't overthink - answer directly when you have sufficient information

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

# Main chat interface
prompt = st.chat_input("Ask me anything about research topics or your uploaded papers...")

if prompt:
    if not groq_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
        st.stop()
    
    # Add user message to chat
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Build tools list
    tools = build_tools()
    
    # Check if at least one tool is enabled
    if not tools:
        st.warning("⚠️ Please enable at least one tool or upload a PDF to use the agent.")
        st.stop()
    
    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
    )
    
    # Initialize agent with custom prompt
    try:
        from langchain.prompts import PromptTemplate
        
        react_prompt = PromptTemplate(
            template=CUSTOM_REACT_PROMPT,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                "tool_names": ", ".join([tool.name for tool in tools])
            }
        )
        
        agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=max_iterations,
            return_intermediate_steps=False
        )
    except:
        # Fallback to old method if new one fails
        agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=max_iterations
        )
    
    # Execute agent with callback
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=True,
            collapse_completed_thoughts=True
        )
        
        try:
            response = agent_executor.invoke(
                {"input": prompt},
                {"callbacks": [st_callback]}
            )
            
            # Extract the output
            if isinstance(response, dict):
                final_response = response.get("output", str(response))
            else:
                final_response = str(response)
            
            st.session_state["messages"].append({"role": "assistant", "content": final_response})
            st.write(final_response)
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.error(error_msg)
            st.session_state["messages"].append({"role": "assistant", "content": error_msg})

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.info(
    """
    **Research Agent Features:**
    
    🌐 **Web Search** - Live web results via DuckDuckGo
    
    📚 **Wikipedia** - Encyclopedic knowledge
    
    📄 **arXiv** - Academic paper search
    
    🔍 **PDF Analysis** - Query uploaded research papers
    
    The agent intelligently combines multiple sources to provide comprehensive research answers.
    """
)

# Display active tools
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Active Tools")
active_tools = build_tools()
if active_tools:
    for tool in active_tools:
        st.sidebar.markdown(f"✅ {tool.name}")
else:
    st.sidebar.markdown("⚠️ No tools enabled")