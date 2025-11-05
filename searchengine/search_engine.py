import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Marshall (Groq)", page_icon="🔎")
st.title("🔎 Marshall — Groq Powered Research Agent")

# Sidebar
st.sidebar.header("Settings")
groq_key = st.sidebar.text_input("Groq API Key", type="password")

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "qwen/qwen3-32b",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
    ],
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
max_tokens = st.sidebar.slider("Max new tokens", 64, 4096, 1024, 32)

enable_search = st.sidebar.toggle("DuckDuckGo Search", True)
enable_wiki = st.sidebar.toggle("Wikipedia", True)
enable_arxiv = st.sidebar.toggle("arXiv", True)

# Tools
tools = []
if enable_search:
    tools.append(DuckDuckGoSearchRun(name="Search"))

if enable_arxiv:
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    tools.append(ArxivQueryRun(api_wrapper=arxiv_wrapper))

if enable_wiki:
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    tools.append(WikipediaQueryRun(api_wrapper=wiki_wrapper))

# Session memory
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm Marshall (Groq). Ask me anything."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
prompt = st.chat_input("Ask me something...")

if prompt:
    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Groq LLM
    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
    )

    # Agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        reply = agent.run(prompt, callbacks=[cb])
        st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.write(reply)
