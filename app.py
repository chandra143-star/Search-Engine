import streamlit as st

from langchain_groq import ChatGroq
from langchain.agents import create_agent

from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)


# ---------------------------------------------------------
# Page
# ---------------------------------------------------------

st.set_page_config(
    page_title="LangChain Search",
    page_icon="🔎",
)

st.title("🔎 LangChain - Chat with Search")

st.write(
    """
    Ask questions and the AI agent can search the web,
    Arxiv, and Wikipedia to find information.
    """
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password",
)

if not api_key:
    st.sidebar.warning("Please enter your Groq API key.")
    st.stop()


# ---------------------------------------------------------
# Tools
# ---------------------------------------------------------

# Arxiv
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200,
)

arxiv = ArxivQueryRun(
    api_wrapper=arxiv_wrapper
)


# Wikipedia
wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200,
)

wiki = WikipediaQueryRun(
    api_wrapper=wiki_wrapper
)


# DuckDuckGo
search = DuckDuckGoSearchRun(
    name="Search"
)


tools = [
    search,
    arxiv,
    wiki,
]


# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm a chatbot that can search the web, "
                "Arxiv, and Wikipedia. How can I help you?"
            ),
        }
    ]


# Display previous messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


# ---------------------------------------------------------
# User input
# ---------------------------------------------------------

if prompt := st.chat_input(
    placeholder="What is machine learning?"
):

    # Add user message
    st.session_state["messages"].append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # Display user message
    st.chat_message("user").write(prompt)


    # -----------------------------------------------------
    # Groq LLM
    # -----------------------------------------------------

    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0,
    )


    # -----------------------------------------------------
    # Create LangChain agent
    # -----------------------------------------------------

    search_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful research assistant. "
            "Use the available search, Arxiv, and Wikipedia "
            "tools when they are useful. "
            "Give a clear and concise final answer."
        ),
    )


    # -----------------------------------------------------
    # Run agent
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        try:

            result = search_agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]
                }
            )

            # Get final AI message
            response = result["messages"][-1].content

            st.write(response)

            # Save response
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

        except Exception as e:

            st.error(
                f"An error occurred while running the agent:\n\n{e}"
            )
