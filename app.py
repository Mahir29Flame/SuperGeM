import os
from pathlib import Path

import streamlit as st

import langchain_google_genai as genai
from langchain_classic.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools

# from READPDF import ASKPDF

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "gemini-3.5-flash-lite"

# Bundled locally: Streamlit serves these itself, so no external CDN is needed.
AVATARS = {
    "user": str(BASE_DIR / "assets" / "icon" / "user.png"),
    "assistant": str(BASE_DIR / "assets" / "icon" / "gemini-avatar.png"),
}


def load_system_prompt() -> str:
    prompt = (BASE_DIR / "Gemini" / "system_prompt.txt").read_text(encoding="utf-8")
    prompt = prompt.rstrip()
    # The file is written as a chat transcript, drop the trailing "User:" so it
    # can be used as the agent prefix.
    if prompt.endswith("User:"):
        prompt = prompt[: -len("User:")].rstrip()
    return prompt


def get_tool_names() -> list[str]:
    # Tools that the Agent can use
    tool_names = ["ddg-search",  # Duckduckgo search for getting current informations
                  "wikipedia",  # Wikipedia search for general knowledge
                  "arxiv",  # Arxiv search for research papers
                  "pubmed"]  # Pubmed search for searching medical informations
    if os.environ.get("SERPAPI_API_KEY"):
        # Google-Scholar search for research papers, needs a SerpAPI key
        tool_names.append("google-scholar")
    return tool_names


@st.cache_resource(show_spinner="Loading the agent...")
def get_agent(api_key: str):
    if not api_key:
        raise RuntimeError(
            "Enter a Gemini API key in the sidebar. Get one from "
            "https://aistudio.google.com/apikey."
        )

    llm = genai.GoogleGenerativeAI(model=MODEL_NAME,
                                   verbose=True,  # to see what the Agent is thinking
                                   temperature=0,  # less creativity
                                   google_api_key=api_key
                                   )

    # tools = load_tools(get_tool_names()) + [ASKPDF(api_key=api_key)]

    return initialize_agent(llm,
                            agent="zero-shot-react-description",
                            agent_kwargs={"prefix": load_system_prompt()},
                            verbose=True,
                            handle_parsing_errors=True)


def build_agent_input(user_input: str, chat_history: list[dict], max_turns: int = 5) -> str:
    """Give the agent the last few messages, it has no memory of its own."""
    lines = []
    for message in chat_history[-2 * max_turns:]:
        for role, text in message.items():
            lines.append(f"{role}: {text}")
    lines.append(f"Human: {user_input}")
    return "\n".join(lines)


def render_message(role: str, text: str) -> None:
    with st.chat_message(role, avatar=AVATARS[role]):
        st.markdown(text)


st.set_page_config(page_title="SuperGeM",
                       page_icon=AVATARS["assistant"])
st.header(
    "SuperGeM",
    text_alignment= "center",
    icon= ":material/flare:",
    divider= "blue"
    )

def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.markdown("## API access")
        api_key = st.text_input(
            "Gemini API key",
            type="password",
            value="",
            help="Your key is kept in this session and is not written to the app files.",
        )
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        api_key = api_key.strip()
        st.caption("Get a key from [Google AI Studio](https://aistudio.google.com/apikey).")
        if not api_key:
            st.info("Enter a Gemini API key to enable chat.")

    prompt = st.chat_input("Chat with SuperGeM", disabled=not api_key)

    for message in st.session_state.chat_history:
        if "Human" in message:
            render_message("user", message["Human"])
        elif "AI" in message:
            render_message("assistant", message["AI"])

    if prompt:
        agent_input = build_agent_input(prompt, st.session_state.chat_history)
        st.session_state.chat_history.append({"Human": prompt})
        render_message("user", prompt)

        error = None
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            try:
                agent = get_agent(api_key)
                with st.spinner("Thinking..."):
                    agent_response = agent.invoke({"input": agent_input})
                answer = agent_response["output"]
            except Exception as e:
                error = str(e)
                answer = f"Error: {error}"

            if error:
                st.error(error)
            else:
                st.markdown(answer)

        st.session_state.chat_history.append({"AI": answer})

    with st.sidebar:
        st.markdown("# ℹ️Info")
        # Feature List
        st.markdown("## Features")
        st.markdown("""
            - **Advanced Language Understanding**: At its core, the Gemini leverages Google's Gemini-pro LLM, ensuring a deep comprehension of human language.
            - **Dynamic Tool Selection**: The agent autonomously decides the most suitable tool or response for each query, optimizing user experience.
            - **Versatile Search Options**: With integrated search capabilities across DuckDuckGo, Wikipedia, Google Scholar, ArXiv, and PubMed, users can find accurate information across the web and academic journals.
            - **Web Interface**: A user-friendly web interface built using the Streamlit library, making it accessible for non-technical users.
            - **Powered by Langchain**: Utilizes the Langchain library for efficient handling of API and LLM calls.
            """)
        # Display Linkedin logo and link
        col1, col2 = st.columns([0.5, 4])
        with col1:
            st.image(str(BASE_DIR / "assets" / "linkedin.png"), width=30)
        with col2:
            st.markdown(
                "[Linkedin](https://www.linkedin.com/in/chanyalcin/)",
                unsafe_allow_html=True,
            )

        # Display Github logo and link
        col1, col2 = st.columns([0.5, 4])
        with col1:
            st.image(str(BASE_DIR / "assets" / "github-sign.png"), width=30)
        with col2:
            st.markdown(
                "[Github](https://github.com/g-hano)",
                unsafe_allow_html=True,
            )


    # with st.expander("Memory"):
    #     for message in st.session_state.chat_history:
    #         if "Human" in message:
    #             st.markdown(f"<span style='color: green;'>Human: {message['Human']}</span>", unsafe_allow_html=True)
    #         elif "AI" in message:
    #             st.markdown(f"<span style='color: blue;'>AI: {message['AI']}</span>", unsafe_allow_html=True)
    #         elif "System" in message:
    #             st.write("System: " + message["System"])

if __name__ == '__main__':
    main()
