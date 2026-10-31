import hashlib
import logging
import mimetypes
import os
import socket
import tempfile
import time
from pathlib import Path
from urllib.parse import quote, unquote

import streamlit as st
import streamlit.components.v1 as components

import langchain_google_genai as genai
from langchain_classic.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import BaseTool
from google import genai as google_genai

import READPDF

BASE_DIR = Path(__file__).resolve().parent
MODELS = {
    "Gemini 3.5 Flash Lite": "gemini-3.5-flash-lite",
    "Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite",
    "Gemini 3 Flash": "gemini-3-flash-preview",
    "Gemini 3.6 Flash": "gemini-3.5-flash",
    "Gemini 3.8 Flash": "gemini-3.8-flash",
    "Gemma 4 31B": "gemma-4-31b-it",
    "Gemma 4 26B": "gemma-4-26b-a4b-it",
}

DEFAULT_MODEL = "Gemini 3.5 Flash Lite"

log = logging.getLogger("supergem")

TOOL_FAILED_PREFIX = "TOOL_UNAVAILABLE"
FRIENDLY_TOOL_FAILURE = (
    "I couldn't reach the search sources for that question. "
    "Please try again in a moment."
)
DNS_ERROR_MARKERS = (
    "gaierror",
    "nameresolutionerror",
    "failed to resolve",
    "getaddrinfo",
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname",
)
TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "remotedisconnected",
    "connectionrefused",
    "econnreset",
    "502",
    "503",
    "504",
)
RAW_ERROR_MARKERS = (
    TOOL_FAILED_PREFIX.lower(),
    "nameresolutionerror",
    "max retries exceeded",
    "getaddrinfo",
    "failed to resolve",
    "httpsconnectionpool",
    "connecttimeout",
    "connectionerror",
    "gaierror",
    "agent stopped due to iteration limit",
    "agent stopped due to time limit",
)

# Bundled locally: Streamlit serves these itself, so no external CDN is needed.
AVATARS = {
    "user": str(BASE_DIR / "assets" / "icon" / "user.png"),
    "assistant": str(BASE_DIR / "assets" / "icon" / "Gemini-Pro.png"),
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


def _exception_text(exc: BaseException) -> str:
    """Flatten an exception and its causes into lowercase text."""
    parts = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__module__}.{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | ".join(parts).lower()


def _failure_reason(exc: BaseException) -> str:
    text = _exception_text(exc)
    if any(marker in text for marker in DNS_ERROR_MARKERS):
        return "network/DNS lookup failed"
    if any(marker in text for marker in TRANSIENT_ERROR_MARKERS):
        return "network timeout"
    return "unexpected tool error"


def _is_transient(exc: BaseException) -> bool:
    text = _exception_text(exc)
    if any(marker in text for marker in DNS_ERROR_MARKERS):
        # A failed name lookup will not succeed on an immediate retry.
        return False
    return isinstance(exc, (socket.timeout, TimeoutError)) or any(
        marker in text for marker in TRANSIENT_ERROR_MARKERS
    )


def _tool_failure_observation(tool_name: str, exc: BaseException) -> str:
    return (
        f"{TOOL_FAILED_PREFIX}: {tool_name} failed ({_failure_reason(exc)}). "
        "Do not retry this tool with the same input and do not show this message "
        "to the user. Use a different tool or answer from your own knowledge."
    )

def _is_rag_file(file) -> bool:
    return Path(file.name).suffix.lower() in READPDF.SUPPORTED_SUFFIXES


def stage_uploads(files: list) -> list[tuple[str, str]]:
    """Write uploads to a content-addressed temp path so the index cache hits."""
    upload_dir = Path(tempfile.gettempdir()) / "supergem_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for file in files:
        data = file.getvalue()
        digest = hashlib.sha256(data).hexdigest()[:16]
        path = upload_dir / f"{digest}{Path(file.name).suffix.lower()}"
        if not path.exists():
            path.write_bytes(data)
        staged.append((str(path), file.name))
    return staged


def document_context(staged: list[tuple[str, str]], question: str, api_key: str) -> str:
    """Retrieve the most relevant passages from each staged document."""
    parts = []
    for path, display_name in staged:
        passages = READPDF.ask_pdf(path, question, api_key)
        parts.append(f"[{display_name}]\n{passages}")
    return "\n\n".join(parts)


def _mime_type(file) -> str:
    """Streamlit uploads are in-memory, so the SDK needs an explicit MIME type."""
    mime = getattr(file, "type", None)
    if mime and mime != "application/octet-stream":
        return mime
    guessed, _ = mimetypes.guess_type(file.name)
    if guessed:
        return guessed
    if Path(file.name).suffix.lower() in READPDF.TEXT_SUFFIXES:
        return "text/plain"
    return "application/octet-stream"


def upload_files(files: list, api_key: str) -> list:
    client = google_genai.Client(api_key=api_key)
    return [
        client.files.upload(
            file=file,
            config={
                "display_name": file.name,
                "mime_type": _mime_type(file),
            },
        )
        for file in files
    ]


def ask_gemini_with_files(
    api_key: str,
    model_name: str,
    user_text: str,
    uploaded_files: list,
) -> str:
    client = google_genai.Client(api_key=api_key)

    contents = [user_text]

    for uploaded_file in uploaded_files:
        contents.append(uploaded_file)

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
    )

    return response.text

class ResilientTool(BaseTool):
    """Report tool failures as observations instead of crashing the agent."""

    name: str
    description: str
    wrapped_tool: BaseTool
    max_retries: int = 1

    def _run(self, tool_input: str = "") -> str:
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                return self.wrapped_tool.run(tool_input)
            except Exception as exc:
                if attempt + 1 < attempts and _is_transient(exc):
                    time.sleep(0.4)
                    continue
                log.warning("Tool %s failed: %s", self.name, exc)
                return _tool_failure_observation(self.name, exc)
        return _tool_failure_observation(self.name, RuntimeError("unknown error"))


def wrap_tools(tools: list[BaseTool]) -> list[BaseTool]:
    return [
        ResilientTool(
            name=tool.name,
            description=tool.description,
            wrapped_tool=tool,
        )
        for tool in tools
    ]


def sanitize_answer(text: str) -> str:
    """Keep raw tool/network error text out of the chat."""
    if not text:
        return FRIENDLY_TOOL_FAILURE
    lines = [
        line for line in text.splitlines()
        if TOOL_FAILED_PREFIX not in line
    ]
    cleaned = "\n".join(lines).strip()
    if not cleaned or any(marker in cleaned.lower() for marker in RAW_ERROR_MARKERS):
        return FRIENDLY_TOOL_FAILURE
    return cleaned


def friendly_error(exc: BaseException) -> str:
    text = _exception_text(exc)
    if any(
        marker in text
        for marker in ("api key", "api_key_invalid", "unauthenticated", "permission denied")
    ):
        return "Your Gemini API key was rejected. Check the key in the sidebar and try again."
    if any(
        marker in text
        for marker in ("resource_exhausted", "quota", "rate limit", "429")
    ):
        return "The Gemini API rate limit or quota was reached. Wait a moment and try again."
    return "Something went wrong while contacting the AI service. Please try again."


@st.cache_resource(show_spinner="Loading the agent...")
def get_agent(api_key: str, model_name: str):
    if not api_key:
        raise RuntimeError(
            "Enter a Gemini API key in the sidebar. Get one from "
            "https://aistudio.google.com/apikey."
        )

    llm = genai.GoogleGenerativeAI(model=model_name,
                                   verbose=True,  # to see what the Agent is thinking
                                   temperature=0.3,  # less creativity
                                   google_api_key=api_key
                                   )

    tools = wrap_tools(load_tools(get_tool_names()))

    return initialize_agent(tools, llm,
                            agent="zero-shot-react-description",
                            agent_kwargs={"prefix": load_system_prompt()},
                            verbose=True,
                            handle_parsing_errors=True,
                            max_iterations=6)


def build_agent_input(user_input: str, chat_history: list[dict], max_turns: int = 5) -> str:
    """Give the agent the last few messages, it has no memory of its own."""
    lines = []
    for message in chat_history[-2 * max_turns:]:
        for role, text in message.items():
            lines.append(f"{role}: {text}")
    lines.append(f"Human: {user_input}")
    return "\n".join(lines)


def run_agent(api_key: str, model_name: str, agent_input: str) -> str:
    agent = get_agent(api_key, model_name)
    return agent.invoke({"input": agent_input})["output"]


def render_message(role: str, text: str) -> None:
    with st.chat_message(role, avatar=AVATARS[role]):
        st.markdown(text)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")

def render_sidebar_hint() -> None:
    """Point first-time users (especially on phones) at the collapsed sidebar."""
    st.markdown(
        """
        <style>
        .sg-sidebar-hint {
            display: none;
            position: fixed;
            top: 4.9rem;
            left: 0.9rem;
            z-index: 1000000;
            align-items: flex-start;
            gap: 0.45rem;
            max-width: min(20rem, calc(100vw - 2rem));
            padding: 0.7rem 0.9rem;
            border-radius: 0.75rem;
            background: rgba(43, 49, 62, 0.96);
            border: 1px solid rgba(232, 163, 61, 0.55);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
            color: #f5f5f5;
            font-size: 0.9rem;
            line-height: 1.35;
            pointer-events: none;
        }
        body:has([data-testid="stExpandSidebarButton"]) .sg-sidebar-hint,
        body:has([aria-label="Expand sidebar"]) .sg-sidebar-hint {
            display: flex;
        }
        .sg-sidebar-hint strong {
            display: block;
            margin-bottom: 0.1rem;
            color: #f0b75a;
        }
        .sg-hint-arrow {
            flex: 0 0 auto;
            width: 1.7rem;
            height: 1.7rem;
            color: #f0b75a;
            animation: sg-hint-bob 1.4s ease-in-out infinite;
        }
        body:has([data-testid="stExpandSidebarButton"]) [data-testid="stExpandSidebarButton"],
        body:has([aria-label="Expand sidebar"]) [aria-label="Expand sidebar"] {
            border-radius: 0.5rem;
            animation: sg-hint-pulse 1.8s ease-in-out infinite;
        }
        @keyframes sg-hint-bob {
            0%, 100% { transform: translate(0, 0); }
            50% { transform: translate(-3px, -3px); }
        }
        @keyframes sg-hint-pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(240, 183, 90, 0); }
            50% { box-shadow: 0 0 0 4px rgba(240, 183, 90, 0.35); }
        }
        @media (prefers-reduced-motion: reduce) {
            .sg-hint-arrow,
            [data-testid="stExpandSidebarButton"],
            [aria-label="Expand sidebar"] {
                animation: none !important;
            }
        }
        </style>
        <div class="sg-sidebar-hint" role="status">
          <svg class="sg-hint-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M20 20 L6 6"></path>
            <path d="M6 13 V6 H13"></path>
          </svg>
          <div>
            <strong>Open the menu</strong>
            Tap the double-arrow at the top left, then paste your Gemini API key.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="SuperGeM",
                       page_icon=AVATARS["assistant"])
st.header(
    "SuperGeM",
    text_alignment= "center",
    icon= ":material/flare:",
    divider= "blue",
    )

st.markdown(
    """
    <style>
    div[data-testid="stSelectbox"] {
        max-width: 240px;
    }

    div[data-testid="stSelectbox"] label {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)    

KEY_COOKIE = "supergem_api_key"


def load_saved_key() -> str:
    """Read the API key this browser saved in a first-party cookie."""
    try:
        value = st.context.cookies.get(KEY_COOKIE, "")
    except Exception:
        return ""
    if not isinstance(value, str):
        return ""
    return unquote(value) if value else ""


def save_key_cookie(value: str) -> None:
    # Same-origin iframe, so this writes the cookie for the app's origin.
    payload = quote(value, safe="")
    components.html(
        f"""<script>
        document.cookie = "{KEY_COOKIE}={payload}; path=/; max-age=31536000; SameSite=Lax" +
          (location.protocol === "https:" ? "; Secure" : "");
        </script>""",
        height=0,
    )


def forget_key_cookie() -> None:
    components.html(
        f"""<script>
        document.cookie = "{KEY_COOKIE}=; path=/; max-age=0; SameSite=Lax" +
          (location.protocol === "https:" ? "; Secure" : "");
        </script>""",
        height=0,
    )


def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    clear_saved = st.session_state.pop("clear_saved_key", False)
    saved_key = "" if clear_saved else load_saved_key()
    if clear_saved:
        st.session_state.api_key_input = ""
        # Render on this run so the delete script is not torn down by a rerun.
        forget_key_cookie()
    elif "api_key_input" not in st.session_state:
        st.session_state.api_key_input = saved_key
    if saved_key:
        st.session_state.saved_key = saved_key

    with st.sidebar:
        st.markdown("## API access")
        typed_key = st.text_input(
            "Gemini API key",
            type="password",
            key="api_key_input",
            help="Your key is kept in this session and saved in this browser.",
        ).strip()
        api_key = typed_key
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        api_key = api_key.strip()
        st.caption("Get a key from [Google AI Studio](https://aistudio.google.com/apikey).")

        # Never persist the server's env-var key into a visitor's browser.
        if typed_key and typed_key != st.session_state.get("saved_key"):
            save_key_cookie(typed_key)
            st.session_state.saved_key = typed_key

        if st.session_state.get("saved_key"):
            st.caption("Saved in this browser")
            if st.button("Forget saved key", key="forget_saved_key"):
                st.session_state.pop("saved_key", None)
                st.session_state.clear_saved_key = True
                st.rerun()

        if not api_key:
            st.info("Enter a Gemini API key to enable chat.")

        st.markdown("### [Check Usage](https://aistudio.google.com/rate-limit?timeRange=last-1-day)")   

    if not api_key:
        render_sidebar_hint()


    for message in st.session_state.chat_history:
        if "Human" in message:
            render_message("user", message["Human"])
        elif "AI" in message:
            render_message("assistant", message["AI"])

    st.markdown(
        """
        <style>
        /* Move the chat input up slightly */
        div[data-testid="stChatInput"] {
            bottom: 3.8rem !important;
        }

        /* Pin the model selector to the bottom-left */
        .st-key-model_selector {
            position: fixed !important;
            bottom: 5rem !important;
            left: max(calc((100vw - 43rem) / 2), 1rem) !important;
            width: 240px !important;
            z-index: 1000 !important;
        }

        /* Keep the selector compact */
        .st-key-model_selector div[data-testid="stSelectbox"] {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    selected_model = st.selectbox(
        "Model",
        options=list(MODELS.keys()),
        index=list(MODELS.keys()).index(DEFAULT_MODEL),
        key="model_selector",
        disabled=not api_key,
        label_visibility="collapsed",
    )

    prompt = st.chat_input(
        "Chat with SuperGeM",
        accept_file= True,
        max_upload_size= 1024,
        accept_audio= True,
        disabled=not api_key,
    )

    if prompt:
        user_text = prompt.text
        files = prompt.files or []

        agent_input = build_agent_input(
            user_text,
            st.session_state.chat_history
        )

        st.session_state.chat_history.append({"Human": user_text})
        render_message("user", user_text)

        failed = False

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            try:
                with st.spinner("Thinking..."):
                    if files and not all(_is_rag_file(f) for f in files):
                        # Non-document attachments go straight to Gemini until a
                        # dedicated multimodal identification step exists.
                        answer = ask_gemini_with_files(
                            api_key,
                            MODELS[selected_model],
                            agent_input,
                            upload_files(files, api_key),
                        )
                    elif files:
                        try:
                            context = document_context(
                                stage_uploads(files),
                                user_text,
                                api_key,
                            )
                        except READPDF.EmptyDocumentError:
                            # Scanned/image-only PDFs have no text to index;
                            # let Gemini read the original file instead.
                            answer = ask_gemini_with_files(
                                api_key,
                                MODELS[selected_model],
                                agent_input,
                                upload_files(files, api_key),
                            )
                        else:
                            agent_input = (
                                "Context retrieved from the user's attached documents:\n"
                                f"<documents>\n{context}\n</documents>\n\n"
                                + agent_input
                            )
                            answer = run_agent(
                                api_key,
                                MODELS[selected_model],
                                agent_input,
                            )
                    else:
                        answer = run_agent(
                            api_key,
                            MODELS[selected_model],
                            agent_input,
                        )

                    answer = sanitize_answer(answer)

            except Exception as e:
                log.exception("Agent request failed")
                failed = True
                answer = friendly_error(e)
                detail = f"{type(e).__name__}: {e}"
                if api_key:
                    detail = detail.replace(api_key, "***")

            if failed:
                st.error(answer)
                with st.expander("Technical details"):
                    st.code(detail[:1500])
            else:
                st.markdown(answer)

        st.session_state.chat_history.append({"AI": answer})

    with st.sidebar:
        st.markdown("## ℹ️Info")
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
                "[Linkedin](https://www.linkedin.com/in/njm29)",
                unsafe_allow_html=True,
            )

        # Display Github logo and link
        col1, col2 = st.columns([0.5, 4])
        with col1:
            st.image(str(BASE_DIR / "assets" / "github-sign.png"), width=30)
        with col2:
            st.markdown(
                "[Github](https://github.com/Mahir29Flame)",
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
