import json
from functools import lru_cache
from pathlib import Path
import pymupdf
from langchain_community.vectorstores import FAISS
from langchain_core.tools import BaseTool
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "Gemini"

# "embedding-001" and "text-embedding-004" have been shut down by Google.
EMBEDDING_MODEL = "models/gemini-embedding-2"


def extract_text(docs):
    text = ""
    for pdf_path in docs:
        # Open the PDF file
        with pymupdf.open(pdf_path) as document:
            # Iterate through each page of the document
            for page in document:
                page_text = page.get_text()  # Extract text from the current page
                text += page_text + "\n"  # Append the extracted text to the cumulative text variable, with a newline for separation
    return text


def split_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    chunks = splitter.split_text(text)
    return chunks


def knowledge_base(chunks, api_key: str):
    # task_type is left to the defaults: embed_documents() uses RETRIEVAL_DOCUMENT
    # and embed_query() uses RETRIEVAL_QUERY.
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    knowledge = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return knowledge


def _default_pdf() -> str:
    """Fallback used when the agent forgets to pass a path."""
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIR}")
    return str(pdfs[0])


@lru_cache(maxsize=4)
def _get_knowledge_base(pdf_path: str, api_key: str) -> FAISS:
    """Build (and cache) the vector index for a PDF file."""
    path = Path(pdf_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    chunks = split_chunks(extract_text([str(path)]))
    if not chunks:
        raise ValueError(f"No extractable text found in {pdf_path}")
    return knowledge_base(chunks, api_key)


def ask_pdf(pdf_path: str, question: str, api_key: str) -> str:
    knowledge = _get_knowledge_base(str(pdf_path), api_key)
    similarity_search = knowledge.similarity_search(question, k=4)
    return "\n\n".join(doc.page_content for doc in similarity_search)


class ASKPDF(BaseTool):
    name: str = "ask_pdf"
    description: str = """Extracts text from PDF files and answers questions.
                    Inputs: a JSON object with the keys "pdf_path" (for example
                    "Gemini/USA-Law.pdf") and "question".
                    Outputs: the passages of the PDF that answer the question."""

    verbose: bool = True
    api_key: str

    @staticmethod
    def _parse_input(tool_input, kwargs: dict) -> dict:
        if isinstance(tool_input, dict):
            return {**tool_input, **kwargs}
        text = str(tool_input).strip()
        if text.startswith("{"):
            try:
                return {**json.loads(text), **kwargs}
            except json.JSONDecodeError:
                pass
        return {"question": text, **kwargs}

    def _run(self, tool_input: str = "", **kwargs) -> str:
        try:
            params = self._parse_input(tool_input, kwargs)
            question = params.get("question") or str(tool_input)
            pdf_path = params.get("pdf_path") or _default_pdf()
            return ask_pdf(pdf_path, question, self.api_key)
        except Exception as e:
            # we return the exception in order to warn the Agent
            return f"Error: {e}"
