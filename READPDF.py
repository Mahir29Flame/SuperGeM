import json
from functools import lru_cache
from pathlib import Path

import pymupdf
from google import genai as google_genai
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool
from langchain_text_splitters import CharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "Gemini"

# "embedding-001" and "text-embedding-004" have been shut down by Google.
EMBEDDING_MODEL = "gemini-embedding-2"

# Formats we can turn into text chunks for embedding.
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".json", ".log", ".rst"}
SUPPORTED_SUFFIXES = {".pdf"} | TEXT_SUFFIXES


def extract_text(docs):
    text = ""
    for pdf_path in docs:
        if Path(pdf_path).suffix.lower() in TEXT_SUFFIXES:
            text += Path(pdf_path).read_text(encoding="utf-8", errors="ignore") + "\n"
            continue
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


class GeminiEmbeddings(Embeddings):
    """Direct google-genai adapter for gemini-embedding-2.

    LangChain's GoogleGenerativeAIEmbeddings always sends ``task_type``, which
    gemini-embedding-2 rejects; the model expects the task in the text prefix.
    """

    batch_size = 100

    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL):
        self.client = google_genai.Client(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepared = [f"title: none | text: {text}" for text in texts]
        return self._embed(prepared)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"task: question answering | query: {text}"])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            result = self.client.models.embed_content(
                model=self.model,
                contents=batch,
            )
            vectors.extend(embedding.values for embedding in result.embeddings)
        return vectors


def knowledge_base(chunks, api_key: str):
    embeddings = GeminiEmbeddings(api_key=api_key)
    knowledge = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return knowledge


def _default_pdf() -> str:
    """Fallback used when the agent forgets to pass a path."""
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIR}")
    return str(pdfs[0])


class EmptyDocumentError(ValueError):
    """The document has no extractable text, e.g. a scanned image-only PDF."""


@lru_cache(maxsize=4)
def _get_knowledge_base(pdf_path: str, api_key: str) -> FAISS:
    """Build (and cache) the vector index for a document."""
    path = Path(pdf_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    text = extract_text([str(path)])
    if not text.strip():
        raise EmptyDocumentError(
            f"No extractable text found in {pdf_path}; it may be a scanned PDF."
        )
    chunks = split_chunks(text)
    if not chunks:
        raise EmptyDocumentError(f"No text chunks could be built from {pdf_path}")
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
