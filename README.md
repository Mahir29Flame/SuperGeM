# 🧠SuperGeM
---

## Introduction
This is the GitHub repository of Super-GemM, a sophisticated agent designed to meet a wide array of user needs through advanced language understanding and decision-making capabilities. Powered by Google's cutting-edge Gemini 3.5 Flash Lite Large Language Model (LLM), this agent integrates a suite of special tools to facilitate searches across various platforms including DuckDuckGo, Wikipedia, Google Scholar, ArXiv, and PubMed, alongside the capability to read and interpret PDF documents. 

## Features

- **Advanced Language Understanding**: At its core, the Gemini-Pro Agent leverages Google's Gemini-pro LLM, ensuring a deep comprehension of human language.
- **Dynamic Tool Selection**: The agent autonomously decides the most suitable tool or response for each query, optimizing user experience.
- **Versatile Search Options**: With integrated search capabilities across DuckDuckGo, Wikipedia, Google Scholar, ArXiv, and PubMed, users can find accurate information across the web and academic journals.
- **PDF Interpretation**: Ability to read and analyze PDF documents, extracting valuable insights and answers.
- **Web Interface**: A user-friendly web interface built using the Streamlit library, making it accessible for non-technical users.
- **Powered by Langchain**: Utilizes the Langchain library for efficient handling of API and LLM calls.

![chart](assets/chart.jpg)

## Installation
To get started with the Gemini-Pro Agent, ensure you have Python installed on your system. Follow these steps to install the required packages:

1. Clone this repository:
```
git clone https://github.com/Mahir29Flame/SuperGeM.git
cd SuperGeM
```

2. Create a virtual environment (Python 3.11 or newer) and install the required packages:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
   With [uv](https://docs.astral.sh/uv/) this is just `uv venv` followed by `uv pip install -r requirements.txt`.

3. Start the app:
```
streamlit run app.py
```
   Enter your Gemini API key in the sidebar when the app starts. The key is kept only in the current Streamlit session. For desktop or automated deployments, `GEMINI_API_KEY` or `GOOGLE_API_KEY` can still be set as a fallback. `SERPAPI_API_KEY` is optional; without it the Google Scholar tool is disabled. You can get a Gemini key from [Google AI Studio](https://aistudio.google.com/apikey).

## Usage
After installation, you can utilize the Gemini-Pro Agent through its Streamlit web interface. Follow these steps to start interacting with the agent:

1. **Enter your API key**: Use the sidebar field, or configure `GEMINI_API_KEY`/`GOOGLE_API_KEY` for a desktop or automated deployment.

2. **Prepare PDF Files**: Place any PDF files you want the agent to read and interpret in a folder named `Gemini` within the root directory of the project. This ensures the agent can easily access and analyze your documents.

3. **Run the Application**: Execute the following command in your terminal to start the web interface:
   ```
   streamlit run app.py
   ```
   
Through the web interface, you can query the agent, request information from integrated search platforms, and have your PDF documents read and interpreted. The agent's advanced language understanding and decision-making capabilities will ensure you receive relevant and accurate responses.

This will start the web interface, where you can interact with the agent directly.

## How It Works

The Gemini-Pro Agent uses the Langchain library to manage API and LLM calls, enabling it to interpret queries, decide the most appropriate tool or answer, and fetch data from various sources. The agent's decision-making process and language understanding capabilities are powered by Google's Gemini-pro LLM, ensuring high-quality responses.

## Resources I used in the project

- all html-css codes are belong to [alejandro-ao](https://github.com/alejandro-ao/ask-multiple-pdfs/tree/main)

## Contributing
We welcome contributions to the Gemini-Pro Agent project. If you're interested in improving the agent or adding new features, please fork the repository and submit a pull request.

## License
[MIT License](LICENSE.md)

---
