# [LLM powered Search Engine](https://github.com/cbxdv/AI_Search/blob/main/LLM%20Powered%20Search%20Engine.pdf)

A LLM powered search engine that searches the internet for you and summarizes content, instead of you going through links and getting details. Provides an interface for further questions based on the content.

## Features
- Playwright, BeautifulSoup, Readability, HTML2Text are some of the documents used for scraping data from the internet.
- Uses RAG technique with ChromaDB, to clarify doubts using the scraped content.
- Responses from the LLM, are checked for relevance so that questions away from the content can also be answered.

## Tools
- Google
- Meta Llama 3
- GroqCloud
- Chroma
- Playwright
- Gradio

## Running

### Install packages
```bash
pip install -r requirements.txt
```

### Create a `.env` file
```env
GROQ_API_KEY=<GROQ_API_KEY>
```

### Gradio UI
```shell
python gradio_ui.py
```
