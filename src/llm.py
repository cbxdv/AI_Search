import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

from src.researcher import RetrieverResult


class LLM:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="my_collection")

    def generate(self, prompt: str) -> str:
        """Generate response from LLM using Ollama"""
        
        print("-----LLM GENERATING-----")
        res = ollama.generate(
            "mistral:instruct", prompt, stream=False, options={"temperature": 0.3}
        )
        return res["response"]

    def generate_summary(self, search_query: str, web_contents: list[RetrieverResult]):
        """Generate a summary for the web search results"""
        prompt = ""
        for wc in web_contents:
            prompt += wc["content"] + "\n"
        prompt += f"""\n\n
            User Query: {search_query}
        """
        prompt = (
            "Using the provided content, and the user's online search query, generate a summary the user can use to get information about the query. The summary should include important things from the content provided. If searched for movie, the summary should highlight the release date, actors and other import information. If searched for a company, then highlights its value, stackholders, location and more. If searched for question, then answer the question using the content provided. The summary should be professional and should include all relevant content only from the web content. Start with what the content is really about the query's main content is using 2 or 3 lines, then continue with sub headings with 3 points in each. Make sure the summary is long and has around 500 words and also has emojis."
            + prompt
        )
        summary = self.generate(prompt)
        return summary

    def build_db(self, web_contents: list[RetrieverResult]):
        """Populates the vectorstore with the given result texts"""

        for result in web_contents:
            # Destructuring url and content from list item
            url = result["url"]
            content = result["content"]

            # Splitting text
            splits = self.text_splitter.split_text(content)

            ids = [f"{url}-{x}" for x in range(len(splits))]
            metadata = [{"url": url} for x in range(len(splits))]
            uris = [url] * len(splits)
            self.collection.add(
                documents=splits, uris=uris, metadatas=metadata, ids=ids
            )

    def answer_followup(self, question: str):
        """CRAG flow"""

        vec_res = self.collection.query(query_texts=question, n_results=3)
        docs = vec_res["documents"][0]

        prompt = f"Using the content below, answer the user's question. Admit if you don't know the accurate answer. Prevent explaination or preamble. The response should be 2 to 3 line only.\nQuestion: {question}.\nContent:\n"
        for i in docs:
            prompt += "- " + i + "\n"

        return self.generate(prompt)
