from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_groq import ChatGroq

from src.researcher import RetrieverResult, Researcher


class LLM:
    def __init__(self, researcher: Researcher):
        self.llm = Groq()
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=50
        )
        self.chroma_client = chromadb.PersistentClient(path="./db")
        try:
            self.chroma_client.create_collection(name="my_collection")
        except:
            pass
        self.collection = self.chroma_client.get_collection(name="my_collection")
        self.researcher = researcher
        self.db_content_urls = []

    def generate_summary(self, search_query: str, web_contents: list[RetrieverResult]):
        """Generate a summary for the web search results"""
        content = ""
        for wc in web_contents[:4]:
            content += wc["content"] + "\n"

        prompt = f"User Query: {search_query}\nContent: {content[:5000]}"
        completion = self.llm.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "Using the provided content, and the user's online search query, generate a summary the user can use to get information about the query. The summary should include important things from the content provided. If searched for movie, the summary should highlight the release date, actors and other import information. If searched for a company, then highlights its value, stackholders, location and more. If searched for question, then answer the question using the content provided. The summary should be professional and should include all relevant content only from the web content. Start with what the content is really about the query's main content is using 2 or 3 lines, then continue with sub headings with 3 points in each. Make sure the summary is long and has around 500 words and also has emojis.",
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
            max_tokens=1024,
            stop=None,
            top_p=1,
        )
        summary =  completion.choices[0].message.content

        summary += "\n\n**Sources:**\n"
        for wc in web_contents[:4]:
            summary += f'- {wc['url']}\n'

        return summary

    def build_db(self, web_contents: list[RetrieverResult]):
        """Populates the vectorstore with the given result texts"""

        for result in web_contents:
            # Destructuring url and content from list item
            url = result["url"]
            content = result["content"]

            if url not in self.db_content_urls:
                # Splitting text
                splits = self.text_splitter.split_text(content)

                if len(splits) == 0:
                    continue

                ids = [f"{url}-{x}" for x in range(len(splits))]
                metadata = [{"url": url} for x in range(len(splits))]
                uris = [url] * len(splits)
                self.collection.add(
                    documents=splits, uris=uris, metadatas=metadata, ids=ids
                )
                self.db_content_urls.append(url)

    def answer_followup(self, question: str) -> str:
        """RAG flow"""

        def gen():
            vec_res = self.collection.query(query_texts=question, n_results=3)
            docs = vec_res["documents"][0]
            prompt = f"\nQuestion: {question}.\nContent:\n"
            for i in docs:
                prompt += "- " + i + "\n"
            completion = self.llm.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": "Using the content below, answer the user's question. Admit if you don't know the accurate answer. Prevent explaination or preamble. The response should be 2 to 3 line only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                max_tokens=1024,
                stop=None,
                top_p=1,
            )
            return completion.choices[0].message.content

        primary_answer = gen()
        relevance = self.check_answer_relevance(question, primary_answer)
        if relevance:
            return primary_answer
        else:
            # New search
            ret_res = self.researcher.search_retrieve_content(question)
            self.build_db(ret_res)
            return gen()

    def check_answer_relevance(self, question: str, answer: str) -> bool:
        """Checks whether the answer provided is relevant to the question"""

        class BinaryResponse(BaseModel):
            response: str

        system_prompt = "You are an expert in finding whether the answer correctly answers a question. Based on the question and answer provided, say whether the answer is accurate, relevant and not partial. The answer should not be incomplete based on the context, if so then the answer is not relevant. If the answer says that its not mentioned in the context, then the answer is invalid. Give a binary output 'yes' or 'no' for the response. The response should be JSON with a single key 'response' with value as 'yes' or 'no'. No explanation or preable. Just output the JSON."

        chat = ChatGroq(model_name="llama3-70b-8192")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Question: {question}\nAnswer: {answer}"),
            ]
        )
        parser = JsonOutputParser(pydantic_object=BinaryResponse)
        chain = prompt | chat | parser
        res = chain.invoke({"question": question, "answer": answer})
        return res["response"] == "yes"
