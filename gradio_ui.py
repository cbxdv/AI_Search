import gradio as gr
from playwright.sync_api import sync_playwright

from src.llm import LLM
from src.researcher import Researcher
from src.browser import Browser


searched = False


def gen_summary(query):
    with sync_playwright() as playwright:
        browser = Browser(playwright)
        researcher = Researcher(get_page_html=browser.get_page_html)
        llm = LLM(researcher=researcher)

        retrival_results = researcher.search_retrieve_content(query)

        if len(retrival_results) == 0:
            return "Unable to fetch content from external sources. Try again later."
        response = llm.generate_summary(query, retrival_results)
        llm.build_db(retrival_results)

        global searched
        searched = True

        return response


def rag(question, _):
    with sync_playwright() as playwright:
        browser = Browser(playwright)
        researcher = Researcher(get_page_html=browser.get_page_html)
        llm = LLM(researcher=researcher)

        if not searched:
            return "Search on something before asking questions....."
        if len(question) == 0:
            return "Ask a valid question..."
        return llm.answer_followup(question)


with gr.Blocks(title="AI Search") as demo:
    inp = gr.Textbox(
        placeholder="Search Query", label="Search anything", autofocus=True
    )
    but = gr.Button("Search", variant="primary")
    out = gr.Markdown("")
    inp.submit(lambda x: gen_summary(x), inp, out)
    but.click(lambda x: gen_summary(x), inp, out)
    gr.ChatInterface(
        fn=rag,
        chatbot=gr.Chatbot(height=500, render=False),
        textbox=gr.Textbox(
            placeholder="Ask something from the content", scale=7, render=False
        ),
        clear_btn=None,
        retry_btn=None,
        undo_btn=None,
        stop_btn=None,
    )

demo.launch()
