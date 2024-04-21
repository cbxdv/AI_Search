import gradio as gr
from playwright.sync_api import sync_playwright

from src.llm import LLM
from src.researcher import Researcher
from src.browser import Browser


searched_query = ""

llm = LLM()


def gen_summary(query):
    with sync_playwright() as playwright:
        browser = Browser(playwright)
        researcher = Researcher(get_page_html=browser.get_page_html)

        retrival_results = researcher.search_retrieve_content(query)

        if len(retrival_results) == 0:
            return "Unable to fetch content from external sources. Try again later."

        response = llm.generate_summary(query, retrival_results)
        llm.build_db(retrival_results)

        global searched_query
        searched_query = query

        return response


def rag(x, _):
    if len(searched_query) == 0:
        return "Search on something before asking questions....."
    return llm.answer_followup(x)


with gr.Blocks() as demo:
    gr.Markdown("# AI Search")
    inp = gr.Textbox(placeholder="Search Query")
    but = gr.Button("Search")
    out = gr.Markdown("")
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
