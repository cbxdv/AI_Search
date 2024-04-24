import gradio as gr
from playwright.sync_api import sync_playwright

from src.llm import LLM
from src.researcher import Researcher
from src.browser import Browser


searched = False


def generate_summary(query):
    with sync_playwright() as playwright:
        browser = Browser(playwright)
        researcher = Researcher(get_page_html=browser.get_page_html)
        llm = LLM(researcher=researcher)

        retrival_results = researcher.search_retrieve_content(query)

        if len(retrival_results) == 0:
            return "Unable to fetch content from external sources. Try again later."
        summary_results = llm.generate_summary(query, retrival_results)
        llm.build_db(retrival_results)

        global searched
        searched = True

        return [
            "",                                 # the loader placeholder
            summary_results["summary"],         # summary of the content
            summary_results["references"],      # references
            gr.Tabs(visible=True),              # for making the output tabs visible
            gr.Row(visible=True)                # for making the QnA section visible
        ]


def content_qna(question):
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
    gr.Markdown("#### 🔎 Start with asking something that you want to web search")
    gr.Markdown("#### ❓ Once summary is generated, questions related to the query can be asked.")

    search_input = gr.Textbox(
        placeholder="Search Query", label="Search anything", autofocus=True
    )
    search_button = gr.Button("Search", variant="primary")

    # A placeholder for loading indicator
    load_ph = gr.Markdown('')

    output_tabs = gr.Tabs(visible=False)
    with output_tabs:
        with gr.Tab(label='Summary'):
            summary_out = gr.Markdown('')
        with gr.Tab(label='References'):
            references_out = gr.Markdown('')

    qa_row = gr.Row(visible=False)
    with qa_row:
        gr.Interface(
            fn=content_qna,
            inputs=[gr.Textbox(label="Question")],
            outputs=[gr.Textbox(label="Answer")],
            allow_flagging='never'
        )

    # Summary
    search_input.submit(generate_summary, search_input, [load_ph, summary_out, references_out, output_tabs, qa_row])
    search_button.click(generate_summary, search_input, [load_ph, summary_out, references_out, output_tabs, qa_row])


demo.launch()
