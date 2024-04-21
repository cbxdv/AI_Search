from playwright.sync_api import sync_playwright

from src.llm import LLM
from src.researcher import Researcher
from src.browser import Browser

query = input("Search Query: ")

with sync_playwright() as playwright:
    browser = Browser(playwright)
    researcher = Researcher(get_page_html=browser.get_page_html)
    llm = LLM()

    retrival_results = researcher.search_retrieve_content(query)
    if len(retrival_results) == 0:
        print("Unable to fetch content from external sources. Try again later.")
        exit()

    summary = llm.generate_summary(query, retrival_results)
    print(summary)

    llm.build_db(retrival_results)

    q = "n"
    while q.lower() == "n":
        question = input("Question: ")
        res = llm.answer_followup(question)
        print("\n" + res + "\n")
        q = input("Quit? (Y or N) : ")
