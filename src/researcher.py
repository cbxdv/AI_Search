import bs4
from readability import Document
import html2text
from typing import TypedDict


class RetrieverResult(TypedDict):
    url: str
    content: str


class Researcher:
    def __init__(self, get_page_html):
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.get_page_html = get_page_html

    def search_retrieve_content(self, query: str) -> list[RetrieverResult]:
        """Searches in google and retrieves content"""

        urls = self.google_search(query)
        retrival_results = []

        for url in urls:
            content = self.get_content(url)
            if content is None:
                continue
            retrival_results.append(content)

        return retrival_results

    def google_search(self, query: str) -> list[str]:
        """Searches the web using Google and returns links for the query"""

        urls = []

        search_text = query.replace(" ", "+")
        url = f"https://www.google.com/search?q={search_text}"
        print(f"Searching for {{{query}}} using Google: {url}")
        page_html = self.get_page_html(url)
        if len(page_html) == 0:
            return []

        # Extracting a tags that have a specific jsname
        soup = bs4.BeautifulSoup(page_html, "html.parser")
        links = soup.find_all("a", jsname="UWckNb")

        for link in links:
            href = link.get("href")

            # Cleaning url
            if href.find("#:~:text=") != -1:
                href = href.split("#:~:text=")[0]

            if href not in urls:
                urls.append(href)

        return urls

    def get_content(self, url: str) -> RetrieverResult:
        """Returns the content in a website in a cleaned format"""

        blacklist = ["youtube.com"]
        for bll in blacklist:
            if url.find(bll) != -1:
                return None

        # Getting HTML from browser
        page_html = self.get_page_html(url)
        if len(page_html) == 0:
            return None

        # Parsing and getting the content only
        doc = Document(page_html)
        content = self.h2t.handle(doc.summary())
        content = content.replace("\n", " ")

        return {"url": url, "content": content}

    def __del__(self):
        self.h2t.close()
