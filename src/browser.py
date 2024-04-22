from playwright.sync_api import Playwright


class Browser:
    def __init__(self, playwright: Playwright):
        print("Starting Chromium in headless mode")
        chromium = playwright.chromium
        self.browser = chromium.launch(headless=False)

    def get_page_html(self, url: str):
        """Navigates to the provided URL and returns the response"""
        try:
            page = self.browser.new_page()
            print(f"Browser going to {url}")
            res = page.goto(url)
            if not res.ok:
                return ""
            content = page.content()
            page.close()
            return content
        except:
            return ""
