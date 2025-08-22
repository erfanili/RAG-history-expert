


from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    # Login
    page.goto("https://auth.openai.com/log-in")
    page.fill("input[name='email']", "er.esmaili2060@gmail.com")
    page.click("button[type='email']")
    breakpoint()
    page.fill("input[name='password']", "your_password")
    page.wait_for_load_state("networkidle")

    # Input text (searching flights)
    page.fill("input[name='origin']", "New York")
    page.fill("input[name='destination']", "Paris")
    page.click("button.search")  # whatever the selector is
    page.wait_for_selector(".results")

    print(page.content())
    browser.close()

