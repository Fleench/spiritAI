from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    page.goto("https://churchwritings.com/category/ante-nicene", wait_until='networkidle', timeout=30000)
    
    # Get all links
    links = page.locator('a').all()
    print(f"Total links on page: {len(links)}\n")
    
    # Print first 20 links
    for i, link in enumerate(links[:20]):
        try:
            href = link.get_attribute('href')
            text = link.text_content()
            print(f"{i}: {href} -> {text}")
        except:
            pass
    
    browser.close()
