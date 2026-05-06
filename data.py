from playwright.sync_api import sync_playwright
import os
import time

base_url = "https://churchwritings.com"
os.makedirs('/workspace/data', exist_ok=True)

visited_urls = set()
downloaded_articles = 0

def scrape_with_playwright(url, depth=0):
    """Scrape using Playwright to handle JavaScript rendering"""
    global downloaded_articles
    
    if depth > 3:  # Limit recursion
        return
    
    if url in visited_urls:
        return
    
    visited_urls.add(url)
    
    if 'churchwritings.com' not in url:
        return
    
    print(f"{'  ' * depth}Fetching: {url}")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            page.goto(url, wait_until='networkidle', timeout=30000)
            time.sleep(2)
            
            # Get full page text
            text = page.content()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
            
            content = soup.get_text(separator=' ', strip=True)
            
            # Save if it's a book/article page (not a category listing)
            if len(content) > 1000 and '/book/' in url:
                filename = url.replace('https://', '').replace('/', '_').replace('.', '_')[:100] + '.txt'
                filepath = f'/workspace/data/{filename}'
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"{'  ' * depth}✓ SAVED: {filename} ({len(content)} chars)")
                downloaded_articles += 1
            
            # Find all links on page
            links = page.locator('a[href]').all()
            found_links = []
            
            for link in links:
                try:
                    href = link.get_attribute('href')
                    if not href:
                        continue
                    
                    # Convert relative to absolute
                    if href.startswith('/'):
                        href = base_url + href
                    
                    # Only scrape churchwritings.com
                    if 'churchwritings.com' not in href:
                        continue
                    
                    # Remove fragments
                    href = href.split('#')[0]
                    
                    # Only follow category and book links
                    if ('/category/ante-nicene/' in href or '/book/' in href) and href not in visited_urls:
                        found_links.append(href)
                except:
                    pass
            
            # Remove duplicates
            found_links = list(set(found_links))
            print(f"{'  ' * depth}Found {len(found_links)} new links")
            
            browser.close()
            
            # Scrape found links
            for link in found_links[:20]:  # Limit to 20 per page
                time.sleep(1)
                scrape_with_playwright(link, depth + 1)
    
    except Exception as e:
        print(f"{'  ' * depth}✗ Error: {e}")

# Start from main Ante-Nicene category
print("Starting scrape of Ante-Nicene writings...")
print("This will take a while...\n")

scrape_with_playwright(f"{base_url}/category/ante-nicene")

print(f"\n✓ Finished!")
print(f"Downloaded {downloaded_articles} articles")
print(f"Total URLs visited: {len(visited_urls)}")
