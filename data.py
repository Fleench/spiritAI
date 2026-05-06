import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, urlparse

base_url = "https://churchwritings.com"
os.makedirs('/workspace/data', exist_ok=True)

visited_urls = set()
downloaded_articles = 0

def extract_main_content(soup):
    """Extract the main article text from the page"""
    # Try common content containers
    selectors = [
        'article',
        'main',
        '.post-content',
        '.entry-content',
        '.content',
        'div.container',
    ]
    
    for selector in selectors:
        content = soup.select_one(selector)
        if content:
            text = content.get_text(separator=' ', strip=True)
            if len(text) > 500:  # Real content
                return text
    
    # Fallback: get all text and filter
    text = soup.get_text(separator=' ', strip=True)
    return text

def is_article_url(url):
    """Check if URL looks like an article (not just a category)"""
    path = urlparse(url).path.lower()
    
    # Skip if it's just a category page
    if path.count('/') <= 2:
        return False
    
    # Skip common non-article pages
    skip_words = ['category', 'page', 'tag', 'author', 'search', 'about', 'contact']
    for word in skip_words:
        if word in path:
            return False
    
    return True

def scrape_page(url, depth=0):
    """Recursively scrape pages and extract articles"""
    global downloaded_articles
    
    if depth > 3:  # Limit recursion depth
        return
    
    if url in visited_urls:
        return
    
    visited_urls.add(url)
    
    # Only scrape churchwritings.com domain
    if 'churchwritings.com' not in url:
        return
    
    print(f"{'  ' * depth}Visiting: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"{'  ' * depth}✗ Error fetching: {e}")
        return
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # If this looks like an article, try to save it
    if is_article_url(url):
        content = extract_main_content(soup)
        
        if len(content) > 500:  # Only save substantial content
            filename = url.replace('https://', '').replace('http://', '').replace('/', '_').replace('.', '_')[:100] + '.txt'
            filepath = f'/workspace/data/{filename}'
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"{'  ' * depth}✓ SAVED: {filename} ({len(content)} chars)")
            downloaded_articles += 1
    
    # Find all links on this page
    links = []
    for a in soup.find_all('a', href=True):
        link = a['href']
        
        # Convert relative to absolute URLs
        if link.startswith('/'):
            link = urljoin(base_url, link)
        elif not link.startswith('http'):
            link = urljoin(url, link)
        
        # Only churchwritings.com
        if 'churchwritings.com' in link:
            # Remove fragments
            link = link.split('#')[0]
            
            if link not in visited_urls:
                links.append(link)
    
    print(f"{'  ' * depth}Found {len(links)} new links")
    
    # Scrape found links
    for link in links[:15]:  # Limit to 15 per page to avoid explosion
        time.sleep(0.3)  # Be nice to server
        scrape_page(link, depth + 1)

# Start from the Ante-Nicene category
print("Starting scrape of Ante-Nicene writings...")
scrape_page(f"{base_url}/category/ante-nicene")

print(f"\n✓ Finished! Downloaded {downloaded_articles} articles")
print(f"Total URLs visited: {len(visited_urls)}")
