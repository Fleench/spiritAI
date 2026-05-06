import requests
from bs4 import BeautifulSoup
import os
import time

base_url = "https://churchwritings.com"
os.makedirs('/workspace/data', exist_ok=True)

# Get all author category pages
print("Fetching author categories...")
url = f"{base_url}/category/ante-nicene"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

category_links = [a['href'] for a in soup.find_all('a', href=True) if '/category/ante-nicene/' in a['href']]
print(f"Found {len(category_links)} author categories")

downloaded = 0

# For each author category, find the actual articles
for category_link in category_links:
    full_url = base_url + category_link
    print(f"\nFetching: {full_url}")
    
    try:
        response = requests.get(full_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find article links (usually in posts or articles)
        article_links = [a['href'] for a in soup.find_all('a', href=True) 
                        if a['href'].startswith('/') and '/category/ante-nicene/' not in a['href'] 
                        and a['href'] != category_link]
        
        print(f"  Found {len(article_links)} potential articles")
        
        # Download each article
        for article_link in article_links[:10]:  # Limit to first 10 per category to avoid spam
            full_article_url = base_url + article_link
            
            try:
                article_response = requests.get(full_article_url)
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                
                # Extract text
                text = article_soup.get_text()
                
                if len(text) > 100:  # Only save if there's actual content
                    filename = article_link.strip('/').replace('/', '_') + '.txt'
                    filepath = f'/workspace/data/{filename}'
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    print(f"    ✓ Downloaded: {filename} ({len(text)} chars)")
                    downloaded += 1
                    
                time.sleep(0.5)  # Be nice to the server
            except Exception as e:
                print(f"    ✗ Error: {article_link}: {e}")
    
    except Exception as e:
        print(f"  Error fetching category: {e}")

print(f"\n✓ Downloaded {downloaded} articles to /workspace/data/")
