import requests
from bs4 import BeautifulSoup
import os

os.makedirs('/tmp/data', exist_ok=True)

# Start with the category page
base_url = "https://churchwritings.com"
url = f"{base_url}/category/ante-nicene"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all article links (adjust selector based on actual HTML structure)
links = [a['href'] for a in soup.find_all('a', href=True) if '/ante-nicene/' in a['href']]

# Remove duplicates
links = list(set(links))

# Download each article
for link in links:
    try:
        # Make absolute URL if it's relative
        if link.startswith('/'):
            full_url = base_url + link
        else:
            full_url = link
        
        article = requests.get(full_url)
        article_soup = BeautifulSoup(article.content, 'html.parser')
        
        # Extract text (adjust based on their HTML structure)
        text = article_soup.get_text()
        
        # Save to file
        filename = link.split('/')[-1] + '.txt'
        with open(f'/tmp/data/{filename}', 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Error downloading {full_url}: {e}")
