import requests
from bs4 import BeautifulSoup
import os

os.makedirs('data', exist_ok=True)

# Start with the category page
url = "https://churchwritings.com/category/ante-nicene"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all article links (adjust selector based on actual HTML structure)
links = [a['href'] for a in soup.find_all('a', href=True) if '/ante-nicene/' in a['href']]

# Download each article
for link in links:
    try:
        article = requests.get(link)
        article_soup = BeautifulSoup(article.content, 'html.parser')
        
        # Extract text (adjust based on their HTML structure)
        text = article_soup.get_text()
        
        # Save to file
        filename = link.split('/')[-2] + '.txt'
        with open(f'data/{filename}', 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Error downloading {link}: {e}")
