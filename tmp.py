import requests
from bs4 import BeautifulSoup

base_url = "https://churchwritings.com"
url = f"{base_url}/category/ante-nicene/barnabas"

response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Print the raw HTML structure
print(soup.prettify()[:2000])  # First 2000 chars of HTML

# Also check for specific content markers
print("\n\n=== Looking for content ===")
print(f"Title: {soup.title}")
print(f"H1: {soup.find('h1')}")
print(f"H2: {soup.find('h2')}")

# Get all text
text = soup.get_text()
print(f"\nTotal text on page: {len(text)} chars")
print(f"First 500 chars:\n{text[:500]}")
