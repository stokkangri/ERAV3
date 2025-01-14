import os
import requests
from bs4 import BeautifulSoup
import re

# Directory to save the extracted Odia text
OUTPUT_DIR = "odia_texts"

# Regex pattern to detect Odia Unicode characters
ODIA_PATTERN = re.compile(r'[\u0B00-\u0B7F]+')

def create_output_directory():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def extract_odia_text(text):
    """Extract Odia text segments from a given text."""
    return " ".join(ODIA_PATTERN.findall(text))

def save_text_to_file(url, content):
    """Save extracted Odia text to a file."""
    # Sanitize filename
    filename = url.replace("https://", "").replace("http://", "").replace("/", "_") + ".txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"Saved Odia text to: {filepath}")

def scrape_website(url, visited_urls, depth, max_depth):
    """Scrape a website to extract Odia text and follow links up to a given depth."""
    if depth > max_depth:
        return

    try:
        print(f"Scraping: {url} at depth {depth}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unnecessary tags
        for tag in soup(["script", "style", "meta", "noscript"]):
            tag.decompose()

        # Extract paragraphs and filter Odia text
        paragraphs = []
        for element in soup.find_all(["p", "div", "span"]):  # Adjust tags as needed
            paragraph_text = element.get_text(separator=" ").strip()  # Combine text with spaces
            odia_text = extract_odia_text(paragraph_text)
            if odia_text:
                paragraphs.append(odia_text)

        if paragraphs:
            save_text_to_file(url, "\n\n".join(paragraphs))  # Join paragraphs with double newlines
        else:
            print(f"No Odia text found on: {url}")

        # Find and follow links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = requests.compat.urljoin(url, href)
            if full_url.startswith("http") and full_url not in visited_urls:
                visited_urls.add(full_url)
                scrape_website(full_url, visited_urls, depth + 1, max_depth)

    except Exception as e:
        print(f"Error scraping {url}: {e}")

def main():
    """Main function to scrape websites specified by the user."""
    with open('website.list', 'r', encoding='utf-8') as file:
        websites = [line.strip() for line in file.readlines()]

    max_depth = 3
    create_output_directory()

    visited_urls = set()
    for website in websites:
        if website not in visited_urls:
            visited_urls.add(website)
            scrape_website(website, visited_urls, depth=0, max_depth=max_depth)

if __name__ == "__main__":
    main()