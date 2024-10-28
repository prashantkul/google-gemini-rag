import requests
from bs4 import BeautifulSoup
import pandas as pd
import tldextract
import time
from urllib.parse import urljoin
from fake_useragent import UserAgent
import concurrent.futures
from requests.exceptions import RequestException
from bs4 import BeautifulSoup


# Define constants
BASE_URL = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"
MAX_DEPTH = 5
DELAY_BETWEEN_REQUESTS = 1
USER_AGENT_ROTATION = True


# Helper function to clean text
def clean_text(text):
    return ' '.join(text.split()).strip()


# Helper function to resolve relative URLs
def resolve_url(base, url):
    return urljoin(base, url)


# Function to extract all links, including from navigation and sidebars
def extract_all_links(soup, base_url):
    """Extract links from all sections including nav, aside, ul, and a."""
    links = []

    # Extract links from left nav (specifically targeting the left nav structure)
    left_nav = soup.find('nav', class_='left-nav')  # Adjust class as needed
    if left_nav:
        for a_tag in left_nav.find_all('a', href=True):
            link = resolve_url(base_url, a_tag['href'])
            links.append(link)

    # Extract links from <nav>, <aside>, <ul>, <li>
    for section in soup.find_all(['nav', 'aside', 'ul', 'li']):
        for a_tag in section.find_all('a', href=True):
            link = resolve_url(base_url, a_tag['href'])
            links.append(link)

    # Extract links from <a> tags outside nav or sidebar sections
    for a_tag in soup.find_all('a', href=True):
        link = resolve_url(base_url, a_tag['href'])
        links.append(link)

    return list(set(links))  # Remove duplicate links


# Function to scrape content from a single page
def scrape_content(url, session):
    try:
        # Rotate User-Agent if enabled
        if USER_AGENT_ROTATION:
            ua = UserAgent()
            user_agent = ua.random
            headers = {'User-Agent': user_agent}
        else:
            headers = None

        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except RequestException as e:
        print(f"Failed to retrieve {url}: {e}")
        return [], []

    try:
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"Error parsing HTML for {url}: {e}")
        return [], []

    content = []

    # Extract headings (h1, h2, etc.)
    for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5']):
        text = clean_text(header.get_text())
        if text:
            content.append({'type': 'header', 'content': text, 'url': url})

    # Extract paragraphs
    for paragraph in soup.find_all('p'):
        text = clean_text(paragraph.get_text())
        if text:
            content.append({'type': 'paragraph', 'content': text, 'url': url})

    # Extract images
    for img in soup.find_all('img'):
        img_url = resolve_url(url, img.get('src', ''))
        alt_text = clean_text(img.get('alt', 'No description'))
        if img_url:
            content.append({'type': 'image', 'content': alt_text, 'url': img_url})

    # Extract embedded videos (YouTube/Vimeo)
    for video in soup.find_all('iframe'):
        video_src = video.get('src', '')
        if 'youtube' in video_src or 'vimeo' in video_src:
            video_url = resolve_url(url, video_src)
            content.append({'type': 'video', 'content': 'Embedded Video', 'url': video_url})

    # Collect all links on the page (including sidebars and navigation)
    links = extract_all_links(soup, url)
    return content, links


# BFS function to crawl the site
def bfs_scrape(start_url):
    session = requests.Session()
    queue = [(start_url, 0)]  # (URL, current depth)
    visited_urls = set()
    all_data = []

    while queue:
        url, depth = queue.pop(0)
        if url in visited_urls or depth > MAX_DEPTH:
            continue

        print(f"Scraping {url} (Depth: {depth})...")
        visited_urls.add(url)

        # Scrape the current page
        content, links = scrape_content(url, session)
        all_data.extend(content)

        # Add new links to the queue
        for link in links:
            if BASE_URL in link and link not in visited_urls:
                queue.append((link, depth + 1))

        # Respect server load
        time.sleep(DELAY_BETWEEN_REQUESTS)

    return all_data


# Start the BFS scraping process
if __name__ == "__main__":
    data = bfs_scrape(BASE_URL)

    # Convert the scraped data to a DataFrame
    df = pd.DataFrame(data)

    # Save the content as CSV for further processing
    df.to_csv('ms_applied_data_science_full_content_6.csv', index=False)

    print("Scraping completed and content saved!")