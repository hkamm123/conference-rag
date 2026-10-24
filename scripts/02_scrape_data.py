"""
Step 2: Scrape Conference Talks
================================
Scrapes 5 years of General Conference talks from the Church website
and saves them to data/talks.json.

Usage:
    python scripts/02_scrape_data.py

Output:
    data/talks.json  — JSON array of talk objects

Prerequisites:
    - Internet connection
    - No API keys needed for this step
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


YEARS_TO_SCRAPE = 5
START_YEAR = 2025 - YEARS_TO_SCRAPE
END_YEAR = 2025
OUTPUT_DIR = os.path.join('scripts', 'output')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'talks.json')


def setup_session():
    """Create an HTTP session with a browser-like user agent."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    return session


def get_conference_urls(start_year, end_year):
    """Generate URLs for each conference (April + October per year)."""
    base_url = 'https://www.churchofjesuschrist.org/study/general-conference/{year}/{month}?lang=eng'
    return [
        (base_url.format(year=year, month=month), str(year), month)
        for year in range(start_year, end_year + 1)
        for month in ['04', '10']
    ]


def get_talk_urls(conference_url, year, month, session):
    """Extract individual talk URLs from a conference index page."""
    try:
        response = session.get(conference_url, timeout=10)
        response.raise_for_status()
    except Exception:
        return []

    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')
    talk_urls = []
    seen = set()

    session_slugs = [
        'saturday-morning', 'saturday-afternoon', 'sunday-morning', 'sunday-afternoon',
        'priesthood-session', 'women-session', 'womens-session', 'session', 'video'
    ]

    month_path = f'/study/general-conference/{year}/{month}/'
    for link in soup.find_all('a', href=True):
        href = link.get('href')
        if not href or month_path not in href or 'lang=eng' not in href:
            continue

        canonical = 'https://www.churchofjesuschrist.org' + href
        if canonical in seen:
            continue
        seen.add(canonical)

        if any(slug in canonical.lower() for slug in session_slugs):
            continue
        # Skip the conference index page itself
        if href.rstrip('?lang=eng').endswith(f'/{month}'):
            continue

        talk_urls.append(canonical)

    return talk_urls


def scrape_talk(talk_url, session):
    """Scrape a single talk page and return structured data."""
    try:
        response = session.get(talk_url, timeout=10)
        response.raise_for_status()
    except Exception:
        return None

    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    title = soup.find("h1").text.strip() if soup.find("h1") else "No Title"

    speaker_tag = soup.find("p", {"class": "author-name"})
    speaker = speaker_tag.text.strip() if speaker_tag else "Unknown"
    # The page shows "By President Russell M. Nelson" with a non-breaking space.
    speaker = re.sub(r'^By\s+', '', speaker.replace('\xa0', ' '))

    calling_tag = soup.find("p", {"class": "author-role"})
    calling = calling_tag.text.strip() if calling_tag else ""

    content_div = soup.find("div", {"class": "body-block"})
    if not content_div:
        return None

    content = " ".join(p.text.strip() for p in content_div.find_all("p"))

    year_match = re.search(r'/(\d{4})/', talk_url)
    year = int(year_match.group(1)) if year_match else None
    season = "April" if "/04/" in talk_url else "October"

    return {
        "title": title,
        "speaker": speaker,
        "calling": calling,
        "year": year,
        "season": season,
        "url": talk_url,
        "text": content
    }


def main():
    print("=" * 60)
    print(f"Scraping Conference Talks ({START_YEAR}–{END_YEAR})")
    print("=" * 60)

    session = setup_session()
    conference_urls = get_conference_urls(START_YEAR, END_YEAR)

    # Phase 1: Find all talk URLs
    print("\nFinding talk URLs...")
    all_talk_urls = []
    for conf_url, year, month in tqdm(conference_urls, desc="Conferences"):
        urls = get_talk_urls(conf_url, year, month, session)
        all_talk_urls.extend(urls)
    print(f"Found {len(all_talk_urls)} talks\n")

    # Phase 2: Scrape each talk in parallel
    print("Scraping talk content...")
    talks_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scrape_talk, url, session): url for url in all_talk_urls}
        for future in tqdm(as_completed(futures), total=len(all_talk_urls), desc="Scraping"):
            talk = future.result()
            if talk:
                talks_data.append(talk)

    if not talks_data:
        print("\n❌ No talks scraped! Check your internet connection or website structure.")
        return

    # Save to JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(talks_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Scraped {len(talks_data)} talks!")
    print(f"   Years: {min(t['year'] for t in talks_data)} – {max(t['year'] for t in talks_data)}")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"\nNext: python scripts/03_import_data.py")


if __name__ == '__main__':
    main()
