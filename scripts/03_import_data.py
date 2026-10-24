"""
Step 3: Import Talk Data to Supabase
======================================
Reads scripts/output/talks.json, splits each talk into paragraph-sized
chunks, imports the talks and chunks to Supabase (without embeddings), and
saves the chunk records locally for the embedding step.

After this step, KEYWORD SEARCH will light up green on your site!
Embeddings are added in the next step to enable semantic search.

Usage:
    python scripts/03_import_data.py

Input:
    scripts/output/talks.json  — from Step 2 (scraping)

Output:
    scripts/output/chunks.json  — chunk records with id, talk_id, chunk_num, text

Prerequisites:
    - config.public.json with Supabase URL
    - config.secret.json with Supabase service key
    - Database schema created (Step 1)
    - Talks scraped (Step 2)
"""

import json
import os
import sys
import time
import uuid

from supabase import create_client
from tqdm import tqdm


INPUT_FILE = os.path.join('scripts', 'output', 'talks.json')
OUTPUT_FILE = os.path.join('scripts', 'output', 'chunks.json')
BATCH_SIZE = 100

# Paragraphs shorter than MIN_CHUNK_CHARS (dialogue lines, short quotes,
# scripture lines) are merged with the following paragraphs, as long as the
# chunk stays under MAX_CHUNK_CHARS. Most paragraphs (median ~250 chars) are
# left as written.
MIN_CHUNK_CHARS = 250
MAX_CHUNK_CHARS = 1500


def load_config():
    with open('config.public.json', 'r') as f:
        public_config = json.load(f)
    with open('config.secret.json', 'r') as f:
        secrets = json.load(f)
    return public_config, secrets


def chunk_paragraphs(paragraphs):
    """Group a talk's paragraphs into chunks of at least MIN_CHUNK_CHARS."""
    chunks = []
    current = ""
    for paragraph in paragraphs:
        if current and (len(current) >= MIN_CHUNK_CHARS
                        or len(current) + 1 + len(paragraph) > MAX_CHUNK_CHARS):
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current} {paragraph}" if current else paragraph
    if current:
        # Fold a short final piece into the previous chunk if it fits.
        if (chunks and len(current) < MIN_CHUNK_CHARS
                and len(chunks[-1]) + 1 + len(current) <= MAX_CHUNK_CHARS):
            chunks[-1] = f"{chunks[-1]} {current}"
        else:
            chunks.append(current)
    return chunks


def insert_in_batches(client, table, records):
    """Insert records in batches; returns the number that failed."""
    errors = 0
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc=f"Importing {table}"):
        batch = records[i:i + BATCH_SIZE]
        try:
            client.table(table).insert(batch).execute()
        except Exception as e:
            print(f"\nError at {table} batch {i // BATCH_SIZE}: {e}")
            errors += len(batch)
        time.sleep(0.1)
    return errors


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} not found. Run scripts/02_scrape_data.py first.")
        sys.exit(1)

    print("=" * 60)
    print("Importing Talk Data to Supabase")
    print("=" * 60)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        talks = json.load(f)
    if talks and 'paragraphs' not in talks[0]:
        print(f"❌ {INPUT_FILE} has no 'paragraphs' field (it was scraped by an older")
        print("   version of the scraper). Re-run scripts/02_scrape_data.py.")
        sys.exit(1)
    print(f"   Loaded {len(talks)} talks\n")

    # Build talk rows and chunk rows (no embeddings yet). IDs are generated
    # here so chunks can reference their talk before anything is inserted.
    print("Splitting talks into paragraph chunks...")
    talk_records = []
    chunk_records = []
    for talk in tqdm(talks, desc="Chunking"):
        talk_id = str(uuid.uuid4())
        talk_records.append({
            'id': talk_id,
            'title': talk['title'],
            'speaker': talk['speaker'],
            'calling': talk['calling'],
            'year': int(talk['year']) if talk['year'] else None,
            'season': talk['season'],
            'url': talk['url'],
        })
        for i, text in enumerate(chunk_paragraphs(talk['paragraphs']), 1):
            chunk_records.append({
                'id': str(uuid.uuid4()),
                'talk_id': talk_id,
                'chunk_num': i,
                'text': text,
                # No 'embedding' field — added in the next steps
            })

    print(f"✅ Split {len(talks)} talks into {len(chunk_records):,} chunks")
    print(f"   Average: {len(chunk_records) / len(talks):.1f} chunks per talk\n")

    # Save chunks locally (for the embedding step to use)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunk_records, f, ensure_ascii=False)
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"💾 Saved {len(chunk_records):,} chunks to {OUTPUT_FILE} ({file_size_mb:.1f} MB)")

    public_config, secrets = load_config()
    client = create_client(public_config['SUPABASE_URL'], secrets['SUPABASE_SERVICE_KEY'])

    # Clear existing data. Deleting talks also deletes their chunks
    # (ON DELETE CASCADE). The Data API refuses a DELETE without a filter,
    # so match every row with a filter that is always true.
    print("\n" + "=" * 60)
    print("Checking for existing data...")
    print("=" * 60)
    result = client.table('talks').select('id', count='exact').limit(1).execute()
    existing_count = result.count or 0
    if existing_count > 0:
        print(f"   Found {existing_count:,} existing talks. Deleting them and their chunks...")
        client.table('talks').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        print("   ✅ Cleared.")
    else:
        print("   Tables are empty — ready for import.")

    # Talks first: every chunk's talk_id must already exist (foreign key).
    print("\n" + "=" * 60)
    print(f"Importing {len(talk_records):,} talks and {len(chunk_records):,} chunks")
    print("=" * 60)
    print("   (Text only — embeddings will be added in the next steps)\n")

    talk_errors = insert_in_batches(client, 'talks', talk_records)
    if talk_errors:
        print(f"\n❌ {talk_errors} talks failed to import; not importing chunks.")
        sys.exit(1)
    chunk_errors = insert_in_batches(client, 'talk_chunks', chunk_records)

    # Final verification
    talk_count = client.table('talks').select('id', count='exact').limit(1).execute().count or 0
    chunk_count = client.table('talk_chunks').select('id', count='exact').limit(1).execute().count or 0
    print(f"\n   Talks in database:  {talk_count:,} (expected {len(talk_records):,})")
    print(f"   Chunks in database: {chunk_count:,} (expected {len(chunk_records):,})")
    if chunk_errors or chunk_count != len(chunk_records):
        print(f"\n❌ Import incomplete ({chunk_errors:,} chunk rows failed). Re-run this script.")
        sys.exit(1)

    print(f"\n🎉 Keyword Search is now ready!")
    print(f"   Refresh your site — the 🔍 Keyword Search panel should turn GREEN.")
    print(f"\nNext: python scripts/04_embed_data.py")


if __name__ == '__main__':
    main()
