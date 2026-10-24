"""
Step 4: Generate Embeddings
===============================
Reads scripts/output/chunks.json and generates OpenAI embeddings
for each chunk. Saves the result to scripts/output/chunks_with_embeddings.json.

This is the most expensive step (~$0.02 in API costs for ~9k chunks). The output is
saved to disk so you won't lose your work if the next step fails.

Usage:
    python scripts/04_embed_data.py

Input:
    scripts/output/chunks.json  — from Step 3 (import)

Output:
    scripts/output/chunks_with_embeddings.json  — chunk records with embedding vectors

Prerequisites:
    - config.secret.json with OPENAI_API_KEY
    - scripts/output/chunks.json from the import step
"""

import json
import os
import sys
import time

from openai import OpenAI
from tqdm import tqdm


INPUT_FILE = os.path.join('scripts', 'output', 'chunks.json')
OUTPUT_FILE = os.path.join('scripts', 'output', 'chunks_with_embeddings.json')
BATCH_SIZE = 100


def load_secrets():
    with open('config.secret.json', 'r') as f:
        return json.load(f)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} not found. Run scripts/03_import_data.py first.")
        sys.exit(1)

    # Load chunks
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        records = json.load(f)

    print("=" * 60)
    print(f"Generating Embeddings for {len(records):,} Chunks")
    print("=" * 60)
    print(f"   Model: text-embedding-3-small (1,536 dimensions)")
    print(f"   Batch size: {BATCH_SIZE}\n")

    # Check if there's a partial result we can resume from
    already_embedded = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        already_embedded = len(existing)
        # 03_import_data.py generates new chunk IDs on every run, so an output
        # file from before a re-import would update rows that no longer exist.
        if existing and existing[0]['id'] != records[0]['id']:
            print(f"❌ {OUTPUT_FILE} is from an earlier import (chunk IDs don't match).")
            print(f"   Delete it and re-run this script.")
            sys.exit(1)
        if already_embedded >= len(records):
            print(f"✅ All {len(records):,} chunks already have embeddings in {OUTPUT_FILE}")
            print(f"   Delete {OUTPUT_FILE} to re-generate.")
            return
        print(f"   ⏩ Resuming from chunk {already_embedded:,} (found partial output)")
        records_to_embed = records[already_embedded:]
    else:
        existing = []
        records_to_embed = records

    # Generate embeddings
    secrets = load_secrets()
    client = OpenAI(api_key=secrets['OPENAI_API_KEY'])

    embedded_records = list(existing)  # Start from any partial results

    for i in tqdm(range(0, len(records_to_embed), BATCH_SIZE), desc="Embedding"):
        batch = records_to_embed[i:i + BATCH_SIZE]
        batch_texts = [r['text'] for r in batch]

        try:
            response = client.embeddings.create(
                model='text-embedding-3-small',
                input=batch_texts
            )

            for record, item in zip(batch, response.data):
                record_with_embedding = dict(record)
                record_with_embedding['embedding'] = item.embedding
                embedded_records.append(record_with_embedding)

        except Exception as e:
            # Stop instead of skipping the batch: resuming assumes the output
            # file holds the first N records in order, with no gaps.
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(embedded_records, f, ensure_ascii=False)
            print(f"\n   ❌ Batch error: {e}")
            print(f"   Saved {len(embedded_records):,} embedded chunks. Re-run this script to resume.")
            sys.exit(1)

        # Save progress every 10 batches (in case of crash)
        if (i // BATCH_SIZE + 1) % 10 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(embedded_records, f, ensure_ascii=False)

        time.sleep(0.1)

    # Final save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(embedded_records, f, ensure_ascii=False)

    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n✅ Embedding complete!")
    print(f"   Embedded: {len(embedded_records):,} chunks")

    # Estimate cost
    total_chars = sum(len(r['text']) for r in embedded_records)
    est_tokens = total_chars / 4  # rough estimate
    cost = (est_tokens / 1_000_000) * 0.020
    print(f"   💰 Estimated cost: ${cost:.4f}")

    print(f"\n💾 Saved to {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
    print(f"   This file is your safety net — embeddings are preserved on disk.")
    print(f"\nNext: python scripts/05_update_embeddings.py")


if __name__ == '__main__':
    main()
