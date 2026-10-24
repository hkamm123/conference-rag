"""
Step 5: Import Embeddings to Database
=========================================
Reads scripts/output/chunks_with_embeddings.json and writes each chunk's
embedding into its existing talk_chunks row.

After this step, SEMANTIC SEARCH will light up green on your site
(once the embed-question Edge Function is deployed and the HNSW index
migration has been applied).

Usage:
    python scripts/05_update_embeddings.py

Input:
    scripts/output/chunks_with_embeddings.json  — from Step 4

Prerequisites:
    - config.public.json with Supabase URL
    - config.secret.json with Supabase service key
    - Talks and chunks imported (Step 3) — chunk IDs must match
    - Embeddings generated (Step 4)
"""

import json
import os
import sys
import time

from supabase import create_client
from tqdm import tqdm


INPUT_FILE = os.path.join('scripts', 'output', 'chunks_with_embeddings.json')
BATCH_SIZE = 100


def load_config():
    with open('config.public.json', 'r') as f:
        public_config = json.load(f)
    with open('config.secret.json', 'r') as f:
        secrets = json.load(f)
    return public_config, secrets


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} not found. Run scripts/04_embed_data.py first.")
        sys.exit(1)

    print("Loading embeddings data...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        records = json.load(f)

    with_embeddings = sum(1 for r in records if r.get('embedding'))
    print(f"   Loaded {len(records):,} records")
    print(f"   With embeddings: {with_embeddings:,}")
    if with_embeddings < len(records):
        print("❌ Some records have no embedding. Re-run scripts/04_embed_data.py.")
        sys.exit(1)

    public_config, secrets = load_config()
    client = create_client(public_config['SUPABASE_URL'], secrets['SUPABASE_SERVICE_KEY'])

    # Upsert by primary key: each row already exists from Step 3, so this
    # UPDATEs it in place rather than deleting and re-inserting the table.
    print("\n" + "=" * 60)
    print(f"Writing {len(records):,} embeddings in batches of {BATCH_SIZE}")
    print("=" * 60 + "\n")

    errors = 0
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Updating"):
        batch = [
            {k: r[k] for k in ('id', 'talk_id', 'chunk_num', 'text', 'embedding')}
            for r in records[i:i + BATCH_SIZE]
        ]
        try:
            client.table('talk_chunks').upsert(batch).execute()
        except Exception as e:
            print(f"\nError at batch {i // BATCH_SIZE}: {e}")
            errors += len(batch)
        time.sleep(0.1)

    # Verify
    total = client.table('talk_chunks').select('id', count='exact').limit(1).execute().count or 0
    with_emb = client.table('talk_chunks') \
        .select('id', count='exact') \
        .not_.is_('embedding', 'null') \
        .limit(1).execute().count or 0

    print(f"\n   Total chunks:          {total:,}")
    print(f"   Chunks with embeddings: {with_emb:,}")
    if errors or with_emb != len(records):
        print(f"\n❌ Incomplete ({errors:,} rows failed). Re-run this script; it is safe to repeat.")
        sys.exit(1)

    print(f"\n🎉 Embeddings loaded!")
    print(f"   Next: apply the HNSW index migration, then refresh your site.")


if __name__ == '__main__':
    main()
