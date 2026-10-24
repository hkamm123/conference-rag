"""
Step 1: Create Database Schema
================================
Creates the talks and talk_chunks tables, Row Level Security policies, and
the match_chunks() function in your Supabase database.

The schema itself lives in supabase/migrations/*_paragraph_schema.sql, so
there is one source of truth. This script runs that file through the
Supabase Management API. If you use the Supabase CLI, `supabase db push`
applies the same file (and records it in the migration history).

The HNSW vector index is a separate migration (*_talk_chunks_hnsw_index.sql),
applied after the data is loaded; see the notes in that file.

Usage:
    python scripts/01_create_schema.py

Prerequisites:
    - config.public.json with Supabase URL
    - config.secret.json with Supabase service key, access token, and project ref
    - Supabase project created
"""

import glob
import json
import sys
import time

# Load configuration from both config files
with open('config.public.json', 'r') as f:
    public_config = json.load(f)
with open('config.secret.json', 'r') as f:
    secrets = json.load(f)

SUPABASE_URL = public_config['SUPABASE_URL']
SUPABASE_SERVICE_KEY = secrets['SUPABASE_SERVICE_KEY']
SUPABASE_ACCESS_TOKEN = secrets['SUPABASE_ACCESS_TOKEN']
SUPABASE_PROJECT_REF = secrets['SUPABASE_PROJECT_REF']

SCHEMA_MIGRATION_GLOB = 'supabase/migrations/*_paragraph_schema.sql'


def create_schema():
    import requests
    from supabase import create_client

    print("=" * 60)
    print("Creating Database Schema")
    print("=" * 60)

    matches = sorted(glob.glob(SCHEMA_MIGRATION_GLOB))
    if not matches:
        print(f"❌ No schema migration found ({SCHEMA_MIGRATION_GLOB})")
        return False
    with open(matches[-1], 'r') as f:
        schema_sql = f.read()
    print(f"   Running {matches[-1]}")

    url = f"https://api.supabase.com/v1/projects/{SUPABASE_PROJECT_REF}/database/query"
    headers = {
        "Authorization": f"Bearer {SUPABASE_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    resp = requests.post(url, headers=headers, json={"query": schema_sql})
    if resp.status_code in (200, 201):
        print("✅ Database schema created successfully!")
    else:
        print(f"❌ Schema creation failed: {resp.status_code}")
        print(resp.text[:500])
        return False

    # Verify tables exist via PostgREST
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    for attempt in range(5):
        try:
            for table in ('talks', 'talk_chunks'):
                result = client.table(table).select('id', count='exact').limit(1).execute()
                print(f"✅ Table {table} verified. Current rows: {result.count or 0}")
            return True
        except Exception:
            if attempt < 4:
                print(f"   Waiting for schema cache to refresh... ({attempt + 1}/5)")
                time.sleep(3)
            else:
                print("⚠️  Tables created but PostgREST cache hasn't refreshed yet.")
                print("   This is normal — they should be ready by import time.")
                return True


if __name__ == '__main__':
    if not create_schema():
        sys.exit(1)
    print("\n✅ Schema ready! Next: python scripts/02_scrape_data.py")
