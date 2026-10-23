from supabase import create_client
import json

with open('config.public.json') as f:
    public = json.load(f)

# Connect with the ANON key (not service key!)
client = create_client(public['SUPABASE_URL'], public['SUPABASE_ANON_KEY'])

try: 
  # This works — page_views has a public SELECT policy
  result = client.table('page_views').select('*').limit(5).execute()
  print(f"page_views: {len(result.data)} rows ✅")
except Exception as e:
  print(f"Exception while trying to select from page_views: {e}")

try: 
  # This should work
  result = client.table('page_views').insert({'page_url': 'rls-test'}).execute()
  print(f"Insert into page views worked.")
except Exception as e:
  print(f"Exception while trying to insert into page_views: {e}")

try:
  # This shouldn't work
  result = client.table('sentence_embeddings').insert({'talk_id': '11111111-2222-4333-8444-555555555555','title': 'some-talk', 'text': 'hello conference'}).execute()
  print(f"Insert into sentence_embeddings worked.")
except Exception as e:
  print(f"Exception while trying to insert into sentence_embeddings: {e}")

try:
  # This returns nothing — sentence_embeddings requires authentication
  result = client.table('sentence_embeddings').select('*').limit(5).execute()
  print(f"sentence_embeddings: {len(result.data)} rows (expected: 0) 🔒")
except Exception as e:
  print(f"Exception while trying to select from sentence_embeddings: {e}")