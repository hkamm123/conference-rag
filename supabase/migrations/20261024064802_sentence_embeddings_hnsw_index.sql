-- HNSW vector index for semantic search (match_sentences).
--
-- Without it, every search compares the question to all ~26k sentence vectors
-- (~10.5 s on the free tier), which exceeds the authenticated role's 8 s
-- statement_timeout, so semantic search fails. With it: ~0.5 s.
--
-- Run this AFTER the data is loaded (scripts/03-05). Building once over the
-- full table is much faster than updating the graph on every insert.
--
-- !! SLOW: the build took about 3 minutes on the free tier. !!
-- Don't run it in the SQL Editor (its HTTP request times out). Use
-- `supabase db push` or psql through the session pooler (port 5432).
--
-- Size: ~200 MB (the index stores its own copy of every vector), bringing the
-- database to ~433 MB of the free tier's 500 MB.

-- The server-wide default statement_timeout is 2 minutes; the build needs more.
SET statement_timeout = 0;

-- Enough memory to build the whole graph (~170 MB) in memory.
SET maintenance_work_mem = '256MB';

-- A parallel build puts the graph in shared memory (/dev/shm), which is too
-- small on this instance ("could not resize shared memory segment ... No space
-- left on device"), so build in a single process.
SET max_parallel_maintenance_workers = 0;

-- vector_cosine_ops matches the <=> (cosine distance) operator used in
-- match_sentences(); an index with a different operator class would not be used.
CREATE INDEX IF NOT EXISTS sentence_embeddings_embedding_idx
ON sentence_embeddings
USING hnsw (embedding vector_cosine_ops);

RESET max_parallel_maintenance_workers;
RESET maintenance_work_mem;
RESET statement_timeout;
