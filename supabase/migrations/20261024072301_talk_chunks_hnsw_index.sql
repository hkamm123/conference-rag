-- HNSW vector index for semantic search (match_chunks).
--
-- Without it, every search compares the question to every chunk vector. With
-- the earlier sentence-level design (26k rows) that took ~10.5 s on the free
-- tier, past the authenticated role's 8 s statement_timeout, so semantic
-- search failed until this index existed.
--
-- Apply this AFTER the data is loaded (scripts/03-05). Building once over the
-- full table is much faster than updating the graph on every insert.
--
-- !! SLOW: building the index over the 26k sentence vectors took about
-- 3 minutes on the free tier (the ~9k paragraph chunks should take roughly a
-- third of that). Don't run it in the SQL Editor (its HTTP request times out).
-- Use `supabase db push` or psql through the session pooler (port 5432).
--
-- Size: the index stores its own copy of every vector (~6 KB each).

-- The server-wide default statement_timeout is 2 minutes; the build may need more.
SET statement_timeout = 0;

-- Enough memory to build the whole graph in memory.
SET maintenance_work_mem = '256MB';

-- A parallel build puts the graph in shared memory (/dev/shm), which is too
-- small on this instance ("could not resize shared memory segment ... No space
-- left on device"), so build in a single process.
SET max_parallel_maintenance_workers = 0;

-- vector_cosine_ops matches the <=> (cosine distance) operator used in
-- match_chunks(); an index with a different operator class would not be used.
CREATE INDEX IF NOT EXISTS talk_chunks_embedding_idx
ON talk_chunks
USING hnsw (embedding vector_cosine_ops);

RESET max_parallel_maintenance_workers;
RESET maintenance_work_mem;
RESET statement_timeout;
