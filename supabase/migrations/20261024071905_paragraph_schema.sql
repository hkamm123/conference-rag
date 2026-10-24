-- Paragraph-level chunks with talk details normalized into their own table.
--
-- Replaces the sentence-level design (one wide sentence_embeddings table with
-- the talk's title/speaker/calling repeated on every row). Talk details are
-- now stored once in `talks`; `talk_chunks` holds one row per paragraph-sized
-- chunk (short paragraphs are merged with their neighbors at import time).
--
-- Safe to re-run: every statement is IF [NOT] EXISTS or DROP-then-CREATE.
-- The HNSW vector index is a separate, later migration, because it should be
-- built after the data is loaded (scripts/03-05).

-- ============================================
-- Remove the sentence-level schema
-- ============================================
DROP FUNCTION IF EXISTS match_sentences(vector, int);
DROP TABLE IF EXISTS sentence_embeddings;

CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- talks: one row per conference talk
-- ============================================
CREATE TABLE IF NOT EXISTS talks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    speaker TEXT,
    calling TEXT,
    year INTEGER,
    season TEXT,
    url TEXT UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- talk_chunks: paragraph-sized pieces of each talk, with embeddings
-- ============================================
CREATE TABLE IF NOT EXISTS talk_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    talk_id UUID NOT NULL REFERENCES talks(id) ON DELETE CASCADE,
    chunk_num INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    -- One chunk per position in a talk. Its index starts with talk_id, so it
    -- also serves lookups by talk_id (Postgres doesn't index foreign keys
    -- automatically).
    UNIQUE (talk_id, chunk_num)
);

-- ============================================
-- Row Level Security: logged-in users can read; only service_role
-- (which bypasses RLS) can write.
-- ============================================
ALTER TABLE talks ENABLE ROW LEVEL SECURITY;
ALTER TABLE talk_chunks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow authenticated users to read" ON talks;
CREATE POLICY "Allow authenticated users to read"
ON talks FOR SELECT
TO authenticated
USING (true);

DROP POLICY IF EXISTS "Allow authenticated users to read" ON talk_chunks;
CREATE POLICY "Allow authenticated users to read"
ON talk_chunks FOR SELECT
TO authenticated
USING (true);

-- ============================================
-- Similarity search
-- ============================================
-- SECURITY INVOKER (the default): runs with the caller's permissions, so the
-- RLS policies above apply to searches too.
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1536),
    match_count int DEFAULT 20
)
RETURNS TABLE (
    id uuid,
    talk_id uuid,
    title text,
    speaker text,
    url text,
    text text,
    similarity float
)
LANGUAGE sql STABLE
SET search_path = public
AS $$
    SELECT
        c.id,
        c.talk_id,
        t.title,
        t.speaker,
        t.url,
        c.text,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM talk_chunks c
    JOIN talks t ON t.id = c.talk_id
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
$$;
