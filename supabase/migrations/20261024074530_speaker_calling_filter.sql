-- Speaker / calling filters for semantic search and RAG.
--
-- The embed-question Edge Function asks an LLM to pull speaker names and
-- calling groups out of a question ("What have apostles said about grace?"),
-- and match_chunks() applies them as exact SQL filters. Who gave a talk is
-- exact data, so it belongs in a WHERE clause, not in the embedding.

-- ============================================
-- callings: lookup table from each calling (as printed on the talk page) to
-- a calling group. The rule "which callings count as apostles" lives here
-- once, as data, instead of being repeated in queries. "Apostles" is not a
-- group of its own: the parser expands it to first_presidency +
-- quorum_of_twelve. A calling missing from this table counts as 'other'.
-- ============================================
CREATE TABLE IF NOT EXISTS callings (
    calling TEXT PRIMARY KEY,
    calling_group TEXT NOT NULL CHECK (calling_group IN (
        'first_presidency', 'quorum_of_twelve', 'seventy',
        'presiding_bishopric', 'general_officers', 'other'
    ))
);

ALTER TABLE callings ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow authenticated users to read" ON callings;
CREATE POLICY "Allow authenticated users to read"
ON callings FOR SELECT
TO authenticated
USING (true);

INSERT INTO callings (calling, calling_group) VALUES
    ('First Counselor in the First Presidency',                               'first_presidency'),
    ('President of The Church of Jesus Christ of Latter-day Saints',          'first_presidency'),
    ('Second Counselor in the First Presidency',                              'first_presidency'),
    ('First Counselor in the Primary General Presidency',                     'general_officers'),
    ('First Counselor in the Relief Society General Presidency',              'general_officers'),
    ('First Counselor in the Sunday School General Presidency',               'general_officers'),
    ('First Counselor in the Young Men General Presidency',                   'general_officers'),
    ('First Counselor in the Young Women General Presidency',                 'general_officers'),
    ('Primary General President',                                             'general_officers'),
    ('Recently Released First Counselor in the Young Men General Presidency', 'general_officers'),
    ('Recently Released Primary General President',                           'general_officers'),
    ('Relief Society General President',                                      'general_officers'),
    ('Second Counselor in the Primary General Presidency',                    'general_officers'),
    ('Second Counselor in the Relief Society General Presidency',             'general_officers'),
    ('Second Counselor in the Sunday School General Presidency',              'general_officers'),
    ('Second Counselor in the Young Men General Presidency',                  'general_officers'),
    ('Second Counselor in the Young Women General Presidency',                'general_officers'),
    ('Sunday School General President',                                       'general_officers'),
    ('Young Men General President',                                           'general_officers'),
    ('Young Women General President',                                         'general_officers'),
    ('Managing Director, Church Auditing Department',                         'other'),
    ('Member of the Meadow Wood Ward, Provo Utah Edgemont Stake',             'other'),
    ('Member of the Slate Canyon 14th Ward (Spanish), Provo Utah Stake',      'other'),
    ('First Counselor in the Presiding Bishopric',                            'presiding_bishopric'),
    ('Presiding Bishop',                                                      'presiding_bishopric'),
    ('Second Counselor in the Presiding Bishopric',                           'presiding_bishopric'),
    ('Acting President of the Quorum of the Twelve Apostles',                 'quorum_of_twelve'),
    ('Of the Quorum of the Twelve Apostles',                                  'quorum_of_twelve'),
    ('President of the Quorum of the Twelve Apostles',                        'quorum_of_twelve'),
    ('Emeritus Member of the Seventy',                                        'seventy'),
    ('Of the Presidency of the Seventy',                                      'seventy'),
    ('Of the Seventy',                                                        'seventy')
ON CONFLICT (calling) DO UPDATE SET calling_group = EXCLUDED.calling_group;

-- ============================================
-- match_chunks with optional filters
-- ============================================
-- filter_speakers:       exact talks.speaker values (NULL or empty = any)
-- filter_calling_groups: callings.calling_group values (NULL or empty = any)
-- filter_mode:           'and' (default) or 'or', used only when both are set
--
-- HNSW normally returns its ef_search (40) nearest candidates and only then
-- applies the WHERE clause, so a selective filter (one speaker is ~2% of the
-- chunks) could leave few or no results. hnsw.iterative_scan makes the index
-- keep scanning until LIMIT rows pass the filter. relaxed_order can return
-- them slightly out of order, so the outer query sorts them again.
DROP FUNCTION IF EXISTS match_chunks(vector, int);

-- pgvector's settings (hnsw.*) only exist once its library is loaded into the
-- session. Until then Postgres treats hnsw.iterative_scan as an unknown
-- placeholder, and a non-superuser may not put one in a function's SET clause
-- ("permission denied to set parameter"). Any vector operation loads it.
SELECT '[1]'::vector;

CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1536),
    match_count int DEFAULT 20,
    filter_speakers text[] DEFAULT NULL,
    filter_calling_groups text[] DEFAULT NULL,
    filter_mode text DEFAULT 'and'
)
RETURNS TABLE (
    id uuid,
    talk_id uuid,
    title text,
    speaker text,
    calling text,
    url text,
    text text,
    similarity float
)
LANGUAGE sql STABLE
SET search_path = public
SET hnsw.iterative_scan = relaxed_order
AS $$
    WITH nearest AS MATERIALIZED (
        SELECT
            c.id,
            c.talk_id,
            t.title,
            t.speaker,
            t.calling,
            t.url,
            c.text,
            c.embedding <=> query_embedding AS distance
        FROM talk_chunks c
        JOIN talks t ON t.id = c.talk_id
        LEFT JOIN callings cl ON cl.calling = t.calling
        WHERE CASE
            WHEN coalesce(cardinality(filter_speakers), 0) = 0
             AND coalesce(cardinality(filter_calling_groups), 0) = 0 THEN true
            WHEN coalesce(cardinality(filter_calling_groups), 0) = 0 THEN
                t.speaker = ANY (filter_speakers)
            WHEN coalesce(cardinality(filter_speakers), 0) = 0 THEN
                coalesce(cl.calling_group, 'other') = ANY (filter_calling_groups)
            WHEN filter_mode = 'or' THEN
                t.speaker = ANY (filter_speakers)
                OR coalesce(cl.calling_group, 'other') = ANY (filter_calling_groups)
            ELSE
                t.speaker = ANY (filter_speakers)
                AND coalesce(cl.calling_group, 'other') = ANY (filter_calling_groups)
        END
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_count
    )
    SELECT id, talk_id, title, speaker, calling, url, text, 1 - distance AS similarity
    FROM nearest
    ORDER BY distance;
$$;
