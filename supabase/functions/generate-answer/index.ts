// generate-answer: the "G" in RAG. Given a question and the IDs of the talks
// that semantic search retrieved, it loads those talks from the database and
// asks GPT to answer using only them.
//
// The browser sends talk IDs, never talk text. The text comes from our own
// database, so a caller can't slip arbitrary "context" (or instructions) into
// the prompt and use this function as a general-purpose GPT proxy.

// Type definitions for the Supabase Edge Runtime (Deno.serve, Deno.env, ...).
import "@supabase/functions-js/edge-runtime.d.ts";
import { withSupabase } from "@supabase/server";

const CHAT_MODEL = "gpt-4o";

// Cost limits. app.js sends the top 3 talks. (The longest talk has 34 chunks,
// so 5 talks stays far below the Data API's 1,000-row limit.)
const MAX_QUESTION_LENGTH = 1000;
const MAX_TALKS = 5;
const MAX_ANSWER_TOKENS = 700;

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const SYSTEM_PROMPT =
  `You answer questions about General Conference talks from The Church of Jesus Christ of Latter-day Saints.

Rules:
- Use ONLY the talks provided in the user's message. Do not add facts from outside them.
- If the talks don't answer the question, say so plainly instead of guessing.
- Cite the talk(s) you draw from by title and speaker, for example: ("Title," Speaker).
- Write plain text with short paragraphs. Do not use Markdown (no **, #, or bullet symbols).`;

const NO_SOURCES_ANSWER =
  "I couldn't find any conference talks related to that question.";

type Talk = { title: string; speaker: string; chunks: string[] };

function errorResponse(status: number, message: string): Response {
  // app.js reads `error` from failed responses.
  return Response.json({ error: message }, { status });
}

// Scraped speaker names start with "By " and use a non-breaking space (U+00A0)
// after the first name, e.g. "By President Russell M. Nelson".
function cleanSpeaker(speaker: string): string {
  return speaker.replace(/\u00a0/g, " ").replace(/^By\s+/, "").trim();
}

export default {
  // auth: "user" verifies the caller's JWT before this handler runs, and gives
  // us ctx.supabase: a database client that acts as THAT user, so RLS on
  // talks and talk_chunks still applies to the queries below.
  fetch: withSupabase({ auth: "user" }, async (req, ctx) => {
    if (req.method !== "POST") {
      return errorResponse(405, "Use POST");
    }

    let body: { question?: unknown; talk_ids?: unknown };
    try {
      body = await req.json();
    } catch {
      return errorResponse(400, "Request body must be JSON");
    }

    const { question, talk_ids } = body;
    if (typeof question !== "string" || question.trim() === "") {
      return errorResponse(400, "Missing or invalid 'question' field");
    }
    if (question.length > MAX_QUESTION_LENGTH) {
      return errorResponse(
        400,
        `Question must be at most ${MAX_QUESTION_LENGTH} characters`,
      );
    }
    if (
      !Array.isArray(talk_ids) ||
      !talk_ids.every((id) => typeof id === "string" && UUID_PATTERN.test(id))
    ) {
      return errorResponse(400, "'talk_ids' must be an array of talk UUIDs");
    }
    const talkIds = [...new Set(talk_ids as string[])];
    if (talkIds.length > MAX_TALKS) {
      return errorResponse(400, `At most ${MAX_TALKS} talks per question`);
    }

    // Nothing retrieved (this is also what the page-load readiness check
    // sends): answer without paying for a GPT call.
    if (talkIds.length === 0) {
      return Response.json({ answer: NO_SOURCES_ANSWER });
    }

    // Load the talks' chunks in order, with each talk's details nested under
    // row.talks (PostgREST follows the talk_id foreign key). Select only the
    // columns we need: `embedding` would add ~30 KB of JSON per row for nothing.
    const { data: rows, error: dbError } = await ctx.supabase
      .from("talk_chunks")
      .select("talk_id, chunk_num, text, talks(title, speaker)")
      .in("talk_id", talkIds)
      .order("talk_id")
      .order("chunk_num");
    if (dbError) {
      console.error("Database error", dbError);
      return errorResponse(500, "Could not load the source talks");
    }

    const talks = new Map<string, Talk>();
    for (const row of rows ?? []) {
      let talk = talks.get(row.talk_id);
      if (!talk) {
        // Each chunk has exactly one talk (many-to-one), so PostgREST returns
        // `talks` as a single object. Without generated database types the
        // client can't know that and types it as an array.
        const info = row.talks as unknown as
          | { title: string; speaker: string | null }
          | null;
        talk = {
          title: info?.title ?? "Untitled talk",
          speaker: cleanSpeaker(info?.speaker ?? "Unknown"),
          chunks: [],
        };
        talks.set(row.talk_id, talk);
      }
      talk.chunks.push(row.text);
    }
    if (talks.size === 0) {
      return Response.json({ answer: NO_SOURCES_ANSWER });
    }

    // Keep search-rank order (talkIds is ordered best match first) and give
    // each talk a clear label so GPT can cite it.
    const context = talkIds
      .filter((id) => talks.has(id))
      .map((id, i) => {
        const talk = talks.get(id)!;
        return `[Talk ${i + 1}] "${talk.title}" by ${talk.speaker}\n${
          talk.chunks.join("\n\n")
        }`;
      })
      .join("\n\n---\n\n");

    // Set with `supabase secrets set --env-file supabase/functions/.env`.
    const openaiKey = Deno.env.get("OPENAI_API_KEY");
    if (!openaiKey) {
      return errorResponse(500, "OPENAI_API_KEY is not set");
    }

    try {
      const openaiResponse = await fetch(
        "https://api.openai.com/v1/chat/completions",
        {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${openaiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: CHAT_MODEL,
            messages: [
              { role: "system", content: SYSTEM_PROMPT },
              {
                role: "user",
                content: `Talks:\n\n${context}\n\n---\n\nQuestion: ${question.trim()}`,
              },
            ],
            // Low temperature keeps the answer close to the sources.
            temperature: 0.3,
            max_tokens: MAX_ANSWER_TOKENS,
          }),
        },
      );

      if (!openaiResponse.ok) {
        // Full details go to the function logs (Dashboard → Edge Functions →
        // generate-answer → Logs); the browser gets a generic message.
        console.error(
          "OpenAI error",
          openaiResponse.status,
          await openaiResponse.text(),
        );
        return errorResponse(502, "Answer service failed");
      }

      const result = await openaiResponse.json();
      return Response.json({ answer: result.choices[0].message.content });
    } catch (err) {
      console.error("Could not reach OpenAI", err);
      return errorResponse(502, "Answer service unavailable");
    }
  }),
};
