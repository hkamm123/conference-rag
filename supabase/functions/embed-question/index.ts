// embed-question: turns a user's question into an embedding vector.
// The OpenAI key stays here on the server; the browser only ever sees the vector.

// Type definitions for the Supabase Edge Runtime (Deno.serve, Deno.env, ...).
import "@supabase/functions-js/edge-runtime.d.ts";
import { withSupabase } from "@supabase/server";

// Must match the model in scripts/04_embed_data.py, or question vectors and
// sentence vectors won't be comparable.
const EMBEDDING_MODEL = "text-embedding-3-small";

// Each call costs money, so cap how much text one request can send to OpenAI.
const MAX_QUESTION_LENGTH = 1000;

function errorResponse(status: number, message: string): Response {
  // app.js reads `error` from failed responses.
  return Response.json({ error: message }, { status });
}

export default {
  // auth: "user" makes withSupabase verify the caller's JWT against the
  // project's public signing keys (SUPABASE_JWKS) before this handler runs.
  // Logged-out callers get a 401 and never reach OpenAI. withSupabase also
  // answers CORS preflight (OPTIONS) requests and adds CORS headers to every
  // response, including errors.
  fetch: withSupabase({ auth: "user" }, async (req, _ctx) => {
    if (req.method !== "POST") {
      return errorResponse(405, "Use POST");
    }

    let question: unknown;
    try {
      ({ question } = await req.json());
    } catch {
      return errorResponse(400, "Request body must be JSON");
    }
    if (typeof question !== "string" || question.trim() === "") {
      return errorResponse(400, "Missing or invalid 'question' field");
    }
    if (question.length > MAX_QUESTION_LENGTH) {
      return errorResponse(
        400,
        `Question must be at most ${MAX_QUESTION_LENGTH} characters`,
      );
    }

    // Set with `supabase secrets set --env-file supabase/functions/.env`.
    const openaiKey = Deno.env.get("OPENAI_API_KEY");
    if (!openaiKey) {
      return errorResponse(500, "OPENAI_API_KEY is not set");
    }

    try {
      const openaiResponse = await fetch("https://api.openai.com/v1/embeddings", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${openaiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model: EMBEDDING_MODEL, input: question.trim() }),
      });

      if (!openaiResponse.ok) {
        // Full details go to the function logs (Dashboard → Edge Functions →
        // embed-question → Logs); the browser gets a generic message.
        console.error(
          "OpenAI error",
          openaiResponse.status,
          await openaiResponse.text(),
        );
        return errorResponse(502, "Embedding service failed");
      }

      const result = await openaiResponse.json();
      return Response.json({ embedding: result.data[0].embedding });
    } catch (err) {
      console.error("Could not reach OpenAI", err);
      return errorResponse(502, "Embedding service unavailable");
    }
  }),
};
