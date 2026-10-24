// embed-question: turns a user's question into an embedding vector.
// The OpenAI key stays here on the server; the browser only ever sees the vector.
//
// With { parse: true }, it first asks an LLM to pull speaker names and calling
// groups out of the question ("What have apostles said about grace?"). Those
// become exact SQL filters for match_chunks(), and only the remaining topic is
// embedded. Questions that name no speaker or calling are embedded unchanged.

// Type definitions for the Supabase Edge Runtime (Deno.serve, Deno.env, ...).
import "@supabase/functions-js/edge-runtime.d.ts";
import { withSupabase } from "@supabase/server";

// Must match the model in scripts/04_embed_data.py, or question vectors and
// chunk vectors won't be comparable.
const EMBEDDING_MODEL = "text-embedding-3-small";

// Small, fast model for turning the question into filters.
const PARSE_MODEL = "gpt-4o-mini";

// Each call costs money, so cap how much text one request can send to OpenAI.
const MAX_QUESTION_LENGTH = 1000;

// Must match the CHECK constraint on callings.calling_group.
const CALLING_GROUPS = [
  "first_presidency",
  "quorum_of_twelve",
  "seventy",
  "presiding_bishopric",
  "general_officers",
  "other",
] as const;

// How long a worker reuses the speaker list before reloading it.
const SPEAKER_CACHE_MS = 10 * 60 * 1000;

const TITLES = ["Elder", "President", "Sister", "Brother", "Bishop"] as const;

type Filters = {
  speakers: string[];
  calling_groups: string[];
  combine: "and" | "or";
};

// One person, with every way their name is stored on talks
// ("Elder Jeffrey R. Holland", "President Jeffrey R. Holland", ...).
export type Person = { variants: string[]; titles: Set<string>; words: string[] };

let speakerCache: { people: Person[]; loadedAt: number } | null = null;

function errorResponse(status: number, message: string): Response {
  // app.js reads `error` from failed responses.
  return Response.json({ error: message }, { status });
}

// POST to the OpenAI API with the server-side key. Throws on failure; the
// full error goes to the function logs, never to the browser. The response
// shape depends on the endpoint, so callers read the fields they need.
// deno-lint-ignore no-explicit-any
async function openai(path: string, body: unknown): Promise<any> {
  // Set with `supabase secrets set --env-file supabase/functions/.env`.
  const openaiKey = Deno.env.get("OPENAI_API_KEY");
  if (!openaiKey) throw new Error("OPENAI_API_KEY is not set");

  const response = await fetch(`https://api.openai.com/v1/${path}`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${openaiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`OpenAI ${path} ${response.status}: ${await response.text()}`);
  }
  return await response.json();
}

// "Presented by Elder Gérald Caussé" -> { title: "Elder", words: ["gerald", "causse"] }
export function splitName(name: string): { title: string; words: string[] } {
  let rest = name.replace(/^Presented by\s+/i, "").trim();
  let title = "";
  for (const t of TITLES) {
    if (rest.startsWith(`${t} `)) {
      title = t;
      rest = rest.slice(t.length + 1);
      break;
    }
  }
  const words = rest
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "") // strip accents
    .toLowerCase()
    .replace(/\./g, "")
    .split(/\s+/)
    .filter(Boolean);
  return { title, words };
}

// Group stored speaker names into people by their name without the title.
export function buildPeople(speakerNames: string[]): Person[] {
  const byName = new Map<string, Person>();
  for (const speaker of speakerNames) {
    const { title, words } = splitName(speaker);
    const key = words.join(" ");
    let person = byName.get(key);
    if (!person) {
      person = { variants: [], titles: new Set(), words };
      byName.set(key, person);
    }
    if (!person.variants.includes(speaker)) person.variants.push(speaker);
    person.titles.add(title);
  }
  return [...byName.values()];
}

async function loadPeople(
  // deno-lint-ignore no-explicit-any
  supabase: any,
): Promise<Person[]> {
  if (speakerCache && Date.now() - speakerCache.loadedAt < SPEAKER_CACHE_MS) {
    return speakerCache.people;
  }
  // One row per talk (501). The Data API returns at most 1,000 rows per
  // request, so this would need paging if the talks table grows past that.
  const { data, error } = await supabase.from("talks").select("speaker");
  if (error) throw new Error(`Loading speakers failed: ${error.message}`);

  const names = (data ?? []).map((row: { speaker: string | null }) => row.speaker)
    .filter(Boolean);
  speakerCache = { people: buildPeople(names), loadedAt: Date.now() };
  return speakerCache.people;
}

// All stored name variants for everyone a mention could refer to. A mention
// matches a person when all its words appear in their name (so "Holland"
// matches both Jeffrey R. and Matthew S. Holland); if nothing matches, the
// surname alone is tried. A title narrows the matches to people who use it,
// or else to people stored without a title (many women are), or else is
// ignored, so "Sister Johnson" finds Camille N. Johnson, not the Elder Johnsons.
export function matchSpeakers(
  mention: { title: string; name: string },
  people: Person[],
): string[] {
  const { words } = splitName(mention.name);
  if (words.length === 0) return [];
  const hasAll = (p: Person, ws: string[]) => ws.every((w) => p.words.includes(w));
  let matches = people.filter((p) => hasAll(p, words));
  if (matches.length === 0) {
    matches = people.filter((p) => hasAll(p, words.slice(-1)));
  }
  if (mention.title) {
    const titled = matches.filter((p) => p.titles.has(mention.title));
    const untitled = matches.filter((p) => p.titles.has(""));
    matches = titled.length ? titled : untitled.length ? untitled : matches;
  }
  return matches.flatMap((p) => p.variants);
}

const PARSE_INSTRUCTIONS =
  `You turn a question about General Conference talks into search filters. Return:

- speakers: each person the question names, as { title, name }. title is the title used in the question (Elder, President, Sister, Brother, Bishop) or "" if none; name is the name as written, without the title (e.g. "Holland", "Jeffrey R. Holland", "Oaks"). If the question refers to a specific person only by role (e.g. "the prophet"), give that person's name if you know it. Do not list people for a group reference like "apostles". [] if none.
- calling_groups: the groups the question limits speakers to, or [] if none:
  - first_presidency: the President of the Church and his counselors in the First Presidency
  - quorum_of_twelve: members of the Quorum of the Twelve Apostles, including its President and Acting President
  - seventy: General Authority Seventies, including the Presidency of the Seventy
  - presiding_bishopric: the Presiding Bishop and his counselors
  - general_officers: general presidencies of the Relief Society, Young Women, Young Men, Primary, and Sunday School
  - other: anyone else
  "Apostles" or "prophets, seers, and revelators" means BOTH first_presidency and quorum_of_twelve. "General authorities" means first_presidency, quorum_of_twelve, seventy, and presiding_bishopric.
- combine: "and" if a talk must match both a named speaker and a calling group ("What did President Oaks teach while in the First Presidency?"); "or" if either is enough ("What have Sister Johnson and the Seventy said about service?"). Use "and" if unsure or if only speakers or only groups are given.
- search_text: a natural question or phrase about the topic alone, with the speakers and groups removed. Examples:
  "What have apostles said about grace?" -> "What has been taught about grace?"
  "What did Elder Uchtdorf and Elder Bednar say about hope?" -> "What has been taught about hope?"
  "What did President Oaks teach while in the First Presidency about families?" -> "What has been taught about families?"
  If the question names no speaker or group, return it unchanged.`;

const FILTER_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["speakers", "calling_groups", "combine", "search_text"],
  properties: {
    speakers: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["title", "name"],
        properties: {
          title: { type: "string", enum: ["", ...TITLES] },
          name: { type: "string" },
        },
      },
    },
    calling_groups: {
      type: "array",
      items: { type: "string", enum: CALLING_GROUPS },
    },
    combine: { type: "string", enum: ["and", "or"] },
    search_text: { type: "string" },
  },
};

// Returns the filters (or null if the question names no speaker or calling)
// and the text to embed. The LLM only reads the question; matching names to
// the stored speakers is done in code, against the real data.
export async function parseQuestion(
  question: string,
  people: Person[],
): Promise<{ filters: Filters | null; searchText: string }> {
  const result = await openai("chat/completions", {
    model: PARSE_MODEL,
    temperature: 0,
    messages: [
      { role: "system", content: PARSE_INSTRUCTIONS },
      { role: "user", content: question },
    ],
    // Structured output: the reply must match FILTER_SCHEMA exactly.
    response_format: {
      type: "json_schema",
      json_schema: { name: "search_filters", strict: true, schema: FILTER_SCHEMA },
    },
  });
  const parsed = JSON.parse(result.choices[0].message.content);

  const filters: Filters = {
    speakers: [
      ...new Set<string>(
        // deno-lint-ignore no-explicit-any
        parsed.speakers.flatMap((m: any) => matchSpeakers(m, people)),
      ),
    ],
    calling_groups: [...new Set<string>(parsed.calling_groups)].filter((g) =>
      (CALLING_GROUPS as readonly string[]).includes(g)
    ),
    combine: parsed.combine === "or" ? "or" : "and",
  };
  if (filters.speakers.length === 0 && filters.calling_groups.length === 0) {
    return { filters: null, searchText: question };
  }
  const searchText = String(parsed.search_text ?? "").trim();
  return { filters, searchText: searchText || question };
}

export default {
  // auth: "user" makes withSupabase verify the caller's JWT against the
  // project's public signing keys (SUPABASE_JWKS) before this handler runs.
  // Logged-out callers get a 401 and never reach OpenAI. withSupabase also
  // answers CORS preflight (OPTIONS) requests and adds CORS headers to every
  // response, including errors. ctx.supabase queries the database as the
  // calling user, so RLS applies.
  fetch: withSupabase({ auth: "user" }, async (req, ctx) => {
    if (req.method !== "POST") {
      return errorResponse(405, "Use POST");
    }

    let body: { question?: unknown; parse?: unknown };
    try {
      body = await req.json();
    } catch {
      return errorResponse(400, "Request body must be JSON");
    }
    const { question, parse } = body;
    if (typeof question !== "string" || question.trim() === "") {
      return errorResponse(400, "Missing or invalid 'question' field");
    }
    if (question.length > MAX_QUESTION_LENGTH) {
      return errorResponse(
        400,
        `Question must be at most ${MAX_QUESTION_LENGTH} characters`,
      );
    }

    let filters: Filters | null = null;
    let searchText = question.trim();
    if (parse === true) {
      try {
        const people = await loadPeople(ctx.supabase);
        ({ filters, searchText } = await parseQuestion(searchText, people));
      } catch (err) {
        // Parsing is an extra; if it fails, search the question unfiltered.
        console.error("Question parsing failed", err);
      }
    }

    try {
      const result = await openai("embeddings", {
        model: EMBEDDING_MODEL,
        input: searchText,
      });
      const embedding = result.data[0].embedding;
      return parse === true
        ? Response.json({ embedding, filters, search_text: searchText })
        : Response.json({ embedding });
    } catch (err) {
      // Full details go to the function logs (Dashboard → Edge Functions →
      // embed-question → Logs); the browser gets a generic message.
      console.error("Embedding failed", err);
      return errorResponse(502, "Embedding service failed");
    }
  }),
};
