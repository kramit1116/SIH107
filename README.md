<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/bis-saarthi-logo-dark.webp" />
    <img src="docs/bis-saarthi-logo.webp" alt="BIS Saarthi" width="360" />
  </picture>
</p>

# BIS Saarthi: BIS Standards & Services Assistant

A conversational assistant for Indian Standards and BIS services (SIH problem
statement 26107). Ask in English or Hindi; every factual answer cites the
official BIS page, product list, lab directory or standard it came from.

What it answers, and from which source:

| Need | Source in the knowledge base |
|---|---|
| Which Indian Standard applies to a product | 24k-row BIS standards catalogue, product manuals, compulsory product lists |
| Is certification compulsory, under which QCO and scheme | BIS lists for Scheme-I (ISI mark), Scheme-II (CRS), Scheme-IV and upcoming QCOs |
| How to get a licence / CRS registration, fees, timelines | BIS product certification FAQ, Grant of Licence guidelines, CRS portal pages |
| Hallmarking, HUID, BIS Care app, complaints | BIS hallmarking, consumer and app pages |
| Which labs can test my product, near me | BIS LIMS (search by IS number) joined with the Group-1 recognised lab list |
| Management systems, FMCS, training, Standards Clubs | BIS pages for each scheme |
| Hindi questions and follow-ups | Query rewritten to English for search; answer in the user's language |

Full paid standard text is never ingested or reproduced.

## How an answer is made

1. **Rewrite** (only for Hindi or follow-up questions): a small model turns the
   message into a standalone English search query, using the conversation.
2. **Retrieve**: SQLite FTS5/BM25 over the corpus plus the catalogue, with exact
   IS-number boosting. Official guidance pages and compulsory lists rank above
   gazette schedules; duplicate passages are removed.
3. **Add context**: lab questions get live BIS LIMS rows for the product's IS
   number (cached in `data/lims_labs.json`); a compulsory-list hit brings the
   matching scheme's process page so "next steps" are grounded.
4. **Generate**: the LLM answers from that evidence only, citing each fact as
   `[Source N]`.
5. **Verify**: every IS number must be supported by a source cited in the same
   sentence, clause numbers must appear in a cited excerpt, and a title-only
   catalogue record cannot carry legal claims. One repair attempt, then a
   refusal that lists related documents.
6. **Cite**: markers are renumbered `[1]..[n]` and the response carries exactly
   the cited sources. The UI shows them as clickable superscripts and chips.

If the model is rate limited the reply is a retryable `model_busy` state;
if it is missing, `model_unavailable`. Retrieved text is never shown as an
answer on its own.

## Run it

Backend (Python 3.10+):

```
pip install -r requirements-dev.txt
cp .env.example .env            # choose one LLM provider block, add the key
set -a; source .env; set +a
PYTHONPATH=src python scripts/import_rag_corpus.py \
  --corpus new_data/bis-rag-text-corpus-2026-09-18 --db kb/bis_rag.db
PYTHONPATH=src uvicorn bis_assistant.server:app --port 8000
```

The import prints `imported 359 documents ... knowledge: 19 pages, ~400 chunks`.
Re-run it whenever `data/knowledge/` changes.

UI (React + TypeScript, in `ui/`):

```
cd ui && npm install && npm run dev   # http://127.0.0.1:5173, /api proxied to :8000
```

The empty screen shows one example question per kind of user. Answers carry
numbered citations; clicking one opens the source excerpt and its official link.

## Configuration

Settings live in `config.yaml`; `BIS_<SECTION>_<KEY>` environment variables win.

| Variable | Default | Meaning |
|---|---|---|
| `BIS_LLM_PROVIDER` | `openai-compatible` | `openai-compatible`, `ollama`, `gemini` or `anthropic` |
| `BIS_LLM_MODEL` | empty | model id; empty means the chatbot is offline |
| `BIS_LLM_API_KEY` | empty | cloud API key (keep it in `.env`) |
| `BIS_LLM_FALLBACK_API_KEYS` | empty | comma-separated backup keys, tried in order after the primary fails; rejected/rate-limited keys immediately move to the next key |
| `BIS_LLM_BASE_URL` | provider default | e.g. `https://api.groq.com/openai/v1` |
| `BIS_LLM_FALLBACK_MODEL` | empty | same-provider model used when the main one is rate limited or failing |
| `BIS_LLM_UTILITY_MODEL` | empty | small model for query rewrites and chat titles |
| `BIS_LLM_TIMEOUT_S` | `20` | request timeout |
| `BIS_RAG_DB_PATH` | `kb/bis_rag.db` | SQLite index built by the import script |
| `BIS_RAG_TOP_K` | `5` | evidence passages per question |
| `BIS_LIMS_LIVE` | `1` | `0` = lab lookups use only the committed cache |

Recommended Groq setup for demos (separate rate-limit buckets):
`BIS_LLM_MODEL=qwen/qwen3.8-27b`, `BIS_LLM_FALLBACK_MODEL=openai/gpt-oss-120b`,
`BIS_LLM_UTILITY_MODEL=openai/gpt-oss-20b`.

Optional dense retrieval: install `sentence-transformers` (CPU torch first) and
pass `--embedding-model BAAI/bge-small-en-v1.5` to the import script. Without
it, search stays lexical. Models are only loaded from the local cache at runtime.

## Data

- `new_data/bis-rag-text-corpus-2026-09-18/`: 359 BIS gazette notifications and
  product manuals (extracted text) plus the 24k-row standards catalogue.
- `data/knowledge/generated/`: official BIS web pages and compulsory-product
  tables, fetched by `python scripts/build_knowledge.py`. Each section keeps
  the URL it came from. Re-run the script when BIS updates its pages and
  review the diff.
- `data/knowledge/curated/`: short pages assembled from the same official
  text for topics spread over several pages (licence steps, finding your
  standard, marks and schemes, Standards Clubs).
- `data/recognised_labs.json`: BIS Group-1 recognised labs with state and OSL code.
- `data/lims_labs.json`: cached LIMS lab lists for common standards
  (`PYTHONPATH=src python scripts/prefetch_lims.py [IS numbers]`).

## Demo questions

- Which Indian Standard applies to steel water bottles, and is BIS certification compulsory?
- I make LED bulbs. Do I need CRS registration? (then: "what are the steps?")
- Suggest BIS recognised labs in Pune to test two-wheeler helmets
- How do I verify the HUID on my gold jewellery?
- How do I complain about a fake ISI mark?
- प्रेशर कुकर के लिए कौन सा मानक अनिवार्य है?
- Is ISI mark mandatory for pressure cookers? (then: "which labs can test it?", "what is the fee?")

## Tests

```
PYTHONPATH=src python -m pytest tests/ -q        # unit + contract tests, no network
cd ui && npm run typecheck && npm run test:contract && npm run build
BIS_EVAL_ALLOW_ENV=1 BIS_EVAL_PAUSE_S=6 PYTHONPATH=src python eval/run_groq_50.py   # live model
```

The UI contract fixtures in `ui/tests/fixtures/` are recorded backend responses.

## Other endpoints

The FastAPI server also provides server-side threads with owner tokens,
feedback, consent, erasure and export (`/me`), speech-to-text (`/transcribe`),
chat titles (`/title`), Prometheus metrics (`/metrics`) and a two-person
review flow for knowledge-base diffs (`/kb/diff`, `/kb/publish`).
