# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file CLI that exports one ChatGPT conversation to Markdown. It calls the same
undocumented backend endpoint the ChatGPT web app uses (`GET /backend-api/conversation/<id>`),
using a bearer token or session cookie for auth. See README.md for full usage and auth setup.

## Commands

```bash
uv sync                       # install deps
uv run python export_chat.py <CONVERSATION_ID> --bearer "$CHATGPT_BEARER"
uv run pytest                 # run tests
uv run pytest test_smoke.py::test_extract_text_parts_string  # run a single test
```

No linter or formatter is configured in this repo.

## Architecture

Everything lives in `export_chat.py`:

- `fetch_conversation` — HTTP call to the backend endpoint, builds auth headers/cookies.
- `_collect_messages` — walks the conversation's `mapping` (a dict of tree nodes keyed by
  node id, each optionally holding a `message`), flattens it into `Message` objects, and
  sorts by `create_time`. This is **not** branch-aware — it takes every message in the
  mapping regardless of which branch is "active" in a conversation with edits/regenerations.
- `_extract_text_parts` — normalizes the several shapes `message.content` can take
  (`{"parts": [...]}`, `{"text": ...}`, plain string/list) into a single string.
- `_to_markdown` — renders the `Message` list into the final Markdown document.
- `main` — argument parsing and file output wiring.

`test_smoke.py` loads `export_chat.py` via `importlib` (not a package import) since the
module lives at the repo root with no package structure.
