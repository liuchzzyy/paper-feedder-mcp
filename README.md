# paper-feed

A modular Python framework for collecting, filtering, and exporting academic papers from RSS feeds and Gmail email alerts.

## Features

- **RSS Feed Source** — Reads feeds from OPML files (Nature, Science, ACS, Wiley, RSC, Elsevier, etc.). Concurrent fetching with deduplication and auto source-name detection.
- **Gmail Source** — Fetches Google Scholar alerts and journal TOC emails via EZGmail. Inline OAuth2 JSON credentials (env-only deployment).
- **Two-Stage Filtering** — Keyword OR first-pass (broad) + AI semantic second-pass (precise, via DeepSeek/OpenAI-compatible API).
- **Metadata Enrichment** — CrossRef and OpenAlex API clients for DOI lookup, author/date/abstract completion.
- **Export Adapters** — JSON file export with metadata. Zotero API export (optional dependency).
- **CLI** — `paper-feed fetch | filter | enrich | export` pipeline commands.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (package manager)

## Installation

```bash
# Clone and install
cd paper-feed
uv sync

# With development tools (pytest, ruff, ty)
uv sync --group dev
```

## Quick Start

### CLI Pipeline

```bash
# 1. Fetch papers from RSS feeds
paper-feed fetch --source rss --limit 200 --output raw.json

# 2. Keyword filter (OR logic — any match passes)
paper-feed filter --input raw.json --output filtered.json \
    --keywords battery zinc electrolyte operando

# 3. AI semantic filter (requires OPENAI_API_KEY in .env)
paper-feed filter --input filtered.json --output ai_filtered.json \
    --keywords battery zinc --ai

# 4. Enrich with CrossRef + OpenAlex metadata
paper-feed enrich --input ai_filtered.json --output enriched.json --source all

# 5. Export
paper-feed export --input enriched.json --output final.json --format json --include-metadata
```

### Python API

```python
import asyncio
from paper_feed import RSSSource, FilterPipeline, JSONAdapter, FilterCriteria

async def main():
    # Fetch
    source = RSSSource("feeds/RSS_official.opml")
    papers = await source.fetch_papers(limit=50)

    # Filter (keywords use OR logic)
    criteria = FilterCriteria(keywords=["battery", "electrode"])
    result = await FilterPipeline().filter(papers, criteria)

    # Export
    await JSONAdapter().export(result.papers, "papers.json", include_metadata=True)

asyncio.run(main())
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Key settings:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for AI filtering (DeepSeek/OpenAI-compatible) |
| `OPENAI_BASE_URL` | Custom API base URL (e.g. `https://api.deepseek.com/v1`) |
| `RESEARCH_PROMPT` | Natural language research interests for AI filtering |
| `GMAIL_TOKEN_JSON` | Gmail OAuth2 token (inline JSON, required for Gmail source) |
| `GMAIL_CREDENTIALS_JSON` | Gmail OAuth2 credentials (inline JSON, required for Gmail source) |
| `POLITE_POOL_EMAIL` | Email for CrossRef/OpenAlex polite pool access |
| `PAPER_FEED_OPML` | Path to OPML file with RSS feeds |

See `.env.example` for all available options.

## Project Structure

```
paper-feed/
├── src/paper_feed/
│   ├── core/           # Models, base classes, CLI, config
│   ├── sources/        # RSS, Gmail, CrossRef, OpenAlex
│   ├── filters/        # Keyword filter + AI filter pipeline
│   ├── ai/             # Keyword generator + AI filter stage
│   ├── adapters/       # JSON, Zotero export
│   └── utils/          # Text cleaning, DOI regex
├── tests/unit/         # 350 tests (pytest + pytest-asyncio)
├── feeds/              # Default OPML file
└── pyproject.toml      # Project configuration
```

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check

# Type check
uv run ty check
```

## License

MIT
