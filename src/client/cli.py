"""Command-line interface for paper-feedder-mcp."""

import argparse
import asyncio
import json
import logging
import sys
from datetime import date
from pathlib import Path
import shutil
from typing import List

from src.config.settings import (
    get_openai_config,
    get_rss_config,
    get_zotero_config,
)
from src.models.responses import FilterCriteria, FilterResult, PaperItem
from src.server import serve

logger = logging.getLogger(__name__)


# -------------------- Helpers --------------------


def _load_papers(path: str) -> List[PaperItem]:
    filepath = Path(path)
    if not filepath.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        raw = json.loads(filepath.read_text(encoding="utf-8"))
        return [PaperItem(**item) for item in raw]
    except Exception as exc:
        print(
            f"Error loading papers from {path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def _save_papers(papers: List[PaperItem], path: str) -> None:
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = [p.model_dump() for p in papers]
    filepath.write_text(
        json.dumps(data, default=str, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# -------------------- Handlers --------------------


async def _handle_fetch(args: argparse.Namespace) -> None:
    since = None
    if args.since:
        since = date.fromisoformat(args.since)

    if args.source == "rss":
        from src.sources.rss import RSSSource

        opml_path = args.opml
        if not opml_path:
            opml_path = get_rss_config()["opml_path"]

        source = RSSSource(opml_path)
        papers = await source.fetch_papers(limit=args.limit, since=since)
    elif args.source == "gmail":
        from src.sources.gmail import GmailSource

        query = args.query
        source = GmailSource(query=query)
        papers = await source.fetch_papers(limit=args.limit, since=since)
    else:
        print(
            f"Error: unknown source: {args.source}",
            file=sys.stderr,
        )
        sys.exit(1)

    _save_papers(papers, args.output)
    print(f"Fetched {len(papers)} papers -> {args.output}")


async def _handle_filter(args: argparse.Namespace) -> None:
    from src.filters.pipeline import FilterPipeline

    papers = _load_papers(args.input)

    min_date = None
    if args.min_date:
        min_date = date.fromisoformat(args.min_date)

    criteria = FilterCriteria(
        keywords=args.keywords or [],
        exclude_keywords=args.exclude or [],
        authors=args.authors or [],
        min_date=min_date,
        has_pdf=args.has_pdf,
    )

    llm_client = None
    if args.ai:
        from openai import OpenAI

        config = get_openai_config()
        api_key = config.get("api_key")
        if api_key:
            kwargs = {"api_key": api_key}
            base_url = config.get("base_url")
            if base_url:
                kwargs["base_url"] = base_url
            llm_client = OpenAI(**kwargs)
        else:
            print(
                "Warning: --ai requested but OPENAI_API_KEY not set. "
                "Skipping AI filter.",
                file=sys.stderr,
            )

    pipeline = FilterPipeline(llm_client=llm_client)
    result: FilterResult = await pipeline.filter(papers, criteria)

    _save_papers(result.papers, args.output)
    print(
        f"Filtered: {result.passed_count} passed, "
        f"{result.rejected_count} rejected "
        f"(from {result.total_count} total) -> {args.output}"
    )


async def _handle_export(args: argparse.Namespace) -> None:
    papers = _load_papers(args.input)

    if args.format == "json":
        from src.adapters.json import JSONAdapter

        adapter = JSONAdapter()
        await adapter.export(
            papers,
            args.output,
            include_metadata=args.include_metadata,
        )
    elif args.format == "zotero":
        from src.adapters.zotero import ZoteroAdapter

        zotero_config = get_zotero_config()
        adapter = ZoteroAdapter(
            library_id=zotero_config["library_id"],
            api_key=zotero_config["api_key"],
            library_type=zotero_config.get("library_type", "user"),
        )
        await adapter.export(papers)
        _delete_output_dir()
    else:
        print(
            f"Error: unknown format: {args.format}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Exported {len(papers)} papers ({args.format}) -> {args.output}")


async def _handle_enrich(args: argparse.Namespace) -> None:
    papers = _load_papers(args.input)

    use_crossref = args.source in ("crossref", "all")
    use_openalex = args.source in ("openalex", "all")

    semaphore = asyncio.Semaphore(args.concurrency)

    crossref_client = None
    openalex_client = None
    if use_crossref:
        from src.sources.crossref import CrossrefClient

        crossref_client = CrossrefClient()
    if use_openalex:
        from src.sources.openalex import OpenAlexClient

        openalex_client = OpenAlexClient()

    async def _enrich_one(paper: PaperItem) -> PaperItem:
        async with semaphore:
            result = paper

            if use_crossref:
                assert crossref_client is not None
                result = await crossref_client.enrich_paper(result)

            if use_openalex:
                assert openalex_client is not None
                result = await openalex_client.enrich_paper(result)

            return result

    try:
        tasks = [_enrich_one(p) for p in papers]
        results = await asyncio.gather(*tasks)
    finally:
        if crossref_client is not None:
            await crossref_client.close()
        if openalex_client is not None:
            await openalex_client.close()

    final_papers = [p for p in results if p is not None]

    _save_papers(list(final_papers), args.output)

    enriched_count = sum(1 for orig, enr in zip(papers, final_papers) if orig != enr)
    print(f"Enriched {enriched_count}/{len(papers)} papers -> {args.output}")


def _delete_output_dir() -> None:
    output_dir = Path("output")
    removed_any = False
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
        removed_any = True

    # Clean up common intermediate files in cwd (if they exist outside output/)
    intermediate_files = [
        "raw.json",
        "filtered.json",
        "enriched.json",
        "export.json",
        "gmail.json",
        "zotero.json",
    ]
    for filename in intermediate_files:
        candidate = Path(filename)
        if candidate.exists() and candidate.is_file():
            candidate.unlink()
            removed_any = True

    if removed_any:
        print("Deleted output/ directory and intermediate files.")
    else:
        print("No output/ directory or intermediate files to delete.")


async def _handle_delete(_args: argparse.Namespace) -> None:
    _delete_output_dir()


# -------------------- CLI Setup --------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-feedder-mcp",
        description="Collect, filter, enrich, and serve papers via MCP.",
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    subparsers.add_parser(
        "serve", help="Run MCP server over stdio (default)"
    )

    fetch_parser = subparsers.add_parser(
        "fetch", help="Fetch papers from a source"
    )
    fetch_parser.add_argument(
        "-s",
        "--source",
        choices=["rss", "gmail"],
        default="rss",
        help="Data source (default: rss)",
    )
    fetch_parser.add_argument(
        "--opml",
        "--rss-feeds",
        dest="opml",
        help="OPML file path (for RSS source)",
    )
    fetch_parser.add_argument(
        "-q",
        "--query",
        "--gmail-query",
        dest="query",
        help="Gmail search query (for Gmail source)",
    )
    fetch_parser.add_argument(
        "-n",
        "--limit",
        "--max-papers",
        dest="limit",
        type=int,
        help="Maximum number of papers to fetch",
    )
    fetch_parser.add_argument(
        "--since",
        "--from-date",
        dest="since",
        help="Only fetch papers since date (YYYY-MM-DD)",
    )
    fetch_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSON file path",
    )

    filter_parser = subparsers.add_parser(
        "filter", help="Filter papers by criteria"
    )
    filter_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input JSON file with papers",
    )
    filter_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSON file path",
    )
    filter_parser.add_argument(
        "-k",
        "--keywords",
        nargs="+",
        help="Required keywords (OR logic, first-pass filter)",
    )
    filter_parser.add_argument(
        "-x",
        "--exclude",
        "--exclude-keywords",
        dest="exclude",
        nargs="+",
        help="Exclude keywords (NOT logic)",
    )
    filter_parser.add_argument(
        "-a",
        "--authors",
        nargs="+",
        help="Author filter (OR logic)",
    )
    filter_parser.add_argument(
        "--min-date",
        "--after",
        "--from",
        dest="min_date",
        help="Minimum publication date (YYYY-MM-DD)",
    )
    filter_parser.add_argument(
        "--pdf",
        "--has-pdf",
        "--require-pdf",
        dest="has_pdf",
        action="store_true",
        help="Require PDF availability",
    )
    filter_parser.add_argument(
        "--ai",
        "--semantic",
        "--use-ai",
        dest="ai",
        action="store_true",
        help="Enable AI-powered relevance filtering",
    )

    export_parser = subparsers.add_parser(
        "export", help="Export papers to a format"
    )
    export_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input JSON file with papers",
    )
    export_parser.add_argument(
        "-f",
        "--format",
        choices=["json", "zotero"],
        default="json",
        help="Export format (default: json)",
    )
    export_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file path",
    )
    export_parser.add_argument(
        "-m",
        "--metadata",
        "--include-metadata",
        "--with-metadata",
        dest="include_metadata",
        action="store_true",
        help="Include metadata in export",
    )

    enrich_parser = subparsers.add_parser(
        "enrich",
        help="Enrich papers with CrossRef/OpenAlex metadata",
    )
    enrich_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input JSON file with papers",
    )
    enrich_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSON file path",
    )
    enrich_parser.add_argument(
        "--api",
        "--source",
        "--provider",
        dest="source",
        choices=["crossref", "openalex", "all"],
        default="all",
        help="Enrichment API provider (default: all)",
    )
    enrich_parser.add_argument(
        "-j",
        "--jobs",
        "--concurrency",
        "--parallel",
        dest="concurrency",
        type=int,
        default=5,
        help="Max concurrent API requests (default: 5)",
    )

    subparsers.add_parser(
        "delete",
        help="Delete output/ directory",
    )

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args()

    if not args.command or args.command == "serve":
        try:
            asyncio.run(serve())
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(130)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            logger.debug("Full traceback:", exc_info=True)
            sys.exit(1)
        return

    handlers = {
        "fetch": _handle_fetch,
        "filter": _handle_filter,
        "export": _handle_export,
        "enrich": _handle_enrich,
        "delete": _handle_delete,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        asyncio.run(handler(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)
