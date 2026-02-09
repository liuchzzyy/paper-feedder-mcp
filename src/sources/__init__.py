"""Paper data sources (RSS, Gmail, CrossRef, OpenAlex, etc.)."""

from src.sources.opml import OPMLParser, parse_opml
from src.sources.rss import RSSSource
from src.sources.rss_parser import RSSParser
from src.sources.gmail import GmailSource
from src.sources.gmail_parser import GmailParser
from src.sources.crossref import CrossrefClient
from src.sources.openalex import OpenAlexClient

__all__ = [
    "RSSSource",
    "RSSParser",
    "OPMLParser",
    "parse_opml",
    "GmailSource",
    "GmailParser",
    "CrossrefClient",
    "OpenAlexClient",
]
