"""Tool name enumeration for paper-feedder-mcp MCP tools."""

from enum import Enum


class ToolName(str, Enum):
    """MCP tool names for paper-feedder-mcp."""

    FETCH_RSS = "paper-feedder-mcp_fetch_rss"
    FETCH_GMAIL = "paper-feedder-mcp_fetch_gmail"
    FILTER_KEYWORDS = "paper-feedder-mcp_filter_keywords"
    FILTER_AI = "paper-feedder-mcp_filter_ai"
    ENRICH = "paper-feedder-mcp_enrich"
    EXPORT_JSON = "paper-feedder-mcp_export_json"
    GENERATE_KEYWORDS = "paper-feedder-mcp_generate_keywords"
    SEARCH_CROSSREF = "paper-feedder-mcp_search_crossref"
    SEARCH_OPENALEX = "paper-feedder-mcp_search_openalex"
