"""Zotero export adapter for paper-feedder-mcp."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.models.responses import ExportAdapter, PaperItem

try:
    from zotero_mcp.clients.zotero.api_client import ZoteroAPIClient
    from zotero_mcp.services.zotero.item_service import ItemService

    zotero_available = True
    _zotero_import_error: Exception | None = None
except Exception as exc:
    zotero_available = False
    _zotero_import_error = exc
    ZoteroAPIClient = None  # type: ignore[assignment,misc]
    ItemService = None  # type: ignore[assignment,misc]


class ZoteroAdapter(ExportAdapter):
    """Export papers to Zotero library via zotero-mcp."""

    adapter_name: str = "zotero"

    def __init__(
        self,
        library_id: str,
        api_key: str,
        library_type: str = "user",
    ):
        if not zotero_available or ZoteroAPIClient is None or ItemService is None:
            raise ImportError(
                "zotero-mcp is required for ZoteroAdapter. "
                "Install it with: uv pip install /path/to/zotero-mcp"
            ) from _zotero_import_error

        self.library_id = library_id
        self.api_key = api_key
        self.library_type = library_type

        self._api_client = ZoteroAPIClient(
            library_id=library_id,
            library_type=library_type,
            api_key=api_key,
        )
        self._item_service = ItemService(api_client=self._api_client)

    async def export(
        self,
        papers: List[PaperItem],
        collection_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not zotero_available:
            raise ImportError(
                "zotero-mcp is required for ZoteroAdapter. "
                "Install it with: uv pip install /path/to/zotero-mcp"
            ) from _zotero_import_error

        success_count = 0
        failures = []
        collection_keys = [collection_id] if collection_id else None

        for paper in papers:
            try:
                zotero_item = self._paper_to_zotero_item(
                    paper, collection_keys=collection_keys
                )
                await self._item_service.create_item(zotero_item)
                success_count += 1
            except Exception as e:
                failures.append({"title": paper.title, "error": str(e)})

        return {
            "success_count": success_count,
            "total": len(papers),
            "failures": failures,
        }

    def _paper_to_zotero_item(
        self, paper: PaperItem, collection_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        creators = [
            {"creatorType": "author", "name": author} for author in paper.authors
        ]

        date_str = None
        if paper.published_date:
            date_str = paper.published_date.isoformat()

        access_date_str = None
        if paper.access_date:
            access_date_str = paper.access_date.isoformat()
        else:
            access_date_str = datetime.now().strftime("%Y-%m-%d")

        zotero_item: Dict[str, Any] = {
            "itemType": paper.item_type or "journalArticle",
            "title": paper.title,
            "creators": creators,
            "abstractNote": paper.abstract or None,
            "publicationTitle": paper.publication_title,
            "journalAbbreviation": paper.journal_abbreviation,
            "publisher": paper.publisher,
            "place": paper.place,
            "volume": paper.volume,
            "issue": paper.issue,
            "pages": paper.pages,
            "section": paper.section,
            "partNumber": paper.part_number,
            "partTitle": paper.part_title,
            "series": paper.series,
            "seriesTitle": paper.series_title,
            "seriesText": paper.series_text,
            "DOI": paper.doi,
            "citationKey": paper.citation_key,
            "url": paper.url,
            "accessDate": access_date_str,
            "PMID": paper.pmid,
            "PMCID": paper.pmcid,
            "ISSN": paper.issn,
            "archive": paper.archive,
            "archiveLocation": paper.archive_location,
            "shortTitle": paper.short_title,
            "language": paper.language,
            "libraryCatalog": paper.library_catalog,
            "callNumber": paper.call_number,
            "rights": paper.rights,
            "date": date_str,
        }

        if collection_keys:
            zotero_item["collections"] = collection_keys

        zotero_item = {k: v for k, v in zotero_item.items() if v is not None}

        return zotero_item
