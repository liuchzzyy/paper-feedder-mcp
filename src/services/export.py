"""Export service wrapping adapters."""

from typing import List

from src.adapters.json import JSONAdapter
from src.models.responses import PaperItem


class ExportService:
    """Service for exporting papers."""

    async def export_json(
        self,
        papers: List[PaperItem],
        filepath: str,
        include_metadata: bool = True,
    ) -> None:
        adapter = JSONAdapter()
        await adapter.export(
            papers,
            filepath,
            include_metadata=include_metadata,
        )
