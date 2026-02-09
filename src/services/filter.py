"""Filtering service wrapping keyword and AI filtering."""

import json
from datetime import date
from typing import List, Optional

from openai import OpenAI

from src.ai.keyword_generator import KeywordGenerator
from src.config.settings import get_openai_config
from src.filters.ai_filter import AIFilterStage
from src.filters.pipeline import FilterPipeline
from src.models.responses import FilterCriteria, FilterResult, PaperItem


def _load_papers_json(papers_json: str) -> List[PaperItem]:
    data = json.loads(papers_json)
    if not isinstance(data, list):
        raise ValueError("papers_json must be a JSON array of paper objects")
    return [PaperItem(**item) for item in data]


class FilterService:
    """Service for filtering papers."""

    async def filter_keywords(
        self,
        papers_json: str,
        keywords: List[str],
        exclude: Optional[List[str]] = None,
        authors: Optional[List[str]] = None,
        min_date: Optional[date] = None,
        has_pdf: bool = False,
    ) -> FilterResult:
        papers = _load_papers_json(papers_json)
        criteria = FilterCriteria(
            keywords=keywords,
            exclude_keywords=exclude or [],
            authors=authors or [],
            min_date=min_date,
            has_pdf=has_pdf,
        )
        pipeline = FilterPipeline(llm_client=None)
        return await pipeline.filter(papers, criteria)

    async def filter_ai(
        self,
        papers_json: str,
        research_prompt: Optional[str] = None,
    ) -> FilterResult:
        papers = _load_papers_json(papers_json)
        criteria = FilterCriteria()

        llm_client: Optional[OpenAI] = None
        config = get_openai_config()
        api_key = config.get("api_key")
        if api_key:
            kwargs = {"api_key": api_key}
            base_url = config.get("base_url")
            if base_url:
                kwargs["base_url"] = base_url
            llm_client = OpenAI(**kwargs)

        ai_stage = AIFilterStage(openai_client=llm_client)
        relevant, messages = await ai_stage.filter(
            papers, criteria, research_prompt=research_prompt
        )

        return FilterResult(
            papers=relevant,
            total_count=len(papers),
            passed_count=len(relevant),
            rejected_count=len(papers) - len(relevant),
            filter_stats={"ai_filter": {"messages": messages}},
        )

    async def generate_keywords(
        self, research_prompt: Optional[str] = None
    ) -> List[str]:
        generator = KeywordGenerator()
        return await generator.extract_keywords(research_prompt=research_prompt)
