"""Keyword-based filter stage for paper filtering."""

from typing import List, Tuple
from src.models.responses import PaperItem, FilterCriteria


class KeywordFilterStage:
    """Filter papers based on keyword, author, and date criteria."""

    def is_applicable(self, criteria: FilterCriteria) -> bool:
        return bool(
            criteria.keywords
            or criteria.exclude_keywords
            or criteria.authors
            or criteria.has_pdf
            or criteria.min_date is not None
        )

    async def filter(
        self, papers: List[PaperItem], criteria: FilterCriteria
    ) -> Tuple[List[PaperItem], List[str]]:
        if not self.is_applicable(criteria):
            return papers, []

        filtered = []
        messages = []

        for paper in papers:
            if criteria.exclude_keywords and self._should_exclude(
                paper, criteria.exclude_keywords
            ):
                messages.append(
                    f"Excluded: '{paper.title[:50]}...' (matched exclude keyword)"
                )
                continue

            if criteria.keywords and not self._matches_keywords(
                paper, criteria.keywords
            ):
                messages.append(
                    f"Filtered: '{paper.title[:50]}...' "
                    f"(no matching keywords)"
                )
                continue

            if criteria.authors and not self._matches_authors(paper, criteria.authors):
                messages.append(
                    f"Filtered: '{paper.title[:50]}...' (no matching authors)"
                )
                continue

            if criteria.has_pdf and not self._has_pdf(paper):
                messages.append(f"Filtered: '{paper.title[:50]}...' (no PDF available)")
                continue

            if criteria.min_date is not None and not self._meets_date_requirement(
                paper, criteria.min_date
            ):
                messages.append(
                    f"Filtered: '{paper.title[:50]}...' "
                    f"(published before {criteria.min_date})"
                )
                continue

            filtered.append(paper)

        return filtered, messages

    def _matches_keywords(self, paper: PaperItem, keywords: List[str]) -> bool:
        text = (paper.title + " " + paper.abstract).lower()
        return any(keyword.lower() in text for keyword in keywords)

    def _matches_authors(self, paper: PaperItem, authors: List[str]) -> bool:
        paper_authors = [author.lower() for author in paper.authors]
        return any(
            any(auth.lower() in paper_auth for paper_auth in paper_authors)
            for auth in authors
        )

    def _should_exclude(self, paper: PaperItem, exclude_keywords: List[str]) -> bool:
        text = (paper.title + " " + paper.abstract).lower()
        return any(keyword.lower() in text for keyword in exclude_keywords)

    def _has_pdf(self, paper: PaperItem) -> bool:
        return paper.pdf_url is not None

    def _meets_date_requirement(self, paper: PaperItem, min_date) -> bool:
        if paper.published_date is None:
            return True
        return paper.published_date >= min_date
