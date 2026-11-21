from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests

from config import OPENALEX_API_KEY, OPENALEX_MAILTO, PER_PAGE, REQUEST_SLEEP_SEC, WORKS_SELECT


class OpenAlexRateLimitError(RuntimeError):
    """Consultas a /works bloqueadas temporariamente pela OpenAlex."""

    def __init__(self, retry_after_sec: int, url: str) -> None:
        self.retry_after_sec = retry_after_sec
        hours = retry_after_sec / 3600
        safe_url = url.split("api_key=")[0].rstrip("&?")
        super().__init__(
            f"OpenAlex retornou 429 (Too Many Requests). "
            f"Aguarde ~{retry_after_sec}s (~{hours:.1f} h) antes de retomar. URL: {safe_url}"
        )


def _short_openalex_id(url_or_id: str | None) -> str | None:
    if not url_or_id:
        return None
    s = str(url_or_id).rstrip("/")
    if "/" in s:
        return s.split("/")[-1]
    return s


class OpenAlexClient:
    BASE = "https://api.openalex.org"

    def __init__(
        self,
        mailto: str = OPENALEX_MAILTO,
        api_key: str = OPENALEX_API_KEY,
    ) -> None:
        self.mailto = mailto
        self.api_key = (api_key or "").strip()
        self.session = requests.Session()

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "mailto": self.mailto}
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{self.BASE}{path}"
        max_wait = 120  # não dormir horas sem avisar (Retry-After pode ser >10k s)
        for attempt in range(12):
            time.sleep(REQUEST_SLEEP_SEC)
            try:
                resp = self.session.get(url, params=params, timeout=90)
            except (requests.ConnectionError, requests.Timeout) as exc:
                # Queda de conexão no meio da paginação: esperar e repetir a mesma página
                wait = min(60, 2**attempt)
                print(f"[OpenAlex] erro de conexão ({exc.__class__.__name__}); nova tentativa em {wait}s", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code == 429:
                retry_raw = resp.headers.get("Retry-After", "0")
                try:
                    retry_after = int(retry_raw)
                except ValueError:
                    retry_after = 0
                if retry_after > max_wait:
                    raise OpenAlexRateLimitError(retry_after, resp.url)
                wait = retry_after or min(60, 2**attempt)
                print(
                    f"[OpenAlex] 429 — nova tentativa em {wait}s (tentativa {attempt + 1}/12)",
                    flush=True,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"OpenAlex: rate limit após várias tentativas ({url})")

    def search_institution_br(self, query: str) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        data = self._get(
            "/institutions",
            {
                "search": query,
                "filter": "country_code:br",
                "per-page": 5,
                "sort": "relevance_score:desc",
            },
        )
        return data.get("results") or []

    def iter_works(self, filter_expr: str) -> Iterator[dict[str, Any]]:
        cursor = "*"
        while cursor:
            data = self._get(
                "/works",
                {
                    "filter": filter_expr,
                    "per-page": PER_PAGE,
                    "cursor": cursor,
                    "select": WORKS_SELECT,
                },
            )
            for work in data.get("results") or []:
                yield work
            cursor = data.get("meta", {}).get("next_cursor")


def pick_best_institution(candidates: list[dict[str, Any]], ies_name: str) -> dict[str, Any] | None:
    if not candidates:
        return None
    name_upper = ies_name.upper()
    for c in candidates:
        if c.get("display_name", "").upper() in name_upper or name_upper in c.get("display_name", "").upper():
            return c
    return candidates[0]


def map_institutions(
    institutions_df: pd.DataFrame,
    cache_path: Path,
    client: OpenAlexClient,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in institutions_df.iterrows():
        ies = row["instituicao_ies"]
        sigla = row["sigla_ies"]
        candidates = client.search_institution_br(ies)
        if not candidates:
            candidates = client.search_institution_br(sigla)
        best = pick_best_institution(candidates, ies)
        if best:
            rows.append(
                {
                    "instituicao_ies": ies,
                    "sigla_ies": sigla,
                    "n_programas": row["n_programas"],
                    "openalex_id": _short_openalex_id(best.get("id")),
                    "openalex_name": best.get("display_name"),
                    "openalex_works_count": best.get("works_count"),
                    "match_status": "auto",
                }
            )
        else:
            rows.append(
                {
                    "instituicao_ies": ies,
                    "sigla_ies": sigla,
                    "n_programas": row["n_programas"],
                    "openalex_id": None,
                    "openalex_name": None,
                    "openalex_works_count": None,
                    "match_status": "not_found",
                }
            )
    out = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False, encoding="utf-8")
    return out


def append_works_jsonl(works: Iterator[dict[str, Any]], path: Path, seen_ids: set[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_new = 0
    with path.open("a", encoding="utf-8") as f:
        for work in works:
            wid = _short_openalex_id(work.get("id"))
            if not wid or wid in seen_ids:
                continue
            seen_ids.add(wid)
            f.write(json.dumps(work, ensure_ascii=False) + "\n")
            n_new += 1
    return n_new


def chunk_list(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
