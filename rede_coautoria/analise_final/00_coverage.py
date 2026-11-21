#!/usr/bin/env python3
"""Mede a completude do corpus contra o que os programas declararam à CAPES.

1. DOIs: que fração da produção declarada pelos programas de Computação (artigos em
   periódico e trabalhos em anais, 2018–2024) está no corpus? Para os DOIs ausentes,
   consulta a OpenAlex e classifica o motivo.
2. Docentes: que fração dos docentes CAPES foi identificada na rede de coautoria, e que
   fração dos autores do escopo são docentes.

Entradas: data/capes/producao_cc.csv, data/capes/docentes_openalex.csv, corpus OpenAlex,
output/colaboracao_cc_brasil.graphml.gz. Saída: analise_final/results/coverage.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.build_network import iter_works_jsonl
from src.openalex_client import OpenAlexClient, chunk_list

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_JSON = RESULTS_DIR / "coverage.json"
CS_FIELD_ID = "17"
DOIS_PER_REQUEST = 50


def normalize_doi(doi: str | None) -> str:
    doi = (doi or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi


def corpus_dois() -> set[str]:
    return {normalize_doi(w.get("doi")) for w in iter_works_jsonl(config.OBRAS_JSONL)} - {""}


def classify_missing(dois: list[str], scope_ids: set[str]) -> tuple[Counter, Counter]:
    """Consulta na OpenAlex os DOIs ausentes do corpus e devolve (motivos, campos fora de CS)."""
    client = OpenAlexClient()
    reasons: Counter = Counter()
    other_fields: Counter = Counter()
    # Vírgula ou barra vertical quebram a sintaxe de filtro da OpenAlex
    queryable = [d for d in dois if "," not in d and "|" not in d]
    reasons["doi_nao_consultavel"] = len(dois) - len(queryable)
    for batch in chunk_list(queryable, DOIS_PER_REQUEST):
        data = client._get(
            "/works",
            {
                "filter": "doi:" + "|".join(batch),
                "per-page": DOIS_PER_REQUEST,
                "select": "doi,publication_year,primary_topic,authorships",
            },
        )
        found = {normalize_doi(w.get("doi")): w for w in data.get("results") or []}
        for doi in batch:
            work = found.get(doi)
            if work is None:
                reasons["nao_esta_na_openalex"] += 1
                continue
            year = work.get("publication_year") or 0
            if not config.PUBLICATION_YEAR_MIN <= year <= config.PUBLICATION_YEAR_MAX:
                reasons["ano_fora_2018_2024"] += 1
                continue
            field = ((work.get("primary_topic") or {}).get("field") or {})
            field_id = str(field.get("id", "")).rstrip("/").split("/")[-1]
            if field_id != CS_FIELD_ID:
                reasons["topico_principal_fora_de_cs"] += 1
                other_fields[field.get("display_name") or "sem tópico"] += 1
                continue
            inst_ids = {
                str(x).split("/")[-1]
                for auth in work.get("authorships") or []
                for inst in auth.get("institutions") or []
                for x in [inst.get("id"), *(inst.get("lineage") or [])]
                if x
            }
            if not inst_ids & scope_ids:
                reasons["sem_afiliacao_ies_escopo"] += 1
                continue
            # Passa em todos os filtros mas não veio na coleta (ex.: mudança na OpenAlex)
            reasons["deveria_estar_no_corpus"] += 1
    return reasons, other_fields


def doi_coverage(scope_ids: set[str]) -> dict:
    production = pd.read_csv(config.DATA_DIR / "capes" / "producao_cc.csv", dtype=str).fillna("")
    in_corpus = corpus_dois()
    print(f"DOIs no corpus: {len(in_corpus)}", flush=True)

    result: dict = {"dois_no_corpus": len(in_corpus), "por_subtipo": {}}
    for subtype, group in production.groupby("SUBTIPO"):
        with_doi = group[group["DOI"] != ""]
        distinct = sorted(set(with_doi["DOI"]))
        present = [d for d in distinct if d in in_corpus]
        missing = [d for d in distinct if d not in in_corpus]
        by_year = {}
        for year, g in with_doi.groupby("AN_BASE_PRODUCAO"):
            year_dois = set(g["DOI"])
            by_year[year] = {
                "dois": len(year_dois),
                "no_corpus": len(year_dois & in_corpus),
            }
        print(f"{subtype}: consultando {len(missing)} DOIs ausentes na OpenAlex...", flush=True)
        reasons, other_fields = classify_missing(missing, scope_ids)
        result["por_subtipo"][subtype] = {
            "registros_capes": len(group),
            "registros_com_doi": len(with_doi),
            "fracao_com_doi": round(len(with_doi) / len(group), 4) if len(group) else None,
            "dois_distintos": len(distinct),
            "dois_no_corpus": len(present),
            "fracao_no_corpus": round(len(present) / len(distinct), 4) if distinct else None,
            "por_ano": by_year,
            "ausentes_por_motivo": dict(reasons.most_common()),
            "ausentes_campo_principal_top10": dict(other_fields.most_common(10)),
        }
    return result


def faculty_coverage() -> dict:
    people = pd.read_csv(config.DATA_DIR / "capes" / "docentes_openalex.csv", dtype=str).fillna("")
    matched = people["openalex_ids"] != ""
    result = {
        "docentes_distintos": int(len(people)),
        "docentes_identificados": int(matched.sum()),
        "fracao_identificados": round(float(matched.mean()), 4),
        "por_nivel": people.loc[matched, "nivel"].value_counts().to_dict(),
        "por_categoria": {
            cat: {
                "docentes": int(len(g)),
                "identificados": int((g["openalex_ids"] != "").sum()),
            }
            for cat, g in people.groupby("categoria")
        },
    }
    graph = nx.read_graphml(config.GRAPHML_PATH)
    scope_nodes = [d for _, d in graph.nodes(data=True) if d.get("afiliacao_escopo")]
    faculty_nodes = [d for d in scope_nodes if d.get("docente_capes")]
    result["autores_escopo"] = len(scope_nodes)
    result["autores_escopo_docentes"] = len(faculty_nodes)
    result["fracao_autores_escopo_docentes"] = round(len(faculty_nodes) / len(scope_nodes), 4)
    # Docentes concentram a produção? fração das autorias do escopo que são de docentes
    papers_scope = sum(int(d.get("n_publicacoes_corpus", 0)) for d in scope_nodes)
    papers_faculty = sum(int(d.get("n_publicacoes_corpus", 0)) for d in faculty_nodes)
    result["fracao_autorias_escopo_de_docentes"] = round(papers_faculty / papers_scope, 4)
    return result


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    institutions = pd.read_csv(config.INSTITUICOES_CSV, dtype=str)
    scope_ids = set(institutions["openalex_id"].dropna())

    coverage = {
        "doi": doi_coverage(scope_ids),
        "docentes": faculty_coverage(),
    }
    OUT_JSON.write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(coverage, indent=2, ensure_ascii=False))
    print(f"Resultado: {OUT_JSON}")


if __name__ == "__main__":
    main()
