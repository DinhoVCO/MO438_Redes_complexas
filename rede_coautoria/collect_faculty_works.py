#!/usr/bin/env python3
"""Acrescenta ao corpus todas as obras 2018–2024 (de qualquer campo) dos docentes CAPES
identificados na OpenAlex.

O corpus base (run_pipeline.py) pega obras com tópico principal em Computer Science das
IES do escopo. A auditoria contra a CAPES mostrou que quase metade da produção dos docentes
é classificada fora de CS; este passo a recupera filtrando por pessoa, e não por IES, para
não trazer departamentos inteiros de outras áreas.

Cada obra gravada leva `coleta_origem`: "docentes" aqui; obras sem o campo vêm da coleta
base ("ies_cs"). Obras já presentes no corpus não são duplicadas.

Ordem: run_pipeline.py -> collect_capes.py -> label_faculty.py -> collect_faculty_works.py
       -> run_pipeline.py (reconstrói as redes com o corpus ampliado) -> label_faculty.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from label_faculty import MATCH_CSV
from src.build_network import load_seen_work_ids
from src.openalex_client import OpenAlexClient, append_works_jsonl, chunk_list

STATE_JSON = config.DATA_DIR / "capes" / "coleta_docentes_estado.json"
AUTHORS_PER_REQUEST = 50
ORIGIN = "docentes"


def tag_origin(works):
    for work in works:
        work["coleta_origem"] = ORIGIN
        yield work


def main() -> None:
    people = pd.read_csv(MATCH_CSV, dtype=str).fillna("")
    author_ids = sorted({a for ids in people["openalex_ids"] for a in ids.split("|") if a})
    batches = chunk_list(author_ids, AUTHORS_PER_REQUEST)
    print(f"Perfis OpenAlex de docentes: {len(author_ids)} ({len(batches)} lotes)", flush=True)

    state = json.loads(STATE_JSON.read_text(encoding="utf-8")) if STATE_JSON.exists() else {"completed": []}
    completed = set(state["completed"])
    seen = load_seen_work_ids(config.OBRAS_JSONL)
    n_before = len(seen)
    client = OpenAlexClient()
    year_filter = f"publication_year:{config.PUBLICATION_YEAR_MIN}-{config.PUBLICATION_YEAR_MAX}"

    for idx, batch in enumerate(batches, start=1):
        if idx in completed:
            continue
        filter_expr = f"authorships.author.id:{'|'.join(batch)},{year_filter}"
        n_new = append_works_jsonl(tag_origin(client.iter_works(filter_expr)), config.OBRAS_JSONL, seen)
        completed.add(idx)
        state["completed"] = sorted(completed)
        STATE_JSON.write_text(json.dumps(state), encoding="utf-8")
        print(f"  lote {idx}/{len(batches)}: +{n_new} obras (corpus: {len(seen)})", flush=True)

    print(f"Obras acrescentadas: {len(seen) - n_before}. Corpus: {len(seen)} obras.")
    if len(completed) < len(batches):
        sys.exit(2)


if __name__ == "__main__":
    main()
