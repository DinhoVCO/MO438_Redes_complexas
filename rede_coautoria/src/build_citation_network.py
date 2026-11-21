"""Rede dirigida de citações: A → B se a publicação A cita B, ambas no corpus."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from src.build_network import _short_id, iter_works_jsonl


def _load_referencias(path: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return refs
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            wid = _short_id(row.get("id"))
            if wid:
                refs[wid] = row
    return refs


def build_citation_graph(
    obras_jsonl: Path,
    referencias_jsonl: Path,
    aliases: dict[str, str] | None = None,
) -> tuple[nx.DiGraph, dict[str, Any]]:
    referencias = _load_referencias(referencias_jsonl)
    # Mesmos aliases da coautoria: autor_ids de cada obra usa o perfil canônico
    aliases = aliases or {}
    corpus: dict[str, dict[str, Any]] = {}

    for work in iter_works_jsonl(obras_jsonl):
        wid = _short_id(work.get("id"))
        if not wid or wid in corpus:
            continue
        author_ids: list[str] = []
        for auth in work.get("authorships") or []:
            aid = _short_id((auth.get("author") or {}).get("id"))
            aid = aliases.get(aid, aid)
            if aid and aid not in author_ids:
                author_ids.append(aid)
        ref_row = referencias.get(wid, {})
        # A própria obra tem prioridade; referencias.jsonl só completa obras
        # coletadas sem referenced_works (coleta antiga, em lotes "magros")
        if "referenced_works" in work:
            cited = [_short_id(r) for r in (work.get("referenced_works") or [])]
            cited_by = work.get("cited_by_count")
        else:
            cited = [_short_id(r) for r in (ref_row.get("referenced_works") or [])]
            cited_by = ref_row.get("cited_by_count", work.get("cited_by_count"))
        cited = [c for c in cited if c]
        corpus[wid] = {
            "year": work.get("publication_year") or ref_row.get("publication_year"),
            "cited_by_count": cited_by if cited_by is not None else -1,
            "author_ids": author_ids,
            "referenced": cited,
            "tem_referencias": wid in referencias or "referenced_works" in work,
            "fwci": work.get("fwci"),
            # "ies_cs" = coleta base (IES + Computer Science); "docentes" = obras dos docentes CAPES
            "origem": work.get("coleta_origem", "ies_cs"),
        }

    g = nx.DiGraph()
    for wid, data in corpus.items():
        year = data["year"] if data["year"] is not None else -1
        g.add_node(
            wid,
            label=wid,
            publication_year=int(year),
            cited_by_count=int(data["cited_by_count"]),
            autor_ids="|".join(data["author_ids"]),
            tem_referencias=bool(data["tem_referencias"]),
            # FWCI: citações normalizadas por campo, tipo e ano (OpenAlex); -1 = ausente
            fwci=float(data["fwci"]) if data["fwci"] is not None else -1.0,
            origem=data["origem"],
        )

    n_edges = 0
    n_external = 0
    for wid, data in corpus.items():
        for cited in data["referenced"]:
            if cited == wid:
                continue
            if cited in corpus:
                if not g.has_edge(wid, cited):
                    g.add_edge(wid, cited)
                    n_edges += 1
            else:
                n_external += 1

    stats = {
        "n_vertices": g.number_of_nodes(),
        "n_arestas": g.number_of_edges(),
        "n_obras_com_referencias": sum(1 for d in corpus.values() if d["tem_referencias"]),
        "n_mencoes_externas": n_external,
        "direcao": "A->B significa que A cita B",
        "arestas": "somente entre publicações do corpus",
    }
    return g, stats


def write_citation_graphml(g: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for _, data in g.nodes(data=True):
        for key, val in list(data.items()):
            if isinstance(val, bool):
                data[key] = int(val)
            elif not isinstance(val, (str, int, float)):
                data[key] = str(val)
    nx.write_graphml(g, path)
