from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


def _short_id(url_or_id: str | None) -> str | None:
    if not url_or_id:
        return None
    s = str(url_or_id).rstrip("/")
    return s.split("/")[-1] if "/" in s else s


def load_seen_work_ids(jsonl_path: Path) -> set[str]:
    seen: set[str] = set()
    if not jsonl_path.exists():
        return seen
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            work = json.loads(line)
            wid = _short_id(work.get("id"))
            if wid:
                seen.add(wid)
    return seen


def iter_works_jsonl(jsonl_path: Path):
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_scope_sigla(instituicoes_csv: Path) -> dict[str, str]:
    """ID OpenAlex do escopo -> sigla CAPES (a primeira sigla, quando vários campi
    compartilham o mesmo ID, ex.: USP)."""
    scope_sigla: dict[str, str] = {}
    if instituicoes_csv.exists():
        inst_df = pd.read_csv(instituicoes_csv).dropna(subset=["openalex_id"])
        for oid, sigla in zip(inst_df["openalex_id"].astype(str).str.strip(), inst_df["sigla_ies"]):
            scope_sigla.setdefault(oid, sigla)
    return scope_sigla


def load_author_aliases(match_csv: Path) -> dict[str, str]:
    """Perfil OpenAlex secundário -> perfil canônico do mesmo docente CAPES.

    A OpenAlex às vezes divide uma pessoa em vários perfis; label_faculty.py escolhe o
    canônico (o de mais obras no corpus). Sem o arquivo, nenhum perfil é mesclado.
    """
    if not match_csv.exists():
        return {}
    df = pd.read_csv(match_csv, dtype=str).fillna("")
    if "openalex_id_canonico" not in df:
        return {}
    aliases: dict[str, str] = {}
    for ids, canonical in zip(df["openalex_ids"], df["openalex_id_canonico"]):
        for aid in filter(None, ids.split("|")):
            if aid != canonical:
                aliases[aid] = canonical
    return aliases


def iter_work_authors(
    obras_jsonl: Path,
    scope_sigla: dict[str, str],
    aliases: dict[str, str] | None = None,
):
    """Para cada obra, devolve (autores, nomes, IES do escopo por autor).

    - autores: IDs já com os aliases aplicados, sem repetição;
    - nomes: lista de (id_original, id_mesclado, nome);
    - IES do escopo: id_mesclado -> IDs do escopo nas afiliações daquela obra.
    """
    aliases = aliases or {}
    for work in iter_works_jsonl(obras_jsonl):
        author_ids: list[str] = []
        names: list[tuple[str, str, str]] = []
        work_scope: dict[str, set[str]] = defaultdict(set)
        for auth in work.get("authorships") or []:
            raw_id = _short_id((auth.get("author") or {}).get("id"))
            if not raw_id:
                continue
            aid = aliases.get(raw_id, raw_id)
            author_ids.append(aid)
            names.append((raw_id, aid, (auth.get("author") or {}).get("display_name") or raw_id))
            for inst in auth.get("institutions") or []:
                # O lineage inclui a instituição-mãe (ex.: IBILCE -> UNESP)
                candidates = [_short_id(inst.get("id"))] + [
                    _short_id(x) for x in (inst.get("lineage") or [])
                ]
                for iid in candidates:
                    if iid in scope_sigla:
                        work_scope[aid].add(iid)
        yield list(dict.fromkeys(author_ids)), names, work_scope


def build_coauthorship_graph(
    obras_jsonl: Path,
    instituicoes_csv: Path,
    aliases: dict[str, str] | None = None,
) -> tuple[nx.Graph, dict[str, Any]]:
    scope_sigla = load_scope_sigla(instituicoes_csv)
    aliases = aliases or {}

    author_names: dict[str, str] = {}
    author_papers: dict[str, int] = defaultdict(int)
    # Quantas obras o autor assinou com afiliação em cada IES do escopo
    author_scope_counts: dict[str, Counter] = defaultdict(Counter)
    # Perfis OpenAlex reunidos em cada nó (mais de um quando houve mesclagem)
    merged_profiles: dict[str, set[str]] = defaultdict(set)
    edge_weights: dict[tuple[str, str], int] = defaultdict(int)

    n_works = 0
    n_works_skipped_no_authors = 0

    for author_ids, names, work_scope in iter_work_authors(obras_jsonl, scope_sigla, aliases):
        n_works += 1
        for raw_id, aid, name in names:
            merged_profiles[aid].add(raw_id)
            # O nó mesclado fica com o nome do perfil canônico
            if raw_id == aid or aid not in author_names:
                author_names[aid] = name
        # Cada obra conta uma vez por autor, mesmo com authorship repetida ou perfis mesclados
        for aid in author_ids:
            author_papers[aid] += 1
        for aid, iids in work_scope.items():
            author_scope_counts[aid].update(iids)
        if len(author_ids) < 2:
            n_works_skipped_no_authors += 1
            continue
        for a, b in combinations(sorted(author_ids), 2):
            edge_weights[(a, b)] += 1

    g = nx.Graph()
    for aid, name in author_names.items():
        counts = author_scope_counts.get(aid)
        # IES principal = IES do escopo em que o autor mais publicou no corpus
        main_id = counts.most_common(1)[0][0] if counts else ""
        g.add_node(
            aid,
            label=name,
            n_publicacoes_corpus=int(author_papers[aid]),
            afiliacao_escopo=bool(counts),
            ies_principal=scope_sigla.get(main_id, ""),
            ies_principal_id=main_id,
            # Todas as IES do escopo em que o autor publicou (usado no cruzamento com docentes CAPES)
            ies_escopo_ids="|".join(sorted(counts)) if counts else "",
            openalex_ids_mesclados="|".join(sorted(merged_profiles[aid])),
        )
    for (a, b), w in edge_weights.items():
        g.add_edge(a, b, weight=int(w))

    stats: dict[str, Any] = {
        "n_obras_corpus": n_works,
        "n_obras_sem_par_coautoria": n_works_skipped_no_authors,
        "n_vertices": g.number_of_nodes(),
        "n_arestas": g.number_of_edges(),
        "n_vertices_afiliacao_escopo": sum(
            1 for _, d in g.nodes(data=True) if d.get("afiliacao_escopo")
        ),
        "n_perfis_mesclados": sum(1 for p in merged_profiles.values() if len(p) > 1),
        "n_perfis_absorvidos": sum(len(p) - 1 for p in merged_profiles.values()),
    }
    return g, stats


def write_graphml(g: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for _, data in g.nodes(data=True):
        for key, val in list(data.items()):
            if not isinstance(val, (str, int, float, bool)):
                data[key] = str(val)
    nx.write_graphml(g, path)
