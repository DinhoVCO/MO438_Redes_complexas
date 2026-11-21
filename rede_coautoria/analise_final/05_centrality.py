#!/usr/bin/env python3
"""Centralidade e influência (objetivos ii e iv, P2).

Três níveis:
- publicações (citações, dirigida): grau de entrada e PageRank × cited_by_count;
- autores na rede de citação (dirigida, peso fracionário): força de entrada, PageRank, HITS;
- autores na coautoria (componente gigante, sem pesos): as seis medidas da AP3 — grau, grau
  médio dos vizinhos, k-core, proximidade (estimada com fontes amostradas), intermediação
  (estimada com fontes amostradas) e autovetor.

Direção: a aresta A → B significa "A cita B", então PageRank e autoridade (HITS) premiam quem
é citado; o hub premia quem cita autores muito citados.

Validação (P2): Spearman/Kendall entre as centralidades e os indicadores tradicionais (citações
totais e índice h no corpus) e Jaccard dos top 50, só com autores do escopo (afiliacao_escopo).

Saídas:
- results/author_centrality.csv     — todas as medidas por autor
- results/centrality.json           — correlações, Jaccard, tops
- figures/centrality_spearman_coauthorship.png e figures/centrality_spearman_validation.png
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from common import (
    AUTHOR_CITATION_PATH,
    COAUTHORSHIP_PATH,
    FIGURES_DIR,
    PUB_CITATION_PATH,
    RESULTS_DIR,
    config,
    load_graph,
    save_json,
    setup_matplotlib,
)
from src.build_network import _short_id, iter_works_jsonl

N_SAMPLE_SOURCES = 1000
TOP_K = 10
JACCARD_K = 50
SEED = 0

AP3_MEASURES = {
    "grau": "Grau",
    "grau_medio_vizinhos": "Grau médio dos vizinhos",
    "k_core": "k-core",
    "proximidade": "Proximidade",
    "intermediacao": "Intermediação",
    "autovetor": "Autovetor",
}
CITATION_MEASURES = {
    "cit_forca_entrada": "Força de entrada (citação)",
    "cit_pagerank": "PageRank (citação)",
    "cit_autoridade": "Autoridade HITS",
    "cit_hub": "Hub HITS",
}
TRADITIONAL = {
    "citacoes_total": "Citações totais",
    "indice_h": "Índice h (corpus)",
    "n_publicacoes_corpus": "Nº de publicações",
}


def to_igraph(graph: nx.Graph, directed: bool, weight: str | None) -> tuple[ig.Graph, list[str]]:
    names = list(graph.nodes())
    index = {v: i for i, v in enumerate(names)}
    g = ig.Graph(n=len(names), edges=[(index[a], index[b]) for a, b in graph.edges()], directed=directed)
    if weight:
        g.es["weight"] = [float(w) for _, _, w in graph.edges(data=weight)]
    return g, names


# ---------------------------------------------------------------- coautoria (AP3)

def sampled_closeness(g: ig.Graph, sources: list[int]) -> np.ndarray:
    """Proximidade estimada (Eppstein–Wang): soma das distâncias a partir de k fontes, escalada
    por n/k. Em grafo não dirigido d(s, v) = d(v, s), então k BFS bastam."""
    n = g.vcount()
    total = np.zeros(n)
    for start in range(0, len(sources), 50):
        batch = sources[start:start + 50]
        total += np.asarray(g.distances(source=batch), dtype=float).sum(axis=0)
    estimated_sum = total * n / len(sources)
    with np.errstate(divide="ignore"):
        return np.where(estimated_sum > 0, (n - 1) / estimated_sum, 0.0)


def coauthorship_measures(graph: nx.Graph) -> pd.DataFrame:
    """As seis medidas da AP3 na componente gigante da coautoria (sem pesos)."""
    giant = graph.subgraph(max(nx.connected_components(graph), key=len))
    g, names = to_igraph(giant, directed=False, weight=None)
    n = g.vcount()
    print(f"Coautoria (componente gigante): {n} vértices, {g.ecount()} arestas", flush=True)
    random.seed(SEED)
    sources = random.sample(range(n), N_SAMPLE_SOURCES)
    times = {}

    start = time.perf_counter()
    degree = np.array(g.degree(), dtype=float)
    knn = np.nan_to_num(np.array(g.knn()[0], dtype=float))
    coreness = np.array(g.coreness(), dtype=float)
    eigen = np.array(g.eigenvector_centrality(scale=True), dtype=float)
    times["grau_knn_kcore_autovetor"] = time.perf_counter() - start

    start = time.perf_counter()
    closeness = sampled_closeness(g, sources)
    times["proximidade"] = time.perf_counter() - start
    print(f"  proximidade: {times['proximidade']:.0f} s", flush=True)

    start = time.perf_counter()
    raw = np.array(g.betweenness(directed=False, sources=sources), dtype=float)
    # Escala para a população inteira e normaliza pelo nº de pares (n-1)(n-2)/2
    betweenness = raw * (n / N_SAMPLE_SOURCES) / ((n - 1) * (n - 2) / 2)
    times["intermediacao"] = time.perf_counter() - start
    print(f"  intermediação: {times['intermediacao']:.0f} s", flush=True)

    table = pd.DataFrame(
        {
            "author_id": names,
            "grau": degree,
            "grau_medio_vizinhos": knn,
            "k_core": coreness,
            "proximidade": closeness,
            "intermediacao": betweenness,
            "autovetor": eigen,
        }
    )
    table.attrs["tempos_s"] = times
    return table


# ---------------------------------------------------------------- citações

def citation_measures(author_graph: nx.DiGraph) -> pd.DataFrame:
    """Força de entrada, PageRank e HITS na rede de citação entre autores (peso fracionário)."""
    g, names = to_igraph(author_graph, directed=True, weight="weight_frac")
    w = g.es["weight"]
    return pd.DataFrame(
        {
            "author_id": names,
            "cit_grau_entrada": g.indegree(),
            "cit_forca_entrada": g.strength(mode="in", weights=w),
            "cit_pagerank": g.pagerank(damping=0.85, weights=w, directed=True),
            "cit_autoridade": g.authority_score(weights=w, scale=True),
            "cit_hub": g.hub_score(weights=w, scale=True),
        }
    )


def h_index(citations: list[int]) -> int:
    ordered = sorted(citations, reverse=True)
    return sum(1 for i, c in enumerate(ordered, start=1) if c >= i)


def traditional_indicators(pub_graph: nx.DiGraph) -> pd.DataFrame:
    """Citações totais e índice h de cada autor, a partir das obras do corpus."""
    per_author: dict[str, list[int]] = defaultdict(list)
    for _, d in pub_graph.nodes(data=True):
        for a in (d.get("autor_ids") or "").split("|"):
            if a:
                per_author[a].append(max(int(d["cited_by_count"]), 0))
    return pd.DataFrame(
        {
            "author_id": list(per_author),
            "citacoes_total": [sum(c) for c in per_author.values()],
            "indice_h": [h_index(c) for c in per_author.values()],
        }
    )


def publication_measures(pub_graph: nx.DiGraph) -> tuple[pd.DataFrame, dict]:
    """Grau de entrada e PageRank das publicações × cited_by_count (impacto interno × global)."""
    g, names = to_igraph(pub_graph, directed=True, weight=None)
    table = pd.DataFrame(
        {
            "work_id": names,
            "grau_entrada": g.indegree(),
            "pagerank": g.pagerank(damping=0.85, directed=True),
            "cited_by_count": [pub_graph.nodes[w]["cited_by_count"] for w in names],
            "ano": [pub_graph.nodes[w]["publication_year"] for w in names],
        }
    )
    cited = table[table["grau_entrada"] > 0]
    stats = {
        "spearman_grau_entrada_x_cited_by": spearmanr(table["grau_entrada"], table["cited_by_count"])[0],
        "spearman_pagerank_x_cited_by": spearmanr(table["pagerank"], table["cited_by_count"])[0],
        "spearman_pagerank_x_cited_by_so_citadas_no_corpus": spearmanr(cited["pagerank"], cited["cited_by_count"])[0],
        "frac_citadas_no_corpus": float((table["grau_entrada"] > 0).mean()),
        # Fração das citações globais que vêm do próprio corpus
        "razao_mediana_grau_entrada_sobre_cited_by": float(
            (cited["grau_entrada"] / cited["cited_by_count"].clip(lower=1)).median()
        ),
        "jaccard_top50_pagerank_x_cited_by": jaccard_top(table, "pagerank", "cited_by_count", JACCARD_K, "work_id"),
    }
    return table, stats


def work_titles(work_ids: set[str]) -> dict[str, str]:
    """Títulos das obras pedidas, lendo o corpus em streaming."""
    titles: dict[str, str] = {}
    for work in iter_works_jsonl(config.OBRAS_JSONL):
        wid = _short_id(work.get("id"))
        if wid in work_ids and wid not in titles:
            titles[wid] = (work.get("display_name") or "")[:120]
            if len(titles) == len(work_ids):
                break
    return titles


# ---------------------------------------------------------------- validação

def jaccard_top(table: pd.DataFrame, a: str, b: str, k: int, key: str = "author_id") -> float:
    top_a = set(table.nlargest(k, a)[key])
    top_b = set(table.nlargest(k, b)[key])
    return len(top_a & top_b) / len(top_a | top_b)


def correlation_matrix(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return table[columns].corr(method="spearman")


def validation(table: pd.DataFrame) -> dict:
    """Centralidades × citações totais e índice h (Spearman, Kendall, Jaccard top 50)."""
    out = {}
    measures = [*CITATION_MEASURES, "grau", "intermediacao", "autovetor", "proximidade", "k_core"]
    for measure in measures:
        sub = table[[measure, "citacoes_total", "indice_h"]].dropna()
        out[measure] = {
            "n": len(sub),
            "spearman_citacoes": float(spearmanr(sub[measure], sub["citacoes_total"])[0]),
            "kendall_citacoes": float(kendalltau(sub[measure], sub["citacoes_total"])[0]),
            "spearman_indice_h": float(spearmanr(sub[measure], sub["indice_h"])[0]),
            "kendall_indice_h": float(kendalltau(sub[measure], sub["indice_h"])[0]),
            f"jaccard_top{JACCARD_K}_citacoes": jaccard_top(sub.assign(author_id=table.loc[sub.index, "author_id"]), measure, "citacoes_total", JACCARD_K),
            f"jaccard_top{JACCARD_K}_indice_h": jaccard_top(sub.assign(author_id=table.loc[sub.index, "author_id"]), measure, "indice_h", JACCARD_K),
        }
    return out


def top_authors(table: pd.DataFrame, measure: str) -> list[dict]:
    cols = list(dict.fromkeys(["nome", "ies_principal", "docente_capes", measure, "citacoes_total", "indice_h", "grau"]))
    top = table.nlargest(TOP_K, measure)[cols]
    return [
        {k: (round(v, 6) if isinstance(v, float) else (bool(v) if isinstance(v, np.bool_) else v)) for k, v in row.items()}
        for row in top.to_dict("records")
    ]


def bridges(table: pd.DataFrame) -> list[dict]:
    """Autores-ponte: entre os 1 % de maior intermediação, os de menor grau."""
    sub = table.dropna(subset=["intermediacao"])
    cut = sub["intermediacao"].quantile(0.99)
    top = sub[sub["intermediacao"] >= cut].nsmallest(TOP_K, "grau")
    return top[["nome", "ies_principal", "docente_capes", "grau", "intermediacao", "comunidade"]].to_dict("records")


def plot_matrix(matrix: pd.DataFrame, labels: dict, title: str, filename: str) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(0.75 * len(matrix) + 2.5, 0.65 * len(matrix) + 1.5))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ticks = [labels.get(c, c) for c in matrix.columns]
    ax.set_xticks(range(len(ticks)), ticks, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ticks)), ticks, fontsize=8)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            value = matrix.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.6 else "black")
    ax.grid(False)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Spearman ρ")
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


def main() -> None:
    coauthor_graph = load_graph(COAUTHORSHIP_PATH)
    author_graph = load_graph(AUTHOR_CITATION_PATH)
    pub_graph = load_graph(PUB_CITATION_PATH)

    # --- autores: atributos + todas as medidas
    attrs = pd.DataFrame(
        [
            {
                "author_id": v,
                "nome": d.get("label", v),
                "ies_principal": d.get("ies_principal", ""),
                "afiliacao_escopo": bool(d.get("afiliacao_escopo")),
                "docente_capes": bool(d.get("docente_capes")),
                "comunidade": d.get("comunidade", -1),
                "n_publicacoes_corpus": d.get("n_publicacoes_corpus", 0),
            }
            for v, d in coauthor_graph.nodes(data=True)
        ]
    )
    coauthor = coauthorship_measures(coauthor_graph)
    times = coauthor.attrs["tempos_s"]
    citation = citation_measures(author_graph)
    traditional = traditional_indicators(pub_graph)
    table = (
        attrs.merge(coauthor, on="author_id", how="left")
        .merge(citation, on="author_id", how="left")
        .merge(traditional, on="author_id", how="left")
    )
    table.to_csv(RESULTS_DIR / "author_centrality.csv", index=False, encoding="utf-8")
    print(f"Centralidades: {RESULTS_DIR / 'author_centrality.csv'}")

    # --- correlações
    giant = table.dropna(subset=list(AP3_MEASURES))
    ap3_matrix = correlation_matrix(giant, list(AP3_MEASURES))
    scope = table[table["afiliacao_escopo"]].copy()
    scope_giant = scope.dropna(subset=list(AP3_MEASURES))
    joint_cols = ["grau", "intermediacao", "autovetor", *CITATION_MEASURES, *TRADITIONAL]
    joint_matrix = correlation_matrix(scope_giant, joint_cols)

    # --- publicações
    pubs, pub_stats = publication_measures(pub_graph)
    top_pub_ids = set(pubs.nlargest(TOP_K, "pagerank")["work_id"]) | set(pubs.nlargest(TOP_K, "cited_by_count")["work_id"]) | set(pubs.nlargest(TOP_K, "grau_entrada")["work_id"])
    print("Lendo títulos das publicações do top...", flush=True)
    titles = work_titles(top_pub_ids)
    pubs["titulo"] = pubs["work_id"].map(titles)

    def top_pubs(measure: str) -> list[dict]:
        cols = ["work_id", "titulo", "ano", "grau_entrada", "pagerank", "cited_by_count"]
        return pubs.nlargest(TOP_K, measure)[cols].round(6).to_dict("records")

    results = {
        "definicoes": {
            "direcao": "A -> B = A cita B; PageRank e autoridade premiam quem é citado",
            "coautoria": f"componente gigante, sem pesos; proximidade e intermediação estimadas com {N_SAMPLE_SOURCES} fontes amostradas",
            "citacao_autores": "peso fracionário (weight_frac)",
            "validacao": "só autores com afiliacao_escopo=True",
        },
        "tempos_coautoria_s": times,
        "publicacoes": {
            **pub_stats,
            "top_pagerank": top_pubs("pagerank"),
            "top_grau_entrada": top_pubs("grau_entrada"),
            "top_cited_by_count": top_pubs("cited_by_count"),
        },
        "spearman_ap3_coautoria": ap3_matrix.round(3).to_dict(),
        "spearman_conjunto_escopo": joint_matrix.round(3).to_dict(),
        "validacao_escopo": validation(scope.reset_index(drop=True)),
        "top_autores_escopo": {
            m: top_authors(scope, m)
            for m in ["cit_pagerank", "cit_autoridade", "cit_forca_entrada", "intermediacao", "grau", "autovetor", "citacoes_total", "indice_h"]
        },
        "autores_ponte": bridges(scope),
        "top_docentes_por_pagerank": top_authors(table[table["docente_capes"]], "cit_pagerank"),
    }
    save_json(json.loads(json.dumps(results, default=lambda o: o.item() if hasattr(o, "item") else str(o))), "centrality.json")

    plot_matrix(ap3_matrix, AP3_MEASURES, "Spearman entre as seis centralidades (coautoria)", "centrality_spearman_coauthorship.png")
    labels = {**AP3_MEASURES, **CITATION_MEASURES, **TRADITIONAL}
    plot_matrix(joint_matrix, labels, "Centralidades × indicadores tradicionais (autores do escopo)", "centrality_spearman_validation.png")

    print("\nValidação (Spearman com citações totais / índice h):")
    for m, v in results["validacao_escopo"].items():
        print(f"  {m}: {v['spearman_citacoes']:.3f} / {v['spearman_indice_h']:.3f}; Jaccard top{JACCARD_K} cit. {v[f'jaccard_top{JACCARD_K}_citacoes']:.2f}")
    print(f"Publicações: Spearman PageRank × cited_by = {pub_stats['spearman_pagerank_x_cited_by']:.3f}")


if __name__ == "__main__":
    main()
