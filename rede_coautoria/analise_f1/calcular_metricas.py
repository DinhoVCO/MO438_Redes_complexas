#!/usr/bin/env python3
"""Métricas da Entrega Parcial (itens 1–9) para coautoria e citações."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import igraph as ig
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figuras"
COAUTH_PATH = ROOT / "output" / "colaboracao_cc_brasil.graphml.gz"
CIT_PATH = ROOT / "output" / "citacoes_cc_brasil.graphml"
JSON_PATH = OUT / "resultados_f1.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def save(results: dict) -> None:
    JSON_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def hist_counts(values, title, xlabel, path, loglog=False) -> None:
    counts = Counter(values)
    xs = sorted(x for x in counts if x > 0 and counts[x] > 0)
    ys = [counts[x] for x in xs]
    fig, ax = plt.subplots(figsize=(6, 4))
    if loglog:
        ax.loglog(xs, ys, "o", markersize=3)
    else:
        ax.bar(xs, ys)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frequência")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def hist_unit_interval(values, title, xlabel, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(list(values), bins=40, range=(0, 1))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("vértices")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def path_metrics(nodes, edges, directed: bool) -> dict:
    if len(nodes) < 2:
        return {
            "tamanho_componente": len(nodes),
            "distancia_media": None,
            "diametro": None,
        }
    index = {n: i for i, n in enumerate(nodes)}
    el = [(index[u], index[v]) for u, v in edges]
    g = ig.Graph(n=len(nodes), edges=el, directed=directed)
    g.simplify(multiple=True, loops=True)
    log(f"  distâncias em {g.vcount()} vértices, {g.ecount()} arestas...")
    return {
        "tamanho_componente": g.vcount(),
        "distancia_media": g.average_path_length(directed=directed, unconn=False),
        "diametro": g.diameter(directed=directed, unconn=False),
    }


def draw_core(g: nx.Graph, score, title, path, k=250) -> None:
    ranked = sorted(g.nodes(), key=score, reverse=True)[:k]
    h = g.subgraph(ranked).copy()
    isolates = [n for n in list(h.nodes()) if h.degree(n) == 0]
    h.remove_nodes_from(isolates)
    if h.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(h, seed=7, k=1.2 / (h.number_of_nodes() ** 0.5))
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx(
        h,
        pos,
        ax=ax,
        node_size=12,
        width=0.3,
        with_labels=False,
        node_color="#1f4e79",
        edge_color="#9aa4b2",
        arrows=g.is_directed(),
        arrowsize=6,
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def analyze_coauthorship(g: nx.Graph, results: dict) -> None:
    log("Coautoria: métricas rápidas")
    n = g.number_of_nodes()
    m = g.number_of_edges()
    components = list(nx.connected_components(g))
    sizes = sorted((len(c) for c in components), reverse=True)
    lcc_nodes = max(components, key=len)
    lcc = g.subgraph(lcc_nodes)
    local = nx.clustering(g)
    results["coautoria"] = {
        "n_vertices": n,
        "n_arestas": m,
        "grau_medio": (2 * m) / n,
        "clusterizacao_media": nx.average_clustering(g),
        "transitividade": nx.transitivity(g),
        "densidade": nx.density(g),
        "n_componentes": len(components),
        "maior_componente": sizes[0],
        "tamanhos_componentes_top10": sizes[:10],
    }
    save(results)
    hist_counts(
        [g.degree(v) for v in g],
        "Coautoria — distribuição dos graus",
        "grau",
        FIG / "coautoria_graus.png",
        loglog=True,
    )
    hist_unit_interval(
        local.values(),
        "Coautoria — distribuição da clusterização local",
        "coeficiente de clusterização",
        FIG / "coautoria_clustering.png",
    )
    hist_counts(
        sizes,
        "Coautoria — distribuição dos tamanhos das componentes",
        "tamanho da componente",
        FIG / "coautoria_componentes.png",
        loglog=True,
    )
    log("Coautoria: distância e diâmetro na maior componente")
    results["coautoria"].update(
        path_metrics(lcc_nodes, lcc.edges(), directed=False)
    )
    save(results)
    log("Coautoria: visualização do núcleo de maior grau")
    draw_core(
        g,
        lambda v: g.degree(v),
        "Coautoria — 250 vértices de maior grau",
        FIG / "coautoria_visualizacao.png",
    )


def analyze_citations(g: nx.DiGraph, results: dict) -> None:
    log("Citações: métricas rápidas")
    n = g.number_of_nodes()
    m = g.number_of_edges()
    local = nx.clustering(g)
    results["citacoes"] = {
        "n_vertices": n,
        "n_arestas": m,
        "grau_medio_entrada": m / n,
        "grau_medio_saida": m / n,
        "clusterizacao_media": nx.average_clustering(g),
        "transitividade": nx.transitivity(g),
        "densidade": nx.density(g),
    }
    save(results)
    hist_counts(
        [d for _, d in g.in_degree()],
        "Citações — distribuição do grau de entrada",
        "grau de entrada",
        FIG / "citacoes_grau_entrada.png",
        loglog=True,
    )
    hist_counts(
        [d for _, d in g.out_degree()],
        "Citações — distribuição do grau de saída",
        "grau de saída",
        FIG / "citacoes_grau_saida.png",
        loglog=True,
    )
    hist_unit_interval(
        local.values(),
        "Citações — distribuição da clusterização local",
        "coeficiente de clusterização",
        FIG / "citacoes_clustering.png",
    )
    log("Citações: distância e diâmetro na maior componente forte")
    sccs = list(nx.strongly_connected_components(g))
    lscc_nodes = max(sccs, key=len)
    lscc = g.subgraph(lscc_nodes)
    results["citacoes"].update(path_metrics(lscc_nodes, lscc.edges(), directed=True))
    save(results)
    log("Citações: visualização do núcleo de maior grau de entrada")
    draw_core(
        g,
        lambda v: g.in_degree(v),
        "Citações — 250 vértices de maior grau de entrada",
        FIG / "citacoes_visualizacao.png",
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    log(f"Lendo {COAUTH_PATH.name}")
    coauth = nx.read_graphml(COAUTH_PATH)
    analyze_coauthorship(coauth, results)
    del coauth
    log(f"Lendo {CIT_PATH.name}")
    cit = nx.read_graphml(CIT_PATH)
    if not cit.is_directed():
        cit = nx.DiGraph(cit)
    analyze_citations(cit, results)
    log("Concluído")
    log(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERRO: {exc}")
        sys.exit(1)
