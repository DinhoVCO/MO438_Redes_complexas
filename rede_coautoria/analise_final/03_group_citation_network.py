#!/usr/bin/env python3
"""Rede dirigida de citações entre grupos.

Agrega a rede de citações entre autores (01_author_citation_network.py) em dois níveis:
- comunidades de coautoria (Louvain, 02_communities.py), só as com ≥ 10 membros;
- IES principal dos autores (rede entre "programas", para a P4).

Aresta X → Y: soma das citações de membros de X a membros de Y. O laço X → X é a citação
interna ao grupo. Peso principal: `weight_frac` (cada citação entre publicações vale 1, repartida
entre os pares de autores), então o peso de X → Y é o nº de citações atribuídas a X → Y.

Métricas: força de entrada/saída, fração interna, PageRank (sem laços), homofilia global
(fração interna observada × esperada pelas forças, análoga à modularidade dirigida) e heatmap
de quem cita quem entre os maiores grupos.

Saídas:
- output/citacoes_comunidades.graphml e output/citacoes_ies.graphml
- results/group_citation_network.json
- results/community_citation_metrics.csv e results/ies_citation_metrics.csv
- figures/community_citation_heatmap.png e figures/ies_citation_heatmap.png
"""
from __future__ import annotations

from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from common import AUTHOR_CITATION_PATH, FIGURES_DIR, RESULTS_DIR, config, load_graph, save_json, setup_matplotlib

MIN_COMMUNITY_SIZE = 10
HEATMAP_SIZE = 20
COMMUNITY_GRAPHML = config.OUTPUT_DIR / "citacoes_comunidades.graphml"
IES_GRAPHML = config.OUTPUT_DIR / "citacoes_ies.graphml"


def aggregate(author_graph: nx.DiGraph, group_of: dict[str, str | int]) -> tuple[nx.DiGraph, dict]:
    """Soma as arestas autor → autor por par de grupos; arestas com autor sem grupo são contadas à parte."""
    weight: Counter = Counter()
    weight_frac: defaultdict = defaultdict(float)
    lost_w = lost_f = total_w = total_f = 0.0
    for a, b, data in author_graph.edges(data=True):
        total_w += data["weight"]
        total_f += data["weight_frac"]
        x, y = group_of.get(a), group_of.get(b)
        if x is None or y is None:
            lost_w += data["weight"]
            lost_f += data["weight_frac"]
            continue
        weight[(x, y)] += data["weight"]
        weight_frac[(x, y)] += data["weight_frac"]
    graph = nx.DiGraph()
    graph.add_nodes_from(set(group_of.values()))
    for (x, y), w in weight.items():
        graph.add_edge(x, y, weight=int(w), weight_frac=round(weight_frac[(x, y)], 4))
    coverage = {
        "frac_weight_coberto": 1 - lost_w / total_w,
        "frac_weight_frac_coberto": 1 - lost_f / total_f,
    }
    return graph, coverage


def group_metrics(graph: nx.DiGraph, weight: str = "weight_frac") -> pd.DataFrame:
    """Forças, fração interna e PageRank (sem laços) de cada grupo."""
    no_loops = graph.copy()
    no_loops.remove_edges_from(nx.selfloop_edges(no_loops))
    pagerank = nx.pagerank(no_loops, alpha=0.85, weight=weight)
    rows = []
    for x in graph.nodes():
        internal = graph.edges[x, x][weight] if graph.has_edge(x, x) else 0.0
        out_total = graph.out_degree(x, weight=weight)
        in_total = graph.in_degree(x, weight=weight)
        rows.append(
            {
                "grupo": x,
                "citacoes_internas": internal,
                "citacoes_feitas_externas": out_total - internal,
                "citacoes_recebidas_externas": in_total - internal,
                # Das citações que o grupo faz, quanto fica no próprio grupo
                "frac_interna_saida": internal / out_total if out_total else np.nan,
                # Das citações que o grupo recebe, quanto vem do próprio grupo
                "frac_interna_entrada": internal / in_total if in_total else np.nan,
                "n_grupos_citantes": sum(1 for y in no_loops.predecessors(x)),
                "pagerank": pagerank[x],
            }
        )
    return pd.DataFrame(rows)


def homophily(graph: nx.DiGraph, weight: str = "weight_frac") -> dict:
    """Fração da citação que fica dentro do grupo, observada × esperada se os alvos fossem
    sorteados proporcionalmente às forças de entrada (a diferença é a modularidade dirigida)."""
    total = graph.size(weight=weight)
    internal = sum(d[weight] for x, y, d in graph.edges(data=True) if x == y)
    expected = sum(
        graph.out_degree(x, weight=weight) * graph.in_degree(x, weight=weight) for x in graph
    ) / total**2
    observed = internal / total
    return {
        "frac_interna_observada": observed,
        "frac_interna_esperada": expected,
        "modularidade_dirigida": observed - expected,
        "razao_observada_esperada": observed / expected if expected else None,
    }


def same_group_share(author_graph: nx.DiGraph, community_of: dict, ies_of: dict) -> dict:
    """Entre arestas autor → autor com comunidade e IES conhecidas nas duas pontas,
    a fração do peso que fica na mesma comunidade e na mesma IES."""
    total = same_comm = same_ies = both = 0.0
    for a, b, d in author_graph.edges(data=True):
        if a in community_of and b in community_of and a in ies_of and b in ies_of:
            w = d["weight_frac"]
            total += w
            sc = community_of[a] == community_of[b]
            si = ies_of[a] == ies_of[b]
            same_comm += w * sc
            same_ies += w * si
            both += w * (sc and si)
    return {
        "frac_mesma_comunidade": same_comm / total,
        "frac_mesma_ies": same_ies / total,
        "frac_mesma_comunidade_e_ies": both / total,
        "frac_mesma_comunidade_outra_ies": (same_comm - both) / total,
    }


def plot_heatmap(graph: nx.DiGraph, groups: list, labels: list[str], title: str, filename: str) -> None:
    """Heatmap normalizado por linha: fração das citações de X (linha) que vão para Y (coluna)."""
    plt = setup_matplotlib()
    from matplotlib.colors import LogNorm

    matrix = np.zeros((len(groups), len(groups)))
    for i, x in enumerate(groups):
        out_total = graph.out_degree(x, weight="weight_frac")
        for j, y in enumerate(groups):
            if graph.has_edge(x, y) and out_total:
                matrix[i, j] = graph.edges[x, y]["weight_frac"] / out_total
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    image = ax.imshow(np.where(matrix > 0, matrix, np.nan), cmap="viridis", norm=LogNorm(vmin=1e-3, vmax=1))
    ax.set_xticks(range(len(groups)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(groups)), labels, fontsize=7)
    ax.set_xlabel("Grupo citado")
    ax.set_ylabel("Grupo citante")
    ax.grid(False)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Fração das citações do grupo citante")
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


def write_group_graphml(graph: nx.DiGraph, attrs: pd.DataFrame, path) -> None:
    """Grava a rede de grupos com os atributos do resumo (IDs como texto)."""
    out = nx.relabel_nodes(graph, {x: str(x) for x in graph.nodes()})
    for row in attrs.to_dict("records"):
        node = out.nodes[str(row["grupo"])]
        for key, value in row.items():
            if key == "grupo":
                continue
            if isinstance(value, (bool, np.bool_)):
                node[key] = bool(value)
            elif isinstance(value, (int, np.integer)):
                node[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                node[key] = float(value) if not np.isnan(value) else -1.0
            else:
                node[key] = str(value)
    nx.write_graphml(out, path)
    print(f"GraphML: {path}")


def short_label(row) -> str:
    # O tópico dominante distingue melhor os grupos que a subárea ("Information Systems" é genérica)
    topic = str(row["topico_dominante"]).replace(" and ", " & ").replace("Techniques", "Tech.")
    if len(topic) > 30:
        topic = topic[:29] + "…"
    return f"C{row['grupo']} {topic} ({row['ies_dominante']})"


def main() -> None:
    author_graph = load_graph(AUTHOR_CITATION_PATH)
    communities = pd.read_csv(RESULTS_DIR / "author_communities.csv", usecols=["author_id", "comunidade"])
    summary = pd.read_csv(RESULTS_DIR / "communities_summary.csv")
    big = set(summary.loc[summary["tamanho"] >= MIN_COMMUNITY_SIZE, "comunidade"])
    all_community_of = dict(zip(communities["author_id"], communities["comunidade"]))
    community_of = {a: c for a, c in all_community_of.items() if c in big}
    ies_of = {
        v: d["ies_principal"] for v, d in author_graph.nodes(data=True) if d.get("ies_principal")
    }
    results: dict = {"tamanho_minimo_comunidade": MIN_COMMUNITY_SIZE}

    # --- comunidades
    comm_graph, coverage = aggregate(author_graph, community_of)
    print(f"Rede de comunidades: {comm_graph.number_of_nodes()} vértices, {comm_graph.number_of_edges()} arestas", flush=True)
    comm_metrics = group_metrics(comm_graph)
    comm_metrics = comm_metrics.merge(
        summary.rename(columns={"comunidade": "grupo"})[
            ["grupo", "tamanho", "n_docentes_capes", "ies_dominante", "pureza_ies", "n_ies", "n_obras", "subarea_dominante", "topico_dominante"]
        ],
        on="grupo",
    ).sort_values("tamanho", ascending=False)
    comm_metrics.to_csv(RESULTS_DIR / "community_citation_metrics.csv", index=False, encoding="utf-8")
    write_group_graphml(comm_graph, comm_metrics, COMMUNITY_GRAPHML)

    weighted = comm_metrics["citacoes_internas"] + comm_metrics["citacoes_feitas_externas"]
    results["comunidades"] = {
        "n_vertices": comm_graph.number_of_nodes(),
        "n_arestas": comm_graph.number_of_edges(),
        "n_arestas_sem_lacos": comm_graph.number_of_edges() - nx.number_of_selfloops(comm_graph),
        "densidade_sem_lacos": (comm_graph.number_of_edges() - nx.number_of_selfloops(comm_graph))
        / (comm_graph.number_of_nodes() * (comm_graph.number_of_nodes() - 1)),
        "cobertura": coverage,
        "homofilia": homophily(comm_graph),
        "frac_interna_saida_mediana": float(comm_metrics["frac_interna_saida"].median()),
        "frac_interna_saida_ponderada": float(comm_metrics["citacoes_internas"].sum() / weighted.sum()),
        "top_pagerank": comm_metrics.nlargest(10, "pagerank")[
            ["grupo", "tamanho", "ies_dominante", "subarea_dominante", "pagerank", "citacoes_recebidas_externas"]
        ].to_dict("records"),
        "top_recebidas_externas": comm_metrics.nlargest(10, "citacoes_recebidas_externas")[
            ["grupo", "tamanho", "ies_dominante", "subarea_dominante", "citacoes_recebidas_externas"]
        ].to_dict("records"),
    }
    top = comm_metrics.head(HEATMAP_SIZE)
    plot_heatmap(
        comm_graph,
        list(top["grupo"]),
        [short_label(r) for _, r in top.iterrows()],
        f"Citações entre as {HEATMAP_SIZE} maiores comunidades (normalizado por linha)",
        "community_citation_heatmap.png",
    )

    # --- IES
    ies_graph, ies_coverage = aggregate(author_graph, ies_of)
    print(f"Rede de IES: {ies_graph.number_of_nodes()} vértices, {ies_graph.number_of_edges()} arestas", flush=True)
    ies_metrics = group_metrics(ies_graph)
    n_authors = Counter(ies_of.values())
    n_faculty = Counter(
        d["ies_principal"] for _, d in author_graph.nodes(data=True) if d.get("docente_capes") and d.get("ies_principal")
    )
    ies_metrics["n_autores"] = ies_metrics["grupo"].map(n_authors)
    ies_metrics["n_docentes_capes"] = ies_metrics["grupo"].map(n_faculty).fillna(0).astype(int)
    ies_metrics = ies_metrics.sort_values("n_autores", ascending=False)
    ies_metrics.to_csv(RESULTS_DIR / "ies_citation_metrics.csv", index=False, encoding="utf-8")
    write_group_graphml(ies_graph, ies_metrics, IES_GRAPHML)
    weighted = ies_metrics["citacoes_internas"] + ies_metrics["citacoes_feitas_externas"]
    results["ies"] = {
        "n_vertices": ies_graph.number_of_nodes(),
        "n_arestas": ies_graph.number_of_edges(),
        "cobertura": ies_coverage,
        "homofilia": homophily(ies_graph),
        "frac_interna_saida_mediana": float(ies_metrics["frac_interna_saida"].median()),
        "frac_interna_saida_ponderada": float(ies_metrics["citacoes_internas"].sum() / weighted.sum()),
        "top_pagerank": ies_metrics.nlargest(10, "pagerank")[
            ["grupo", "n_autores", "n_docentes_capes", "pagerank", "citacoes_recebidas_externas"]
        ].to_dict("records"),
    }
    top_ies = ies_metrics.head(HEATMAP_SIZE)
    plot_heatmap(
        ies_graph,
        list(top_ies["grupo"]),
        [str(g) for g in top_ies["grupo"]],
        f"Citações entre as {HEATMAP_SIZE} maiores IES (normalizado por linha)",
        "ies_citation_heatmap.png",
    )

    results["mesmo_grupo"] = same_group_share(author_graph, community_of, ies_of)
    save_json(results, "group_citation_network.json")

    for level in ("comunidades", "ies"):
        h = results[level]["homofilia"]
        print(
            f"{level}: citação interna {h['frac_interna_observada']:.1%} "
            f"(esperado {h['frac_interna_esperada']:.1%}, razão {h['razao_observada_esperada']:.1f})"
        )
    s = results["mesmo_grupo"]
    print(f"Pares com comunidade e IES: mesma comunidade {s['frac_mesma_comunidade']:.1%}, mesma IES {s['frac_mesma_ies']:.1%}")


if __name__ == "__main__":
    main()
