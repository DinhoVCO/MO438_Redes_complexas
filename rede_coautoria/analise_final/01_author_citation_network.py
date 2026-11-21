#!/usr/bin/env python3
"""Rede dirigida de citações entre autores.

Agrega a rede de citações entre publicações ao nível de autores: a aresta a → b indica
que alguma obra de a cita alguma obra de b, com as duas obras no corpus.

Pesos da aresta a → b:
- weight: nº de pares (obra citante, obra citada) em que a assina a citante e b a citada;
- weight_frac: soma de 1 / (n_autores_citante × n_autores_citado) sobre esses pares, de modo
  que cada citação entre publicações distribui peso total 1 entre os pares de autores
  (obras com muitos autores não inflam as arestas).

Autocitações (a → a) são descartadas e contadas. Os vértices usam os mesmos IDs da rede de
coautoria (com os perfis de docentes já mesclados), então as duas redes se cruzam.

Entradas: output/citacoes_cc_brasil.graphml e output/colaboracao_cc_brasil.graphml.gz
Saídas:   output/citacoes_autores_cc_brasil.graphml.gz
          analise_final/results/author_citation_network.json
          analise_final/figures/author_citation_degree_ccdf.png
"""
from __future__ import annotations

from collections import Counter, defaultdict

import networkx as nx
import numpy as np

from common import (
    AUTHOR_CITATION_PATH,
    COAUTHORSHIP_PATH,
    FIGURES_DIR,
    PUB_CITATION_PATH,
    load_graph,
    save_json,
    setup_matplotlib,
)

# Atributos copiados da rede de coautoria (mesmos nomes, em português, para as redes casarem)
COAUTHOR_ATTRS = (
    "label",
    "ies_principal",
    "ies_principal_id",
    "afiliacao_escopo",
    "docente_capes",
    "programa_capes",
    "id_pessoa_capes",
)


def work_authors(pub_graph: nx.DiGraph) -> dict[str, list[str]]:
    """Lista de autores (IDs canônicos) de cada publicação."""
    return {
        wid: [a for a in (data.get("autor_ids") or "").split("|") if a]
        for wid, data in pub_graph.nodes(data=True)
    }


def build_author_citation_graph(
    pub_graph: nx.DiGraph, coauthor_graph: nx.Graph
) -> tuple[nx.DiGraph, dict]:
    """Agrega as citações publicação → publicação em citações autor → autor."""
    authors = work_authors(pub_graph)

    weight: Counter = Counter()
    weight_frac: defaultdict[tuple[str, str], float] = defaultdict(float)
    n_pub_citations = 0
    n_pub_self = 0  # citações entre publicações que compartilham algum autor
    n_pairs = 0
    n_pairs_self = 0  # pares a → a descartados
    frac_self = 0.0
    n_pub_no_authors = 0

    for citing, cited in pub_graph.edges():
        src, dst = authors[citing], authors[cited]
        n_pub_citations += 1
        if not src or not dst:
            n_pub_no_authors += 1
            continue
        if set(src) & set(dst):
            n_pub_self += 1
        share = 1.0 / (len(src) * len(dst))
        for a in src:
            for b in dst:
                n_pairs += 1
                if a == b:
                    n_pairs_self += 1
                    frac_self += share
                    continue
                weight[(a, b)] += 1
                weight_frac[(a, b)] += share

    # Produção e citações globais (OpenAlex) de cada autor dentro do corpus
    n_works: Counter = Counter()
    cited_by_sum: Counter = Counter()
    for wid, data in pub_graph.nodes(data=True):
        cites = max(int(data.get("cited_by_count", 0)), 0)
        for a in authors[wid]:
            n_works[a] += 1
            cited_by_sum[a] += cites

    graph = nx.DiGraph()
    for author in n_works:
        attrs = dict(coauthor_graph.nodes[author]) if author in coauthor_graph else {}
        node = {key: attrs.get(key, "") for key in COAUTHOR_ATTRS}
        node["label"] = node["label"] or author
        node["afiliacao_escopo"] = bool(attrs.get("afiliacao_escopo", False))
        node["docente_capes"] = bool(attrs.get("docente_capes", False))
        node["n_publicacoes_corpus"] = n_works[author]
        node["soma_cited_by_count"] = cited_by_sum[author]
        graph.add_node(author, **node)
    for (a, b), w in weight.items():
        graph.add_edge(a, b, weight=w, weight_frac=round(weight_frac[(a, b)], 6))

    missing = sum(1 for a in n_works if a not in coauthor_graph)
    if missing:
        print(f"Aviso: {missing} autores da rede de citações não estão na rede de coautoria")

    self_stats = {
        "citacoes_entre_publicacoes": n_pub_citations,
        "citacoes_sem_autores": n_pub_no_authors,
        "citacoes_publicacao_com_autor_em_comum": n_pub_self,
        "frac_citacoes_publicacao_com_autor_em_comum": n_pub_self / max(n_pub_citations, 1),
        "pares_autor_gerados": n_pairs,
        "pares_autocitacao_descartados": n_pairs_self,
        "frac_pares_autocitacao": n_pairs_self / max(n_pairs, 1),
        # Cada citação entre publicações vale 1 na contagem fracionária
        "frac_peso_fracionario_autocitacao": frac_self / max(n_pub_citations - n_pub_no_authors, 1),
    }
    return graph, self_stats


def connectivity_summary(graph: nx.DiGraph) -> dict:
    """|V|, |E|, grau médio, densidade, isolados, componentes fraca e forte, reciprocidade."""
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    wcc = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
    scc = sorted((len(c) for c in nx.strongly_connected_components(graph)), reverse=True)
    isolated = sum(1 for v in graph if graph.degree(v) == 0)
    return {
        "n_vertices": n,
        "n_arestas": m,
        "grau_medio_saida": m / n,
        "densidade": nx.density(graph),
        "n_isolados": isolated,
        "frac_isolados": isolated / n,
        "n_componentes_fracas": len(wcc),
        "maior_componente_fraca": wcc[0],
        "frac_maior_componente_fraca": wcc[0] / n,
        "n_componentes_fortes_nao_triviais": sum(1 for s in scc if s > 1),
        "maior_componente_forte": scc[0],
        "frac_maior_componente_forte": scc[0] / n,
        "reciprocidade": nx.reciprocity(graph) if m else 0.0,
    }


def subset_summary(graph: nx.DiGraph, attr: str) -> dict:
    """Mesmo resumo restrito ao subgrafo induzido pelos vértices com attr verdadeiro."""
    nodes = [v for v, d in graph.nodes(data=True) if d.get(attr)]
    return connectivity_summary(graph.subgraph(nodes))


def strength_summary(graph: nx.DiGraph) -> dict:
    """Estatísticas dos graus e forças, para comparar contagem inteira e fracionária."""
    out = {}
    for name, values in (
        ("grau_entrada", [d for _, d in graph.in_degree()]),
        ("grau_saida", [d for _, d in graph.out_degree()]),
        ("forca_entrada", [d for _, d in graph.in_degree(weight="weight")]),
        ("forca_entrada_frac", [d for _, d in graph.in_degree(weight="weight_frac")]),
    ):
        arr = np.asarray(values, dtype=float)
        out[name] = {
            "media": float(arr.mean()),
            "mediana": float(np.median(arr)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
        }
    weights = np.asarray([w for _, _, w in graph.edges(data="weight")], dtype=float)
    out["peso_aresta"] = {
        "media": float(weights.mean()),
        "max": float(weights.max()),
        "frac_peso_1": float((weights == 1).mean()),
    }
    return out


def coauthor_overlap(author_graph: nx.DiGraph, coauthor_graph: nx.Graph) -> dict:
    """Parcela das citações autor → autor entre pares que também são coautores.

    Mede quanto da rede de citações repete a de coautoria (citar o próprio grupo).
    """
    n_edges = w_total = f_total = 0.0
    n_co = w_co = f_co = 0.0
    for a, b, data in author_graph.edges(data=True):
        n_edges += 1
        w_total += data["weight"]
        f_total += data["weight_frac"]
        if coauthor_graph.has_edge(a, b):
            n_co += 1
            w_co += data["weight"]
            f_co += data["weight_frac"]
    return {
        "frac_arestas_entre_coautores": n_co / n_edges,
        "frac_weight_entre_coautores": w_co / w_total,
        "frac_weight_frac_entre_coautores": f_co / f_total,
    }


def top_cited(graph: nx.DiGraph, weight: str, k: int = 15) -> list[dict]:
    """Autores com maior força de entrada (citações recebidas dentro do corpus)."""
    ranking = sorted(graph.in_degree(weight=weight), key=lambda x: -x[1])[:k]
    return [
        {
            "author_id": a,
            "nome": graph.nodes[a]["label"],
            "ies_principal": graph.nodes[a]["ies_principal"],
            "docente_capes": graph.nodes[a]["docente_capes"],
            weight: round(s, 3),
        }
        for a, s in ranking
    ]


def plot_degree_ccdf(author_graph: nx.DiGraph, pub_graph: nx.DiGraph) -> None:
    """CCDF dos graus de entrada e saída nas redes de publicações e de autores."""
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (title, degree_fn) in zip(
        axes, (("Grau de entrada", "in_degree"), ("Grau de saída", "out_degree"))
    ):
        for label, graph in (("Publicações", pub_graph), ("Autores", author_graph)):
            degrees = np.asarray([d for _, d in getattr(graph, degree_fn)()])
            degrees = np.sort(degrees[degrees > 0])
            ccdf = 1.0 - np.arange(len(degrees)) / len(degrees)
            ax.loglog(degrees, ccdf, marker=".", linestyle="none", markersize=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("k")
    axes[0].set_ylabel("P(K ≥ k)")
    axes[0].legend()
    fig.suptitle("Distribuição de graus (CCDF): rede de citações de publicações × de autores")
    path = FIGURES_DIR / "author_citation_degree_ccdf.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


def main() -> None:
    pub_graph = load_graph(PUB_CITATION_PATH)
    coauthor_graph = load_graph(COAUTHORSHIP_PATH)

    author_graph, self_stats = build_author_citation_graph(pub_graph, coauthor_graph)
    print(
        f"Rede de autores: {author_graph.number_of_nodes()} vértices, "
        f"{author_graph.number_of_edges()} arestas",
        flush=True,
    )

    author_summary = connectivity_summary(author_graph)
    results = {
        "definicao": {
            "aresta": "a -> b: alguma publicação de a cita alguma publicação de b (ambas no corpus)",
            "weight": "nº de pares (publicação citante, publicação citada)",
            "weight_frac": "soma de 1/(n_autores_citante * n_autores_citado)",
            "autocitacao": "laços a -> a descartados",
        },
        "autocitacao": self_stats,
        "rede_publicacoes": connectivity_summary(pub_graph),
        "rede_autores": author_summary,
        "rede_autores_afiliacao_escopo": subset_summary(author_graph, "afiliacao_escopo"),
        "rede_autores_docentes_capes": subset_summary(author_graph, "docente_capes"),
        "sobreposicao_coautoria": coauthor_overlap(author_graph, coauthor_graph),
        "graus_e_forcas": strength_summary(author_graph),
        "mais_citados_weight": top_cited(author_graph, "weight"),
        "mais_citados_weight_frac": top_cited(author_graph, "weight_frac"),
    }
    save_json(results, "author_citation_network.json")

    plot_degree_ccdf(author_graph, pub_graph)

    nx.write_graphml(author_graph, AUTHOR_CITATION_PATH)
    size_mb = AUTHOR_CITATION_PATH.stat().st_size / 1e6
    print(f"GraphML: {AUTHOR_CITATION_PATH} ({size_mb:.1f} MB)")

    pub = results["rede_publicacoes"]
    print("\nComparação (publicações → autores):")
    for key in ("n_vertices", "n_arestas", "grau_medio_saida", "frac_isolados",
                "frac_maior_componente_fraca", "frac_maior_componente_forte", "reciprocidade"):
        print(f"  {key}: {pub[key]:.4g} → {author_summary[key]:.4g}")
    print(f"  autocitações (pares a→a): {self_stats['frac_pares_autocitacao']:.1%}; "
          f"citações com autor em comum: {self_stats['frac_citacoes_publicacao_com_autor_em_comum']:.1%}")
    overlap = results["sobreposicao_coautoria"]
    print(f"  arestas entre coautores: {overlap['frac_arestas_entre_coautores']:.1%} "
          f"(peso fracionário: {overlap['frac_weight_frac_entre_coautores']:.1%})")


if __name__ == "__main__":
    main()
