#!/usr/bin/env python3
"""Propriedades estruturais das redes.

Prioridade alta:
- mundo pequeno (Aula 15): σ = (C/C_rand)/(L/L_rand) contra Erdős–Rényi e contra o modelo de
  configuração;
- lei de potência (Aula 16): ajuste com `powerlaw` (α, k_min, comparação com lognormal,
  exponencial e lei de potência truncada) do grau na coautoria, do grau de entrada na citação
  entre autores e de `cited_by_count`;
- homofilia e assortatividade (Aula 10): por IES e por grau, com modelo nulo;
- Granovetter (Aula 4): sobreposição de vizinhança × peso, arestas dentro × entre comunidades e
  remoção de arestas fracas/fortes primeiro.
Prioridade média:
- centralização de Freeman (grau) por comunidade (Aula 9);
- paradoxo da amizade (Aula 8).
Prioridade baixa:
- snapshots cumulativos anuais 2018–2024 da coautoria (Aula 14);
- reciprocidade e censo de tríades na citação entre docentes, contra o modelo nulo (Aula 4).

Saídas: results/structural_properties.json, results/community_centralization.csv e as figuras
figures/structural_powerlaw.png, figures/structural_granovetter.png, figures/structural_snapshots.png
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from itertools import combinations

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
from scipy.stats import spearmanr

from common import (
    AUTHOR_CITATION_PATH,
    COAUTHORSHIP_PATH,
    FIGURES_DIR,
    PUB_CITATION_PATH,
    RESULTS_DIR,
    ROOT,
    load_graph,
    save_json,
    setup_matplotlib,
)

SEED = 0
N_DISTANCE_SOURCES = 500
N_LABEL_PERMUTATIONS = 200
N_NULL_TRIADS = 10
MIN_COMMUNITY_SIZE = 10
WEIGHT_BINS = [1, 2, 3, 4, 5, 10, 20, math.inf]


def to_igraph(graph: nx.Graph, directed: bool, weight: str | None = None) -> tuple[ig.Graph, list[str]]:
    names = list(graph.nodes())
    index = {v: i for i, v in enumerate(names)}
    g = ig.Graph(n=len(names), edges=[(index[a], index[b]) for a, b in graph.edges()], directed=directed)
    if weight:
        g.es["weight"] = [float(w) for _, _, w in graph.edges(data=weight, default=1)]
    return g, names


def sampled_mean_distance(g: ig.Graph, n_sources: int) -> float:
    """Distância média estimada a partir de fontes amostradas (só pares alcançáveis)."""
    sources = random.sample(range(g.vcount()), n_sources)
    total, count = 0.0, 0
    for start in range(0, n_sources, 50):
        d = np.asarray(g.distances(source=sources[start:start + 50]), dtype=float)
        mask = np.isfinite(d) & (d > 0)
        total += d[mask].sum()
        count += mask.sum()
    return total / count


# ---------------------------------------------------------------- mundo pequeno

def small_world(giant: ig.Graph, f1: dict) -> dict:
    n, m = giant.vcount(), giant.ecount()
    k = 2 * m / n
    c_real = giant.transitivity_avglocal_undirected(mode="zero")
    l_real = f1["distancia_media"]  # exata, calculada na etapa 2 (igraph, todos os pares)

    # Erdős–Rényi: fórmulas analíticas da aula
    c_er = k / n
    l_er = math.log(n) / math.log(k)

    # Modelo de configuração: mesmos graus, arestas embaralhadas
    random.seed(SEED)
    null = giant.copy()
    null.rewire(n=10 * m, allowed_edge_types="simple")
    c_conf = null.transitivity_avglocal_undirected(mode="zero")
    l_conf = sampled_mean_distance(null, N_DISTANCE_SOURCES)

    return {
        "n": n,
        "m": m,
        "grau_medio": k,
        "C_real": c_real,
        "L_real": l_real,
        "C_ER": c_er,
        "L_ER": l_er,
        "sigma_ER": (c_real / c_er) / (l_real / l_er),
        "C_configuracao": c_conf,
        "L_configuracao_estimada": l_conf,
        "sigma_configuracao": (c_real / c_conf) / (l_real / l_conf),
    }


# ---------------------------------------------------------------- lei de potência

def fit_power_law(values: np.ndarray, name: str) -> tuple[dict, powerlaw.Fit]:
    values = values[values > 0]
    fit = powerlaw.Fit(values, discrete=True, verbose=False)
    out = {
        "n": int(len(values)),
        "alpha": float(fit.power_law.alpha),
        "sigma_alpha": float(fit.power_law.sigma),
        "xmin": float(fit.power_law.xmin),
        "n_cauda": int((values >= fit.power_law.xmin).sum()),
    }
    # R > 0 favorece a lei de potência; p = significância do sinal de R
    for alt in ("lognormal", "exponential", "truncated_power_law"):
        r, p = fit.distribution_compare("power_law", alt, normalized_ratio=True)
        out[f"R_vs_{alt}"] = float(r)
        out[f"p_vs_{alt}"] = float(p)
    print(f"  {name}: α = {out['alpha']:.2f}, k_min = {out['xmin']:.0f}, cauda = {out['n_cauda']}", flush=True)
    return out, fit


def plot_power_laws(fits: dict) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, len(fits), figsize=(4 * len(fits), 3.6))
    for ax, (label, fit) in zip(axes, fits.items()):
        fit.plot_ccdf(ax=ax, marker=".", linestyle="none", markersize=3, label="dados")
        fit.power_law.plot_ccdf(ax=ax, color="C3", linestyle="--", label=f"lei de potência (α = {fit.power_law.alpha:.2f})")
        fit.lognormal.plot_ccdf(ax=ax, color="C2", linestyle=":", label="lognormal")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("k")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("P(K ≥ k) na cauda")
    fig.suptitle("Ajuste das caudas (pacote powerlaw)")
    path = FIGURES_DIR / "structural_powerlaw.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


# ---------------------------------------------------------------- assortatividade

def ies_homophily(g: ig.Graph, ies: list[str]) -> dict:
    """Assortatividade nominal por IES nas arestas com IES conhecida nas duas pontas,
    com dois modelos nulos: o analítico pelos graus (Σ e_i²) e a permutação dos rótulos."""
    labelled = [i for i, x in enumerate(ies) if x]
    sub = g.induced_subgraph(labelled)
    labels = [ies[i] for i in labelled]
    codes = {x: c for c, x in enumerate(sorted(set(labels)))}
    types = np.array([codes[x] for x in labels])
    edges = np.array(sub.get_edgelist())
    same = types[edges[:, 0]] == types[edges[:, 1]]
    observed = float(same.mean())

    # Fração esperada se as pontas fossem sorteadas proporcionalmente ao grau
    degree = np.array(sub.degree(), dtype=float)
    share = np.bincount(types, weights=degree) / degree.sum()
    expected_degree = float((share**2).sum())

    rng = np.random.default_rng(SEED)
    permuted = []
    for _ in range(N_LABEL_PERMUTATIONS):
        p = rng.permutation(types)
        permuted.append((p[edges[:, 0]] == p[edges[:, 1]]).mean())
    permuted = np.array(permuted)
    return {
        "n_vertices": sub.vcount(),
        "n_arestas": sub.ecount(),
        "assortatividade_ies": sub.assortativity_nominal(types=types.tolist(), directed=False),
        "frac_arestas_mesma_ies": observed,
        "frac_esperada_pelos_graus": expected_degree,
        "frac_esperada_permutacao_media": float(permuted.mean()),
        "frac_esperada_permutacao_desvio": float(permuted.std()),
        "z_permutacao": float((observed - permuted.mean()) / permuted.std()),
        "frac_arestas_entre_ies": 1 - observed,
    }


def degree_assortativity(coauthor: ig.Graph, author_cit: ig.Graph, pub_cit: ig.Graph) -> dict:
    return {
        "coautoria": coauthor.assortativity_degree(directed=False),
        # Dirigida: correlação entre o grau de saída de quem cita e o de entrada de quem é citado
        "citacao_autores_saida_entrada": author_cit.assortativity(
            types1=author_cit.outdegree(), types2=author_cit.indegree(), directed=True
        ),
        "citacao_publicacoes_saida_entrada": pub_cit.assortativity(
            types1=pub_cit.outdegree(), types2=pub_cit.indegree(), directed=True
        ),
    }


def friendship_paradox(g: ig.Graph) -> dict:
    degree = np.array(g.degree(), dtype=float)
    knn = np.array(g.knn()[0], dtype=float)
    mask = degree > 0
    return {
        "grau_medio": float(degree[mask].mean()),
        "grau_medio_dos_vizinhos_media": float(np.nanmean(knn[mask])),
        # <k²>/<k>: grau médio de um vizinho escolhido ao acaso (a "amizade" da aula)
        "k2_sobre_k": float((degree[mask] ** 2).mean() / degree[mask].mean()),
        "frac_com_vizinhos_de_grau_maior": float((knn[mask] > degree[mask]).mean()),
    }


# ---------------------------------------------------------------- Granovetter

def neighborhood_overlap(g: ig.Graph) -> np.ndarray:
    """O_ij = |N(i) ∩ N(j)| / (k_i − 1 + k_j − 1 − |N(i) ∩ N(j)|) para cada aresta."""
    neighbors = [set(nb) for nb in g.get_adjlist()]
    overlap = np.zeros(g.ecount())
    for eid, (a, b) in enumerate(g.get_edgelist()):
        na, nb = neighbors[a], neighbors[b]
        common = len(na & nb) if len(na) < len(nb) else len(nb & na)
        denom = (len(na) - 1) + (len(nb) - 1) - common
        overlap[eid] = common / denom if denom > 0 else 0.0
    return overlap


def removal_curve(g: ig.Graph, order: np.ndarray, checkpoints: np.ndarray) -> list[float]:
    """Fração de vértices na maior componente depois de remover as arestas na ordem dada.

    Implementado ao contrário (union-find adicionando as arestas que sobram)."""
    n = g.vcount()
    edges = g.get_edgelist()
    parent = list(range(n))
    size = [1] * n
    largest = 1

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    m = len(order)
    remaining_needed = sorted({int(round((1 - f) * m)) for f in checkpoints})
    result: dict[int, float] = {}
    added = 0
    targets = iter(remaining_needed)
    target = next(targets)
    while target == 0:
        result[0] = largest / n
        target = next(targets, None)
    for eid in order[::-1]:  # as últimas a serem removidas entram primeiro
        a, b = edges[eid]
        ra, rb = find(a), find(b)
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
            largest = max(largest, size[ra])
        added += 1
        while target is not None and added == target:
            result[target] = largest / n
            target = next(targets, None)
    return [result[int(round((1 - f) * m))] for f in checkpoints]


def granovetter(g: ig.Graph, communities: list[int]) -> tuple[dict, dict]:
    start = time.perf_counter()
    overlap = neighborhood_overlap(g)
    print(f"  sobreposição de vizinhança: {time.perf_counter() - start:.0f} s", flush=True)
    weight = np.array(g.es["weight"])
    edges = np.array(g.get_edgelist())
    comm = np.array(communities)
    between = comm[edges[:, 0]] != comm[edges[:, 1]]

    bins = []
    for lo, hi in zip(WEIGHT_BINS[:-1], WEIGHT_BINS[1:]):
        mask = (weight >= lo) & (weight < hi)
        if mask.any():
            bins.append({
                "peso": f"{lo}" if hi == lo + 1 else (f"{lo}+" if hi == math.inf else f"{lo}-{hi - 1}"),
                "n_arestas": int(mask.sum()),
                "overlap_medio": float(overlap[mask].mean()),
                "frac_entre_comunidades": float(between[mask].mean()),
            })
    rho, p = spearmanr(weight, overlap)

    # Remoção (Onnela et al.): fracas primeiro × fortes primeiro, por peso e por overlap
    checkpoints = np.linspace(0, 0.95, 20)
    rng = np.random.default_rng(SEED)
    tie_break = rng.random(len(weight))  # desempate aleatório entre pesos iguais
    curves = {
        "fracas_primeiro_peso": removal_curve(g, np.lexsort((tie_break, weight)), checkpoints),
        "fortes_primeiro_peso": removal_curve(g, np.lexsort((tie_break, -weight)), checkpoints),
        "fracas_primeiro_overlap": removal_curve(g, np.lexsort((tie_break, overlap)), checkpoints),
        "fortes_primeiro_overlap": removal_curve(g, np.lexsort((tie_break, -overlap)), checkpoints),
    }
    stats = {
        "spearman_peso_overlap": {"rho": float(rho), "p_valor": float(p)},
        "por_faixa_de_peso": bins,
        "dentro_comunidade": {
            "frac_arestas": float((~between).mean()),
            "peso_medio": float(weight[~between].mean()),
            "overlap_medio": float(overlap[~between].mean()),
        },
        "entre_comunidades": {
            "frac_arestas": float(between.mean()),
            "peso_medio": float(weight[between].mean()),
            "overlap_medio": float(overlap[between].mean()),
        },
        "remocao": {"frac_removida": checkpoints.round(3).tolist(), **curves},
    }
    plot_data = {"weight": weight, "overlap": overlap, "between": between, "checkpoints": checkpoints, "curves": curves}
    return stats, plot_data


def granovetter_small_teams(pub_graph: nx.DiGraph, max_authors: int = 10) -> dict:
    """Robustez: peso × sobreposição na coautoria montada só com obras de até `max_authors`
    autores (sem os cliques gigantes dos artigos com dezenas de autores)."""
    weight: dict[tuple[str, str], int] = defaultdict(int)
    for _, d in pub_graph.nodes(data=True):
        authors = sorted({a for a in (d.get("autor_ids") or "").split("|") if a})
        if 2 <= len(authors) <= max_authors:
            for pair in combinations(authors, 2):
                weight[pair] += 1
    index: dict[str, int] = {}
    edges = [(index.setdefault(a, len(index)), index.setdefault(b, len(index))) for a, b in weight]
    g = ig.Graph(n=len(index), edges=edges)
    g.es["weight"] = list(weight.values())
    giant = g.connected_components().giant()
    overlap = neighborhood_overlap(giant)
    w = np.array(giant.es["weight"])
    rho, p = spearmanr(w, overlap)
    return {
        "max_autores_por_obra": max_authors,
        "n_vertices": giant.vcount(),
        "n_arestas": giant.ecount(),
        "spearman_peso_overlap": {"rho": float(rho), "p_valor": float(p)},
        "overlap_medio_por_peso": {
            label: float(overlap[(w >= lo) & (w < hi)].mean())
            for label, lo, hi in (("1", 1, 2), ("2", 2, 3), ("3-4", 3, 5), ("5-9", 5, 10), ("10+", 10, math.inf))
        },
    }


def plot_granovetter(data: dict) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    weight, overlap, between = data["weight"], data["overlap"], data["between"]
    levels = np.unique(np.clip(weight, 1, 30))
    for mask, label, color in ((~between, "dentro da comunidade", "C0"), (between, "entre comunidades", "C1")):
        w = np.clip(weight[mask], 1, 30)
        means = [overlap[mask][w == lv].mean() if (w == lv).sum() >= 20 else np.nan for lv in levels]
        axes[0].plot(levels, means, marker=".", label=label, color=color)
    axes[0].set_xlabel("Peso da aresta (nº de coautorias, truncado em 30)")
    axes[0].set_ylabel("Sobreposição de vizinhança média")
    axes[0].set_title("Força do vínculo × sobreposição")
    axes[0].legend(fontsize=8)

    axes[1].hist([overlap[~between], overlap[between]], bins=30, density=True, label=["dentro", "entre"], alpha=0.7)
    axes[1].set_xlabel("Sobreposição de vizinhança")
    axes[1].set_ylabel("Densidade")
    axes[1].set_title("Arestas dentro × entre comunidades")
    axes[1].legend(fontsize=8)

    styles = {
        "fracas_primeiro_peso": ("Fracas primeiro (peso)", "C3", "-"),
        "fortes_primeiro_peso": ("Fortes primeiro (peso)", "C0", "-"),
        "fracas_primeiro_overlap": ("Fracas primeiro (overlap)", "C3", "--"),
        "fortes_primeiro_overlap": ("Fortes primeiro (overlap)", "C0", "--"),
    }
    for key, curve in data["curves"].items():
        label, color, ls = styles[key]
        axes[2].plot(data["checkpoints"], curve, color=color, linestyle=ls, label=label)
    axes[2].set_xlabel("Fração de arestas removidas")
    axes[2].set_ylabel("Fração na maior componente")
    axes[2].set_title("Remoção de vínculos")
    axes[2].legend(fontsize=7)
    fig.suptitle("Granovetter na coautoria (componente gigante)")
    path = FIGURES_DIR / "structural_granovetter.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


# ---------------------------------------------------------------- Freeman

def freeman_by_community(coauthor_graph: nx.Graph) -> pd.DataFrame:
    """Centralização de grau de Freeman do subgrafo de coautoria de cada comunidade."""
    members: dict[int, list[str]] = defaultdict(list)
    for v, d in coauthor_graph.nodes(data=True):
        members[int(d.get("comunidade", -1))].append(v)
    rows = []
    for cid, nodes in members.items():
        n = len(nodes)
        if n < MIN_COMMUNITY_SIZE:
            continue
        sub = coauthor_graph.subgraph(nodes)
        degrees = np.array([d for _, d in sub.degree()], dtype=float)
        rows.append({
            "comunidade": cid,
            "tamanho": n,
            # Σ (k_max − k_i) / ((n−1)(n−2)): 1 = estrela, 0 = todos com o mesmo grau
            "centralizacao_freeman": float((degrees.max() - degrees).sum() / ((n - 1) * (n - 2))),
            "densidade_interna": nx.density(sub),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- snapshots

def yearly_snapshots(pub_graph: nx.DiGraph) -> list[dict]:
    """Coautoria cumulativa até cada ano: n, m, ⟨k⟩, C médio e maior componente."""
    by_year: dict[int, list[list[str]]] = defaultdict(list)
    for _, d in pub_graph.nodes(data=True):
        authors = [a for a in (d.get("autor_ids") or "").split("|") if a]
        if authors:
            by_year[d["publication_year"]].append(authors)
    index: dict[str, int] = {}
    edges: set[tuple[int, int]] = set()
    rows = []
    for year in sorted(y for y in by_year if y > 0):
        for authors in by_year[year]:
            ids = sorted({index.setdefault(a, len(index)) for a in authors})
            edges.update(combinations(ids, 2))
        g = ig.Graph(n=len(index), edges=list(edges))
        components = g.connected_components()
        rows.append({
            "ano": year,
            "n_vertices": g.vcount(),
            "n_arestas": g.ecount(),
            "grau_medio": 2 * g.ecount() / g.vcount(),
            "clusterizacao_media": g.transitivity_avglocal_undirected(mode="zero"),
            "frac_maior_componente": max(components.sizes()) / g.vcount(),
            "n_componentes": len(components),
        })
        print(f"  {year}: {rows[-1]['n_vertices']} vértices, maior componente {rows[-1]['frac_maior_componente']:.1%}", flush=True)
    return rows


def plot_snapshots(rows: list[dict]) -> None:
    plt = setup_matplotlib()
    data = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot(data["ano"], data["n_vertices"] / 1000, marker="o", label="vértices (mil)")
    axes[0].plot(data["ano"], data["n_arestas"] / 10000, marker="s", label="arestas (10 mil)")
    axes[0].set_title("Tamanho")
    axes[0].legend(fontsize=8)
    axes[1].plot(data["ano"], data["grau_medio"], marker="o", label="⟨k⟩")
    axes[1].set_title("Grau médio")
    axes[2].plot(data["ano"], data["frac_maior_componente"], marker="o", label="maior componente")
    axes[2].plot(data["ano"], data["clusterizacao_media"], marker="s", label="clusterização média")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Conectividade")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Ano (cumulativo)")
    fig.suptitle("Evolução da rede de coautoria, 2018–2024")
    path = FIGURES_DIR / "structural_snapshots.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


# ---------------------------------------------------------------- tríades

TRIAD_NAMES = ["003", "012", "102", "021D", "021U", "021C", "111D", "111U",
               "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]


def triads_vs_null(author_graph: nx.DiGraph) -> dict:
    """Reciprocidade e censo de tríades na citação entre docentes CAPES, contra grafos
    aleatorizados que preservam os graus de entrada e saída."""
    faculty = [v for v, d in author_graph.nodes(data=True) if d.get("docente_capes")]
    # Subgrafo montado em ordem fixa: o subgraph() do networkx percorre os nós na ordem de um
    # set (muda a cada execução por causa do hash de strings), e a ordem das arestas muda o
    # resultado do rewire mesmo com a semente fixa
    faculty_set = set(faculty)
    sub = nx.DiGraph()
    sub.add_nodes_from(faculty)
    sub.add_edges_from((u, v) for u in faculty for v in author_graph.successors(u) if v in faculty_set)
    g, _ = to_igraph(sub, directed=True)
    real = np.array(list(g.triad_census()), dtype=float)
    reciprocity = g.reciprocity()
    random.seed(SEED)
    null_counts, null_recip = [], []
    for _ in range(N_NULL_TRIADS):
        h = g.copy()
        h.rewire(n=10 * g.ecount(), allowed_edge_types="simple")
        null_counts.append(list(h.triad_census()))
        null_recip.append(h.reciprocity())
    null_counts = np.array(null_counts, dtype=float)
    mean, std = null_counts.mean(axis=0), null_counts.std(axis=0)
    z = np.where(std > 0, (real - mean) / np.where(std > 0, std, 1), np.nan)
    return {
        "n_vertices": g.vcount(),
        "n_arestas": g.ecount(),
        "reciprocidade": reciprocity,
        "reciprocidade_nula_media": float(np.mean(null_recip)),
        "triades": {
            name: {"real": int(real[i]), "nulo_medio": float(mean[i]), "z": float(z[i]) if np.isfinite(z[i]) else None}
            for i, name in enumerate(TRIAD_NAMES)
        },
    }


def full_reciprocity_vs_null(author_cit: ig.Graph) -> dict:
    random.seed(SEED)
    h = author_cit.copy()
    h.rewire(n=3 * author_cit.ecount(), allowed_edge_types="simple")
    return {"real": author_cit.reciprocity(), "nulo": h.reciprocity()}


# ---------------------------------------------------------------- principal

def main() -> None:
    coauthor_graph = load_graph(COAUTHORSHIP_PATH)
    author_graph = load_graph(AUTHOR_CITATION_PATH)
    pub_graph = load_graph(PUB_CITATION_PATH)
    f1 = json.loads((ROOT / "analise_f1" / "resultados_f1.json").read_text(encoding="utf-8"))["coautoria"]

    giant_nodes = max(nx.connected_components(coauthor_graph), key=len)
    giant_nx = coauthor_graph.subgraph(giant_nodes)
    giant, giant_names = to_igraph(giant_nx, directed=False, weight="weight")
    coauthor, coauthor_names = to_igraph(coauthor_graph, directed=False, weight="weight")
    author_cit, _ = to_igraph(author_graph, directed=True)
    pub_cit, _ = to_igraph(pub_graph, directed=True)
    results: dict = {}

    print("Mundo pequeno...", flush=True)
    results["mundo_pequeno"] = small_world(giant, f1)
    print(f"  σ_ER = {results['mundo_pequeno']['sigma_ER']:.0f}; σ_config = {results['mundo_pequeno']['sigma_configuracao']:.1f}", flush=True)

    print("Lei de potência...", flush=True)
    fits, results["lei_de_potencia"] = {}, {}
    for key, label, values in (
        ("grau_coautoria", "Grau na coautoria", np.array(coauthor.degree())),
        ("grau_entrada_citacao_autores", "Grau de entrada (citação entre autores)", np.array(author_cit.indegree())),
        ("cited_by_count_publicacoes", "Citações globais das publicações", np.array([d["cited_by_count"] for _, d in pub_graph.nodes(data=True)])),
    ):
        results["lei_de_potencia"][key], fits[label] = fit_power_law(values, label)
    plot_power_laws(fits)

    print("Assortatividade...", flush=True)
    ies = [coauthor_graph.nodes[v].get("ies_principal") or "" for v in coauthor_names]
    results["assortatividade"] = {
        "grau": degree_assortativity(coauthor, author_cit, pub_cit),
        "ies_coautoria": ies_homophily(coauthor, ies),
        "reciprocidade_citacao_autores": full_reciprocity_vs_null(author_cit),
    }
    results["paradoxo_da_amizade"] = friendship_paradox(coauthor)

    print("Granovetter...", flush=True)
    communities = [int(giant_nx.nodes[v].get("comunidade", -1)) for v in giant_names]
    results["granovetter"], plot_data = granovetter(giant, communities)
    results["granovetter"]["robustez_equipes_pequenas"] = granovetter_small_teams(pub_graph)
    plot_granovetter(plot_data)

    print("Centralização de Freeman...", flush=True)
    freeman = freeman_by_community(coauthor_graph)
    impact = pd.read_csv(RESULTS_DIR / "group_impact.csv", usecols=["grupo", "concentracao_membro_top", "gini_membros", "fwci_mediano"])
    freeman = freeman.merge(impact.rename(columns={"grupo": "comunidade"}), on="comunidade", how="left")
    freeman.to_csv(RESULTS_DIR / "community_centralization.csv", index=False, encoding="utf-8")
    results["freeman"] = {
        "n_comunidades": len(freeman),
        "mediana": float(freeman["centralizacao_freeman"].median()),
        "spearman_tamanho": float(spearmanr(freeman["tamanho"], freeman["centralizacao_freeman"])[0]),
        "spearman_gini_impacto": float(spearmanr(freeman["centralizacao_freeman"], freeman["gini_membros"], nan_policy="omit")[0]),
        "spearman_concentracao_membro_top": float(spearmanr(freeman["centralizacao_freeman"], freeman["concentracao_membro_top"], nan_policy="omit")[0]),
        "spearman_fwci_mediano": float(spearmanr(freeman["centralizacao_freeman"], freeman["fwci_mediano"], nan_policy="omit")[0]),
        "mais_centralizadas_ge_100": freeman[freeman["tamanho"] >= 100].nlargest(5, "centralizacao_freeman").round(3).to_dict("records"),
    }

    print("Snapshots anuais...", flush=True)
    snapshots = yearly_snapshots(pub_graph)
    results["snapshots_anuais"] = snapshots
    plot_snapshots(snapshots)

    print("Tríades (citação entre docentes)...", flush=True)
    results["triades_docentes"] = triads_vs_null(author_graph)

    save_json(json.loads(json.dumps(results, default=float)), "structural_properties.json")


if __name__ == "__main__":
    main()
