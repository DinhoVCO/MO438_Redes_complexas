#!/usr/bin/env python3
"""Detecção de comunidades na rede de coautoria.

Duas escalas:
- componente gigante (ponderada por `weight` = nº de coautorias): Louvain com 50 sementes,
  Leiden, Fast-Greedy (CNM), Label Propagation e Fluid Communities;
- subgrafo reduzido (coautoria entre docentes CAPES das maiores IES, sem pesos): os mesmos
  algoritmos e mais Girvan–Newman, com a curva Q × nº de comunidades.

Também: significância de Q contra o modelo de configuração, NMI/ARI entre algoritmos,
validação contra a IES principal e o programa CAPES, e caracterização das maiores comunidades.

Partição principal: Louvain com a semente de maior Q. Os outros componentes conexos (fora da
componente gigante) viram, cada um, uma comunidade própria.

Saídas:
- results/communities.json                 — tabelas e métricas
- results/author_communities.csv           — autor → comunidade (todos os algoritmos)
- results/communities_summary.csv          — uma linha por comunidade (principal)
- figures/communities_gn_modularity.png    — curva de Girvan–Newman
- figures/communities_size_ccdf.png        — distribuição de tamanhos
- output/colaboracao_cc_brasil.graphml.gz  — atributo `comunidade` adicionado
"""
from __future__ import annotations

import random
import time
from collections import Counter, defaultdict

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from common import COAUTHORSHIP_PATH, FIGURES_DIR, RESULTS_DIR, config, load_graph, save_json, setup_matplotlib
from src.build_network import _short_id, iter_works_jsonl, load_author_aliases

N_SEEDS = 50
N_NULL = 10
MIN_COMMUNITY_SIZE = 10
SUBGRAPH_TARGET = 300  # nº mínimo de docentes no subgrafo reduzido
GN_MAX_COMMUNITIES = 60
GN_TIME_LIMIT = 900  # segundos
TOP_COMMUNITIES = 15


# ---------------------------------------------------------------- algoritmos

def louvain_seeds(g: ig.Graph, weights: str | None, n_seeds: int) -> dict:
    """Louvain (igraph multilevel) com seed = 0…n-1; o igraph usa o RNG do módulo random."""
    runs = []
    for seed in range(n_seeds):
        random.seed(seed)
        start = time.perf_counter()
        membership = g.community_multilevel(weights=weights).membership
        elapsed = time.perf_counter() - start
        runs.append((seed, g.modularity(membership, weights=weights), membership, elapsed))
    best_seed, best_q, best_membership, _ = max(runs, key=lambda r: r[1])
    qs = np.array([r[1] for r in runs])
    # Estabilidade: quão parecidas são as partições das outras sementes com a melhor
    nmi_vs_best = [normalized_mutual_info_score(best_membership, r[2]) for r in runs if r[0] != best_seed]
    return {
        "membership": best_membership,
        "seed": best_seed,
        "q_max": float(qs.max()),
        "q_medio": float(qs.mean()),
        "q_desvio": float(qs.std()),
        "q_min": float(qs.min()),
        "nmi_medio_vs_melhor": float(np.mean(nmi_vs_best)),
        "tempo_medio_s": float(np.mean([r[3] for r in runs])),
    }


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def fluid_membership(g: ig.Graph, k: int, seed: int = 0) -> list[int]:
    """Fluid Communities (networkx; exige grafo conexo e k fixo)."""
    nx_graph = nx.Graph(g.get_edgelist())
    nx_graph.add_nodes_from(range(g.vcount()))
    communities = list(nx.community.asyn_fluidc(nx_graph, k, max_iter=100, seed=seed))
    membership = [-1] * g.vcount()
    for cid, nodes in enumerate(communities):
        for v in nodes:
            membership[v] = cid
    # Vértices que o algoritmo não alcançou (não deveria ocorrer com max_iter alto)
    leftover = [v for v, c in enumerate(membership) if c < 0]
    for v in leftover:
        membership[v] = len(communities)
    return membership


def run_all(g: ig.Graph, weights: str | None) -> tuple[dict, dict]:
    """Roda todos os algoritmos rápidos; devolve partições e a tabela Q / nº / tempo."""
    partitions: dict[str, list[int]] = {}
    table: dict[str, dict] = {}

    louvain = louvain_seeds(g, weights, N_SEEDS)
    partitions["louvain"] = louvain.pop("membership")
    table["louvain"] = {**louvain, "tempo_s": louvain["tempo_medio_s"]}

    random.seed(0)
    leiden, t = timed(lambda: g.community_leiden(objective_function="modularity", weights=weights, n_iterations=-1))
    partitions["leiden"] = leiden.membership
    table["leiden"] = {"tempo_s": t}

    fastgreedy, t = timed(lambda: g.community_fastgreedy(weights=weights).as_clustering())
    partitions["fastgreedy"] = fastgreedy.membership
    table["fastgreedy"] = {"tempo_s": t}

    random.seed(0)
    lpa, t = timed(lambda: g.community_label_propagation(weights=weights))
    partitions["label_propagation"] = lpa.membership
    table["label_propagation"] = {"tempo_s": t}

    k = len(set(partitions["louvain"]))
    fluid, t = timed(lambda: fluid_membership(g, k))
    partitions["fluid"] = fluid
    table["fluid"] = {"tempo_s": t, "k": k}

    for name, membership in partitions.items():
        sizes = Counter(membership)
        table[name].update(
            {
                "q": g.modularity(membership, weights=weights),
                "n_comunidades": len(sizes),
                "maior_comunidade": max(sizes.values()),
                f"n_comunidades_ge_{MIN_COMMUNITY_SIZE}": sum(1 for s in sizes.values() if s >= MIN_COMMUNITY_SIZE),
            }
        )
        print(f"  {name}: Q = {table[name]['q']:.4f}, {len(sizes)} comunidades, {table[name]['tempo_s']:.1f} s", flush=True)
    return partitions, table


def pairwise_agreement(partitions: dict[str, list[int]]) -> dict:
    names = list(partitions)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a} x {b}"] = {
                "nmi": normalized_mutual_info_score(partitions[a], partitions[b]),
                "ari": adjusted_rand_score(partitions[a], partitions[b]),
            }
    return out


def girvan_newman_curve(g: ig.Graph) -> dict:
    """Girvan–Newman (networkx) com a curva Q × nº de comunidades, como my_GN da AP3."""
    nx_graph = nx.Graph(g.get_edgelist())
    nx_graph.add_nodes_from(range(g.vcount()))
    curve = []
    best = None
    start = time.perf_counter()
    for communities in nx.community.girvan_newman(nx_graph):
        q = nx.community.modularity(nx_graph, communities)
        curve.append((len(communities), q))
        if best is None or q > best[1]:
            best = (communities, q)
        if len(communities) >= GN_MAX_COMMUNITIES or time.perf_counter() - start > GN_TIME_LIMIT:
            break
    elapsed = time.perf_counter() - start
    membership = [0] * g.vcount()
    for cid, nodes in enumerate(best[0]):
        for v in nodes:
            membership[v] = cid
    return {"membership": membership, "curve": curve, "tempo_s": elapsed, "q": best[1]}


def null_model_modularity(g: ig.Graph, real_q: float) -> dict:
    """Q do Louvain em grafos aleatorizados pelo modelo de configuração (troca de arestas).

    Sem pesos dos dois lados: a troca de arestas preserva os graus, não os pesos.
    """
    qs = []
    for rep in range(N_NULL):
        random.seed(1000 + rep)
        shuffled = g.copy()
        shuffled.rewire(n=10 * g.ecount(), allowed_edge_types="simple")
        membership = shuffled.community_multilevel().membership
        qs.append(shuffled.modularity(membership))
    qs = np.array(qs)
    return {
        "q_real_sem_pesos": real_q,
        "q_nulo_medio": float(qs.mean()),
        "q_nulo_desvio": float(qs.std()),
        "z": float((real_q - qs.mean()) / qs.std()) if qs.std() > 0 else None,
        "n_aleatorizacoes": N_NULL,
        "trocas_por_aresta": 10,
    }


# ---------------------------------------------------------------- validação

def validation(membership: list[int], labels: list[str | None]) -> dict:
    """NMI/ARI contra um rótulo de referência, só nos vértices rotulados, com linha de base."""
    pairs = [(m, l) for m, l in zip(membership, labels) if l]
    comm = [p[0] for p in pairs]
    lab = [p[1] for p in pairs]
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(comm)
    return {
        "n_rotulados": len(pairs),
        "n_rotulos": len(set(lab)),
        "nmi": normalized_mutual_info_score(lab, comm),
        "ari": adjusted_rand_score(lab, comm),
        # Mesmas comunidades, rótulos embaralhados: o NMI de partições com muitas classes não é zero
        "nmi_embaralhado": normalized_mutual_info_score(lab, shuffled),
        "ari_embaralhado": adjusted_rand_score(lab, shuffled),
    }


def purity_stats(membership: list[int], labels: list[str | None]) -> dict:
    """Pureza (fração da IES dominante) e nº de IES por comunidade; fragmentação de cada IES."""
    by_comm: dict[int, Counter] = defaultdict(Counter)
    for m, l in zip(membership, labels):
        if l:
            by_comm[m][l] += 1
    sizes = Counter(membership)
    purities, n_labels, weights = [], [], []
    for cid, counts in by_comm.items():
        total = sum(counts.values())
        if sizes[cid] < MIN_COMMUNITY_SIZE or total < 5:
            continue
        purities.append(counts.most_common(1)[0][1] / total)
        n_labels.append(len(counts))
        weights.append(total)
    # Para cada IES: em quantas comunidades estão 50 % dos seus membros
    by_label: dict[str, Counter] = defaultdict(Counter)
    for m, l in zip(membership, labels):
        if l:
            by_label[l][m] += 1
    spread = []
    for counts in by_label.values():
        total, acc = sum(counts.values()), 0
        for n_comm, (_, c) in enumerate(counts.most_common(), start=1):
            acc += c
            if acc >= total / 2:
                spread.append(n_comm)
                break
    return {
        "n_comunidades_avaliadas": len(purities),
        "pureza_media": float(np.mean(purities)),
        "pureza_media_ponderada": float(np.average(purities, weights=weights)),
        "pureza_mediana": float(np.median(purities)),
        "n_ies_por_comunidade_mediana": float(np.median(n_labels)),
        "comunidades_para_50pct_da_ies_mediana": float(np.median(spread)),
        "comunidades_para_50pct_da_ies_media": float(np.mean(spread)),
    }


# ---------------------------------------------------------------- caracterização

def community_topics(community_of: dict[str, int]) -> tuple[dict, dict, dict]:
    """Conta obras, subáreas e tópicos (primary_topic) por comunidade, lendo o corpus em streaming."""
    aliases = load_author_aliases(config.FACULTY_MATCH_CSV)
    n_works: Counter = Counter()
    subfields: dict[int, Counter] = defaultdict(Counter)
    topics: dict[int, Counter] = defaultdict(Counter)
    seen: set[str] = set()
    for work in iter_works_jsonl(config.OBRAS_JSONL):
        wid = _short_id(work.get("id"))
        if not wid or wid in seen:
            continue
        seen.add(wid)
        comms = set()
        for auth in work.get("authorships") or []:
            aid = _short_id((auth.get("author") or {}).get("id"))
            aid = aliases.get(aid, aid)
            if aid in community_of:
                comms.add(community_of[aid])
        topic = work.get("primary_topic") or {}
        subfield = (topic.get("subfield") or {}).get("display_name", "")
        for cid in comms:
            n_works[cid] += 1
            if topic:
                topics[cid][topic.get("display_name", "")] += 1
                subfields[cid][subfield] += 1
    return n_works, subfields, topics


def characterize(graph: nx.Graph, community_of: dict[str, int]) -> tuple[pd.DataFrame, list[dict]]:
    """Resumo de todas as comunidades e descrição detalhada das maiores."""
    members: dict[int, list[str]] = defaultdict(list)
    for author, cid in community_of.items():
        members[cid].append(author)
    print("Lendo tópicos do corpus...", flush=True)
    n_works, subfields, topics = community_topics(community_of)

    rows, top = [], []
    for cid in sorted(members, key=lambda c: -len(members[c])):
        nodes = members[cid]
        ies = Counter(graph.nodes[a]["ies_principal"] for a in nodes if graph.nodes[a].get("ies_principal"))
        n_scope = sum(ies.values())
        dominant, dominant_n = ies.most_common(1)[0] if ies else ("", 0)
        sub = subfields[cid].most_common(1)
        rows.append(
            {
                "comunidade": cid,
                "tamanho": len(nodes),
                "n_docentes_capes": sum(1 for a in nodes if graph.nodes[a].get("docente_capes")),
                "n_afiliacao_escopo": n_scope,
                "ies_dominante": dominant,
                "pureza_ies": dominant_n / n_scope if n_scope else np.nan,
                "n_ies": len(ies),
                "n_obras": n_works[cid],
                "subarea_dominante": sub[0][0] if sub else "",
                "topico_dominante": topics[cid].most_common(1)[0][0] if topics[cid] else "",
            }
        )
        if len(top) < TOP_COMMUNITIES:
            sub_graph = graph.subgraph(nodes)
            central = sorted(sub_graph.degree(weight="weight"), key=lambda x: -x[1])[:5]
            top.append(
                {
                    "comunidade": cid,
                    "tamanho": len(nodes),
                    "n_docentes_capes": rows[-1]["n_docentes_capes"],
                    "ies": [(k, round(v / n_scope, 3)) for k, v in ies.most_common(3)] if n_scope else [],
                    "subareas": [(k, v) for k, v in subfields[cid].most_common(3)],
                    "topicos": [(k, v) for k, v in topics[cid].most_common(3)],
                    "autores_centrais": [graph.nodes[a]["label"] for a, _ in central],
                }
            )
    return pd.DataFrame(rows), top


# ---------------------------------------------------------------- figuras

def plot_gn_curve(curve: list[tuple[int, float]], louvain_q: float) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ks, qs = zip(*curve)
    ax.plot(ks, qs, marker=".", label="Girvan–Newman")
    best = int(np.argmax(qs))
    ax.scatter([ks[best]], [qs[best]], color="C3", zorder=3, label=f"Q máx. = {qs[best]:.3f} ({ks[best]} com.)")
    ax.axhline(louvain_q, color="C2", linestyle="--", label=f"Louvain (Q = {louvain_q:.3f})")
    ax.set_xlabel("Número de comunidades")
    ax.set_ylabel("Modularidade Q")
    ax.set_title("Girvan–Newman no subgrafo de docentes")
    ax.legend()
    path = FIGURES_DIR / "communities_gn_modularity.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


def plot_size_ccdf(partitions: dict[str, list[int]]) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for name, membership in partitions.items():
        sizes = np.sort(np.array(list(Counter(membership).values())))
        ccdf = 1.0 - np.arange(len(sizes)) / len(sizes)
        ax.loglog(sizes, ccdf, marker=".", linestyle="none", markersize=4, label=name)
    ax.set_xlabel("Tamanho da comunidade s")
    ax.set_ylabel("P(S ≥ s)")
    ax.set_title("Tamanhos das comunidades na componente gigante")
    ax.legend(fontsize=8)
    path = FIGURES_DIR / "communities_size_ccdf.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


# ---------------------------------------------------------------- principal

def faculty_subgraph(graph: nx.Graph) -> tuple[nx.Graph, list[str]]:
    """Coautoria entre docentes CAPES das IES com mais docentes (maior componente conexa)."""
    faculty = [v for v, d in graph.nodes(data=True) if d.get("docente_capes")]
    by_ies = Counter(graph.nodes[v]["ies_principal"] for v in faculty if graph.nodes[v].get("ies_principal"))
    chosen, total = [], 0
    for ies, n in by_ies.most_common():
        chosen.append(ies)
        total += n
        if total >= SUBGRAPH_TARGET:
            break
    nodes = [v for v in faculty if graph.nodes[v].get("ies_principal") in chosen]
    sub = graph.subgraph(nodes)
    lcc = max(nx.connected_components(sub), key=len)
    return nx.Graph(sub.subgraph(lcc)), chosen


def to_igraph(graph: nx.Graph) -> tuple[ig.Graph, list[str]]:
    names = list(graph.nodes())
    index = {v: i for i, v in enumerate(names)}
    g = ig.Graph(n=len(names), edges=[(index[a], index[b]) for a, b in graph.edges()])
    g.es["weight"] = [float(w) for _, _, w in graph.edges(data="weight", default=1)]
    g.vs["name"] = names
    return g, names


def main() -> None:
    graph = load_graph(COAUTHORSHIP_PATH)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    giant = graph.subgraph(components[0])
    g, names = to_igraph(giant)
    print(f"Componente gigante: {g.vcount()} vértices, {g.ecount()} arestas", flush=True)
    results: dict = {
        "componentes": {
            "n_componentes": len(components),
            "componente_gigante": len(components[0]),
            "n_componentes_ge_10_fora_da_gigante": sum(1 for c in components[1:] if len(c) >= MIN_COMMUNITY_SIZE),
        }
    }

    # --- escala 1: componente gigante, com pesos
    print("Componente gigante (ponderada):", flush=True)
    partitions, table = run_all(g, "weight")
    results["componente_gigante"] = {"tabela": table, "concordancia": pairwise_agreement(partitions)}

    # Q do Louvain sem pesos, para comparar com o modelo nulo (também sem pesos)
    random.seed(table["louvain"]["seed"])
    q_unweighted = g.modularity(g.community_multilevel().membership)
    print("Modelo de configuração...", flush=True)
    results["modelo_configuracao"] = null_model_modularity(g, q_unweighted)
    print(f"  Q real = {q_unweighted:.4f}; Q nulo = {results['modelo_configuracao']['q_nulo_medio']:.4f}", flush=True)

    # Validação contra IES principal (autores do escopo) e programa CAPES (docentes de um só programa)
    ies_labels = [giant.nodes[v].get("ies_principal") or None for v in names]
    program_labels = []
    for v in names:
        prog = giant.nodes[v].get("programa_capes") or ""
        program_labels.append(prog if prog and "|" not in prog else None)
    results["validacao"] = {
        name: {
            "ies_principal": validation(m, ies_labels),
            "programa_capes": validation(m, program_labels),
            "pureza_ies": purity_stats(m, ies_labels),
        }
        for name, m in partitions.items()
    }
    louvain_validation = results["validacao"]["louvain"]
    print(
        f"  Louvain × IES: NMI = {louvain_validation['ies_principal']['nmi']:.3f}, "
        f"ARI = {louvain_validation['ies_principal']['ari']:.3f}; "
        f"pureza ponderada = {louvain_validation['pureza_ies']['pureza_media_ponderada']:.3f}",
        flush=True,
    )

    # --- escala 2: subgrafo reduzido, sem pesos, com Girvan–Newman
    sub_nx, chosen_ies = faculty_subgraph(graph)
    sub_g, sub_names = to_igraph(sub_nx)
    print(f"Subgrafo de docentes ({', '.join(chosen_ies)}): {sub_g.vcount()} vértices, {sub_g.ecount()} arestas", flush=True)
    sub_partitions, sub_table = run_all(sub_g, None)
    print("  Girvan–Newman...", flush=True)
    gn = girvan_newman_curve(sub_g)
    sub_partitions["girvan_newman"] = gn["membership"]
    sub_table["girvan_newman"] = {
        "q": gn["q"],
        "n_comunidades": len(set(gn["membership"])),
        "tempo_s": gn["tempo_s"],
        "niveis_percorridos": len(gn["curve"]),
    }
    print(f"  girvan_newman: Q = {gn['q']:.4f}, {sub_table['girvan_newman']['n_comunidades']} comunidades, {gn['tempo_s']:.0f} s")
    sub_ies = [sub_nx.nodes[v].get("ies_principal") or None for v in sub_names]
    results["subgrafo_docentes"] = {
        "ies": chosen_ies,
        "n_vertices": sub_g.vcount(),
        "n_arestas": sub_g.ecount(),
        "tabela": sub_table,
        "concordancia": pairwise_agreement(sub_partitions),
        "validacao_ies": {name: validation(m, sub_ies) for name, m in sub_partitions.items()},
        "curva_gn": gn["curve"],
    }

    # --- partição principal: Louvain na gigante + um grupo por componente menor
    louvain = partitions["louvain"]
    order = [cid for cid, _ in Counter(louvain).most_common()]  # id 0 = maior comunidade
    relabel = {cid: i for i, cid in enumerate(order)}
    community_of = {names[i]: relabel[c] for i, c in enumerate(louvain)}
    next_id = len(order)
    for comp in components[1:]:
        for v in comp:
            community_of[v] = next_id
        next_id += 1

    summary, top = characterize(graph, community_of)
    summary.to_csv(RESULTS_DIR / "communities_summary.csv", index=False, encoding="utf-8")
    results["maiores_comunidades"] = top
    big = summary[summary["tamanho"] >= MIN_COMMUNITY_SIZE]
    results["corte_tamanho"] = {
        "tamanho_minimo": MIN_COMMUNITY_SIZE,
        "n_comunidades": int(len(big)),
        "frac_autores_cobertos": float(big["tamanho"].sum() / graph.number_of_nodes()),
        "frac_docentes_cobertos": float(
            big["n_docentes_capes"].sum() / max(summary["n_docentes_capes"].sum(), 1)
        ),
    }

    rows = []
    giant_index = {v: i for i, v in enumerate(names)}
    for author in graph.nodes():
        row = {"author_id": author, "comunidade": community_of[author], "componente_gigante": author in giant_index}
        for name, membership in partitions.items():
            row[f"comunidade_{name}"] = membership[giant_index[author]] if author in giant_index else -1
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "author_communities.csv", index=False, encoding="utf-8")
    print(f"Partições: {RESULTS_DIR / 'author_communities.csv'}")

    save_json(results, "communities.json")
    plot_gn_curve(gn["curve"], sub_table["louvain"]["q"])
    plot_size_ccdf(partitions)

    for author, cid in community_of.items():
        graph.nodes[author]["comunidade"] = cid
    nx.write_graphml(graph, COAUTHORSHIP_PATH)
    print(f"Atributo `comunidade` gravado em {COAUTHORSHIP_PATH}")


if __name__ == "__main__":
    main()
