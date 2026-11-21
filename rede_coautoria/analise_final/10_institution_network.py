#!/usr/bin/env python3
"""Rede de coautoria entre os docentes CAPES de uma instituição, agrupada por comunidade.

Mostra quantos grupos (comunidades de coautoria, Louvain) existem dentro de uma IES e quem
liga um grupo ao outro. Vértice = docente CAPES com a IES como IES principal; aresta = obras
em coautoria entre dois docentes; espessura = nº de obras.

Desenho em dois níveis: cada comunidade é um círculo (área ∝ nº de docentes), os círculos
são empacotados em espiral (maiores no centro) e os docentes se distribuem dentro do seu
círculo pelo layout de molas. Arestas entre grupos ficam tracejadas.

O tópico de cada grupo é o tópico principal mais frequente nas obras dos docentes da IES
naquele grupo (e não o da comunidade nacional inteira, que mistura outras IES). Os tópicos
ficam guardados em results/ para não reler o corpus a cada execução.

Uso: python 10_institution_network.py [--ies UNICAMP]

Saídas: figures/network_<ies>_docentes.png, results/<ies>_docentes_grupos.csv e
        results/<ies>_grupos_topicos.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from common import COAUTHORSHIP_PATH, FIGURES_DIR, RESULTS_DIR, config, load_graph, setup_matplotlib
from src.build_network import _short_id, iter_works_jsonl, load_author_aliases

MIN_GROUP_DOCENTES = 3
N_COLORED = 3  # em grafo, só 3 cores categóricas passam no teste de daltonismo; o resto vai em cinza
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
OTHER = "#9d9c95"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
EDGE = "#8a8984"
RING = "#b9b8b2"
UNIT = 0.55  # raio de um grupo com 1 docente; cresce com √n
RING_MIN = 6  # grupos com pelo menos 6 docentes viram anel
CENTER_LABEL_MIN = 10  # anéis com pelo menos 10 docentes levam o rótulo no centro, que fica vazio
RING_RADIUS = UNIT * math.sqrt(RING_MIN)
RING_PAD = 0.9  # folga extra por anel no empacotamento (nomes ficam do lado de fora)


def short_name(name: str) -> str:
    """Primeiro e último nome, para rótulos curtos."""
    parts = [p for p in name.replace(".", ". ").split() if p]
    return parts[0] if len(parts) == 1 else f"{parts[0]} {parts[-1]}"


def group_topics(faculty: set[str], community_of: dict[str, int]) -> dict[int, list[tuple[str, int]]]:
    """Tópicos principais das obras dos docentes da IES, por comunidade (corpus em streaming)."""
    aliases = load_author_aliases(config.FACULTY_MATCH_CSV)
    topics: dict[int, Counter] = defaultdict(Counter)
    seen: set[str] = set()
    for work in iter_works_jsonl(config.OBRAS_JSONL):
        wid = _short_id(work.get("id"))
        if not wid or wid in seen:
            continue
        seen.add(wid)
        groups = set()
        for auth in work.get("authorships") or []:
            aid = _short_id((auth.get("author") or {}).get("id"))
            aid = aliases.get(aid, aid)
            if aid in faculty:
                groups.add(community_of[aid])
        topic = (work.get("primary_topic") or {}).get("display_name")
        if topic:
            for g in groups:
                topics[g][topic] += 1
    return {g: c.most_common(3) for g, c in topics.items()}


def pack_circles(radii: dict[int, float], gap: float = 0.45) -> dict[int, np.ndarray]:
    """Empacota círculos em espiral, do maior para o menor, sem sobreposição."""
    centers: dict[int, np.ndarray] = {}
    for c in sorted(radii, key=lambda c: (-radii[c], c)):
        r = radii[c]
        if not centers:
            centers[c] = np.zeros(2)
            continue
        t = 0.0
        while True:
            t += 0.05
            p = np.array([math.cos(t), 0.85 * math.sin(t)]) * (0.35 * t)
            # Anéis levam os nomes para fora do círculo: precisam de folga extra
            if all(np.linalg.norm(p - centers[o]) >= r + radii[o] + gap + RING_PAD * ((r >= RING_RADIUS) + (radii[o] >= RING_RADIUS))
                   for o in centers):
                centers[c] = p
                break
    return centers


def grouped_layout(sub: nx.Graph, community_of: dict[str, int]) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray], dict[int, float]]:
    members: dict[int, list[str]] = defaultdict(list)
    for v in sub:
        members[community_of[v]].append(v)
    radii = {c: UNIT * math.sqrt(len(nodes)) for c, nodes in members.items()}
    centers = pack_circles(radii)
    pos: dict[str, np.ndarray] = {}
    for c, nodes in members.items():
        if len(nodes) == 1:
            pos[nodes[0]] = centers[c]
            continue
        group = sub.subgraph(nodes)
        # Sem coautor dentro do grupo: vão para a borda do círculo, sem distorcer a escala dos demais
        loose = [v for v in nodes if group.degree(v) == 0]
        linked = [v for v in nodes if group.degree(v) > 0]
        for i, v in enumerate(loose):
            angle = -math.pi / 2 + 2 * math.pi * i / max(len(loose), 1)
            pos[v] = centers[c] + 0.82 * radii[c] * np.array([math.cos(angle), math.sin(angle)])
        if len(nodes) >= RING_MIN:
            # Grupo grande: anel ordenado pelo ângulo no layout de molas (coautores ficam vizinhos)
            local = nx.spring_layout(group, weight="weight", seed=0, iterations=500)
            center = np.median(np.array(list(local.values())), axis=0)
            order = sorted(nodes, key=lambda v: math.atan2(*(local[v] - center)[::-1]))
            for i, v in enumerate(order):
                angle = math.pi / 2 - 2 * math.pi * i / len(order)
                pos[v] = centers[c] + 0.86 * radii[c] * np.array([math.cos(angle), math.sin(angle)])
            continue
        if len(linked) == 1:
            pos[linked[0]] = centers[c]
        elif linked:
            local = nx.kamada_kawai_layout(group.subgraph(linked)) if nx.is_connected(group.subgraph(linked)) else \
                nx.spring_layout(group.subgraph(linked), seed=0, k=2.0 / math.sqrt(len(linked)), iterations=500)
            coords = np.array([local[v] for v in linked])
            coords -= np.median(coords, axis=0)
            # Escala pelo percentil 80 (um docente periférico não pode espremer o núcleo) e
            # quem passa do raio fica na borda
            norms = np.linalg.norm(coords, axis=1)
            coords /= np.percentile(norms, 80) or 1.0
            norms = np.linalg.norm(coords, axis=1)
            coords[norms > 1] = coords[norms > 1] / norms[norms > 1, None]
            inner = 0.62 if loose else 0.78
            for v, xy in zip(linked, coords):
                pos[v] = centers[c] + xy * radii[c] * inner
    return pos, centers, radii


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ies", default="UNICAMP")
    args = parser.parse_args()
    ies = args.ies
    slug = ies.lower().replace(" ", "_")

    graph = load_graph(COAUTHORSHIP_PATH)
    faculty = [v for v, d in graph.nodes(data=True) if d.get("docente_capes") and d.get("ies_principal") == ies]
    sub = nx.Graph(graph.subgraph(faculty))
    community_of = {v: int(graph.nodes[v]["comunidade"]) for v in faculty}
    print(f"{ies}: {len(faculty)} docentes, {sub.number_of_edges()} arestas de coautoria entre eles")

    sizes = Counter(community_of.values())
    groups = [c for c, n in sizes.most_common() if n >= MIN_GROUP_DOCENTES]
    color_of = {c: SERIES[i] for i, c in enumerate(groups[:N_COLORED])}

    topics_path = RESULTS_DIR / f"{slug}_grupos_topicos.json"
    if topics_path.exists():
        topics = {int(k): [tuple(t) for t in v] for k, v in json.loads(topics_path.read_text(encoding="utf-8")).items()}
    else:
        print("Lendo tópicos das obras dos docentes...", flush=True)
        topics = group_topics(set(faculty), community_of)
        topics_path.write_text(json.dumps(topics, ensure_ascii=False, indent=1), encoding="utf-8")

    pos, centers, radii = grouped_layout(sub, community_of)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_axis_off()
    ax.set_aspect("equal")

    # Círculo de cada grupo e rótulo (identidade do grupo sem depender da cor)
    for c, center in centers.items():
        r = radii[c]
        color = color_of.get(c, OTHER)
        ax.add_patch(plt.Circle(center, r, facecolor=color, alpha=0.06, edgecolor=RING, linewidth=0.9, linestyle=":", zorder=0))
        if sizes[c] >= MIN_GROUP_DOCENTES:
            # Grupos grandes: dois tópicos; pequenos: um, para os rótulos não se sobreporem
            names = [t for t, _ in topics.get(c, [])[:2 if sizes[c] >= 6 else 1]]
            ring = sizes[c] >= CENTER_LABEL_MIN
            width = 24 if ring else 34
            topic = "\n".join(n if len(n) <= width else n[:width - 1] + "…" for n in names)
            above = center[1] + r + (0.45 if sizes[c] >= RING_MIN else 0.08)  # anel: acima dos nomes de fora
            ax.text(center[0], center[1] if ring else above, f"C{c} · {sizes[c]} docentes\n{topic}",
                    ha="center", va="center" if ring else "bottom", fontsize=7.5, color=INK,
                    fontweight="bold" if ring else "normal", linespacing=1.2, zorder=5)
        else:
            ax.text(center[0], center[1] + r + 0.04, f"C{c}", ha="center", va="bottom", fontsize=6.5, color=INK_SECONDARY, zorder=5)

    # Arestas: espessura ∝ obras em conjunto; entre grupos ficam tracejadas (pontes)
    wmax = max((d["weight"] for _, _, d in sub.edges(data=True)), default=1)
    for a, b, d in sub.edges(data=True):
        bridge = community_of[a] != community_of[b]
        s = math.sqrt(d["weight"] / wmax)
        (xa, ya), (xb, yb) = pos[a], pos[b]
        ax.plot([xa, xb], [ya, yb], color=EDGE, alpha=(0.3 if bridge else 0.45) + 0.4 * s,
                linewidth=0.5 + 3.0 * s, linestyle=(0, (3, 2)) if bridge else "-", zorder=1)

    # Docentes: área ∝ nº de coautores na rede inteira; cor = grupo (3 maiores) ou cinza
    total_degree = np.array([graph.degree(v) for v in faculty], dtype=float)
    node_sizes = 25 + 330 * np.sqrt(total_degree / total_degree.max())
    colors = [color_of.get(community_of[v], OTHER) for v in faculty]
    ax.scatter([pos[v][0] for v in faculty], [pos[v][1] for v in faculty], s=node_sizes, c=colors,
               edgecolor="white", linewidth=1.3, zorder=3)
    # Nos anéis, a distância do nome ao círculo alterna entre vizinhos (perto/longe)
    ring_rank: dict[str, int] = {}
    for c in {community_of[v] for v in faculty if sizes[community_of[v]] >= RING_MIN}:
        members_c = [v for v in faculty if community_of[v] == c]
        members_c.sort(key=lambda v: math.atan2(*(pos[v] - centers[c])[::-1]))
        ring_rank.update({v: i for i, v in enumerate(members_c)})
    for v in faculty:
        c = community_of[v]
        if sizes[c] >= RING_MIN:
            # Anel: nome para fora do círculo, na direção do docente, para vizinhos não se sobreporem
            dx, dy = pos[v] - centers[c]
            norm = math.hypot(dx, dy) or 1.0
            ux, uy = dx / norm, dy / norm
            ax.annotate(short_name(graph.nodes[v]["label"]), pos[v], xytext=((8 + 9 * (ring_rank[v] % 2)) * ux, (8 + 9 * (ring_rank[v] % 2)) * uy), textcoords="offset points",
                        ha="left" if ux > 0.3 else ("right" if ux < -0.3 else "center"),
                        va="bottom" if uy > 0.3 else ("top" if uy < -0.3 else "center"), fontsize=6.3, color=INK, zorder=4)
        else:
            ax.annotate(short_name(graph.nodes[v]["label"]), pos[v], xytext=(0, -7), textcoords="offset points",
                        ha="center", va="top", fontsize=6.3, color=INK, zorder=4)

    xs = [p[0] for p in centers.values()]
    ys = [p[1] for p in centers.values()]
    margin = max(radii.values()) + 0.6
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin + 0.6)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=color_of[c],
                          markeredgecolor="white", label=f"C{c}") for c in groups[:N_COLORED]]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=OTHER, markeredgecolor="white", label="outros grupos"))
    handles.append(plt.Line2D([], [], color=EDGE, linewidth=1.5, label="coautoria no mesmo grupo"))
    handles.append(plt.Line2D([], [], color=EDGE, linewidth=1.5, linestyle=(0, (3, 2)), label="coautoria entre grupos"))
    fig.legend(handles=handles, loc="lower right", fontsize=8, frameon=False, ncol=3)
    n_groups = len(sizes)
    ax.set_title(f"{ies}: coautoria entre os {len(faculty)} docentes do programa, por comunidade\n"
                 f"{n_groups} comunidades; {len(groups)} com pelo menos {MIN_GROUP_DOCENTES} docentes", fontsize=11, color=INK)
    fig.text(0.02, 0.02, "Círculo pontilhado = comunidade (área ∝ docentes). Ponto = docente (área ∝ coautores na rede inteira).\n"
             "Linha: publicações em conjunto entre dois docentes. Comunidades com 1–2 docentes: o resto do grupo está em outras IES.",
             fontsize=7.5, color=INK_SECONDARY, va="bottom")
    path = FIGURES_DIR / f"network_{slug}_docentes.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")

    rows = [{
        "author_id": v,
        "nome": graph.nodes[v]["label"],
        "comunidade": community_of[v],
        "topico_grupo": topics.get(community_of[v], [("", 0)])[0][0],
        "coautores_docentes_ies": sub.degree(v),
        "coautores_total": graph.degree(v),
        "pontes": sum(1 for u in sub[v] if community_of[u] != community_of[v]),
    } for v in faculty]
    table = pd.DataFrame(rows).sort_values(["comunidade", "coautores_total"], ascending=[True, False])
    table.to_csv(RESULTS_DIR / f"{slug}_docentes_grupos.csv", index=False, encoding="utf-8")
    print(f"Tabela: {RESULTS_DIR / f'{slug}_docentes_grupos.csv'}")
    for c in groups:
        print(f"  C{c}: {sizes[c]} docentes; tópicos: {topics.get(c)}")


if __name__ == "__main__":
    main()
