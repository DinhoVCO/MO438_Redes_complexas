#!/usr/bin/env python3
"""Mapas do Brasil: colaboração, citação e comunidades por instituição (IES).

Vértice = IES principal dos autores, posicionada pelas coordenadas da OpenAlex. Os autores
não têm localização própria; cada um conta na IES do escopo onde mais publica.

Dados geográficos (baixados uma vez e guardados em data/geo/, para rodar sem rede depois):
- data/geo/instituicoes_geo.csv  — latitude/longitude das IES (OpenAlex /institutions)
- data/geo/brasil_ufs.geojson    — contorno das UFs (malha do IBGE, qualidade mínima)

Saídas:
- figures/map_ies_coauthorship.png — coautoria entre IES (pares de coautores)
- figures/map_ies_citation.png     — citação entre IES (peso fracionário, nos dois sentidos)
- figures/map_communities.png      — onde estão os membros das 6 maiores comunidades
- mapas/mapa_interativo.html       — versão interativa (Leaflet) para a apresentação
- results/ies_coauthorship_edges.csv — arestas de coautoria entre IES
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import requests

from common import ANALYSIS_DIR, COAUTHORSHIP_PATH, FIGURES_DIR, RESULTS_DIR, config, load_graph, setup_matplotlib
from src.build_network import load_scope_sigla
from src.openalex_client import OpenAlexClient

GEO_DIR = config.DATA_DIR / "geo"
IES_GEO_CSV = GEO_DIR / "instituicoes_geo.csv"
UF_GEOJSON = GEO_DIR / "brasil_ufs.geojson"
IBGE_URL = "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR"
MAPS_DIR = ANALYSIS_DIR / "mapas"

# A OpenAlex dá as coordenadas da sede da instituição. A UNESP entra inteira no escopo, mas o
# programa CAPES é o do campus de São José do Rio Preto (IBILCE), então o ponto vai para lá.
# A USP fica na sede (São Paulo): na rede, os campi de São Paulo, São Carlos e Ribeirão Preto
# são uma só IES.
GEO_OVERRIDES = {
    "UNESP-SÃO JOSÉ DO RIO PRETO": {"cidade": "São José do Rio Preto", "latitude": -20.8197, "longitude": -49.3797},
}

N_EDGES_FIGURE = 150
N_LABELS = 16
N_COMMUNITIES_FIGURE = 6
N_COMMUNITIES_HTML = 15
N_EDGES_HTML = 600

# Paleta (referência do skill de visualização): um tom por camada
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
LAND = "#ecebe7"
BORDER = "#ffffff"
MUTED = "#b9b8b2"


# ---------------------------------------------------------------- dados geográficos

def fetch_geo() -> None:
    """Baixa as coordenadas das IES e o contorno das UFs, se ainda não estão em disco."""
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    if not IES_GEO_CSV.exists():
        institutions = pd.read_csv(config.INSTITUICOES_CSV, dtype=str)
        ids = sorted(set(institutions["openalex_id"].dropna()))
        data = OpenAlexClient()._get(
            "/institutions",
            {"filter": "openalex:" + "|".join(ids), "per-page": 100, "select": "id,display_name,geo"},
        )
        scope = load_scope_sigla(config.INSTITUICOES_CSV)
        rows = []
        for inst in data["results"]:
            oa_id = inst["id"].rsplit("/", 1)[-1]
            geo = inst.get("geo") or {}
            rows.append({
                "openalex_id": oa_id,
                "ies_principal": scope.get(oa_id, ""),
                "nome": inst["display_name"],
                "cidade": geo.get("city", ""),
                "regiao": geo.get("region", ""),
                "latitude": geo.get("latitude"),
                "longitude": geo.get("longitude"),
            })
        pd.DataFrame(rows).to_csv(IES_GEO_CSV, index=False, encoding="utf-8")
        print(f"Coordenadas: {IES_GEO_CSV} ({len(rows)} IES)")
    if not UF_GEOJSON.exists():
        resp = requests.get(
            IBGE_URL,
            params={"formato": "application/vnd.geo+json", "qualidade": "minima", "intrarregiao": "UF"},
            timeout=60,
        )
        resp.raise_for_status()
        UF_GEOJSON.write_text(json.dumps(resp.json()), encoding="utf-8")
        print(f"Malha das UFs: {UF_GEOJSON}")


def spread_overlaps(geo: pd.DataFrame, radius: float = 0.45) -> pd.DataFrame:
    """IES da mesma cidade ficariam sobrepostas: distribui em círculo ao redor do ponto."""
    geo = geo.copy()
    geo["lat_plot"], geo["lon_plot"] = geo["latitude"], geo["longitude"]
    key = geo["latitude"].round(1).astype(str) + "," + geo["longitude"].round(1).astype(str)
    for _, idx in geo.groupby(key).groups.items():
        if len(idx) < 2:
            continue
        for k, i in enumerate(sorted(idx, key=lambda i: -geo.at[i, "n_autores"])):
            angle = 2 * math.pi * k / len(idx)
            geo.at[i, "lat_plot"] = geo.at[i, "latitude"] + radius * math.sin(angle)
            geo.at[i, "lon_plot"] = geo.at[i, "longitude"] + radius * math.cos(angle)
    return geo


# ---------------------------------------------------------------- agregações

def build_tables(graph: nx.Graph) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """IES (atributos), arestas de coautoria e de citação entre IES, e membros por comunidade."""
    geo = pd.read_csv(IES_GEO_CSV)
    geo = geo[geo["ies_principal"].notna()].drop_duplicates("ies_principal").reset_index(drop=True)
    for sigla, values in GEO_OVERRIDES.items():
        for column, value in values.items():
            geo.loc[geo["ies_principal"] == sigla, column] = value

    impact = pd.read_csv(RESULTS_DIR / "institution_vs_capes.csv")
    citation = pd.read_csv(RESULTS_DIR / "ies_citation_metrics.csv").rename(columns={"grupo": "ies_principal"})
    ies = (
        geo.merge(impact[["ies_principal", "n_membros", "n_docentes_capes", "nota_maxima", "fwci_mediano", "citacoes_total"]], on="ies_principal", how="left")
        .merge(citation[["ies_principal", "pagerank", "frac_interna_saida"]], on="ies_principal", how="left")
        .rename(columns={"n_membros": "n_autores"})
    )
    ies["n_autores"] = ies["n_autores"].fillna(0)

    # Coautoria entre IES: pares de coautores com IES principal diferente
    pairs: Counter = Counter()
    internal: Counter = Counter()
    for a, b in graph.edges():
        ia, ib = graph.nodes[a].get("ies_principal"), graph.nodes[b].get("ies_principal")
        if not ia or not ib:
            continue
        if ia == ib:
            internal[ia] += 1
        else:
            pairs[tuple(sorted((ia, ib)))] += 1
    coauthor_edges = pd.DataFrame([{"origem": a, "destino": b, "pares_coautores": w} for (a, b), w in pairs.items()])
    ies["pares_internos"] = ies["ies_principal"].map(internal).fillna(0)
    total_pairs = defaultdict(int)
    for (a, b), w in pairs.items():
        total_pairs[a] += w
        total_pairs[b] += w
    ies["pares_externos"] = ies["ies_principal"].map(total_pairs).fillna(0)
    ies["frac_coautoria_interna"] = ies["pares_internos"] / (ies["pares_internos"] + ies["pares_externos"]).replace(0, np.nan)

    # Citação entre IES (rede da etapa 3.3), sem laços
    cit_graph = nx.read_graphml(config.OUTPUT_DIR / "citacoes_ies.graphml")
    citation_edges = pd.DataFrame([
        {"origem": a, "destino": b, "citacoes": d["weight_frac"]} for a, b, d in cit_graph.edges(data=True) if a != b
    ])

    # Membros de cada comunidade por IES
    communities = pd.read_csv(RESULTS_DIR / "author_communities.csv", usecols=["author_id", "comunidade"])
    communities["ies_principal"] = communities["author_id"].map(lambda a: graph.nodes[a].get("ies_principal") or None)
    members = communities.dropna(subset=["ies_principal"]).groupby(["comunidade", "ies_principal"]).size().rename("membros").reset_index()
    return ies, coauthor_edges, citation_edges, members


# ---------------------------------------------------------------- figuras

def draw_states(ax) -> None:
    shapes = json.loads(UF_GEOJSON.read_text(encoding="utf-8"))
    for feature in shapes["features"]:
        geom = feature["geometry"]
        polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for polygon in polygons:
            ring = np.array(polygon[0])
            ax.fill(ring[:, 0], ring[:, 1], facecolor=LAND, edgecolor=BORDER, linewidth=0.8, zorder=0)
    ax.set_aspect(1 / math.cos(math.radians(15)))
    ax.set_xlim(-74.5, -34)
    ax.set_ylim(-34.5, 6)
    ax.set_axis_off()


def node_size(n: pd.Series, scale: float = 0.12) -> pd.Series:
    return np.sqrt(n.clip(lower=1)) * scale * 30


def draw_edges(ax, edges: pd.DataFrame, pos: dict, weight: str, color: str, top: int) -> None:
    edges = edges.nlargest(top, weight)
    wmax = edges[weight].max()
    for row in edges.sort_values(weight).itertuples():
        a, b = getattr(row, "origem"), getattr(row, "destino")
        if a not in pos or b not in pos:
            continue
        w = getattr(row, weight) / wmax
        (xa, ya), (xb, yb) = pos[a], pos[b]
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=0.3 + 3.2 * math.sqrt(w), alpha=0.15 + 0.5 * math.sqrt(w), solid_capstyle="round", zorder=1)


# Rótulos curtos para siglas longas (o nome completo fica no HTML)
SHORT_LABELS = {"UNESP-SÃO JOSÉ DO RIO PRETO": "UNESP", "UFPB-JOÃO PESSOA": "UFPB"}
# Região ampliada no recorte: eixo São Paulo–Rio–Minas (lon_min, lon_max, lat_min, lat_max)
INSET_BOUNDS = (-49.6, -41.6, -24.4, -19.2)
# Deslocamentos alternados dos rótulos (pontos), para vizinhos não se sobreporem
LABEL_OFFSETS = [(5, 3), (5, -9), (-5, 3), (-5, -9)]


def label_of(sigla: str) -> str:
    return SHORT_LABELS.get(sigla, sigla)


def in_inset(lon: float, lat: float) -> bool:
    x0, x1, y0, y1 = INSET_BOUNDS
    return x0 <= lon <= x1 and y0 <= lat <= y1


def draw_labels(ax, rows: pd.DataFrame, fontsize: float = 6.5, min_dist: float = 1.0) -> None:
    """Rotula as linhas na ordem dada, pulando as que ficam a menos de `min_dist` graus de um
    rótulo já posto (prioridade para as primeiras, que são as maiores)."""
    placed: list[tuple[float, float]] = []
    k = 0
    for row in rows.itertuples():
        if any(math.hypot(row.lon_plot - x, row.lat_plot - y) < min_dist for x, y in placed):
            continue
        placed.append((row.lon_plot, row.lat_plot))
        dx, dy = LABEL_OFFSETS[k % len(LABEL_OFFSETS)]
        k += 1
        ax.annotate(label_of(row.ies_principal), (row.lon_plot, row.lat_plot), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", fontsize=fontsize, color=INK, zorder=4)


def draw_nodes(ax, ies: pd.DataFrame, size_col: str, color: str, labels: pd.DataFrame | None = None,
               scale: float = 0.12, min_dist: float = 1.0) -> None:
    ax.scatter(ies["lon_plot"], ies["lat_plot"], s=node_size(ies[size_col], scale), color=color, edgecolor=BORDER, linewidth=1.0, zorder=3)
    if labels is not None:
        draw_labels(ax, labels, min_dist=min_dist)


def plot_network_map(ies: pd.DataFrame, edges: pd.DataFrame, weight: str, color: str, title: str, note: str, filename: str) -> None:
    """Mapa do Brasil com as IES e as ligações; o eixo SP–RJ–MG ganha um recorte ampliado."""
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    draw_states(ax)
    pos = {r.ies_principal: (r.lon_plot, r.lat_plot) for r in ies.itertuples()}
    draw_edges(ax, edges, pos, weight, color, N_EDGES_FIGURE)
    inside = ies.apply(lambda r: in_inset(r.lon_plot, r.lat_plot), axis=1)
    draw_nodes(ax, ies, "n_autores", color, labels=ies[~inside].nlargest(12, "n_autores"))
    x0, x1, y0, y1 = INSET_BOUNDS
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=INK_SECONDARY, linewidth=0.8, linestyle="--", zorder=5))

    # Recorte ampliado no canto inferior esquerdo (oceano/vizinhos, sem dados)
    inset = ax.inset_axes([0.0, 0.05, 0.46, 0.36])
    draw_states(inset)
    inset.set_xlim(x0, x1)
    inset.set_ylim(y0, y1)
    draw_edges(inset, edges, pos, weight, color, N_EDGES_FIGURE)
    draw_nodes(inset, ies[inside], "n_autores", color, labels=ies[inside].nlargest(16, "n_autores"), min_dist=0.55)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(INK_SECONDARY)
        spine.set_linestyle("--")
        spine.set_linewidth(0.8)
    inset.set_axis_on()
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("white")
    inset.text(0.02, 0.97, "SP – RJ – MG", transform=inset.transAxes, fontsize=7, color=INK_SECONDARY, va="top")

    ax.set_title(title, fontsize=11, color=INK)
    ax.text(0.0, 0.0, note, transform=ax.transAxes, fontsize=7, color=INK_SECONDARY, va="bottom")
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


def plot_communities_map(ies: pd.DataFrame, members: pd.DataFrame, summary: pd.DataFrame) -> None:
    plt = setup_matplotlib()
    top = summary.nlargest(N_COMMUNITIES_FIGURE, "tamanho")
    fig, axes = plt.subplots(2, 3, figsize=(11, 8))
    for ax, row in zip(axes.flat, top.itertuples()):
        draw_states(ax)
        ax.scatter(ies["lon_plot"], ies["lat_plot"], s=4, color=MUTED, zorder=2)
        m = members[members["comunidade"] == row.comunidade].merge(ies, on="ies_principal")
        ax.scatter(m["lon_plot"], m["lat_plot"], s=node_size(m["membros"], 0.2), color=BLUE, edgecolor=BORDER, linewidth=0.8, zorder=3)
        draw_labels(ax, m.nlargest(4, "membros"), min_dist=2.5)
        topic = str(row.topico_dominante)
        topic = topic if len(topic) <= 34 else topic[:33] + "…"
        ax.set_title(f"C{row.comunidade}: {topic}\n{int(row.tamanho)} membros, {len(m)} IES", fontsize=8.5, color=INK)
    fig.suptitle("Onde estão os membros das 6 maiores comunidades de coautoria (área ∝ membros na IES)", fontsize=11)
    path = FIGURES_DIR / "map_communities.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


# ---------------------------------------------------------------- HTML

def write_html(ies: pd.DataFrame, coauthor_edges: pd.DataFrame, citation_edges: pd.DataFrame,
               members: pd.DataFrame, summary: pd.DataFrame) -> None:
    def clean(v):
        if isinstance(v, (float, np.floating)):
            return None if np.isnan(v) else round(float(v), 4)
        if isinstance(v, (np.integer,)):
            return int(v)
        return v

    top = summary.nlargest(N_COMMUNITIES_HTML, "tamanho")
    data = {
        "ies": [{k: clean(v) for k, v in r.items()} for r in ies[[
            "ies_principal", "rotulo", "nome", "cidade", "regiao", "lat_plot", "lon_plot", "n_autores", "n_docentes_capes",
            "nota_maxima", "fwci_mediano", "pagerank", "frac_coautoria_interna", "frac_interna_saida",
        ]].to_dict("records")],
        "coautoria": [[r.origem, r.destino, int(r.pares_coautores)] for r in coauthor_edges.nlargest(N_EDGES_HTML, "pares_coautores").itertuples()],
        "citacao": [[r.origem, r.destino, round(float(r.citacoes), 1)] for r in citation_edges.nlargest(N_EDGES_HTML, "citacoes").itertuples()],
        "comunidades": [
            {
                "id": int(r.comunidade),
                "topico": str(r.topico_dominante),
                "tamanho": int(r.tamanho),
                "membros": {m.ies_principal: int(m.membros) for m in members[members["comunidade"] == r.comunidade].itertuples()},
            }
            for r in top.itertuples()
        ],
        "ufs": json.loads(UF_GEOJSON.read_text(encoding="utf-8")),
    }
    template = (ANALYSIS_DIR / "mapas" / "template_mapa.html").read_text(encoding="utf-8")
    html = template.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    path = MAPS_DIR / "mapa_interativo.html"
    path.write_text(html, encoding="utf-8")
    print(f"HTML: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    fetch_geo()
    graph = load_graph(COAUTHORSHIP_PATH)
    ies, coauthor_edges, citation_edges, members = build_tables(graph)
    missing = ies["latitude"].isna().sum()
    if missing:
        print(f"Aviso: {missing} IES sem coordenadas; ficam fora dos mapas")
    ies = spread_overlaps(ies.dropna(subset=["latitude"]))
    ies["rotulo"] = ies["ies_principal"].map(label_of)
    coauthor_edges.sort_values("pares_coautores", ascending=False).to_csv(RESULTS_DIR / "ies_coauthorship_edges.csv", index=False, encoding="utf-8")
    summary = pd.read_csv(RESULTS_DIR / "communities_summary.csv")

    # Citação: soma os dois sentidos para o traço do mapa (a direção fica no HTML)
    undirected = citation_edges.assign(
        a=citation_edges[["origem", "destino"]].min(axis=1), b=citation_edges[["origem", "destino"]].max(axis=1)
    ).groupby(["a", "b"])["citacoes"].sum().reset_index().rename(columns={"a": "origem", "b": "destino"})

    plot_network_map(
        ies, coauthor_edges, "pares_coautores", BLUE,
        "Coautoria entre instituições",
        f"Círculo: IES (área ∝ autores). Linha: pares de coautores entre IES ({N_EDGES_FIGURE} maiores).",
        "map_ies_coauthorship.png",
    )
    plot_network_map(
        ies, undirected, "citacoes", ORANGE,
        "Citação entre instituições",
        f"Círculo: IES (área ∝ autores). Linha: citações entre IES, nos dois sentidos ({N_EDGES_FIGURE} maiores).",
        "map_ies_citation.png",
    )
    plot_communities_map(ies, members, summary)
    write_html(ies, coauthor_edges, citation_edges, members, summary)

    share = coauthor_edges["pares_coautores"].sum()
    print(f"Pares de coautores entre IES diferentes: {share}; arestas IES–IES: {len(coauthor_edges)}")


if __name__ == "__main__":
    main()
