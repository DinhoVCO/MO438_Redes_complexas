#!/usr/bin/env python3
"""Figuras e tabelas finais do relatório.

Gera as figuras que ainda faltavam e as tabelas em LaTeX a partir de results/, sem recalcular
análises:
- figures/report_network_diagram.png   — publicações → autores → comunidades → IES/programas
- figures/report_community_map.png     — mapa das maiores comunidades da coautoria
- figures/report_pagerank_citations.png — PageRank de citação × citações totais (autores)
- tables/*.tex                         — tabelas do relatório (só \\hline, sem pacotes extras)
- FIGURAS_E_TABELAS.md                 — índice: figura/tabela → seção do relatório → script
"""
from __future__ import annotations

import json
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import ANALYSIS_DIR, COAUTHORSHIP_PATH, FIGURES_DIR, RESULTS_DIR, ROOT, load_graph, setup_matplotlib

TABLES_DIR = ANALYSIS_DIR / "tables"
N_MAP_COMMUNITIES = 30


def load(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def fmt(x, nd: int = 2) -> str:
    """Número no padrão brasileiro: vírgula decimal e espaço fino nos milhares."""
    if isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer() and abs(x) >= 100):
        return f"{int(x):,}".replace(",", "\\,")
    if abs(x) < 0.5 * 10 ** (-nd):
        x = 0.0  # evita "-0,00"
    return f"{x:.{nd}f}".replace(".", "{,}")


def fmt_p(p: float) -> str:
    return "$<$0{,}001" if p < 0.001 else fmt(p, 3)


def shorten(text: str, limit: int) -> str:
    """Corta no limite de palavra e acrescenta reticências."""
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" :,-") + "\\ldots"


def plain(x) -> str:
    """Inteiro com espaço nos milhares, para texto de figura (não LaTeX)."""
    return f"{int(x):,}".replace(",", " ")


def tex_escape(text: str) -> str:
    for a, b in (("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")):
        text = str(text).replace(a, b)
    return text


def write_table(name: str, caption: str, label: str, header: list[str], rows: list[list[str]], align: str) -> None:
    lines = [
        "% Gerado por analise_final/08_report_assets.py — não editar à mão",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\hline",
        " & ".join(header) + " \\\\",
        "\\hline",
        *[" & ".join(r) + " \\\\" for r in rows],
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / name).write_text("\n".join(lines), encoding="utf-8")
    print(f"Tabela: {TABLES_DIR / name}")


# ---------------------------------------------------------------- figuras

def plot_network_diagram() -> None:
    """Esquema das redes do trabalho e de como cada uma deriva da anterior."""
    f1 = json.loads((ROOT / "analise_f1" / "resultados_f1.json").read_text(encoding="utf-8"))
    author = load("author_citation_network.json")["rede_autores"]
    group = load("group_citation_network.json")
    programs = load("institution_vs_capes.json")["programas"]
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.set_axis_off()
    boxes = {
        "pub": (0.10, 0.70, f"Publicações\n{plain(f1['citacoes']['n_vertices'])} no corpus\n{plain(f1['citacoes']['n_arestas'])} citações (A→B)"),
        "aut_cit": (0.42, 0.70, f"Autores — citação\n{plain(author['n_vertices'])} autores\n{plain(author['n_arestas'])} arestas a→b"),
        "aut_co": (0.42, 0.22, f"Autores — coautoria\n{plain(f1['coautoria']['n_vertices'])} autores\n{plain(f1['coautoria']['n_arestas'])} arestas"),
        "comm": (0.80, 0.22, f"Comunidades (Louvain)\n{group['comunidades']['n_vertices']} grupos ≥ 10 membros\nlaço = citação interna"),
        "ies": (0.80, 0.70, f"IES / programas CAPES\n{group['ies']['n_vertices']} IES, {programs['n_programas']} programas\nnota da Quadrienal"),
    }
    for key, (x, y, text) in boxes.items():
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.6", "facecolor": "#eef3fb" if "aut" in key else "#f6f6f6", "edgecolor": "#4a6fa5"})
    arrows = [
        ("pub", "aut_cit", "agrega por autor\n(peso fracionário)"),
        ("pub", "aut_co", "autores da mesma\npublicação"),
        ("aut_co", "comm", "detecção de\ncomunidades"),
        ("aut_cit", "comm", "agrega por\ncomunidade"),
        ("aut_cit", "ies", "agrega por\nIES principal"),
    ]
    for a, b, label in arrows:
        xa, ya, _ = boxes[a]
        xb, yb, _ = boxes[b]
        ax.annotate("", xy=(xb - 0.12 * np.sign(xb - xa), yb + 0.11 * np.sign(ya - yb)),
                    xytext=(xa + 0.12 * np.sign(xb - xa), ya - 0.11 * np.sign(ya - yb)),
                    arrowprops={"arrowstyle": "->", "color": "#4a6fa5", "lw": 1.3})
        offset = 0.07 if ya == yb else 0.0
        ax.text((xa + xb) / 2 + (0.06 if ya != yb else 0.0), (ya + yb) / 2 + offset, label, ha="center", va="center", fontsize=7, color="#333333")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.08, 0.84)
    ax.set_title("Redes construídas e como cada uma deriva da anterior")
    path = FIGURES_DIR / "report_network_diagram.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


def plot_community_map() -> None:
    """Supergrafo das maiores comunidades: vértice = comunidade (área ∝ membros, cor = FWCI
    mediano), aresta = nº de coautorias entre membros das duas comunidades."""
    graph = load_graph(COAUTHORSHIP_PATH)
    impact = pd.read_csv(RESULTS_DIR / "group_impact.csv").set_index("grupo")
    top = impact.nlargest(N_MAP_COMMUNITIES, "n_membros").index.tolist()
    top_set = set(top)
    between: Counter = Counter()
    for a, b in graph.edges():
        ca, cb = graph.nodes[a].get("comunidade"), graph.nodes[b].get("comunidade")
        if ca != cb and ca in top_set and cb in top_set:
            between[tuple(sorted((ca, cb)))] += 1
    super_graph = nx.Graph()
    super_graph.add_nodes_from(top)
    for (a, b), w in between.items():
        super_graph.add_edge(a, b, weight=w)
    pos = nx.spring_layout(super_graph, weight="weight", seed=0, k=2.0, iterations=300)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 8))
    weights = np.array([d["weight"] for _, _, d in super_graph.edges(data=True)], dtype=float)
    nx.draw_networkx_edges(super_graph, pos, ax=ax, width=0.3 + 3 * np.sqrt(weights / weights.max()), alpha=0.25, edge_color="gray")
    fwci = impact.loc[top, "fwci_mediano"].to_numpy()
    nodes = nx.draw_networkx_nodes(
        super_graph, pos, nodelist=top, ax=ax, node_size=impact.loc[top, "n_membros"].to_numpy() / 6,
        node_color=fwci, cmap="viridis", alpha=0.9,
    )
    labels = {
        c: f"C{c}\n{impact.loc[c, 'ies_dominante']}" for c in top
    }
    nx.draw_networkx_labels(super_graph, pos, labels, ax=ax, font_size=7)
    fig.colorbar(nodes, ax=ax, shrink=0.6, label="FWCI mediano das publicações")
    ax.set_title(f"As {N_MAP_COMMUNITIES} maiores comunidades de coautoria (área ∝ membros; aresta ∝ coautorias entre grupos)")
    ax.set_axis_off()
    path = FIGURES_DIR / "report_community_map.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


def plot_pagerank_citations() -> None:
    table = pd.read_csv(RESULTS_DIR / "author_centrality.csv", usecols=["afiliacao_escopo", "cit_pagerank", "citacoes_total", "indice_h"])
    scope = table[table["afiliacao_escopo"]].dropna()
    rho_c = spearmanr(scope["cit_pagerank"], scope["citacoes_total"])[0]
    rho_h = spearmanr(scope["cit_pagerank"], scope["indice_h"])[0]
    data = scope[(scope["citacoes_total"] > 0)]
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    hb = ax.hexbin(data["citacoes_total"], data["cit_pagerank"], xscale="log", yscale="log", gridsize=45, bins="log", cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, label="nº de autores (log)")
    ax.set_xlabel("Citações totais do autor (OpenAlex)")
    ax.set_ylabel("PageRank na rede de citação entre autores")
    ax.set_title(f"Autores do escopo: Spearman ρ = {rho_c:.2f} (citações), {rho_h:.2f} (índice h)", fontsize=9)
    path = FIGURES_DIR / "report_pagerank_citations.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


# ---------------------------------------------------------------- tabelas

def table_network_metrics() -> None:
    """Métricas topológicas das duas redes base (analise_f1, corpus final)."""
    f1 = json.loads((ROOT / "analise_f1" / "resultados_f1.json").read_text(encoding="utf-8"))
    co, ci = f1["coautoria"], f1["citacoes"]

    def sci(x: float) -> str:
        mantissa, exponent = f"{x:.2e}".split("e")
        return f"${mantissa.replace('.', '{,}')} \\times 10^{{{int(exponent)}}}$"

    giant_share = fmt(100 * co["tamanho_componente"] / co["n_vertices"], 1) + "\\%"
    rows = [
        ["Vértices", fmt(co["n_vertices"]), fmt(ci["n_vertices"])],
        ["Arestas", fmt(co["n_arestas"]), fmt(ci["n_arestas"])],
        ["Grau médio", fmt(co["grau_medio"], 2), fmt(ci["grau_medio_entrada"], 2) + " (entrada)"],
        ["Clusterização média", fmt(co["clusterizacao_media"], 3), fmt(ci["clusterizacao_media"], 3)],
        ["Transitividade", fmt(co["transitividade"], 3), fmt(ci["transitividade"], 3)],
        ["Densidade", sci(co["densidade"]), sci(ci["densidade"])],
        ["Componentes", fmt(co["n_componentes"]), "---"],
        ["Maior componente", f"{fmt(co['tamanho_componente'])} ({giant_share})", f"{fmt(ci['tamanho_componente'])} (forte)"],
        ["Distância média", fmt(co["distancia_media"], 2), fmt(ci["distancia_media"], 2)],
        ["Diâmetro", fmt(co["diametro"]), fmt(ci["diametro"])],
    ]
    write_table("tab_metricas_redes.tex", "Métricas das redes de coautoria e de citação entre publicações. Distância e diâmetro na maior componente conexa (coautoria) e na maior componente fortemente conexa (citações).",
                "tab:metricas", ["Métrica", "Coautoria", "Citações"], rows, "lrr")


def table_algorithms() -> None:
    comm = load("communities.json")
    gc = comm["componente_gigante"]["tabela"]
    sub = comm["subgrafo_docentes"]["tabela"]
    val = comm["validacao"]
    names = {"louvain": "Louvain (50 sementes)", "leiden": "Leiden", "fastgreedy": "Fast-Greedy (CNM)",
             "label_propagation": "Label Propagation", "fluid": "Fluid ($k$ = 122)", "girvan_newman": "Girvan--Newman"}
    rows = []
    for key, label in names.items():
        g = gc.get(key)
        s = sub[key]
        v = val.get(key)
        rows.append([
            label,
            fmt(g["q"], 3) if g else "---", fmt(g["n_comunidades"]) if g else "---", (fmt(g["tempo_s"], 1) + " s") if g else "---",
            fmt(v["ies_principal"]["nmi"], 2) if v else "---", fmt(v["ies_principal"]["ari"], 2) if v else "---",
            fmt(s["q"], 3), str(s["n_comunidades"]),
        ])
    write_table(
        "tab_algoritmos.tex",
        "Algoritmos de comunidades. Componente gigante da coautoria (94\\,179 vértices, ponderada): modularidade $Q$, nº de comunidades, tempo e concordância com a IES principal (NMI/ARI). Subgrafo de docentes de USP, UFPE e UTFPR (320 vértices, sem pesos): $Q$ e nº de comunidades.",
        "tab:algoritmos", ["Algoritmo", "$Q$", "Com.", "Tempo", "NMI IES", "ARI IES", "$Q$ sub.", "Com. sub."], rows, "lrrrrrrr",
    )


def table_structural() -> None:
    s = load("structural_properties.json")
    c = load("communities.json")["modelo_configuracao"]
    g = load("group_citation_network.json")
    sw, pl, a = s["mundo_pequeno"], s["lei_de_potencia"], s["assortatividade"]
    rows = [
        ["Mundo pequeno", "$C$ real / ER / configuração", f"{fmt(sw['C_real'], 3)} / {fmt(sw['C_ER'], 4)} / {fmt(sw['C_configuracao'], 4)}"],
        ["", "$L$ real / ER / configuração", f"{fmt(sw['L_real'], 2)} / {fmt(sw['L_ER'], 2)} / {fmt(sw['L_configuracao_estimada'], 2)}"],
        ["", "$\\sigma$ (ER / configuração)", f"{fmt(int(round(sw['sigma_ER'])))} / {fmt(int(round(sw['sigma_configuracao'])))}"],
        ["Lei de potência", "$\\alpha$, $k_{min}$: grau na coautoria", f"{fmt(pl['grau_coautoria']['alpha'], 2)}, {int(pl['grau_coautoria']['xmin'])}"],
        ["", "grau de entrada (citação autores)", f"{fmt(pl['grau_entrada_citacao_autores']['alpha'], 2)}, {int(pl['grau_entrada_citacao_autores']['xmin'])}"],
        ["", "citações globais das publicações", f"{fmt(pl['cited_by_count_publicacoes']['alpha'], 2)}, {int(pl['cited_by_count_publicacoes']['xmin'])}"],
        ["Assortatividade", "Grau: coautoria / citação autores", f"{fmt(a['grau']['coautoria'], 3)} / {fmt(a['grau']['citacao_autores_saida_entrada'], 3)}"],
        ["", "IES na coautoria ($r$ nominal)", fmt(a["ies_coautoria"]["assortatividade_ies"], 2)],
        ["", "Arestas na mesma IES: real / esp.", f"{fmt(100 * a['ies_coautoria']['frac_arestas_mesma_ies'], 1)}\\% / {fmt(100 * a['ies_coautoria']['frac_esperada_pelos_graus'], 1)}\\%"],
        ["Modularidade", "$Q$ Louvain real / configuração", f"{fmt(c['q_real_sem_pesos'], 3)} / {fmt(c['q_nulo_medio'], 3)} $\\pm$ {fmt(c['q_nulo_desvio'], 3)}"],
        ["Citação interna", "Interna: comunidades / IES", f"{fmt(100 * g['comunidades']['homofilia']['frac_interna_observada'], 1)}\\% / {fmt(100 * g['ies']['homofilia']['frac_interna_observada'], 1)}\\%"],
        ["", "Esperado pelas forças", f"{fmt(100 * g['comunidades']['homofilia']['frac_interna_esperada'], 1)}\\% / {fmt(100 * g['ies']['homofilia']['frac_interna_esperada'], 1)}\\%"],
        ["Reciprocidade", "Citação autores: real / nulo", f"{fmt(a['reciprocidade_citacao_autores']['real'], 3)} / {fmt(a['reciprocidade_citacao_autores']['nulo'], 3)}"],
        ["Granovetter", "Overlap: dentro / entre comunidades", f"{fmt(s['granovetter']['dentro_comunidade']['overlap_medio'], 2)} / {fmt(s['granovetter']['entre_comunidades']['overlap_medio'], 2)}"],
        ["", "Spearman peso $\\times$ overlap", fmt(s["granovetter"]["spearman_peso_overlap"]["rho"], 2)],
        ["Paradoxo amizade", "$\\langle k\\rangle$ / grau médio dos vizinhos", f"{fmt(s['paradoxo_da_amizade']['grau_medio'], 1)} / {fmt(s['paradoxo_da_amizade']['grau_medio_dos_vizinhos_media'], 1)}"],
    ]
    write_table("tab_propriedades.tex", "Propriedades estruturais das redes e comparação com modelos nulos. Na lei de potência, a lognormal ajusta melhor os dois graus e empata com a lei de potência nas citações globais.",
                "tab:propriedades", ["Conceito", "Medida", "Valor"], rows, "lll")


def table_impact() -> None:
    r = load("group_impact.json")["comunidades"]["rankings"]
    absolute = r["top_citacoes_total"]
    normalized = r["top_fwci_mediano"]
    rows = []
    for i, (a, n) in enumerate(zip(absolute, normalized), start=1):
        rows.append([
            str(i),
            f"C{a['grupo']} ({tex_escape(a['ies_dominante'])})", fmt(a["n_membros"]), fmt(a["citacoes_total"]),
            f"C{n['grupo']} ({tex_escape(n['ies_dominante'])})", fmt(n["n_membros"]), fmt(n["fwci_mediano"], 2),
        ])
    write_table("tab_impacto_ranking.tex", "Top 10 comunidades por citações totais (absoluto) e por FWCI mediano das publicações (normalizado); só comunidades com $\\geq$ 50 publicações.",
                "tab:impacto", ["\\#", "Absoluto", "Membros", "Citações", "Normalizado", "Membros", "FWCI med."], rows, "rlrrlrr")

    corr = load("group_impact.json")["comunidades"]
    labels = {
        "citacoes_total": "Citações totais", "obras_por_membro": "Publicações por membro", "citacoes_por_membro": "Citações por membro",
        "citacoes_por_obra": "Citações por publicação", "mncs": "MNCS (ajuste ao ano)", "fwci_mediano": "FWCI mediano",
        "frac_obras_top10pct": "Fração no top 10\\%",
    }
    rows = [[label, fmt(corr["spearman_tamanho"][k]["rho"], 2), fmt(corr["spearman_tamanho_so_componente_gigante"][k]["rho"], 2),
             fmt_p(corr["spearman_tamanho_so_componente_gigante"][k]["p_valor"])] for k, label in labels.items()]
    write_table("tab_impacto_tamanho.tex", "Correlação de Spearman entre o tamanho da comunidade e cada métrica de impacto: todas as 240 comunidades e só as 121 da componente gigante (p-valor por permutação).",
                "tab:tamanho", ["Métrica", "$\\rho$ todas", "$\\rho$ gigante", "$p$ gigante"], rows, "lrrr")


def table_capes() -> None:
    r = load("institution_vs_capes.json")
    prog = r["programas"]["spearman_nota"]
    ies = r["ies"]["spearman"]["nota_maxima"]
    labels = [
        ("n_obras", None, "Nº de publicações"), ("obras_por_docente", None, "Publicações por docente"),
        ("indice_h_mediano_docentes", None, "Índice $h$ mediano dos docentes"),
        ("citacoes_por_docente", "citacoes_por_docente", "Citações por docente"),
        ("citacoes_por_obra", None, "Citações por publicação"), ("mncs", "mncs", "MNCS"),
        ("fwci_mediano", "fwci_mediano", "FWCI mediano"), ("frac_obras_top10pct", "frac_obras_top10pct", "Fração no top 10\\%"),
    ]
    rows = []
    for pk, ik, label in labels:
        p = prog[pk]
        i = ies.get(ik) if ik else None
        rows.append([label, fmt(p["rho"], 2), fmt_p(p["p_valor"]),
                     fmt(i["rho"], 2) if i else "---"])
    write_table("tab_capes.tex", f"Spearman entre a nota CAPES e o impacto: {r['programas']['n_elegiveis']} programas (publicações dos docentes) e {r['ies']['n_ies']} IES (nota máxima).",
                "tab:capes", ["Métrica", "$\\rho$ programa", "$p$", "$\\rho$ IES"], rows, "lrrr")


def table_centrality() -> None:
    v = load("centrality.json")["validacao_escopo"]
    labels = {"cit_pagerank": "PageRank (citação)", "cit_forca_entrada": "Força de entrada (citação)", "cit_autoridade": "Autoridade HITS",
              "cit_hub": "Hub HITS", "grau": "Grau (coautoria)", "intermediacao": "Intermediação", "proximidade": "Proximidade",
              "k_core": "$k$-core", "autovetor": "Autovetor"}
    rows = [[label, fmt(v[k]["spearman_citacoes"], 2), fmt(v[k]["kendall_citacoes"], 2), fmt(v[k]["spearman_indice_h"], 2),
             fmt(v[k]["jaccard_top50_citacoes"], 2), fmt(v[k]["jaccard_top50_indice_h"], 2)] for k, label in labels.items()]
    write_table("tab_centralidade.tex", "Centralidades $\\times$ indicadores tradicionais (autores do escopo): Spearman e Kendall com citações totais, Spearman com o índice $h$ e Jaccard dos top 50.",
                "tab:centralidade", ["Medida", "$\\rho$ cit.", "$\\tau$ cit.", "$\\rho$ $h$", "J50 cit.", "J50 $h$"], rows, "lrrrrr")


def table_top_publications() -> None:
    p = load("centrality.json")["publicacoes"]
    rows = []
    for i, (a, b) in enumerate(zip(p["top_pagerank"][:5], p["top_cited_by_count"][:5]), start=1):
        rows.append([str(i), tex_escape(shorten(a["titulo"], 75)), fmt(a["grau_entrada"]), tex_escape(shorten(b["titulo"], 75)), fmt(b["cited_by_count"])])
    write_table("tab_top_publicacoes.tex", "Top 5 publicações por PageRank na rede de citação do corpus (impacto interno; ``Corpus'' = citações recebidas do corpus) e por citações globais da OpenAlex.",
                "tab:toppub", ["\\#", "PageRank (interno)", "Corpus", "Citações globais", "Citações"], rows, "rp{5.0cm}rp{5.0cm}r")


# ---------------------------------------------------------------- índice

INDEX = """# Figuras e tabelas do relatório

Gerado por `08_report_assets.py`. Cada item indica a seção do relatório e o script que o produz.

## Figuras

| # | Arquivo | Seção | Conteúdo | Script |
|---|---------|-------|----------|--------|
| 1 | `figures/report_network_diagram.png` | 5 | Redes construídas e como derivam umas das outras | 08 |
| 2 | `figures/structural_powerlaw.png` | 4 / 6.3 | Caudas das distribuições de grau e citações, com ajuste de lei de potência e lognormal | 07 |
| 3 | `figures/communities_size_ccdf.png` | 6.2 | Distribuição de tamanhos das comunidades por algoritmo | 02 |
| 4 | `figures/communities_gn_modularity.png` | 6.2 | Curva $Q$ × nº de comunidades do Girvan–Newman (protocolo AP3) | 02 |
| 5 | `figures/report_community_map.png` | 6.2 | Mapa das 30 maiores comunidades (cor = FWCI mediano) | 08 |
| 6 | `figures/community_citation_heatmap.png` | 6.1 / 6.4 | Quem cita quem entre as 20 maiores comunidades | 03 |
| 7 | `figures/group_impact_size.png` | 6.4 | Tamanho da comunidade × impacto absoluto e normalizado | 04 |
| 8 | `figures/group_impact_ranking.png` | 6.4 | Ranking absoluto × normalizado das comunidades | 04 |
| 9 | `figures/report_pagerank_citations.png` | 6.3 | PageRank de citação × citações totais dos autores | 08 |
| 10 | `figures/capes_program_impact.png` | 6.5 | Impacto dos programas × nota CAPES | 06 |
| 11 | `figures/structural_granovetter.png` | 6.2 | Granovetter: overlap × peso, pontes entre comunidades, remoção de vínculos | 07 |
| 12 | `figures/centrality_spearman_coauthorship.png` | 6.3 | Spearman entre as seis centralidades da AP3 | 05 |

Mapas (gerados por `09_maps.py`, ainda fora do relatório e dos slides): `map_ies_coauthorship.png`, `map_ies_citation.png`, `map_communities.png` e a versão interativa `mapas/mapa_interativo.html`.

Figuras de apoio (anexo ou apresentação): `author_citation_degree_ccdf.png`, `ies_citation_heatmap.png`, `capes_ies_impact.png`, `centrality_spearman_validation.png`, `structural_snapshots.png`, e as figuras da F1 em `analise_f1/figuras/`.

## Tabelas (`tables/`, LaTeX com `\\hline`, sem pacotes extras)

| Arquivo | Seção | Conteúdo |
|---------|-------|----------|
| `tab_metricas_redes.tex` | 4 | Métricas topológicas das redes de coautoria e de citação |
| `tab_algoritmos.tex` | 6.2 | Algoritmos de comunidades nas duas escalas, com NMI/ARI contra a IES |
| `tab_propriedades.tex` | 4 / 6 | Mundo pequeno, lei de potência, assortatividade, modelos nulos, Granovetter |
| `tab_impacto_ranking.tex` | 6.4 | Top 10 comunidades: absoluto × normalizado |
| `tab_impacto_tamanho.tex` | 6.4 | Spearman tamanho × métricas de impacto |
| `tab_centralidade.tex` | 6.3 | Centralidades × citações e índice $h$ |
| `tab_top_publicacoes.tex` | 6.3 | Top 10 publicações: impacto interno × global |
| `tab_capes.tex` | 6.5 | Spearman nota CAPES × impacto (programas e IES) |

Todos os números vêm de `results/`; para regenerar: rodar os scripts 01–07 e depois este.
"""


def main() -> None:
    plot_network_diagram()
    plot_pagerank_citations()
    plot_community_map()
    table_network_metrics()
    table_algorithms()
    table_structural()
    table_impact()
    table_capes()
    table_centrality()
    table_top_publications()
    (ANALYSIS_DIR / "FIGURAS_E_TABELAS.md").write_text(INDEX, encoding="utf-8")
    print(f"Índice: {ANALYSIS_DIR / 'FIGURAS_E_TABELAS.md'}")


if __name__ == "__main__":
    main()
