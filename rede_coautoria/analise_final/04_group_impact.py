#!/usr/bin/env python3
"""Impacto normalizado dos grupos.

Para cada comunidade de coautoria com ≥ 10 membros (e, à parte, para cada IES principal):
- citações por membro: Σ cited_by_count das obras distintas do grupo / nº de membros;
- citações por publicação: Σ cited_by_count / nº de obras distintas;
- citações por docente CAPES (robustez: os membros incluem alunos e colaboradores externos);
- citações internas por membro: citações vindas do próprio corpus (grau de entrada na rede
  de publicações) / nº de membros;
- PageRank por membro: Σ PageRank dos membros na rede de citação entre autores / nº de membros;
- MNCS: média, nas obras do grupo, de citações / média de citações do corpus no mesmo ano;
- FWCI médio (OpenAlex: normalizado por campo, tipo e ano), nas obras que o têm;
- concentração: fração das citações do grupo nas obras do membro mais citado, e Gini entre membros.

Cada obra conta uma vez por grupo; uma obra com autores de dois grupos conta para os dois.
`cited_by_count` é a contagem global da OpenAlex (não só citações do corpus).

Saídas:
- results/group_impact.csv e results/ies_impact.csv
- results/group_impact.json — rankings absoluto × normalizado e Spearman tamanho × impacto
- figures/group_impact_size.png e figures/group_impact_ranking.png
"""
from __future__ import annotations

from collections import defaultdict

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd

from common import (
    AUTHOR_CITATION_PATH,
    FIGURES_DIR,
    PUB_CITATION_PATH,
    RESULTS_DIR,
    load_graph,
    save_json,
    setup_matplotlib,
    work_table,
)

MIN_MEMBERS = 10
MIN_WORKS_RANKING = 50  # grupos com poucas obras têm médias instáveis (um artigo atípico domina)
MIN_FACULTY = 3
N_PERMUTATIONS = 10_000
TOP_K = 10

NORMALIZED_METRICS = [
    "obras_por_membro",
    "citacoes_por_membro",
    "citacoes_por_membro_sem_obra_top",
    "citacoes_por_obra",
    "citacoes_por_docente",
    "citacoes_internas_por_membro",
    "pagerank_por_membro",
    "mncs",
    "fwci_medio",
    "fwci_mediano",
    "frac_obras_top10pct",
]


def gini(values: np.ndarray) -> float:
    """Coeficiente de Gini (0 = igual entre membros, → 1 = concentrado em um)."""
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    cum = np.cumsum(values)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def author_pagerank(author_graph: nx.DiGraph) -> dict[str, float]:
    """PageRank (α = 0,85, peso fracionário) na rede de citação entre autores, via igraph."""
    names = list(author_graph.nodes())
    index = {v: i for i, v in enumerate(names)}
    g = ig.Graph(
        n=len(names),
        edges=[(index[a], index[b]) for a, b in author_graph.edges()],
        directed=True,
    )
    weights = [w for _, _, w in author_graph.edges(data="weight_frac")]
    scores = g.pagerank(damping=0.85, weights=weights, directed=True)
    return dict(zip(names, scores))


def group_impact(
    works: pd.DataFrame,
    group_of: dict[str, object],
    members: dict[object, list[str]],
    pagerank: dict[str, float],
    faculty: set[str],
) -> pd.DataFrame:
    """Métricas de impacto por grupo, contando cada obra uma vez por grupo."""
    group_works: dict[object, set[int]] = defaultdict(set)
    author_works: dict[str, list[int]] = defaultdict(list)
    for idx, authors in works["autores"].items():
        for a in authors:
            author_works[a].append(idx)
            g = group_of.get(a)
            if g is not None:
                group_works[g].add(idx)

    citations = works["citacoes"].to_numpy()
    rows = []
    for g, nodes in members.items():
        idx = sorted(group_works.get(g, ()))
        sub = works.loc[idx]
        total = float(sub["citacoes"].sum())
        n_faculty = sum(1 for a in nodes if a in faculty)
        # Citações de cada membro (suas obras no corpus); obras em coautoria contam para todos
        member_cites = np.array([citations[author_works[a]].sum() for a in nodes], dtype=float)
        # Uma obra muito citada (ex.: SciPy 1.0, Astropy) pode dominar o grupo inteiro
        top_work = float(sub["citacoes"].max()) if idx else 0.0
        rows.append(
            {
                "grupo": g,
                "n_membros": len(nodes),
                "n_docentes_capes": n_faculty,
                "n_obras": len(idx),
                "citacoes_total": total,
                # Produtividade: separa "publica mais" de "é mais citado por obra"
                "obras_por_membro": len(idx) / len(nodes),
                "citacoes_por_membro": total / len(nodes),
                "citacoes_por_membro_sem_obra_top": (total - top_work) / len(nodes),
                "citacoes_por_obra": total / len(idx) if idx else np.nan,
                "citacoes_por_docente": total / n_faculty if n_faculty >= MIN_FACULTY else np.nan,
                "citacoes_internas_por_membro": float(sub["citacoes_corpus"].sum()) / len(nodes),
                # Arredonda a 10 algarismos: grupos com o PageRank mínimo empatam, e o ruído de
                # ponto flutuante (~1e-11) não pode desempatar os postos do Spearman
                "pagerank_total": float(f"{sum(pagerank.get(a, 0.0) for a in nodes):.10g}"),
                "pagerank_por_membro": float(f"{sum(pagerank.get(a, 0.0) for a in nodes) / len(nodes):.10g}"),
                "mncs": float(sub["citacao_normalizada_ano"].mean()) if idx else np.nan,
                "fwci_medio": float(sub["fwci"].mean()) if idx else np.nan,
                "fwci_mediano": float(sub["fwci"].median()) if idx else np.nan,
                "frac_obras_com_fwci":float(sub["fwci"].notna().mean()) if idx else np.nan,
                "frac_obras_top10pct": np.nan,  # preenchido abaixo
                "concentracao_obra_top": top_work / total if total else np.nan,
                "concentracao_membro_top":float(member_cites.max() / total) if total else np.nan,
                "gini_membros": gini(member_cites),
                "_idx": idx,
            }
        )
    table = pd.DataFrame(rows)
    # Fração das obras do grupo entre as 10 % mais citadas do corpus no mesmo ano
    threshold = works.groupby("ano")["citacoes"].transform(lambda s: s.quantile(0.9))
    top = (works["citacoes"] > threshold).to_numpy()
    table["frac_obras_top10pct"] = table["_idx"].map(lambda idx: float(top[idx].mean()) if idx else np.nan)
    return table.drop(columns="_idx")


def spearman_with_pvalue(x: pd.Series, y: pd.Series, seed: int = 0) -> dict:
    """Spearman com p-valor por permutação (bicaudal); sem depender do scipy."""
    mask = x.notna() & y.notna()
    rx = x[mask].rank().to_numpy()
    ry = y[mask].rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    perms = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(N_PERMUTATIONS)])
    p = float((np.sum(np.abs(perms) >= abs(rho)) + 1) / (N_PERMUTATIONS + 1))
    return {"rho": rho, "p_valor": p, "n": int(mask.sum())}


def rankings(table: pd.DataFrame, label_cols: list[str]) -> dict:
    """Top 10 absoluto × normalizado e posição de cada grupo nos dois rankings."""
    eligible = table[table["n_obras"] >= MIN_WORKS_RANKING].copy()
    robust = ("citacoes_por_membro_sem_obra_top", "fwci_mediano", "frac_obras_top10pct")
    cols = ["grupo", *label_cols, "n_membros", "n_obras", "citacoes_total", "citacoes_por_membro",
            *robust, "mncs", "fwci_medio", "concentracao_obra_top"]
    out = {}
    for metric in ("citacoes_total", "citacoes_por_membro", "mncs", "fwci_medio", *robust):
        top = eligible.nlargest(TOP_K, metric)[cols]
        out[f"top_{metric}"] = top.round(3).to_dict("records")
    top_abs = set(eligible.nlargest(TOP_K, "citacoes_total")["grupo"])
    out["sobreposicao_top10"] = {
        metric: len(top_abs & set(eligible.nlargest(TOP_K, metric)["grupo"]))
        for metric in ("citacoes_por_membro", "mncs", "fwci_medio", *robust)
    }
    return out


def plot_size(table: pd.DataFrame, correlations: dict) -> None:
    """Tamanho × impacto absoluto e normalizado (log-log)."""
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [
        ("citacoes_total", "Citações totais"),
        ("citacoes_por_membro_sem_obra_top", "Citações por membro\n(sem a publicação mais citada)"),
        ("fwci_mediano", "FWCI mediano"),
    ]
    for ax, (metric, ylabel) in zip(axes, panels):
        data = table[table[metric] > 0]
        ax.scatter(data["n_membros"], data[metric], s=12, alpha=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Nº de membros")
        ax.set_ylabel(ylabel)
        rho = correlations[metric]
        ax.set_title(f"Spearman ρ = {rho['rho']:.2f} (p = {rho['p_valor']:.3g})", fontsize=9)
    fig.suptitle("Tamanho da comunidade × impacto")
    path = FIGURES_DIR / "group_impact_size.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


def plot_ranking(table: pd.DataFrame, labels: dict) -> None:
    """Posição de cada comunidade no ranking absoluto × no ranking por FWCI e por membro."""
    plt = setup_matplotlib()
    eligible = table[table["n_obras"] >= MIN_WORKS_RANKING].copy()
    eligible["rank_abs"] = eligible["citacoes_total"].rank(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for ax, (metric, title) in zip(axes, (("citacoes_por_membro_sem_obra_top", "Citações por membro"), ("fwci_mediano", "FWCI mediano"))):
        rank_norm = eligible[metric].rank(ascending=False)
        ax.scatter(eligible["rank_abs"], rank_norm, s=np.sqrt(eligible["n_membros"]) * 2, alpha=0.5)
        lim = len(eligible) + 5
        ax.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=0.8)
        for g, r_abs, r_norm in zip(eligible["grupo"], eligible["rank_abs"], rank_norm):
            if r_abs <= 3 or r_norm <= 3:
                ax.annotate(labels.get(g, str(g)), (r_abs, r_norm), fontsize=7, xytext=(4, 2), textcoords="offset points")
        ax.set_xlim(0, lim)
        ax.set_ylim(lim, 0)
        ax.set_xlabel("Posição no ranking por citações totais")
        ax.set_ylabel(f"Posição no ranking por {title.lower()}")
        ax.set_title(title)
    fig.suptitle("Ranking absoluto × normalizado das comunidades (tamanho do ponto ∝ nº de membros)")
    path = FIGURES_DIR / "group_impact_ranking.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


def main() -> None:
    pub_graph = load_graph(PUB_CITATION_PATH)
    author_graph = load_graph(AUTHOR_CITATION_PATH)
    works = work_table(pub_graph)
    print("PageRank dos autores...", flush=True)
    pagerank = author_pagerank(author_graph)
    faculty = {v for v, d in author_graph.nodes(data=True) if d.get("docente_capes")}

    # --- comunidades
    communities = pd.read_csv(RESULTS_DIR / "author_communities.csv", usecols=["author_id", "comunidade"])
    summary = pd.read_csv(RESULTS_DIR / "communities_summary.csv")
    big = set(summary.loc[summary["tamanho"] >= MIN_MEMBERS, "comunidade"])
    community_of = {a: c for a, c in zip(communities["author_id"], communities["comunidade"]) if c in big}
    members: dict = defaultdict(list)
    for a, c in community_of.items():
        members[c].append(a)
    table = group_impact(works, community_of, members, pagerank, faculty)
    table = table.merge(
        summary.rename(columns={"comunidade": "grupo"})[["grupo", "ies_dominante", "pureza_ies", "topico_dominante"]],
        on="grupo",
    ).sort_values("n_membros", ascending=False)
    table.to_csv(RESULTS_DIR / "group_impact.csv", index=False, encoding="utf-8")
    print(f"Comunidades avaliadas: {len(table)}")

    correlations = {
        metric: spearman_with_pvalue(table["n_membros"], table[metric])
        for metric in ["citacoes_total", *NORMALIZED_METRICS, "concentracao_obra_top", "gini_membros"]
    }
    # Estratos: comunidades da componente gigante × componentes isolados (grupos periféricos)
    n_giant = int(pd.read_csv(RESULTS_DIR / "author_communities.csv", usecols=["comunidade", "componente_gigante"])
                  .query("componente_gigante")["comunidade"].nunique())
    table["componente_gigante"] = table["grupo"] < n_giant
    giant = table[table["componente_gigante"]]
    correlations_giant = {
        metric: spearman_with_pvalue(giant["n_membros"], giant[metric])
        for metric in ["citacoes_total", *NORMALIZED_METRICS]
    }
    strata = (
        table.groupby("componente_gigante")[
            ["n_membros", "n_obras", "obras_por_membro", "citacoes_por_obra", "mncs", "fwci_medio", "frac_obras_top10pct"]
        ]
        .median()
        .rename(index={True: "componente_gigante", False: "componentes_isolados"})
    )
    table.to_csv(RESULTS_DIR / "group_impact.csv", index=False, encoding="utf-8")

    # Concordância entre as métricas normalizadas (robustez da escolha)
    metric_corr = table[NORMALIZED_METRICS].corr(method="spearman").round(3)

    # --- IES (P4): mesmo cálculo com a IES principal como grupo
    ies_of = {v: d["ies_principal"] for v, d in author_graph.nodes(data=True) if d.get("ies_principal")}
    ies_members: dict = defaultdict(list)
    for a, i in ies_of.items():
        ies_members[i].append(a)
    ies_table = group_impact(works, ies_of, ies_members, pagerank, faculty).sort_values("n_membros", ascending=False)
    ies_table.to_csv(RESULTS_DIR / "ies_impact.csv", index=False, encoding="utf-8")
    ies_correlations = {
        metric: spearman_with_pvalue(ies_table["n_membros"], ies_table[metric])
        for metric in ["citacoes_total", "citacoes_por_membro", "citacoes_por_membro_sem_obra_top", "citacoes_por_docente", "mncs", "fwci_medio", "fwci_mediano"]
    }

    results = {
        "definicoes": {
            "membros": f"comunidades Louvain (ou componentes) com >= {MIN_MEMBERS} membros",
            "citacoes": "cited_by_count global da OpenAlex, cada publicação uma vez por grupo",
            "mncs": "média de citações / média do corpus no mesmo ano",
            "fwci": "Field-Weighted Citation Impact da OpenAlex",
            "ranking": f"só grupos com >= {MIN_WORKS_RANKING} publicações",
        },
        "cobertura_fwci": float(works["fwci"].notna().mean()),
        "comunidades": {
            "n_grupos": len(table),
            "spearman_tamanho": correlations,
            "spearman_tamanho_so_componente_gigante": correlations_giant,
            "medianas_por_estrato": strata.to_dict("index"),
            "n_grupos_por_estrato": table["componente_gigante"].value_counts().rename({True: "componente_gigante", False: "componentes_isolados"}).to_dict(),
            "spearman_entre_metricas": metric_corr.to_dict(),
            "rankings": rankings(table, ["ies_dominante", "topico_dominante"]),
            "concentracao": {
                "gini_mediano": float(table["gini_membros"].median()),
                "membro_top_mediano": float(table["concentracao_membro_top"].median()),
            },
        },
        "ies": {
            "n_grupos": len(ies_table),
            "spearman_tamanho": ies_correlations,
            "rankings": rankings(ies_table, []),
        },
    }
    save_json(results, "group_impact.json")

    labels = {
        row.grupo: f"C{row.grupo} ({row.ies_dominante})"
        for row in table.itertuples()
    }
    plot_size(table, correlations)
    plot_ranking(table, labels)

    print("\nSpearman tamanho × métrica (comunidades):")
    for metric, r in correlations.items():
        print(f"  {metric}: ρ = {r['rho']:.3f} (p = {r['p_valor']:.3g})")
    print("Só componente gigante:")
    for metric, r in correlations_giant.items():
        print(f"  {metric}: ρ = {r['rho']:.3f} (p = {r['p_valor']:.3g})")
    print("Sobreposição com o top 10 absoluto:", results["comunidades"]["rankings"]["sobreposicao_top10"])


if __name__ == "__main__":
    main()
