#!/usr/bin/env python3
"""Impacto dos programas e das instituições × conceito CAPES (P4).

Dois níveis de comparação com a `nota_final` da Avaliação Quadrienal CAPES:

1. Programa (mais preciso): o impacto de cada um dos 87 programas é medido nas obras dos seus
   docentes CAPES identificados na OpenAlex (docentes_openalex.csv). Uma obra conta para o
   programa se algum docente dele é autor e o ano da obra está entre os anos em que o docente
   aparece na CAPES. Cada obra conta uma vez por programa.
2. IES principal (a agregação do plano): o impacto por IES de 04_group_impact.py contra a nota
   máxima e a nota média dos programas da IES. Campi que a OpenAlex trata como uma só
   instituição (ex.: USP São Paulo e São Carlos) são somados.

Saídas:
- results/program_impact.csv e results/institution_vs_capes.csv
- results/institution_vs_capes.json — Spearman com p-valor, medianas por nota
- figures/capes_program_impact.png e figures/capes_ies_impact.png
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (
    FIGURES_DIR,
    PUB_CITATION_PATH,
    RESULTS_DIR,
    config,
    load_graph,
    save_json,
    setup_matplotlib,
    work_table,
)
from src.build_network import load_scope_sigla

MIN_FACULTY = 5
MIN_COVERAGE = 0.5
MIN_WORKS = 30

PROGRAM_METRICS = [
    "obras_por_docente",
    "citacoes_por_docente",
    "citacoes_por_docente_sem_obra_top",
    "citacoes_por_obra",
    "mncs",
    "fwci_medio",
    "fwci_mediano",
    "frac_obras_top10pct",
    "indice_h_mediano_docentes",
    "pagerank_por_docente",
]
IES_METRICS = [
    "citacoes_total",
    "citacoes_por_membro",
    "citacoes_por_membro_sem_obra_top",
    "citacoes_por_docente",
    "mncs",
    "fwci_mediano",
    "frac_obras_top10pct",
]


def spearman_table(table: pd.DataFrame, x: str, metrics: list[str]) -> dict:
    out = {}
    for metric in metrics:
        sub = table[[x, metric]].dropna()
        rho, p = spearmanr(sub[x], sub[metric])
        out[metric] = {"rho": float(rho), "p_valor": float(p), "n": len(sub)}
    return out


def program_impact(works: pd.DataFrame, centrality: pd.DataFrame) -> pd.DataFrame:
    """Métricas de impacto por programa CAPES a partir das obras dos seus docentes."""
    programs = pd.read_csv(config.PROGRAMAS_CSV, dtype={"codigo_programa": str})
    faculty = pd.read_csv(config.FACULTY_MATCH_CSV, dtype=str).fillna("")

    # Docente -> (programas, anos); só os identificados têm perfil canônico
    docentes_by_program: dict[str, list[str]] = defaultdict(list)
    matched_by_program: dict[str, list[str]] = defaultdict(list)
    years_of: dict[str, set[int]] = {}
    for row in faculty.itertuples():
        for code in row.programas.split("|"):
            docentes_by_program[code].append(row.id_pessoa)
            if row.openalex_id_canonico:
                matched_by_program[code].append(row.openalex_id_canonico)
        if row.openalex_id_canonico:
            years_of[row.openalex_id_canonico] = {int(y) for y in row.anos.split("|") if y}

    # Obras de cada docente, só nos anos em que ele está na CAPES
    works_of: dict[str, list[int]] = defaultdict(list)
    years = works["ano"].to_numpy()
    for idx, authors in works["autores"].items():
        for a in authors:
            if a in years_of and years[idx] in years_of[a]:
                works_of[a].append(idx)

    threshold = works.groupby("ano")["citacoes"].transform(lambda s: s.quantile(0.9))
    top10 = (works["citacoes"] > threshold).to_numpy()
    h_index = dict(zip(centrality["author_id"], centrality["indice_h"]))
    pagerank = dict(zip(centrality["author_id"], centrality["cit_pagerank"]))

    rows = []
    for program in programs.itertuples():
        code = program.codigo_programa
        docentes = matched_by_program.get(code, [])
        idx = sorted({i for d in docentes for i in works_of.get(d, [])})
        sub = works.loc[idx]
        total = float(sub["citacoes"].sum())
        n_doc = len(docentes)
        rows.append(
            {
                "codigo_programa": code,
                "sigla_ies": program.sigla_ies,
                "nome_programa": program.nome_programa,
                "nivel": program.nivel,
                "nota_final": int(program.nota_final),
                "n_docentes_capes": len(set(docentes_by_program.get(code, []))),
                "n_docentes_identificados": n_doc,
                "n_obras": len(idx),
                "obras_por_docente": len(idx) / n_doc if n_doc else np.nan,
                "citacoes_total": total,
                "citacoes_por_docente": total / n_doc if n_doc else np.nan,
                "citacoes_por_docente_sem_obra_top": (total - (sub["citacoes"].max() if idx else 0)) / n_doc if n_doc else np.nan,
                "citacoes_por_obra": total / len(idx) if idx else np.nan,
                "mncs": float(sub["citacao_normalizada_ano"].mean()) if idx else np.nan,
                "fwci_medio": float(sub["fwci"].mean()) if idx else np.nan,
                "fwci_mediano": float(sub["fwci"].median()) if idx else np.nan,
                "frac_obras_top10pct": float(top10[idx].mean()) if idx else np.nan,
                "indice_h_mediano_docentes": float(np.median([h_index.get(d, 0) for d in docentes])) if docentes else np.nan,
                "pagerank_por_docente": float(np.mean([pagerank.get(d, 0.0) for d in docentes])) if docentes else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table["cobertura_docentes"] = table["n_docentes_identificados"] / table["n_docentes_capes"].replace(0, np.nan)
    table["elegivel"] = (
        (table["n_docentes_identificados"] >= MIN_FACULTY)
        & (table["cobertura_docentes"] >= MIN_COVERAGE)
        & (table["n_obras"] >= MIN_WORKS)
    )
    return table


def ies_vs_capes() -> pd.DataFrame:
    """Nota máxima e média por IES principal (sigla canônica da rede) + impacto da etapa 3.4."""
    programs = pd.read_csv(config.PROGRAMAS_CSV, dtype=str)
    institutions = pd.read_csv(config.INSTITUICOES_CSV, dtype=str)
    scope_sigla = load_scope_sigla(config.INSTITUICOES_CSV)
    sigla_to_canonical = {
        sigla: scope_sigla.get(oa_id)
        for sigla, oa_id in zip(institutions["sigla_ies"], institutions["openalex_id"])
        if isinstance(oa_id, str)
    }
    programs["ies_principal"] = programs["sigla_ies"].map(sigla_to_canonical)
    programs["nota_final"] = programs["nota_final"].astype(int)
    notes = (
        programs.dropna(subset=["ies_principal"])
        .groupby("ies_principal")
        .agg(
            nota_maxima=("nota_final", "max"),
            nota_media=("nota_final", "mean"),
            n_programas=("codigo_programa", "count"),
            tem_doutorado=("nivel", lambda s: any("DO" in v for v in s)),
        )
        .reset_index()
    )
    impact = pd.read_csv(RESULTS_DIR / "ies_impact.csv").rename(columns={"grupo": "ies_principal"})
    return notes.merge(impact, on="ies_principal", how="left")


def plot_programs(table: pd.DataFrame, correlations: dict) -> None:
    plt = setup_matplotlib()
    eligible = table[table["elegivel"]]
    panels = [
        ("citacoes_por_docente", "Citações por docente", True),
        ("fwci_mediano", "FWCI mediano das publicações", False),
        ("frac_obras_top10pct", "Fração das publicações no top 10 % do ano", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    rng = np.random.default_rng(0)
    for ax, (metric, label, log) in zip(axes, panels):
        notes = eligible["nota_final"].to_numpy()
        ax.scatter(notes + rng.uniform(-0.15, 0.15, len(notes)), eligible[metric], s=14, alpha=0.6)
        medians = eligible.groupby("nota_final")[metric].median()
        ax.plot(medians.index, medians.values, color="C3", marker="D", label="mediana")
        if log:
            ax.set_yscale("log")
        ax.set_xticks(sorted(eligible["nota_final"].unique()))
        ax.set_xlabel("Nota CAPES do programa")
        ax.set_ylabel(label)
        r = correlations[metric]
        ax.set_title(f"Spearman ρ = {r['rho']:.2f} (p = {r['p_valor']:.2g})", fontsize=9)
    axes[0].legend()
    fig.suptitle(f"Impacto dos programas (publicações dos docentes) × nota CAPES — {len(eligible)} programas")
    path = FIGURES_DIR / "capes_program_impact.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


def plot_ies(table: pd.DataFrame, correlations: dict) -> None:
    plt = setup_matplotlib()
    data = table.dropna(subset=["citacoes_por_membro"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (metric, label) in zip(axes, (("citacoes_por_docente", "Citações por docente"), ("fwci_mediano", "FWCI mediano"))):
        ax.scatter(data["nota_maxima"], data[metric], s=np.sqrt(data["n_membros"]) * 1.5, alpha=0.5)
        for row in data.itertuples():
            if getattr(row, metric) == data[metric].max() or row.nota_maxima == 7:
                ax.annotate(row.ies_principal, (row.nota_maxima, getattr(row, metric)), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("Nota CAPES máxima da IES")
        ax.set_ylabel(label)
        r = correlations[metric]
        ax.set_title(f"Spearman ρ = {r['rho']:.2f} (p = {r['p_valor']:.2g})", fontsize=9)
    axes[0].set_yscale("log")
    fig.suptitle("Impacto por IES principal × nota CAPES (tamanho do ponto ∝ nº de autores)")
    path = FIGURES_DIR / "capes_ies_impact.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Figura: {path}")


def main() -> None:
    pub_graph = load_graph(PUB_CITATION_PATH)
    works = work_table(pub_graph)
    centrality = pd.read_csv(RESULTS_DIR / "author_centrality.csv", usecols=["author_id", "indice_h", "cit_pagerank"])

    programs = program_impact(works, centrality)
    programs.to_csv(RESULTS_DIR / "program_impact.csv", index=False, encoding="utf-8")
    eligible = programs[programs["elegivel"]]
    print(f"Programas elegíveis: {len(eligible)} / {len(programs)}")
    program_corr = spearman_table(eligible, "nota_final", [*PROGRAM_METRICS, "n_docentes_identificados", "n_obras"])
    by_note = eligible.groupby("nota_final")[
        ["n_docentes_identificados", "obras_por_docente", "citacoes_por_docente", "citacoes_por_obra", "fwci_mediano", "frac_obras_top10pct", "indice_h_mediano_docentes"]
    ].median()
    by_note["n_programas"] = eligible.groupby("nota_final").size()

    ies = ies_vs_capes()
    ies.to_csv(RESULTS_DIR / "institution_vs_capes.csv", index=False, encoding="utf-8")
    ies_eval = ies.dropna(subset=["citacoes_por_membro"])
    ies_corr = {
        "nota_maxima": spearman_table(ies_eval, "nota_maxima", IES_METRICS),
        "nota_media": spearman_table(ies_eval, "nota_media", IES_METRICS),
    }

    top_cols = ["codigo_programa", "sigla_ies", "nome_programa", "nota_final", "n_docentes_identificados", "citacoes_por_docente", "fwci_mediano", "frac_obras_top10pct"]
    results = {
        "definicoes": {
            "programa": "publicações dos docentes CAPES identificados, nos anos em que estão no programa; cada publicação uma vez por programa",
            "elegivel": f">= {MIN_FACULTY} docentes identificados, cobertura >= {MIN_COVERAGE:.0%}, >= {MIN_WORKS} publicações",
            "ies": "IES principal dos autores (etapa 3.4); nota máxima e média dos programas da IES",
        },
        "programas": {
            "n_programas": len(programs),
            "n_elegiveis": len(eligible),
            "cobertura_docentes_mediana": float(programs["cobertura_docentes"].median()),
            "spearman_nota": program_corr,
            "medianas_por_nota": by_note.round(3).to_dict("index"),
            "top_fwci_mediano": eligible.nlargest(10, "fwci_mediano")[top_cols].round(3).to_dict("records"),
            "top_citacoes_por_docente": eligible.nlargest(10, "citacoes_por_docente")[top_cols].round(3).to_dict("records"),
            "nao_elegiveis": programs.loc[~programs["elegivel"], ["codigo_programa", "sigla_ies", "nota_final", "n_docentes_identificados", "cobertura_docentes", "n_obras"]].round(3).to_dict("records"),
        },
        "ies": {
            "n_ies": len(ies_eval),
            "spearman": ies_corr,
        },
    }
    save_json(results, "institution_vs_capes.json")
    plot_programs(programs, program_corr)
    plot_ies(ies_eval, ies_corr["nota_maxima"])

    print("\nProgramas — Spearman com a nota CAPES:")
    for metric, r in program_corr.items():
        print(f"  {metric}: ρ = {r['rho']:.3f} (p = {r['p_valor']:.3g}, n = {r['n']})")
    print("IES — Spearman com a nota máxima:")
    for metric, r in ies_corr["nota_maxima"].items():
        print(f"  {metric}: ρ = {r['rho']:.3f} (p = {r['p_valor']:.3g}, n = {r['n']})")


if __name__ == "__main__":
    main()
