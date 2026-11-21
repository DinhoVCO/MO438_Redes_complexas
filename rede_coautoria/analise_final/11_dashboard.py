#!/usr/bin/env python3
"""Dashboard com as principais conclusões, publicado pelo GitHub Pages (pasta docs/ da raiz).

Lê só arquivos de results/ (não recalcula nada) e grava:
- docs/index.html — dashboard (Chart.js), gerado a partir de dashboard/template_dashboard.html
- docs/mapa.html  — cópia do mapa interativo (mapas/mapa_interativo.html), embutido no dashboard
- docs/.nojekyll  — para o GitHub Pages servir os arquivos como estão
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from common import ANALYSIS_DIR, RESULTS_DIR, ROOT

DOCS_DIR = ROOT.parent / "docs"
REPO_URL = "https://github.com/DinhoVCO/MO438_Redes_complexas"
N_TOP_COMMUNITIES = 12

CENTRALITY_NAMES = {
    "cit_pagerank": ("PageRank", "citação"),
    "cit_forca_entrada": ("Citações recebidas de autores", "citação"),
    "cit_autoridade": ("Autoridade (HITS)", "citação"),
    "cit_hub": ("Hub (HITS)", "citação"),
    "grau": ("Nº de coautores", "coautoria"),
    "intermediacao": ("Intermediação", "coautoria"),
    "k_core": ("k-core", "coautoria"),
    "proximidade": ("Proximidade", "coautoria"),
    "autovetor": ("Autovetor", "coautoria"),
}


def load_json(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def short(text: str, n: int) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def main() -> None:
    summary = pd.read_csv(RESULTS_DIR / "communities_summary.csv")
    impact = pd.read_csv(RESULTS_DIR / "group_impact.csv")
    programs = pd.read_csv(RESULTS_DIR / "program_impact.csv")
    communities = load_json("communities.json")
    structural = load_json("structural_properties.json")
    group_citation = load_json("group_citation_network.json")
    centrality = load_json("centrality.json")
    group_impact = load_json("group_impact.json")
    capes = load_json("institution_vs_capes.json")
    stats = json.loads((ROOT / "output" / "estatisticas_rede.json").read_text(encoding="utf-8"))

    # Conclusão 1: comunidades × instituições
    top = summary.nlargest(N_TOP_COMMUNITIES, "tamanho")
    validation = communities["validacao"]["louvain"]["pureza_ies"]
    c1 = {
        "comunidades": [
            {
                "rotulo": f"C{r.comunidade} · {short(r.topico_dominante, 30)}",
                "topico": r.topico_dominante,
                "tamanho": int(r.tamanho),
                "n_ies": int(r.n_ies),
                "ies": r.ies_dominante,
                "pureza": round(float(r.pureza_ies), 4),
            }
            for r in top.itertuples()
        ],
        "pureza_media": validation["pureza_media_ponderada"],
        "n_ies_mediana": validation["n_ies_por_comunidade_mediana"],
        "coautoria_mesma_ies": structural["assortatividade"]["ies_coautoria"]["frac_arestas_mesma_ies"],
    }

    # Conclusão 2: citação interna
    c2 = {
        level: {
            "observado": group_citation[key]["homofilia"]["frac_interna_observada"],
            "esperado": group_citation[key]["homofilia"]["frac_interna_esperada"],
        }
        for level, key in (("comunidades", "comunidades"), ("ies", "ies"))
    }
    c2["mesma_comunidade"] = group_citation["mesmo_grupo"]["frac_mesma_comunidade"]
    c2["mesma_ies"] = group_citation["mesmo_grupo"]["frac_mesma_ies"]

    # Conclusão 3: centralidades × citações
    validation_c = centrality["validacao_escopo"]
    measures = sorted(
        ({"nome": CENTRALITY_NAMES[k][0], "rede": CENTRALITY_NAMES[k][1], "rho": round(v["spearman_citacoes"], 3)}
         for k, v in validation_c.items() if k in CENTRALITY_NAMES),
        key=lambda m: -m["rho"],
    )
    c3 = {
        "medidas": measures,
        "pagerank_rho": validation_c["cit_pagerank"]["spearman_citacoes"],
        "jaccard_top50": validation_c["cit_pagerank"]["jaccard_top50_citacoes"],
    }

    # Conclusão 4: tamanho × impacto (comunidades da componente gigante)
    giant = impact[impact["componente_gigante"]].dropna(subset=["fwci_mediano"])
    rho_giant = group_impact["comunidades"]["spearman_tamanho_so_componente_gigante"]
    c4 = {
        "comunidades": [
            {
                "id": int(r.grupo),
                "topico": short(r.topico_dominante, 45),
                "ies": r.ies_dominante,
                "membros": int(r.n_membros),
                "citacoes_total": float(r.citacoes_total),
                "fwci_mediano": round(float(r.fwci_mediano), 4),
            }
            for r in giant.itertuples()
        ],
        "rho": {"citacoes_total": rho_giant["citacoes_total"]["rho"], "fwci_mediano": rho_giant["fwci_mediano"]["rho"]},
    }

    # Conclusão 5: programas × nota CAPES
    eligible = programs[programs["elegivel"]].copy()
    rng = np.random.default_rng(0)  # deslocamento horizontal fixo, só para os pontos não se sobreporem
    eligible["jitter"] = rng.uniform(-0.18, 0.18, len(eligible)).round(3)
    medians = eligible.groupby("nota_final")[["n_obras", "fwci_mediano"]].median()
    spearman = capes["programas"]["spearman_nota"]
    c5 = {
        "programas": [
            {
                "ies": r.sigla_ies,
                "nome": str(r.nome_programa).capitalize(),
                "nota": int(r.nota_final),
                "jitter": float(r.jitter),
                "n_obras": int(r.n_obras),
                "fwci_mediano": round(float(r.fwci_mediano), 4),
            }
            for r in eligible.itertuples()
        ],
        "medianas": [{"nota": int(n), "n_obras": float(m.n_obras), "fwci_mediano": round(float(m.fwci_mediano), 4)} for n, m in medians.iterrows()],
        "rho": {"n_obras": spearman["n_obras"]["rho"], "fwci_mediano": spearman["fwci_mediano"]["rho"]},
        "fwci_nota3": float(medians.loc[3, "fwci_mediano"]),
        "fwci_nota7": float(medians.loc[7, "fwci_mediano"]),
    }

    data = {
        "repo": REPO_URL,
        "kpis": [
            {"valor": 87, "rotulo": "programas de pós-graduação"},
            {"valor": stats["n_obras_corpus"], "rotulo": "publicações (2018–2024)"},
            {"valor": stats["n_vertices"], "rotulo": "autores na rede de coautoria"},
            {"valor": int((summary["tamanho"] >= 10).sum()), "rotulo": "comunidades de pesquisa"},
        ],
        "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
    }

    template = (ANALYSIS_DIR / "dashboard" / "template_dashboard.html").read_text(encoding="utf-8")
    html = template.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    shutil.copyfile(ANALYSIS_DIR / "mapas" / "mapa_interativo.html", DOCS_DIR / "mapa.html")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Dashboard: {DOCS_DIR / 'index.html'}")
    print(f"Mapa: {DOCS_DIR / 'mapa.html'}")


if __name__ == "__main__":
    main()
