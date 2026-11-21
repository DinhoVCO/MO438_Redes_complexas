"""Utilidades comuns das análises da entrega final: caminhos, carga dos grafos, saída e estilo."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT = ANALYSIS_DIR.parent
sys.path.insert(0, str(ROOT))
# O console do Windows usa cp1252; acentos e setas precisam de UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config  # noqa: E402

RESULTS_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"

COAUTHORSHIP_PATH = config.GRAPHML_PATH
PUB_CITATION_PATH = config.CITACOES_GRAPHML_PATH
AUTHOR_CITATION_PATH = config.OUTPUT_DIR / "citacoes_autores_cc_brasil.graphml.gz"


def load_graph(path: Path) -> nx.Graph:
    """Lê um GraphML (comprimido ou não) e informa o tamanho e o tempo de leitura."""
    start = time.perf_counter()
    graph = nx.read_graphml(path)
    print(
        f"Lido {path.name}: {graph.number_of_nodes()} vértices, {graph.number_of_edges()} arestas "
        f"({time.perf_counter() - start:.0f} s)",
        flush=True,
    )
    return graph


def save_json(data: dict[str, Any], name: str) -> Path:
    """Grava um dicionário de resultados em results/<name>."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados: {path}")
    return path


def setup_matplotlib():
    """Backend sem janela e estilo único para as figuras do relatório."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return plt


def work_table(pub_graph: nx.DiGraph) -> pd.DataFrame:
    """Uma linha por obra: autores, ano, citações globais e do corpus, citação normalizada, FWCI."""
    rows = []
    for wid, d in pub_graph.nodes(data=True):
        rows.append(
            {
                "work_id": wid,
                "autores": [a for a in (d.get("autor_ids") or "").split("|") if a],
                "ano": d["publication_year"],
                "citacoes": max(int(d["cited_by_count"]), 0),
                "citacoes_corpus": pub_graph.in_degree(wid),
                "fwci": d["fwci"] if d["fwci"] >= 0 else np.nan,
            }
        )
    works = pd.DataFrame(rows)
    # Normalização por ano: obras antigas tiveram mais tempo para ser citadas
    year_mean = works.groupby("ano")["citacoes"].transform("mean")
    works["citacao_normalizada_ano"] = works["citacoes"] / year_mean.replace(0, np.nan)
    return works
