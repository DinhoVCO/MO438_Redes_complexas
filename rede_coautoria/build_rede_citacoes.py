#!/usr/bin/env python3
"""Monta o GraphML dirigido de citações a partir do corpus (referenced_works de cada obra).

referencias.jsonl é só um complemento opcional da coleta antiga (entrega parcial).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.build_citation_network import build_citation_graph, write_citation_graphml
from src.build_network import load_author_aliases


def main() -> None:
    # referencias.jsonl é opcional: a coleta atual já grava referenced_works em obras.jsonl
    aliases = load_author_aliases(config.FACULTY_MATCH_CSV)
    graph, stats = build_citation_graph(config.OBRAS_JSONL, config.REFERENCIAS_JSONL, aliases)
    write_citation_graphml(graph, config.CITACOES_GRAPHML_PATH)
    config.ESTATISTICAS_CITACOES_JSON.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=True))
    print(f"GraphML: {config.CITACOES_GRAPHML_PATH}")


if __name__ == "__main__":
    main()
