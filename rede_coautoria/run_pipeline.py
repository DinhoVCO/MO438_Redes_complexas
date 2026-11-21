#!/usr/bin/env python3
"""Pipeline: CAPES (CC Brasil) -> OpenAlex -> rede de coautoria -> GraphML."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd

import config
from src.build_network import (
    build_coauthorship_graph,
    load_author_aliases,
    load_seen_work_ids,
    write_graphml,
)
from src.capes import load_programas_cc, unique_institutions
from src.openalex_client import (
    OpenAlexClient,
    append_works_jsonl,
    chunk_list,
    map_institutions,
)
from src.relatorio import RelatorioConstrucao


def load_coleta_state() -> dict:
    if config.COLETA_STATE_JSON.exists():
        return json.loads(config.COLETA_STATE_JSON.read_text(encoding="utf-8"))
    return {"completed_chunks": []}


def save_coleta_state(state: dict) -> None:
    config.COLETA_STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.COLETA_STATE_JSON.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Apaga obras.jsonl e estado de coleta antes de buscar de novo.",
    )
    args = parser.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = ROOT / "logs" / "execucao_pipeline.md"
    rel = RelatorioConstrucao(log_path)

    # --- 1. CAPES ---
    rel.section("1. Escopo CAPES — programas de Computação")
    if not config.CAPES_XLSX.exists():
        raise FileNotFoundError(f"Arquivo CAPES não encontrado: {config.CAPES_XLSX}")

    programas = load_programas_cc(config.CAPES_XLSX)
    programas.to_csv(config.PROGRAMAS_CSV, index=False, encoding="utf-8")
    instituicoes = unique_institutions(programas)

    rel.paragraph(
        "Fonte: Avaliação Quadrienal CAPES 2025 (aba GERAL). "
        "Filtro: área de avaliação **COMPUTAÇÃO** (programas de Ciência da Computação no Brasil)."
    )
    rel.bullet(f"Programas de pós-graduação no escopo: **{len(programas)}**")
    rel.bullet(f"Instituições (IES) distintas: **{len(instituicoes)}**")
    rel.bullet(f"Arquivo gerado: `{config.PROGRAMAS_CSV.relative_to(ROOT)}`")
    rel.bullets_end()

    # --- 2. OpenAlex institutions ---
    rel.section("2. Mapeamento IES → OpenAlex Institution")
    client = OpenAlexClient()
    if config.INSTITUICOES_CSV.exists() and not args.fresh:
        mapped = pd.read_csv(config.INSTITUICOES_CSV)
        rel.paragraph("Reutilizando mapeamento existente (`data/instituicoes_openalex.csv`).")
    else:
        mapped = map_institutions(instituicoes, config.INSTITUICOES_CSV, client)
    ok = mapped["openalex_id"].notna().sum()
    missing = mapped[mapped["openalex_id"].isna()]

    rel.paragraph(
        "Cada IES do escopo é buscada na API OpenAlex (`/institutions`, `country_code:br`). "
        "Escolhe-se o melhor candidato por relevância/nome."
    )
    rel.bullet(f"Instituições mapeadas: **{ok}** / {len(mapped)}")
    rel.bullet(f"Saída: `{config.INSTITUICOES_CSV.relative_to(ROOT)}`")
    rel.bullets_end()
    if len(missing):
        rel.paragraph("**IES sem match automático (revisar manualmente):**")
        for _, row in missing.iterrows():
            rel.bullet(f"{row['sigla_ies']} — {row['instituicao_ies']}")
        rel.bullets_end()

    # Várias IES CAPES podem apontar para o mesmo ID (ex.: campi da USP); deduplicar.
    inst_ids = list(dict.fromkeys(mapped["openalex_id"].dropna().astype(str)))
    if not inst_ids:
        raise RuntimeError("Nenhuma instituição OpenAlex mapeada; abortando coleta.")

    # --- 3. Coleta de obras ---
    rel.section("3. Coleta de publicações (OpenAlex Works)")
    year_filter = f"publication_year:{config.PUBLICATION_YEAR_MIN}-{config.PUBLICATION_YEAR_MAX}"
    field_filter = f"primary_topic.field.id:{config.OPENALEX_FIELD_CS}"
    rel.paragraph(
        "Para cada bloco de instituições do escopo, consulta-se `/works` com filtros combinados:"
    )
    rel.bullet(
        f"`authorships.institutions.lineage` — OR de IDs OpenAlex (lotes de {config.INSTITUTIONS_PER_REQUEST}); "
        "o lineage inclui subunidades (ex.: IBILCE dentro da UNESP)"
    )
    rel.bullet(f"`{field_filter}` — Computer Science")
    rel.bullet(f"`{year_filter}`")
    rel.bullet("Paginação via `cursor`; deduplicação por ID da obra em `obras.jsonl`.")
    rel.bullets_end()

    if args.fresh and config.OBRAS_JSONL.exists():
        config.OBRAS_JSONL.unlink()
        save_coleta_state({"completed_chunks": []})

    state = load_coleta_state()
    completed = set(state.get("completed_chunks", []))
    seen = load_seen_work_ids(config.OBRAS_JSONL)
    chunks = chunk_list(inst_ids, config.INSTITUTIONS_PER_REQUEST)

    for idx, chunk in enumerate(chunks, start=1):
        if idx in completed:
            rel.bullet(f"Lote {idx}/{len(chunks)}: já concluído (pulando)")
            continue
        inst_filter = "authorships.institutions.lineage:" + "|".join(chunk)
        filter_expr = f"{inst_filter},{field_filter},{year_filter}"
        try:
            n_chunk = append_works_jsonl(
                client.iter_works(filter_expr), config.OBRAS_JSONL, seen
            )
        except Exception as exc:
            rel.bullet(f"Lote {idx}/{len(chunks)}: **interrompido** ({exc})")
            rel.bullets_end()
            break
        completed.add(idx)
        state["completed_chunks"] = sorted(completed)
        save_coleta_state(state)
        rel.bullet(
            f"Lote {idx}/{len(chunks)}: +{n_chunk} obras novas "
            f"(acumulado único: {len(seen)})"
        )
    rel.bullets_end()
    rel.bullet(f"Total de obras únicas no corpus: **{len(seen)}**")
    rel.bullet(f"Lotes concluídos: **{len(completed)}** / {len(chunks)}")
    rel.bullet(f"Arquivo: `{config.OBRAS_JSONL.relative_to(ROOT)}`")
    rel.bullets_end()

    # Não construir a rede com o corpus incompleto
    if len(completed) < len(chunks):
        print(
            f"Coleta incompleta ({len(completed)}/{len(chunks)} lotes). "
            "Rode de novo `python run_pipeline.py` para retomar; a rede não foi reconstruída.",
            flush=True,
        )
        sys.exit(2)

    # --- 4. Rede ---
    rel.section("4. Construção da rede de coautoria")
    rel.paragraph(
        "A partir de cada obra, extrai-se a lista de autores (OpenAlex Author ID). "
        "Para cada par de autores na mesma obra, adiciona-se uma aresta não direcionada; "
        "o peso `weight` conta coautorias conjuntas no corpus. "
        "**Todos os coautores** presentes nas obras coletadas entram como vértices "
        "(inclui colaboradores internacionais ligados aos programas via publicações)."
    )
    rel.paragraph(
        "Atributos de nó: `label` (nome), `n_publicacoes_corpus`, `afiliacao_escopo` "
        "(True se o autor tem afiliação em alguma IES mapeada do escopo CAPES)."
    )

    # Perfis OpenAlex de um mesmo docente CAPES viram um só nó (se label_faculty.py já rodou)
    aliases = load_author_aliases(config.FACULTY_MATCH_CSV)
    graph, stats = build_coauthorship_graph(config.OBRAS_JSONL, config.INSTITUICOES_CSV, aliases)
    write_graphml(graph, config.GRAPHML_PATH)
    config.ESTATISTICAS_JSON.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rel.table(
        ["Métrica", "Valor"],
        [
            ["Obras no corpus", str(stats["n_obras_corpus"])],
            ["Vértices (autores)", str(stats["n_vertices"])],
            ["Arestas (pares coautoria)", str(stats["n_arestas"])],
            ["Autores com afiliação no escopo IES", str(stats["n_vertices_afiliacao_escopo"])],
            ["GraphML", f"`{config.GRAPHML_PATH.relative_to(ROOT)}`"],
        ],
    )

    min_v = 10_000
    min_e = 10_000
    if stats["n_vertices"] >= min_v and stats["n_arestas"] >= min_e:
        rel.paragraph(
            f"Critério da F1 atendido (≥ {min_v} vértices e ≥ {min_e} arestas, coleta própria)."
        )
    else:
        rel.paragraph(
            f"Ainda abaixo do mínimo da F1 ({min_v} vértices / {min_e} arestas). "
            "Executar novamente `python run_pipeline.py` para retomar lotes pendentes."
        )

    rel.section("5. Reprodutibilidade")
    rel.bullet(f"Executar: `python run_pipeline.py` (retoma coleta) ou `--fresh` para recomeçar")
    rel.bullet(f"mailto OpenAlex: `{config.OPENALEX_MAILTO}`")
    rel.bullets_end()

    print("Pipeline concluído.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
