#!/usr/bin/env python3
"""Cruza os docentes CAPES dos programas de Computação com os autores OpenAlex da rede
de coautoria e grava o programa de cada docente identificado como atributo do nó.

Entrada:
- data/capes/docentes_cc.csv (gerado por collect_capes.py)
- corpus OpenAlex (os perfis de autor são lidos do corpus, não da rede)
- output/colaboracao_cc_brasil.graphml.gz (gerado por run_pipeline.py)

Saídas:
- data/capes/docentes_openalex.csv — um registro por docente, com os IDs OpenAlex encontrados
  e o perfil canônico (o de mais obras), que run_pipeline.py usa para mesclar os perfis
- output/colaboracao_cc_brasil.graphml.gz — atualizado com docente_capes, programa_capes, id_pessoa_capes

Regras do cruzamento (só entre autores que publicaram numa IES do programa do docente):
- estrito: mesmo primeiro nome, mesmo último sobrenome, nomes do meio compatíveis
  (iniciais contam);
- flexível: mesmo primeiro nome (ou inicial), e o último sobrenome OpenAlex é algum dos
  sobrenomes do docente — comum em quem publica com um só sobrenome.
Um autor OpenAlex que casa com mais de um docente é descartado (ambíguo). Um docente pode
receber vários IDs OpenAlex (a OpenAlex às vezes divide uma pessoa em vários perfis).
"""
from __future__ import annotations

import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from collect_capes import FACULTY_CSV
from src.build_network import iter_work_authors, load_scope_sigla

MATCH_CSV = config.FACULTY_MATCH_CSV

PARTICLES = {"da", "de", "do", "das", "dos", "e", "di", "du", "del", "della", "van", "von", "der", "la", "le"}
SUFFIXES = {"junior", "jr", "filho", "neto", "sobrinho", "segundo", "terceiro"}


def name_tokens(name: str) -> list[str]:
    """Normaliza um nome: sem acentos, minúsculas, sem partículas nem sufixos."""
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = "".join(ch if ch.isalpha() else " " for ch in text)
    tokens = [t for t in text.split() if t not in PARTICLES]
    while len(tokens) > 2 and tokens[-1] in SUFFIXES:
        tokens.pop()
    return tokens


def _middle_compatible(oa_middle: list[str], faculty_rest: list[str]) -> bool:
    # Cada nome do meio OpenAlex precisa existir (completo ou como inicial) no docente
    for token in oa_middle:
        if len(token) == 1:
            if not any(t.startswith(token) for t in faculty_rest):
                return False
        elif token not in faculty_rest:
            return False
    return True


def match_tier(faculty: list[str], author: list[str]) -> str | None:
    """Devolve 'estrito', 'flexivel' ou None."""
    if len(faculty) < 2 or len(author) < 2:
        return None
    f_first, f_rest, f_last = faculty[0], faculty[1:], faculty[-1]
    a_first, a_middle, a_last = author[0], author[1:-1], author[-1]
    if len(a_last) == 1:
        return None
    first_ok = a_first == f_first
    if first_ok and a_last == f_last and _middle_compatible(a_middle, faculty[1:-1]):
        return "estrito"
    first_initial_ok = first_ok or (len(a_first) == 1 and f_first.startswith(a_first))
    if first_initial_ok and a_last in f_rest and _middle_compatible(a_middle, f_rest):
        return "flexivel"
    return None


def main() -> None:
    faculty = pd.read_csv(FACULTY_CSV, dtype=str)
    programs = pd.read_csv(config.PROGRAMAS_CSV, dtype=str)
    institutions = pd.read_csv(config.INSTITUICOES_CSV, dtype=str)

    # Programa CAPES -> ID OpenAlex da IES
    sigla_to_oa = dict(zip(institutions["sigla_ies"], institutions["openalex_id"]))
    program_to_oa = {
        code: sigla_to_oa.get(sigla)
        for code, sigla in zip(programs["codigo_programa"], programs["sigla_ies"])
    }

    # Um registro por docente (ID_PESSOA), com todos os programas e anos
    people = (
        faculty.groupby("ID_PESSOA")
        .agg(
            nm_docente=("NM_DOCENTE", "first"),
            programas=("CD_PROGRAMA_IES", lambda s: sorted(set(s))),
            siglas=("SG_ENTIDADE_ENSINO", lambda s: sorted(set(s))),
            anos=("AN_BASE", lambda s: sorted(set(s))),
            categoria=("DS_CATEGORIA_DOCENTE", lambda s: s.mode().iat[0]),
        )
        .reset_index()
    )
    print(f"Docentes distintos (2018–2024): {len(people)}")

    # Os perfis OpenAlex vêm do corpus, não da rede: na rede os perfis de um mesmo
    # docente já podem estar mesclados, e o cruzamento precisa ver todos
    scope_sigla = load_scope_sigla(config.INSTITUICOES_CSV)
    profile_name: dict[str, str] = {}
    profile_works: Counter = Counter()
    profile_scope: dict[str, set[str]] = defaultdict(set)
    for author_ids, names, work_scope in iter_work_authors(config.OBRAS_JSONL, scope_sigla):
        profile_works.update(author_ids)
        for raw_id, _, name in names:
            profile_name[raw_id] = name
        for aid, iids in work_scope.items():
            profile_scope[aid] |= iids
    # Perfis com afiliação no escopo, indexados por IES
    authors_by_inst: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for aid, iids in profile_scope.items():
        tokens = name_tokens(profile_name.get(aid, ""))
        for iid in iids:
            authors_by_inst[iid].append((aid, tokens))

    # Candidatos por docente
    matches: dict[str, dict[str, str]] = {}  # id_pessoa -> {author_id: tier}
    for row in people.itertuples():
        f_tokens = name_tokens(row.nm_docente)
        inst_ids = {program_to_oa.get(code) for code in row.programas} - {None, ""}
        found: dict[str, str] = {}
        for iid in inst_ids:
            for author_id, a_tokens in authors_by_inst.get(iid, []):
                tier = match_tier(f_tokens, a_tokens)
                if tier and (found.get(author_id) != "estrito"):
                    found[author_id] = tier
        # Se há casamentos estritos, os flexíveis do mesmo docente são descartados
        if any(t == "estrito" for t in found.values()):
            found = {a: t for a, t in found.items() if t == "estrito"}
        matches[row.ID_PESSOA] = found

    # Autor OpenAlex reivindicado por mais de um docente -> ambíguo, descartar
    claims: dict[str, list[str]] = defaultdict(list)
    for person, found in matches.items():
        for author_id in found:
            claims[author_id].append(person)
    ambiguous = {a for a, persons in claims.items() if len(persons) > 1}
    for person in matches:
        matches[person] = {a: t for a, t in matches[person].items() if a not in ambiguous}

    people["openalex_ids"] = people["ID_PESSOA"].map(lambda p: "|".join(sorted(matches[p])))
    people["n_openalex_ids"] = people["ID_PESSOA"].map(lambda p: len(matches[p]))
    # Perfil canônico = o de mais obras no corpus; os demais são mesclados nele na rede
    people["openalex_id_canonico"] = people["ID_PESSOA"].map(
        lambda p: min(matches[p], key=lambda a: (-profile_works[a], a)) if matches[p] else ""
    )
    people["nivel"] = people["ID_PESSOA"].map(
        lambda p: (sorted(set(matches[p].values()))[0] if matches[p] else "")
    )
    for col in ("programas", "siglas", "anos"):
        people[col] = people[col].map("|".join)
    people.rename(columns={"ID_PESSOA": "id_pessoa"}).to_csv(MATCH_CSV, index=False, encoding="utf-8")

    # Atributos nos nós. Se a rede já foi construída com a mesclagem, só o canônico existe;
    # se não, todos os perfis do docente recebem o rótulo.
    graph = nx.read_graphml(config.GRAPHML_PATH)
    for _, data in graph.nodes(data=True):
        data["docente_capes"] = False
        data["programa_capes"] = ""
        data["id_pessoa_capes"] = ""
    for row in people.itertuples():
        for author_id in matches[row.ID_PESSOA]:
            if author_id not in graph:
                continue
            node = graph.nodes[author_id]
            node["docente_capes"] = True
            node["programa_capes"] = row.programas
            node["id_pessoa_capes"] = row.ID_PESSOA
    nx.write_graphml(graph, config.GRAPHML_PATH)
    stale = sum(1 for a in people["openalex_id_canonico"] if a and a not in graph)
    if stale:
        print(f"Aviso: {stale} perfis canônicos não estão na rede; rode run_pipeline.py de novo.")

    matched = people["n_openalex_ids"] > 0
    print(f"Docentes com pelo menos um autor OpenAlex: {matched.sum()} / {len(people)} ({matched.mean():.1%})")
    print(f"  por nível: {people.loc[matched, 'nivel'].value_counts().to_dict()}")
    perm = people["categoria"] == "PERMANENTE"
    print(f"  permanentes: {(matched & perm).sum()} / {perm.sum()} ({(matched & perm).sum() / perm.sum():.1%})")
    print(f"Autores OpenAlex descartados por ambiguidade: {len(ambiguous)}")
    print(f"Nós marcados como docente: {sum(1 for _, d in graph.nodes(data=True) if d['docente_capes'])}")
    print(f"Saídas: {MATCH_CSV} e {config.GRAPHML_PATH}")


if __name__ == "__main__":
    main()
