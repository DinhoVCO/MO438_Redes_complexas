#!/usr/bin/env python3
"""Baixa dos dados abertos da CAPES (Sucupira) os docentes e a produção bibliográfica
declarada pelos programas de Computação do escopo, 2018–2024.

Os arquivos da CAPES cobrem todas as áreas do Brasil (centenas de MB cada); aqui eles
são lidos em streaming e só as linhas dos programas de `programas_cc_brasil.csv` são
gravadas em `data/capes/`.

Saídas:
- data/capes/docentes_cc.csv  — uma linha por (ano, programa, docente)
- data/capes/producao_cc.csv  — artigos em periódico e trabalhos em anais, com DOI extraído
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config

CKAN = "https://dadosabertos.capes.gov.br/api/3/action/package_show"
CAPES_DIR = config.DATA_DIR / "capes"
FACULTY_CSV = CAPES_DIR / "docentes_cc.csv"
PRODUCTION_CSV = CAPES_DIR / "producao_cc.csv"

FACULTY_PACKAGES = [
    "2017-a-2020-docentes-da-pos-graduacao-stricto-sensu-no-brasil",
    "2021-a-2024-docentes-da-pos-graduacao-stricto-sensu-no-brasil",
]
PRODUCTION_PACKAGES = [
    "2017-a-2020-detalhes-da-producao-intelectual-bibliografica-de-programas-de-pos-graduacao",
    "2021-a-2024-detalhes-da-producao-intelectual-bibliografica-de-programas-de-pos-graduacao",
]
YEARS = {str(y) for y in range(config.PUBLICATION_YEAR_MIN, config.PUBLICATION_YEAR_MAX + 1)}

# Colunas mantidas em cada saída (as demais são descartadas)
FACULTY_COLUMNS = [
    "AN_BASE", "CD_PROGRAMA_IES", "NM_PROGRAMA_IES", "SG_ENTIDADE_ENSINO", "ID_PESSOA",
    "NM_DOCENTE", "DS_CATEGORIA_DOCENTE", "CD_CAT_BOLSA_PRODUTIVIDADE",
]
PRODUCTION_COLUMNS = [
    "AN_BASE_PRODUCAO", "CD_PROGRAMA_IES", "SG_ENTIDADE_ENSINO", "ID_ADD_PRODUCAO_INTELECTUAL",
    "SUBTIPO", "DOI",
]

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\[\]\"';]+", re.IGNORECASE)


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin1")


def list_csv_resources(package: str) -> list[tuple[str, str]]:
    data = requests.get(CKAN, params={"id": package}, timeout=60).json()["result"]
    return [
        (r["name"], r["url"])
        for r in data["resources"]
        if (r.get("format") or "").upper() == "CSV"
    ]


def stream_rows(url: str):
    """Itera as linhas de um CSV da CAPES (separador ';') como dicionários."""
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        lines = (_decode(raw) for raw in resp.iter_lines())
        reader = csv.reader(lines, delimiter=";")
        header = next(reader)
        for parts in reader:
            yield dict(zip(header, parts))


def extract_doi(row: dict[str, str]) -> str:
    # O DOI aparece em DS_DOI, DS_URL_DOI ou dentro de DS_URL ("[DOI:10.xxx/...]")
    for col in ("DS_DOI", "DS_URL_DOI", "DS_URL", "DS_OBSERVACOES"):
        match = DOI_RE.search(row.get(col) or "")
        if match:
            return match.group(0).lower().rstrip(".,)")
    return ""


def production_subtype(resource_name: str) -> str | None:
    name = resource_name.upper()
    if "ARTPE" in name or "ARTIGO EM PERI" in name:
        return "ARTIGO_PERIODICO"
    if "ANAIS" in name:
        return "TRABALHO_ANAIS"
    return None


def main() -> None:
    CAPES_DIR.mkdir(parents=True, exist_ok=True)
    with config.PROGRAMAS_CSV.open(encoding="utf-8") as f:
        programs = {r["codigo_programa"] for r in csv.DictReader(f)}
    print(f"Programas no escopo: {len(programs)}", flush=True)

    # --- Docentes: um arquivo por ano ---
    n_faculty = 0
    with FACULTY_CSV.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FACULTY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for package in FACULTY_PACKAGES:
            for name, url in list_csv_resources(package):
                year = re.search(r"DOCENTE-(\d{4})", name)
                if not year or year.group(1) not in YEARS:
                    continue
                n_before = n_faculty
                for row in stream_rows(url):
                    if row.get("CD_PROGRAMA_IES") in programs:
                        writer.writerow(row)
                        n_faculty += 1
                print(f"  docentes {year.group(1)}: {n_faculty - n_before} linhas", flush=True)

    # --- Produção bibliográfica: artigos em periódico e trabalhos em anais ---
    n_prod = 0
    with PRODUCTION_CSV.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=PRODUCTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for package in PRODUCTION_PACKAGES:
            for name, url in list_csv_resources(package):
                subtype = production_subtype(name)
                if subtype is None:
                    continue
                n_before = n_prod
                for row in stream_rows(url):
                    if row.get("CD_PROGRAMA_IES") not in programs:
                        continue
                    if row.get("AN_BASE_PRODUCAO") not in YEARS:
                        continue
                    row["SUBTIPO"] = subtype
                    row["DOI"] = extract_doi(row)
                    writer.writerow(row)
                    n_prod += 1
                print(f"  {subtype} ({name[:60]}): {n_prod - n_before} linhas", flush=True)

    print(f"Docentes: {n_faculty} linhas -> {FACULTY_CSV}")
    print(f"Produção: {n_prod} linhas -> {PRODUCTION_CSV}")


if __name__ == "__main__":
    main()
