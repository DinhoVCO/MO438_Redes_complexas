from __future__ import annotations

from pathlib import Path

import pandas as pd


def _normalize_area(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().upper()


def is_area_computacao(area: str) -> bool:
    """Área CAPES 'COMPUTAÇÃO' (exclui COMUNICAÇÃO E INFORMAÇÃO)."""
    if not area:
        return False
    if "COMUNICA" in area and "COMPUT" not in area.replace("COMUNICA", ""):
        return False
    return area.startswith("COMPUT") or area == "COMPUTAÇÃO"


def load_programas_cc(capes_xlsx: Path) -> pd.DataFrame:
    raw = pd.read_excel(capes_xlsx, sheet_name="GERAL", header=4)
    area_col = raw.columns[0]
    areas = raw[area_col].map(_normalize_area)
    mask = areas.map(is_area_computacao)
    cc = raw.loc[mask].copy()
    cc = cc.rename(
        columns={
            area_col: "area_avaliacao",
            raw.columns[1]: "sigla_ies",
            raw.columns[2]: "instituicao_ies",
            raw.columns[3]: "codigo_programa",
            raw.columns[4]: "nome_programa",
            raw.columns[5]: "nivel",
            raw.columns[8]: "nota_final",
        }
    )
    keep = [
        "area_avaliacao",
        "sigla_ies",
        "instituicao_ies",
        "codigo_programa",
        "nome_programa",
        "nivel",
        "nota_final",
    ]
    cc = cc[keep].dropna(subset=["instituicao_ies"])
    cc["instituicao_ies"] = cc["instituicao_ies"].astype(str).str.strip()
    cc["sigla_ies"] = cc["sigla_ies"].astype(str).str.strip()
    return cc.reset_index(drop=True)


def unique_institutions(programas: pd.DataFrame) -> pd.DataFrame:
    inst = (
        programas.groupby(["instituicao_ies", "sigla_ies"], as_index=False)
        .agg(n_programas=("codigo_programa", "count"))
        .sort_values("instituicao_ies")
    )
    return inst.reset_index(drop=True)
