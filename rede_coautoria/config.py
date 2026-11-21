"""Parâmetros do pipeline de coautoria (CC Brasil / CAPES + OpenAlex)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

CAPES_XLSX = PROJECT_ROOT / "data" / (
    "27052026_Resultado_2826627_Resultados_da_Avaliacao_Quadrienal_2025_para_publicacao_25_5___senha (1).xlsx"
)

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

PROGRAMAS_CSV = DATA_DIR / "programas_cc_brasil.csv"
INSTITUICOES_CSV = DATA_DIR / "instituicoes_openalex.csv"
# Corpus da coleta de 2026-11 (mapeamento de IES corrigido, filtro por lineage).
# A coleta de 2026-10, usada na entrega parcial, fica em data/coleta_2026-10/ e data/obras.jsonl.
OBRAS_JSONL = DATA_DIR / "obras_2026-11.jsonl"
REFERENCIAS_JSONL = DATA_DIR / "referencias.jsonl"
ESTATISTICAS_JSON = OUTPUT_DIR / "estatisticas_rede.json"
ESTATISTICAS_CITACOES_JSON = OUTPUT_DIR / "estatisticas_citacoes.json"
GRAPHML_PATH = OUTPUT_DIR / "colaboracao_cc_brasil.graphml.gz"  # comprimido: o .graphml passa de 100 MB
CITACOES_GRAPHML_PATH = OUTPUT_DIR / "citacoes_cc_brasil.graphml"
RELATORIO_PATH = ROOT / "relatorio_construcao.md"

# Docentes CAPES × perfis OpenAlex (label_faculty.py); também define a mesclagem de perfis
FACULTY_MATCH_CSV = DATA_DIR / "capes" / "docentes_openalex.csv"

# OpenAlex: chave em variável de ambiente ou em rede_coautoria/.env (não versionar).
# Crie em https://openalex.org — a chave gratuita aumenta o orçamento diário.
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")

OPENALEX_MAILTO = "r183585@dac.unicamp.br"
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()

# Campo OpenAlex fields/17 — display_name oficial: "Computer Science"
OPENALEX_FIELD_CS = "https://openalex.org/fields/17"

PUBLICATION_YEAR_MIN = 2018
PUBLICATION_YEAR_MAX = 2024

# Campos pedidos a /works. Inclui referências, citações e FWCI para que a rede de
# citações e as métricas de impacto não precisem de uma segunda passagem na API.
WORKS_SELECT = ",".join([
    "id", "doi", "display_name", "publication_year", "type", "authorships",
    "primary_topic", "cited_by_count", "fwci", "citation_normalized_percentile",
    "referenced_works",
])

# Quantas instituições por filtro OR (limite prático de URL)
INSTITUTIONS_PER_REQUEST = 15

REQUEST_SLEEP_SEC = 0.25
# Limite atual da API OpenAlex (com api_key).
PER_PAGE = 100

COLETA_STATE_JSON = DATA_DIR / "coleta_estado.json"

# IDs OpenAlex por consulta ao completar referenced_works (limite de OR = 100).
REFS_IDS_PER_REQUEST = 50
