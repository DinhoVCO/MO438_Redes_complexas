# Pipeline — redes de coautoria e de citação (Computação, Brasil)

Constrói as redes a partir dos programas de **Computação** da Avaliação Quadrienal CAPES 2025 e das publicações da **OpenAlex** (2018–2024). Como reproduzir o projeto inteiro: `README.md` na raiz. Metodologia e números da coleta: `relatorio_construcao.md`.

## Uso

```bash
cd rede_coautoria
pip install -r requirements.txt
```

Ordem completa (coleta do zero, com `OPENALEX_API_KEY` em `.env`):

| Passo | Script | O que faz |
|------:|--------|-----------|
| 1 | `run_pipeline.py --fresh` | Programas CAPES → IES na OpenAlex → obras com afiliação nas IES e tópico principal em Computer Science → rede de coautoria |
| 2 | `collect_capes.py` | Baixa docentes e produção 2018–2024 dos 87 programas (dados abertos CAPES) |
| 3 | `label_faculty.py` | Casa cada docente com perfis OpenAlex (nome + IES) |
| 4 | `collect_faculty_works.py` | Acrescenta ao corpus as obras 2018–2024 dos docentes identificados (`coleta_origem="docentes"`) |
| 5 | `label_faculty.py` | Escolhe o perfil canônico de cada docente no corpus ampliado |
| 6 | `run_pipeline.py` | Reconstrói a coautoria com os perfis de um mesmo docente mesclados |
| 7 | `build_rede_citacoes.py` | Rede de citação entre publicações |
| 8 | `label_faculty.py` | Grava `docente_capes` e `programa_capes` na rede mesclada |

`run_pipeline.py` e `collect_faculty_works.py` retomam de onde pararam (`data/coleta_estado.json`, `data/capes/coleta_docentes_estado.json`).

## Arquivos

| Arquivo | Descrição | No git |
|---------|-----------|:------:|
| `data/programas_cc_brasil.csv` | 87 programas CAPES no escopo, com nota | sim |
| `data/instituicoes_openalex.csv` | Mapeamento IES → OpenAlex (com correções manuais e observação) | sim |
| `data/obras_2026-11.jsonl` | Corpus: 84 676 obras (~480 MB) | **não** (link no README da raiz) |
| `data/capes/docentes_cc.csv` | Docentes dos programas, 2018–2024 | sim |
| `data/capes/producao_cc.csv` | Artigos e trabalhos em anais declarados à CAPES, com DOI | sim |
| `data/capes/docentes_openalex.csv` | Docente → perfis OpenAlex e perfil canônico | sim |
| `output/colaboracao_cc_brasil.graphml.gz` | Coautoria (atributos: IES principal, docente, programa, comunidade) | sim |
| `output/citacoes_cc_brasil.graphml` | Citação entre publicações (ano, citações, FWCI, autores) | sim |
| `output/citacoes_autores_cc_brasil.graphml.gz` | Citação entre autores | sim |
| `output/citacoes_comunidades.graphml`, `output/citacoes_ies.graphml` | Citação entre grupos | sim |
| `output/estatisticas_*.json` | Contagens das redes | sim |
| `logs/execucao_pipeline.md` | Log de cada execução do pipeline | não |

## Módulos

| Módulo | Função |
|--------|--------|
| `config.py` | Caminhos, anos, filtro de campo, tamanho de lote, campos pedidos à OpenAlex |
| `src/capes.py` | Extração dos programas de Computação da planilha CAPES |
| `src/openalex_client.py` | Cliente HTTP da OpenAlex (paginação por cursor, `api_key`, novas tentativas) |
| `src/build_network.py` | Rede de coautoria, aliases de perfis, IES principal de cada autor |
| `src/build_citation_network.py` | Rede de citação entre publicações |
| `src/relatorio.py` | Log das execuções |
