# Impacto científico dos programas de pós-graduação em Computação no Brasil

Projeto da disciplina **MO438 — Redes Complexas** (UNICAMP, 2026). Estuda o impacto científico dos 87 programas de pós-graduação da área de **Computação** avaliados pela CAPES (Quadrienal 2025), com redes de **coautoria** e de **citação** construídas a partir da **OpenAlex** (publicações de 2018 a 2024).

Perguntas de pesquisa:

- **P1.** Os grupos de colaboração (comunidades de coautoria) coincidem com as instituições?
- **P2.** Centralidades de rede (PageRank, intermediação…) medem impacto como os indicadores tradicionais (citações, índice *h*)?
- **P3.** Grupos maiores têm mais impacto quando o tamanho é levado em conta?
- **P4.** O impacto medido nas redes acompanha a nota CAPES dos programas?

Relatório final: `relatorio_final/main.pdf` (fonte LaTeX na mesma pasta).

Dashboard com as principais conclusões: https://dinhovco.github.io/MO438_Redes_complexas/ (código em `docs/`, gerado por `analise_final/11_dashboard.py`).

## Redes

| Rede | Arquivo | Vértices | Arestas |
|------|---------|---------:|--------:|
| Coautoria (não dirigida, ponderada) | `rede_coautoria/output/colaboracao_cc_brasil.graphml.gz` | 108 251 autores | 774 165 |
| Citação entre publicações (dirigida, A → B = A cita B) | `rede_coautoria/output/citacoes_cc_brasil.graphml` | 84 676 obras | 101 117 |
| Citação entre autores (dirigida, peso inteiro e fracionário) | `rede_coautoria/output/citacoes_autores_cc_brasil.graphml.gz` | 108 251 autores | 979 561 |
| Citação entre comunidades de coautoria | `rede_coautoria/output/citacoes_comunidades.graphml` | 240 grupos | 5 321 |
| Citação entre instituições (IES) | `rede_coautoria/output/citacoes_ies.graphml` | 72 IES | 4 613 |

Os arquivos `.graphml.gz` são lidos direto pelo NetworkX (`nx.read_graphml`) e pelo Gephi.

## Estrutura

```
rede_coautoria/            pipeline de coleta e construção das redes
  run_pipeline.py          CAPES → IES na OpenAlex → obras (IES + campo CS) → rede de coautoria
  collect_capes.py         docentes e produção dos programas (dados abertos CAPES)
  label_faculty.py         casa docentes CAPES com perfis OpenAlex e mescla perfis duplicados
  collect_faculty_works.py acrescenta ao corpus as obras 2018–2024 dos docentes identificados
  build_rede_citacoes.py   rede de citação entre publicações
  src/                     módulos (cliente OpenAlex, construção das redes)
  data/                    escopo CAPES, mapeamento IES, docentes (o corpus fica fora do git)
  output/                  redes em GraphML
  analise_f1/              métricas da entrega parcial, recalculadas no corpus final
  analise_final/           análises da entrega final (scripts 00–11, results/, figures/, tables/, mapas/)
relatorio_final/           relatório final (LaTeX e PDF)
docs/                      dashboard publicado pelo GitHub Pages
data/                      planilha da Avaliação Quadrienal CAPES 2025
```

## Como reproduzir

Requisitos: Python 3.10+ e as dependências de `rede_coautoria/requirements.txt`.

```bash
cd rede_coautoria
pip install -r requirements.txt
```

### Opção A — usar o corpus já coletado (recomendado)

O corpus (`obras_2026-11.jsonl`, ~480 MB) passa do limite do GitHub. Baixe a versão comprimida (~56 MB) em **[Google Drive](https://drive.google.com/file/d/1WN_Jv52gMtoQ9jd55ewSkwZHD2_1qAzO/view?usp=sharing)**, coloque em `rede_coautoria/data/` e descomprima:

```bash
gunzip -k data/obras_2026-11.jsonl.gz
```

Com o corpus e os arquivos versionados de `data/` (incluindo `data/capes/docentes_openalex.csv`), as redes já estão em `output/`. Para refazer só as análises:

```bash
python analise_f1/calcular_metricas.py        # métricas da F1 (a distância média leva ~1 h)
cd analise_final
python 00_coverage.py                          # completude contra a produção CAPES (consulta a OpenAlex)
python 01_author_citation_network.py
python 02_communities.py
python 03_group_citation_network.py
python 04_group_impact.py
python 05_centrality.py
python 06_institution_vs_capes.py
python 07_structural_properties.py
python 08_report_assets.py                     # figuras e tabelas finais do relatório
python 09_maps.py                              # mapas do Brasil (PNG) e mapa interativo (mapas/mapa_interativo.html)
python 10_institution_network.py --ies UNICAMP # rede de coautoria entre os docentes de uma IES, por comunidade
python 11_dashboard.py                         # dashboard com as conclusões principais (docs/, GitHub Pages)
```

Cada script grava números em `analise_final/results/` e figuras em `analise_final/figures/`; `analise_final/FIGURAS_E_TABELAS.md` diz qual figura e tabela entra em cada seção do relatório.

### Opção B — coletar tudo de novo

Precisa de uma chave gratuita da OpenAlex em `rede_coautoria/.env` (veja `.env.example`). A coleta leva algumas horas e o resultado pode diferir um pouco, porque a OpenAlex é atualizada continuamente.

```bash
python run_pipeline.py --fresh       # escopo CAPES, IES na OpenAlex, obras IES + CS, coautoria
python collect_capes.py              # docentes e produção dos programas (CAPES)
python label_faculty.py              # casa docentes com perfis OpenAlex
python collect_faculty_works.py      # obras dos docentes identificados (união com o corpus)
python label_faculty.py              # escolhe os perfis canônicos no corpus ampliado
python run_pipeline.py               # reconstrói a coautoria com os perfis mesclados
python build_rede_citacoes.py        # rede de citação entre publicações
python label_faculty.py              # rótulos de docente na rede mesclada
```

Depois, seguir a opção A a partir de `analise_f1/`.

## Dados e fontes

- **CAPES:** Avaliação Quadrienal 2025 (área COMPUTAÇÃO, 87 programas em 75 IES) e dados abertos de docentes e produção 2018–2024 (dadosabertos.capes.gov.br).
- **OpenAlex:** obras, autores, instituições, citações e FWCI (openalex.org). Mapeamento IES → OpenAlex: 74 de 75 IES (falta a UNIFACCAMP).
- **Completude:** 87,6 % dos DOIs de artigos em periódico e 91,6 % dos DOIs de trabalhos em anais declarados à CAPES estão no corpus; 91,4 % dos docentes foram identificados na OpenAlex (93,2 % dos permanentes).

Detalhes da construção: `rede_coautoria/relatorio_construcao.md`.
