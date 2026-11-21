# Figuras e tabelas do relatório

Gerado por `08_report_assets.py`. Cada item indica a seção do relatório e o script que o produz.

## Figuras

| # | Arquivo | Seção | Conteúdo | Script |
|---|---------|-------|----------|--------|
| 1 | `figures/report_network_diagram.png` | 5 | Redes construídas e como derivam umas das outras | 08 |
| 2 | `figures/structural_powerlaw.png` | 4 / 6.3 | Caudas das distribuições de grau e citações, com ajuste de lei de potência e lognormal | 07 |
| 3 | `figures/communities_size_ccdf.png` | 6.2 | Distribuição de tamanhos das comunidades por algoritmo | 02 |
| 4 | `figures/communities_gn_modularity.png` | 6.2 | Curva $Q$ × nº de comunidades do Girvan–Newman (protocolo AP3) | 02 |
| 5 | `figures/report_community_map.png` | 6.2 | Mapa das 30 maiores comunidades (cor = FWCI mediano) | 08 |
| 6 | `figures/community_citation_heatmap.png` | 6.1 / 6.4 | Quem cita quem entre as 20 maiores comunidades | 03 |
| 7 | `figures/group_impact_size.png` | 6.4 | Tamanho da comunidade × impacto absoluto e normalizado | 04 |
| 8 | `figures/group_impact_ranking.png` | 6.4 | Ranking absoluto × normalizado das comunidades | 04 |
| 9 | `figures/report_pagerank_citations.png` | 6.3 | PageRank de citação × citações totais dos autores | 08 |
| 10 | `figures/capes_program_impact.png` | 6.5 | Impacto dos programas × nota CAPES | 06 |
| 11 | `figures/structural_granovetter.png` | 6.2 | Granovetter: overlap × peso, pontes entre comunidades, remoção de vínculos | 07 |
| 12 | `figures/centrality_spearman_coauthorship.png` | 6.3 | Spearman entre as seis centralidades da AP3 | 05 |

Mapas (gerados por `09_maps.py`, ainda fora do relatório e dos slides): `map_ies_coauthorship.png`, `map_ies_citation.png`, `map_communities.png` e a versão interativa `mapas/mapa_interativo.html`.

Figuras de apoio (anexo ou apresentação): `author_citation_degree_ccdf.png`, `ies_citation_heatmap.png`, `capes_ies_impact.png`, `centrality_spearman_validation.png`, `structural_snapshots.png`, e as figuras da F1 em `analise_f1/figuras/`.

## Tabelas (`tables/`, LaTeX com `\hline`, sem pacotes extras)

| Arquivo | Seção | Conteúdo |
|---------|-------|----------|
| `tab_metricas_redes.tex` | 4 | Métricas topológicas das redes de coautoria e de citação |
| `tab_algoritmos.tex` | 6.2 | Algoritmos de comunidades nas duas escalas, com NMI/ARI contra a IES |
| `tab_propriedades.tex` | 4 / 6 | Mundo pequeno, lei de potência, assortatividade, modelos nulos, Granovetter |
| `tab_impacto_ranking.tex` | 6.4 | Top 10 comunidades: absoluto × normalizado |
| `tab_impacto_tamanho.tex` | 6.4 | Spearman tamanho × métricas de impacto |
| `tab_centralidade.tex` | 6.3 | Centralidades × citações e índice $h$ |
| `tab_top_publicacoes.tex` | 6.3 | Top 10 publicações: impacto interno × global |
| `tab_capes.tex` | 6.5 | Spearman nota CAPES × impacto (programas e IES) |

Todos os números vêm de `results/`; para regenerar: rodar os scripts 01–07 e depois este.
