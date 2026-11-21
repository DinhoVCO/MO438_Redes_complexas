# Relatório de construção das redes

Pipeline: `rede_coautoria/`. Atualizado em **2026-11-20** (coleta de novembro, versão da entrega final). A versão da entrega parcial (coleta de 2026-10) está no histórico do git.

---

## 1. Objetivo

Construir redes que permitam medir o impacto científico dos programas de pós-graduação em **Computação** no Brasil:

- rede **não dirigida de coautoria** entre autores;
- rede **dirigida de citação** entre publicações, depois agregada por autor, comunidade e instituição (pasta `analise_final/`).

---

## 2. Escopo (CAPES)

| Item | Valor |
|------|------:|
| Fonte | `data/27052026_Resultado_...xlsx` na raiz (Avaliação Quadrienal 2025, aba GERAL) |
| Filtro | Área de avaliação **COMPUTAÇÃO** |
| Programas | **87** |
| IES distintas | **75** |
| Saída | `data/programas_cc_brasil.csv` (com nível e `nota_final`) |

Script: `src/capes.py`, etapa 1 de `run_pipeline.py`.

Dados abertos da CAPES (`collect_capes.py`, API CKAN de dadosabertos.capes.gov.br), filtrados pelos 87 programas e pelos anos 2018–2024:

| Arquivo | Conteúdo | Registros |
|---------|----------|----------:|
| `data/capes/docentes_cc.csv` | Docentes por programa e ano (ID_PESSOA, categoria) | 14 364 linhas, 2 491 docentes |
| `data/capes/producao_cc.csv` | Artigos em periódico e trabalhos em anais, com DOI extraído | 59 831 |

---

## 3. Publicações (OpenAlex)

### 3.1 Mapeamento institucional

Cada IES do escopo foi buscada em `GET /institutions` (`country_code:br`). Seis casos foram corrigidos à mão (`match_status=manual`, com a justificativa na coluna `observacao`):

| IES | Correção |
|-----|----------|
| FURG | O match automático apontava para a UFRGS |
| USP-São Carlos | Apontava para o Instituto de Química de São Carlos; o ICMC não existe como entidade na OpenAlex, então usa-se a USP |
| USP-Ribeirão Preto | Campus sem entidade própria; usa-se a USP |
| UNESP-São José do Rio Preto | UNESP inteira; o IBILCE é subunidade e entra pelo filtro de *lineage* |
| UFPB-João Pessoa | Campus sem entidade própria; usa-se a UFPB |
| FCE (CESAR) | Registrada na OpenAlex como CESAR |

Resultado: **74 de 75 IES** mapeadas (72 IDs distintos, porque os campi da USP compartilham o ID). Falta só a UNIFACCAMP, que não está na OpenAlex.

### 3.2 Coleta base (IES + Computer Science)

`run_pipeline.py` consulta `GET /works` em lotes de 15 instituições, com:

- `authorships.institutions.lineage:I1|I2|…`: o *lineage* inclui subunidades (antes usava-se `.id`, que perdia obras atribuídas a institutos e faculdades);
- `primary_topic.field.id:17` (Computer Science);
- `publication_year:2018-2024`.

A paginação usa `cursor`, e cada obra é guardada uma vez (deduplicação pelo ID OpenAlex). O cliente usa `api_key` (em `.env`, fora do git) e repete a requisição em erros de conexão; se algum lote ficar incompleto, o script sai com código 2 e não reconstrói a rede.

**Auditoria do filtro de campo.** Numa amostra de autores do escopo, 43,4 % da produção estava em CS. Contagens alternativas: 53 423 obras com o filtro atual, ~91 mil somando subáreas vizinhas, e 93 272 com "qualquer um dos três tópicos em CS". Decisão: manter o filtro de campo na coleta base e recuperar a produção dos docentes fora de CS pela coleta por pessoa (§3.3), para não trazer departamentos inteiros de outras áreas.

### 3.3 Docentes CAPES e coleta por pessoa

1. `label_faculty.py` casa cada docente CAPES com perfis OpenAlex que publicaram numa IES do seu programa.
   - **Nível estrito:** mesmo primeiro nome, mesmo último sobrenome e nomes do meio compatíveis.
   - **Nível flexível:** inicial do primeiro nome, e o último sobrenome OpenAlex é algum dos sobrenomes do docente.
   - Um perfil reivindicado por dois docentes é descartado como ambíguo.
2. `collect_faculty_works.py` busca **todas** as obras 2018–2024 dos perfis identificados, de qualquer campo, e as acrescenta ao corpus com `coleta_origem="docentes"`.

### 3.4 Corpus final

| Item | Coleta base (IES + CS) | União (base + docentes) |
|------|-----------------------:|------------------------:|
| Obras | 53 419 | **84 676** |
| DOIs de artigos CAPES no corpus | 43,1 % | **87,6 %** |
| DOIs de trabalhos em anais CAPES no corpus | 58,0 % | **91,6 %** |
| Docentes identificados na OpenAlex | 87,1 % | **91,4 %** (permanentes: 93,2 %) |

Arquivo: `data/obras_2026-11.jsonl` (~480 MB, fora do git; versão `.gz` de ~56 MB no link do README da raiz). Medição: `analise_final/00_coverage.py` → `analise_final/results/coverage.json` (`coverage_base.json` guarda a medição antes da união).

Motivos dos DOIs de artigos que faltam (1 640): tópico principal fora de CS (935), ano fora de 2018–2024 na OpenAlex (340), DOI ausente da OpenAlex (332), sem afiliação no escopo (31).

---

## 4. Regras de construção

### 4.1 Coautoria (`src/build_network.py`)

Para cada obra do corpus:

1. Extrair os autores (OpenAlex Author ID). Perfis duplicados de um mesmo docente são substituídos pelo **perfil canônico** (o de mais obras) **antes** de criar as arestas, então os pesos se somam e não aparecem laços.
2. Para cada par de autores distintos, incrementar a aresta `(A, B)`; `weight` = nº de obras em conjunto.

Vértices: todos os autores das obras coletadas, inclusive coautores de fora do Brasil.

| Atributo de nó | Significado |
|----------------|-------------|
| `label` | Nome do autor |
| `n_publicacoes_corpus` | Obras do corpus em que o autor aparece |
| `afiliacao_escopo` | Tem alguma authorship numa IES do escopo |
| `ies_principal`, `ies_principal_id` | IES do escopo mais frequente nas suas authorships |
| `ies_escopo_ids` | Todas as IES do escopo em que aparece |
| `openalex_ids_mesclados` | Perfis OpenAlex reunidos neste vértice |
| `docente_capes`, `programa_capes`, `id_pessoa_capes` | Rótulos de docente CAPES (`label_faculty.py`) |
| `comunidade` | Comunidade Louvain (`analise_final/02_communities.py`) |

**Mescla de perfis:** 1 030 perfis OpenAlex foram absorvidos em 688 docentes (ex.: uma docente tinha 9 perfis). Resultado: 2 276 vértices de docente para 2 276 docentes identificados.

### 4.2 Citação entre publicações (`src/build_citation_network.py`)

- Aresta `A → B` se a obra A cita a obra B e as duas estão no corpus (`referenced_works` da própria obra).
- Atributos: `publication_year`, `cited_by_count` (global, OpenAlex), `fwci` (−1 = ausente; presente em 85 % das obras), `autor_ids` (perfis canônicos), `origem` (`ies_cs` ou `docentes`).
- Referências para fora do corpus: 1 949 955 menções. É uma limitação de projeto: só as citações internas formam arestas.

---

## 5. Resultado atual

| Rede | Vértices | Arestas | Arquivo |
|------|---------:|--------:|---------|
| Coautoria | 108 251 | 774 165 | `output/colaboracao_cc_brasil.graphml.gz` |
| Citação entre publicações | 84 676 | 101 117 | `output/citacoes_cc_brasil.graphml` |

- Autores com afiliação no escopo: **62 905**.
- Obras sem par de coautoria (autor único): 10 592.
- Contagens: `output/estatisticas_rede.json` e `output/estatisticas_citacoes.json`.
- Métricas topológicas (F1, recalculadas no corpus final): `analise_f1/resultados_f1.json`; as da entrega parcial ficam em `analise_f1/resultados_f1_entrega_parcial.json`.

As redes agregadas (citação entre autores, comunidades e IES) são geradas em `analise_final/`; veja o `README.md` da raiz.

### Por que os números mudaram desde a entrega parcial

| Rede | Parcial (2026-10) | Final (2026-11) |
|------|------------------:|----------------:|
| Coautoria: vértices / arestas | 79 458 / 402 568 | 108 251 / 774 165 |
| Citação: vértices / arestas | 57 300 / 42 143 | 84 676 / 101 117 |

1. Correção do mapeamento IES → OpenAlex (§3.1) e filtro por *lineage*.
2. Nova coleta completa, com `select` que já traz `referenced_works` e `fwci`. Isso dispensou o antigo `collect_referencias.py`.
3. União com as obras dos docentes CAPES (§3.3).
4. Mescla dos perfis duplicados de docentes (§4.1).

---

## 6. Como reproduzir

Ver `README.md` (ordem dos scripts) e o `README.md` da raiz (corpus para download e análises).

---

## 7. Limitações conhecidas

1. **Proxy institucional na coleta base:** obras de autores da IES em CS que não são do programa entram; obras de docentes classificadas em outros campos só entram pela coleta por pessoa (que cobre os docentes identificados, não alunos).
2. **Desambiguação de autores:** a OpenAlex às vezes divide uma pessoa em vários perfis. Só os perfis de docentes CAPES foram mesclados; alunos e coautores externos podem ter duplicatas.
3. **Casamento por nome:** pode juntar homônimos da mesma IES. O risco é limitado pela exigência de nome e IES e pelo descarte de perfis ambíguos.
4. **UNIFACCAMP** não está na OpenAlex; 3 programas recentes (UERN, UNIOESTE e um da UFMA) não têm docentes na série 2018–2024.
5. **Citações:** só as internas ao corpus viram arestas; `cited_by_count` é global e muda com o tempo (coleta de novembro de 2026).
6. **BrCris** não foi usado (sem API pública para coleta em massa).
