# Brainstorm — Evolução do TSP soja → matching marketplace

> **Status (atualizado 2026-05-08):** Etapas 1 e 2 entregues na branch `feat/matching-marketplace`.
>
> - **Etapa 1** (plano em [docs/superpowers/plans/2026-05-06-matching-marketplace-v1.md](superpowers/plans/2026-05-06-matching-marketplace-v1.md)) — single-objective GA + greedy baseline + cenários "Balanceado" e "Crise de Qualidade" (6×4) + visualização Leaflet animada com tema dark.
> - **Etapa 2** — cenários "Comprador Dominante" e "Preço Alto", baseline Hungarian (per-lot argmax sem capacidade) com card pink + warning de violação, modo 60×6 (60 produtores procedurais MT-GO + 6 traders reais incluindo LDC e Amaggi), toggle de escala na UI.
>
> Próximas etapas (NSGA-II multi-objetivo, calibração com dados reais Comex/CONAB/CEPEA, OSRM real para rotas matching, IOSCO, modal switching, robustez estocástica) ficam para etapas posteriores.

> Transferência de contexto da conversa entre Kiwi e Claude (claude.ai web) para o ambiente Claude Code. Lê isso pra entender de onde estamos saindo, pra onde vamos, e por quê. Não prescreve stack — adapta às convenções do projeto.

---

## TL;DR

O projeto atual é um **TSP genético** rodando na rota Rondonópolis → Santos com 11 cidades, OSRM pra distâncias reais, animação por geração e fitness com penalidades hard. Funciona, fica visualmente legal, mas é **conceitualmente fraco**: TSP não modela o problema real do escoamento de soja. Caminhão de soja não faz tour fechado visitando todas as cidades — ele faz coleta-e-entrega, normalmente com vários caminhões em paralelo, todos convergindo no porto.

A direção nova é tratar o problema como **matching marketplace multi-agente**: N produtores ofertando lotes heterogêneos, M traders comprando pra encher navios em Santos, GA decidindo quem compra de quem sob restrições não-lineares (capacidade, blend de qualidade, logística, janelas). É um problema de bilhões de reais em fricção real, e GA é a ferramenta certa porque LP/Hungarian quebram nas não-linearidades.

---

## O que já existe no projeto

- TSP genético com 11 cidades fixas (Rondonópolis MT → ... → Porto de Santos → volta)
- Distâncias via OSRM (estradas reais)
- Operadores: OX (Order Crossover), 2-opt mutação, torneio k=4
- População 80, 300 gerações, Pc=0.85, Pm=0.15, elitismo p=2
- Fitness com penalidade quadrática pra exceder jornada do motorista (10h ANTT)
- Visual: mapa Leaflet com animação por geração, gráfico de evolução, painel de stats
- Resultado típico: ~5077 km, 75h, leg máximo ~1391 km

**O que tá bom**: a infra de visualização e animação. **O que tá ruim**: o problema modelado.

---

## Por que TSP é a classe errada

1. Caminhão graneleiro de soja **não faz tour visitando todos os silos**. Ele pega num silo, descarrega no porto, volta (normalmente com fertilizante de back-haul).
2. Forçar "Porto de Santos por último" via penalidade é gambiarra pra empurrar TSP a se comportar como pickup-and-delivery.
3. O tour de 5077 km com volta a Rondonópolis pressupõe caminhão voltando vazio — coisa que ninguém faz.
4. Não captura o que importa de verdade no agro: **multi-agente** (vários traders concorrendo), **multi-restrição** (qualidade, capacidade, janela), **multi-objetivo** (custo, CO₂, inclusão).

---

## Direção nova: cenário matching marketplace

### Setup do problema

- N produtores no cinturão MT-GO (caso demo: 6, caso real: 40-60)
- Cada produtor com 1+ lote: `(volume_t, proteína%, umidade%, impurezas%, município, preço_reserva, janela_entrega)`
- M traders compradores (caso demo: 4, caso real: ~6: Cargill, Bunge, ADM, COFCO, LDC, Amaggi)
- Cada trader com: `(programa_embarque_t, spec_mínima_qualidade, janela_navio_porto, preço_máximo, hub_geográfico)`
- Destino comum: Porto de Santos

### Cromossomo

Vetor de tamanho N. `gene[i] ∈ {0..M-1, -1}` indica qual trader comprou o lote i, ou -1 (não casou → vai pro spot CEPEA).

**Vantagens**: simples, permite matching parcial (importante: forçar casar tudo subotimiza), mutação trivial, crossover uniforme válido com repair.

### Função fitness (single-objective primeiro, depois multi)

```
fitness(c) = superávit_total(c)
           - λ_log × custo_logístico(c)
           - λ_qual × penalidade_blend(c)
           - M_BIG × violações_hard(c)

superávit_total = Σ_matched [preço_pago(i,j) - preço_reserva_i] × sacas_lote_i

preço_pago(i,j) = preço_base_dia
                + prêmio_qualidade(lote_i)        # +R$/saca por ponto de proteína acima de 36
                - desconto_logístico(dist_ij)     # quem tá longe do hub do trader recebe menos
```

### Restrições hard (penalidade quadrática no excesso)

1. `Σ_{i: gene[i]=j} volume_i ≤ capacidade_j` (capacidade do trader)
2. Blend ponderado de qualidade do trader j ≥ spec mínima (proteína média ponderada por volume)
3. Janela compatível (lote disponível na semana do navio)

### Operadores

- **Crossover**: uniform crossover + repair. Se o filho viola capacidade, remove iterativamente os matches menos rentáveis até voltar a caber.
- **Mutação**: 50% swap (troca destinatário de 2 lotes), 30% reassign aleatório, 20% force unmatch (-1).
- **Local search opcional pós-crossover**: 2-opt nos matchings (testa trocar destinatário de pares vizinhos no cromossomo, aceita se melhora).

---

## Baselines obrigatórias para comparar

Sem baselines, não dá pra dizer se o GA é bom. Implementa pelo menos 2:

1. **Greedy por preço**: ordena lotes por preço_reserva descendente, atribui ao trader que paga mais e ainda tem capacidade
2. **Hungarian (LP exato)**: assignment puro, ignora não-linearidades de blend e desconto por volume — tende a violar capacidade/spec, bom comparativo
3. **Random matching**: limite inferior, sanity check

Métrica esperada: GA deve dominar em 15-30% de superávit nos cenários com não-linearidades fortes.

---

## Cenários de teste para mostrar variância

Cada um destaca um aspecto diferente do GA:

1. **Balanceado**: traders com programação simétrica, lotes razoavelmente uniformes — caso base, pra mostrar que o GA *converge*.
2. **Comprador dominante**: 1 trader com 60% da demanda, restantes pequenos — clássico do agro brasileiro. Mostra como o GA distribui anti-monopolização (especialmente com obj de Herfindahl-Hirschman ativo).
3. **Crise de qualidade**: chuva ruim → 40% dos lotes com proteína < 35. Força o GA a fazer blends inteligentes (combinar lote ruim com lote excelente pra puxar a média acima da spec).
4. **Preço alto**: papel Santos disparou (`maxPrice` dos traders sobe), produtores menos dispostos a vender barato. Mostra dinâmica de mercado apertado.

---

## Dados externos (calibração realista, nem tudo precisa ser real)

Para defender o trabalho academicamente, calibra dados sintéticos com fontes públicas:

| Fonte | URL/endpoint | Uso |
|-------|--------------|-----|
| **Comex Stat** | `comexstat.mdic.gov.br/api/general` (POST JSON) | Volume mensal de soja (NCM 12019000) por país_destino e URF (BRSSZ Santos, BRPNG Paranaguá, BRRIG Rio Grande). Calibra demanda agregada. |
| **CONAB** | conab.gov.br/info-agro/safras | Produção municipal de soja por safra. Calibra distribuição geográfica de produtores. |
| **CEPEA** | cepea.esalq.usp.br/br/indicador/soja.aspx | Indicador diário Santos FOB e Paranaguá FOB. Sem API, parsing leve. Calibra preço base do dia. |
| **BCB** | `api.bcb.gov.br/dados/serie/bcdata.sgs.{COD}/dados` | PTAX USD/BRL diário (cód 1) e Selic (cód 432). |
| **B3** | b3.com.br | Futuros SFI (soja FOB Santos). Volume/ajuste diários, raspável. |
| **Antaq** | antaq.gov.br/anuario | Atracações em Santos, fila, capacidade efetiva. CSV download. |
| **OSRM** | router.project-osrm.org (já em uso) | Distâncias rodoviárias reais entre municípios. |
| **OpenWeather** | openweathermap.org/api (free tier) | Chuva nas próximas 48h em cada nó produtor. Vira penalidade na BR-163. |

**Limitação importante do Comex**: não revela comprador final (sigilo fiscal). Te dá país_destino, não trader. Pra simulação acadêmica isso é OK — você simula a atribuição trader↔país sinteticamente e calibra os totais com Comex real. Documenta a metodologia com clareza no relatório.

---

## Visualização-alvo

A visualização é o que **vende o trabalho**. O modelo mental:

- **Mapa Leaflet** (mesmo já em uso) com 6+ produtores plotados em municípios reais do MT (lat/lng do CONAB)
- **Trader hubs** como marcadores diferenciados (forma quadrada, ou ícone customizado):
  - Cargill em Rondonópolis (-16.47, -54.64)
  - Bunge em Cuiabá (-15.60, -56.10) — ou outro hub real, ajustar
  - ADM em Alto Araguaia (-17.31, -53.21)
  - COFCO em Rio Verde-GO (-17.79, -50.93)
- **Porto de Santos** (-23.96, -46.33) como destino destacado
- **Múltiplas rotas coloridas** — uma cor por trader. Cada produtor envia pra um trader (linha fina, cor do trader). Cada trader consolidado envia pro porto (linha grossa, mesma cor).
- **Linha tracejada vermelha** sobre rota trader→porto = trader em overload (violação de capacidade)
- **Painel lateral** mostrando, por trader: nome, volume carregado/capacidade (barra), nº lotes adquiridos, blend de qualidade resultante, status (OK / over-capacity / under-spec)
- **Animação por geração** (igual ao TSP atual): a cada frame o GA evolui, as cores dos produtores mudam, linhas vermelhas viram coloridas. Visualmente impressionante.

Demo conceitual da conversa (versão SVG estilizada) pode ser referência visual; replica o conceito com Leaflet real e fica imbatível.

---

## Extensões opcionais (em ordem de impacto pedagógico)

### 1. NSGA-II multi-objetivo (alto impacto)

Em vez de soma ponderada na fitness, otimiza **múltiplos objetivos simultaneamente** e produz fronteira de Pareto:

- Obj 1: maximizar superávit (R$)
- Obj 2: maximizar nº produtores casados (inclusão — evita só pegar grandes)
- Obj 3: minimizar índice Herfindahl-Hirschman da concentração por trader (anti-monopolização)
- Obj 4: minimizar CO₂eq (se incluir escolha modal)

**Saída**: fronteira 4D, plot em parallel coordinates ou pares 2D. Apresenta 3-4 pontos da fronteira em vez de "a solução". Vira conversa de engenharia em vez de algoritmo.

Biblioteca pronta: `pymoo` (Python). Tem NSGA-II, NSGA-III, MOEA/D, indicadores de qualidade (hypervolume), tudo. Se o projeto for JS-only, dá pra portar a lógica do NSGA-II direto (não é complexo: non-dominated sorting + crowding distance).

### 2. Cenário composição IOSCO do índice Santos FOB

Outra direção independente, mas complementar e fortíssima academicamente. Em vez de matching, o GA decide **quais transações observadas entram na amostra** que computa o índice de preço diário de soja FOB Santos — problema real da Grainsights/Grão Direto.

Cromossomo binário: `gene[t] ∈ {0,1}` (inclui transação t na amostra). Fitness multi-objetivo (NSGA-II): robustez estatística (jackknife sobre amostra), representatividade (entropia geográfica + de contraparte), aderência a fundamentos (CBOT × câmbio + prêmio FOB).

Restrições hard inspiradas em IOSCO:
- ≥3 vendedores independentes
- ≥3 compradores independentes
- Nenhuma contraparte > 40% do volume amostrado
- Volume amostrado ≥ 60% do volume diário total
- Sem operações intragrupo

Pode rodar em paralelo ao matching como "demonstração 2" do trabalho.

### 3. Modal switching (cromossomo híbrido)

Inclui Alto Araguaia/Alto Taquari (terminais Rumo Malha Norte) como nós de transbordo possíveis. Cromossomo passa a ser `(matching, modal_por_aresta)` onde modal ∈ {rodo, ferro, hidro}. GA decide onde compensa transbordar baseado em custo (rodo ~R$ 0,15-0,20/t·km, ferro ~R$ 0,08/t·km, transbordo ~R$ 12-18/t fixo) vs flexibilidade vs CO₂.

### 4. Robustez estocástica

Cada aresta com `(μ_tempo, σ_tempo)`. Roda Monte Carlo dentro do cálculo de fitness (50 simulações por solução), otimiza `μ + k·σ`. Mostra rotas robustas vs frágeis. Bônus: dá probabilidade de cumprir janela do navio.

---

## Decisões em aberto / discutir antes de codar

1. **Escopo do trabalho**: ficar só no matching ou incluir índice IOSCO? Sugestão: começa matching, depois adiciona índice se sobrar tempo.
2. **NSGA-II ou single-objective**: NSGA-II é mais impressionante mas adiciona complexidade. Sugestão: implementa single-objective primeiro pra ter algo rodando, depois evolui pra NSGA-II.
3. **Quantos produtores e traders no demo**: 6×4 (caso de aula, visual claro) vs 60×6 (caso real, GA tem mais o que fazer). Idealmente: tem ambos os modos no UI (slider ou preset).
4. **Manter ou descartar o TSP atual**: pode virar uma "feature comparativa" ("veja como TSP modela o mesmo problema mal, e como matching modela melhor"), ou aposentar e focar no novo. Sugestão: manter como toggle pra mostrar evolução conceitual.
5. **Front: Leaflet + JS puro vs componentizar**: o projeto atual já tem Leaflet rodando. Ver se vale refatorar ou adicionar em cima.

---

## Mini pitch (pra abrir apresentação ou pra README)

> Mercado de soja brasileiro como problema de otimização combinatória multi-agente. Aplicação de algoritmo genético ao matching real entre produtores do MT e traders exportadores em Santos: ~60 produtores ofertando lotes heterogêneos, ~6 traders globais comprando pra encher navios com specs rígidas, sob restrições não-lineares de capacidade, blend de qualidade e janelas de embarque que LP/Hungarian não resolvem bem. Calibrado com Comex Stat, CONAB e CEPEA. Visualização final: mapa do MT com rotas multi-trader convergindo em Santos. Resultado esperado: GA domina baselines clássicas em 15-30% de superávit, com restrições regulatórias e operacionais respeitadas.

---

## Por onde começar quando ler isso

Sugestão de ordem de ataque (não obrigatória, ajustar ao contexto do projeto):

1. **Não começar codando.** Ler esse documento, abrir o código atual do TSP, mapear o que reaproveita (renderização Leaflet, animação por geração, gráfico de evolução, OSRM client) e o que precisa de novo (modelo de domínio, cromossomo, fitness, repair operator).
2. **Validar arquitetura com Kiwi antes de mexer**: confirmar se vai ser feature paralela ou substituição, qual escopo (só matching ou inclui IOSCO), single-objective ou NSGA-II. Ver decisões em aberto acima.
3. **Implementar o domínio puro primeiro** (`Producer`, `Trader`, `Lot`, `Match`) sem GA — só estruturas de dados + cálculo de fitness puro. Testar com cromossomo hardcoded.
4. **Adicionar GA por cima**: população, operadores, loop evolutivo. Reutilizar abstrações do TSP se houver.
5. **Ligar na visualização**: adaptar/clonar o componente Leaflet pra renderizar matching multi-rota em vez de TSP single-route.
6. **Baselines depois**: implementa greedy e Hungarian (pode ser numa lib JS tipo `munkres-js`) pra comparar números.
7. **Cenários de teste e dados Comex/CONAB**: vem por último, depois que o pipeline tá estável.

---

## Histórico da conversa (resumo das iterações)

1. Começou com pergunta sobre "papel Paranaguá" → contexto de jargão do mercado físico de soja (FOB Paranaguá como referência de preço, terminologia "papel" = contrato/posição)
2. Discussão sobre função de fitness pra TSP rota Rondonópolis-Santos → identificou que TSP é modelagem fraca, sugeriu margem (não custo) como função objetivo, importância de robustez (variância) e multi-objetivo
3. Kiwi mostrou print do TSP rodando → propostas de cenários A-E (VRP, NSGA-II, multi-modal, dados reais, robustez estocástica)
4. Kiwi pediu coisas mais relacionadas a Grão Direto/Grainsights → cenários de matching, índice IOSCO, back-haul, alocação de armazenagem, blending pra contrato, amostragem
5. Kiwi escolheu aprofundar matching e índice IOSCO → spec técnica completa de cromossomo, fitness, operadores, baselines, cenários de teste, dados externos
6. Demonstração interativa do matching (cards com cromossomo + GA rodando)
7. Kiwi pediu visualização "produto final" → demo no formato mapa multi-trader convergindo em Santos
8. Mini pitch produzido
9. Esse documento

---

## Referências consultadas durante a conversa

- Comex Stat: estatísticas de exportação de soja por porto e país de destino (NCM 12019000)
- Distribuição modal de exportação de soja: Paranaguá ~37%, Santos ~21% em jan/2024
- Frete rodoviário Ponta Porã→Paranaguá pode ser ~27% mais barato que Ponta Porã→Santos (caso citado mostrando arbitragem entre portos)
- Custos típicos: rodo ~R$ 0,15-0,20/t·km, ferro ~R$ 0,08/t·km, transbordo ~R$ 12-18/t
- Spec ABIOVE de soja: proteína ≥ 36%, umidade ≤ 14%, impurezas ≤ 1%
- Capacidade Panamax típico: ~60kt; demurrage Panamax: US$ 15-30k/dia parado
