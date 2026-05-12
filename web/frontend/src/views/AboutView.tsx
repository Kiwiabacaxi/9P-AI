import Card from '../components/shared/Card';

export default function AboutView() {
  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">
            Arquitetura <span>RNA</span>
          </div>
          <div className="page-sub">
            Visao comparativa de todos os algoritmos da disciplina
          </div>
        </div>
      </div>

      {/* Tabela comparativa completa */}
      <Card title="Comparativo Geral" pulse style={{ marginBottom: 16 }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Algoritmo</th>
              <th>Aula</th>
              <th>Pacote</th>
              <th>Arquitetura</th>
              <th>Ativacao</th>
              <th>Regra de atualizacao</th>
              <th>Convergencia</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="td-cyan">Hebb</td>
              <td>02</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>hebb</td>
              <td>{'2 \u2192 1'}</td>
              <td>sign(y_in)</td>
              <td className="td-white">
                {'w \u2190 w + x\u00B7t '}
                <span style={{ color: 'var(--surface-top)' }}>(sempre)</span>
              </td>
              <td className="td-pink">Nao garantida</td>
            </tr>
            <tr>
              <td className="td-cyan">Perceptron Portas</td>
              <td>03</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>perceptronportas</td>
              <td>{'2 \u2192 1'}</td>
              <td>sign(y_in)</td>
              <td className="td-white">
                {'w \u2190 w + \u03B1(t\u2212y)x '}
                <span style={{ color: 'var(--surface-top)' }}>(so no erro)</span>
              </td>
              <td className="td-green">Garantida (sep. linear)</td>
            </tr>
            <tr>
              <td className="td-cyan">Perceptron Letras</td>
              <td>03</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>perceptronletras</td>
              <td>
                {'49 \u2192 1 '}
                <span style={{ color: 'var(--surface-top)' }}>(7x7)</span>
              </td>
              <td>sign(y_in)</td>
              <td className="td-white">
                {'w \u2190 w + \u03B1(t\u2212y)x '}
                <span style={{ color: 'var(--surface-top)' }}>(so no erro)</span>
              </td>
              <td className="td-green">Garantida (sep. linear)</td>
            </tr>
            <tr>
              <td className="td-cyan">MADALINE</td>
              <td>04</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>madaline</td>
              <td>
                {'35 \u2192 13 ADALINE \u2192 13 '}
                <span style={{ color: 'var(--surface-top)' }}>(5x7)</span>
              </td>
              <td>{'sign \u00B7 argmin'}</td>
              <td className="td-white">MRII — atualiza unidade com menor |y_in|</td>
              <td className="td-green">Garantida (sep. linear)</td>
            </tr>
            <tr>
              <td className="td-cyan">MLP Desafio</td>
              <td>05</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>mlp</td>
              <td>{'3 \u2192 2 \u2192 3'}</td>
              <td>tanh</td>
              <td className="td-white">{'Backpropagation — \u03B4\u00B7\u03B1\u00B7a'}</td>
              <td className="td-green">Geralmente sim</td>
            </tr>
            <tr>
              <td className="td-cyan">MLP Letras</td>
              <td>05</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>letras</td>
              <td>
                {'35 \u2192 15 \u2192 26 '}
                <span style={{ color: 'var(--surface-top)' }}>(5x7, A-Z)</span>
              </td>
              <td>tanh</td>
              <td className="td-white">{'Backpropagation — \u03B4\u00B7\u03B1\u00B7a'}</td>
              <td className="td-green">Geralmente sim</td>
            </tr>
            <tr>
              <td className="td-cyan">MLP Image Reg.</td>
              <td>05</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>imgreg</td>
              <td>
                {'2 \u2192 [NxM] \u2192 3 '}
                <span style={{ color: 'var(--surface-top)' }}>(configuravel)</span>
              </td>
              <td>{'ReLU \u00B7 Sigmoid'}</td>
              <td className="td-white">{'SGD estocastico — He init \u00B7 MSE loss'}</td>
              <td className="td-green">Aproximacao universal</td>
            </tr>
            <tr>
              <td className="td-cyan">MLP Funções</td>
              <td>06</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>mlpfunc</td>
              <td>
                {'1 \u2192 [N] \u2192 1 '}
                <span style={{ color: 'var(--surface-top)' }}>(regressão)</span>
              </td>
              <td>{'tanh \u00B7 sigmoid \u00B7 relu'}</td>
              <td className="td-white">{'Backpropagation — \u03B4\u00B7\u03B1\u00B7a'}</td>
              <td className="td-green">Geralmente sim</td>
            </tr>
            <tr>
              <td className="td-cyan">MLP Ortogonal</td>
              <td>06</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>mlport</td>
              <td>
                {'35 \u2192 nHid \u2192 32 '}
                <span style={{ color: 'var(--surface-top)' }}>(5x7, A-Z)</span>
              </td>
              <td>tanh</td>
              <td className="td-white">{'Backprop + dist. euclidiana (sem limiar)'}</td>
              <td className="td-green">Geralmente sim</td>
            </tr>
            <tr>
              <td className="td-cyan">CNN EMNIST</td>
              <td>07</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>cnn</td>
              <td>
                {'28\u00B2 \u2192 Conv \u2192 Pool \u2192 Dense \u2192 26 '}
                <span style={{ color: 'var(--surface-top)' }}>(EMNIST A-Z)</span>
              </td>
              <td>{'ReLU \u00B7 Softmax'}</td>
              <td className="td-white">{'Backprop conv + SGD \u00B7 Cross-Entropy'}</td>
              <td className="td-green">Geralmente sim</td>
            </tr>
            <tr>
              <td className="td-cyan">GA \u00B7 f(x)</td>
              <td>10</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>genetico</td>
              <td>
                {'cromossomo bin. \u2192 x \u2208 [0, 512] '}
                <span style={{ color: 'var(--surface-top)' }}>(otim. f(x))</span>
              </td>
              <td>{'\u2014'}</td>
              <td className="td-white">
                {'Roleta \u00B7 cruz. 1pt \u00B7 mut. bit '}
                <span style={{ color: 'var(--surface-top)' }}>(s/ elite)</span>
              </td>
              <td className="td-pink">Estoc\u00E1stica</td>
            </tr>
            <tr>
              <td className="td-cyan">GA v2 \u00B7 f(x)</td>
              <td>11</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>genetico2</td>
              <td>
                {'cromossomo bin. \u2192 x \u2208 [0, dMax] '}
                <span style={{ color: 'var(--surface-top)' }}>(dom\u00EDnio configur\u00E1vel)</span>
              </td>
              <td>{'\u2014'}</td>
              <td className="td-white">
                {'Torneio/roleta \u00B7 cruz. 1\u20132pt \u00B7 elitismo '}
                <span style={{ color: 'var(--surface-top)' }}>(p melhores)</span>
              </td>
              <td className="td-pink">Estoc\u00E1stica</td>
            </tr>
            <tr>
              <td className="td-cyan">GA \u00B7 Hor\u00E1rio</td>
              <td>12</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>horario</td>
              <td>
                {'matriz [slot \u00D7 turma] '}
                <span style={{ color: 'var(--surface-top)' }}>(cromossomo 2D)</span>
              </td>
              <td>{'\u2014'}</td>
              <td className="td-white">
                {'Torneio \u00B7 troca de linhas \u00B7 flip por c\u00E9lula '}
                <span style={{ color: 'var(--surface-top)' }}>(b\u00F4nus encadeada \u2212 choque)</span>
              </td>
              <td className="td-pink">Estoc\u00E1stica</td>
            </tr>
            <tr>
              <td className="td-cyan">GA \u00B7 TSP</td>
              <td>13</td>
              <td style={{ color: 'var(--on-surface)', fontSize: 10 }}>tsp</td>
              <td>
                {'permuta\u00E7\u00E3o de N cidades '}
                <span style={{ color: 'var(--surface-top)' }}>(OSRM real)</span>
              </td>
              <td>{'\u2014'}</td>
              <td className="td-white">
                {'OX/PMX \u00B7 invers\u00E3o \u00B7 fitness composta '}
                <span style={{ color: 'var(--surface-top)' }}>(\u03BB, \u03C9, \u03B3, \u03BC)</span>
              </td>
              <td className="td-pink">Estoc\u00E1stica</td>
            </tr>
          </tbody>
        </table>
      </Card>

      {/* Cards por grupo de algoritmo */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <Card title="Aula 02 — Hebb">
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--on-surface)',
              lineHeight: 2,
            }}
          >
            <div style={{ color: 'var(--on-surface)', marginBottom: 4 }}>
              Regra de Hebb (1949) — aprendizado hebbiano:
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>
                {'Δw'}
                <sub>i</sub>
              </span>
              {' = x'}
              <sub>i</sub>
              {' \u00B7 t'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>{'Δbias'}</span>
              {' \u00A0= t'}
            </div>
            <div style={{ color: 'var(--surface-top)', marginTop: 6 }}>
              {'// passo unico, sem iteracao'}
            </div>
            <div style={{ color: 'var(--surface-top)' }}>
              {'// nao converge para XOR'}
            </div>
          </div>
        </Card>

        <Card title="Aula 03 — Perceptron">
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--on-surface)',
              lineHeight: 2,
            }}
          >
            <div style={{ color: 'var(--on-surface)', marginBottom: 4 }}>
              Regra do Perceptron — corrige apenas erros:
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>y_in</span>
              {' = bias + \u03A3 x'}
              <sub>i</sub>
              {'\u00B7w'}
              <sub>i</sub>
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>y</span>
              {' \u00A0\u00A0 = sign(y_in)'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>
                {'Δw'}
                <sub>i</sub>
              </span>
              {' = \u03B1\u00B7(t\u2212y)\u00B7x'}
              <sub>i</sub>
            </div>
            <div style={{ color: 'var(--surface-top)', marginTop: 6 }}>
              {'// so atualiza se t \u2260 y'}
            </div>
          </div>
        </Card>

        <Card title="Aula 04 — MADALINE">
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--on-surface)',
              lineHeight: 2,
            }}
          >
            <div style={{ color: 'var(--on-surface)', marginBottom: 4 }}>
              MRII — minimo impacto:
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>
                {'y_in'}
                <sub>j</sub>
              </span>
              {' = bias'}
              <sub>j</sub>
              {' + \u03A3 x'}
              <sub>i</sub>
              {'\u00B7w'}
              <sub>ij</sub>
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>escolhe</span>
              {' j* = argmin |y_in'}
              <sub>j</sub>
              {'|'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>
                {'Δw'}
                <sub>ij*</sub>
              </span>
              {' = \u03B1\u00B7(t\u2212y'}
              <sub>j*</sub>
              {')\u00B7x'}
              <sub>i</sub>
            </div>
            <div style={{ color: 'var(--surface-top)', marginTop: 6 }}>
              {'// 13 unidades ADALINE, saida OR'}
            </div>
          </div>
        </Card>

        <Card title="Aula 05 — MLP Backprop">
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--on-surface)',
              lineHeight: 2,
            }}
          >
            <div>
              <span style={{ color: 'var(--pink)' }}>Forward:</span>
              {' z = w\u00B7a + b \u00A0\u00B7\u00A0 a = tanh(z)'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>
                {'\u03B4 saida:'}
              </span>
              {" (t\u2212y) \u00B7 tanh'(y)"}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>
                {'\u03B4 oculta:'}
              </span>
              {" (\u03A3 \u03B4\u00B7w) \u00B7 tanh'(z)"}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Update:</span>
              {' \u00A0w += \u03B1 \u00B7 \u03B4 \u00B7 a_anterior'}
            </div>
            <div style={{ color: 'var(--surface-top)', marginTop: 4 }}>
              {"// tanh'(x) = 1 \u2212 tanh\u00B2(x)"}
            </div>
          </div>
        </Card>

        <Card title="Aula 05 — Image Regression">
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--on-surface)',
              lineHeight: 2,
            }}
          >
            <div>
              <span style={{ color: 'var(--pink)' }}>Input:</span>
              {' \u00A0(x,y) \u2208 [\u22121,1]\u00B2'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Oculta:</span>
              {' ReLU(w\u00B7a + b)'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Saida:</span>
              {' \u00A0\u03C3(w\u00B7a + b) \u2192 (R,G,B)'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Loss:</span>
              {' \u00A0 0.5\u00B7\u03A3(t\u2212y)\u00B2'}
            </div>
            <div style={{ color: 'var(--surface-top)', marginTop: 4 }}>
              {'// He init \u00B7 SGD por pixel \u00B7 16x16'}
            </div>
          </div>
        </Card>

        <Card title="Aula 06 — MLP Funções">
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--on-surface)',
              lineHeight: 2,
            }}
          >
            <div style={{ color: 'var(--on-surface)', marginBottom: 4 }}>
              Regressão — aproximação de funções:
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Input:</span>
              {' \u00A0x \u2208 [\u22121, 1]'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Target:</span>
              {' f(x) = sin(x)\u00B7sin(2x)'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Saída:</span>
              {' \u00A0y \u2248 f(x) — valor contínuo'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Erro:</span>
              {' \u00A0 E = 0.5\u00B7(t\u2212y)\u00B2'}
            </div>
            <div style={{ color: 'var(--surface-top)', marginTop: 4 }}>
              {'// 50 pontos \u00B7 N camadas configuráveis'}
            </div>
          </div>
        </Card>

        <Card title="Aula 06 — MLP Ortogonal">
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--on-surface)',
              lineHeight: 2,
            }}
          >
            <div style={{ color: 'var(--on-surface)', marginBottom: 4 }}>
              Vetores bipolares ortogonais (Fausett 1994):
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Target:</span>
              {' 32 vetores ortogonais de 32 dims'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Classif:</span>
              {' D = \u221A\u03A3(t_k\u2212y_k)\u00B2'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Regra:</span>
              {' \u00A0MENOR distância euclidiana'}
            </div>
            <div style={{ color: 'var(--surface-top)', marginTop: 4 }}>
              {'// sem limiar \u00B7 tanh puro \u00B7 A-Z'}
            </div>
          </div>
        </Card>

        <Card title="Aula 07 — CNN Convolucional">
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--on-surface)',
              lineHeight: 2,
            }}
          >
            <div style={{ color: 'var(--on-surface)', marginBottom: 4 }}>
              Rede Neural Convolucional (EMNIST Letters):
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Conv:</span>
              {' 8\u00D73\u00D73 \u2192 Pool \u2192 16\u00D73\u00D73 \u2192 Pool'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Dense:</span>
              {' 400 \u2192 64 (ReLU) \u2192 26 (Softmax)'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Loss:</span>
              {' \u00A0Cross-Entropy: \u2212log(p[target])'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Init:</span>
              {' \u00A0He init: \u221A(2/fan_in)'}
            </div>
            <div style={{ color: 'var(--surface-top)', marginTop: 4 }}>
              {'// EMNIST 28\u00D728 \u00B7 26 classes \u00B7 save/load'}
            </div>
          </div>
        </Card>

        <Card title="Aula 10 — GA f(x)">
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--on-surface)',
              lineHeight: 2,
            }}
          >
            <div style={{ color: 'var(--on-surface)', marginBottom: 4 }}>
              Otimização de f(x) = -|x · sin(√|x|)|:
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Cromossomo:</span>
              {' bits binários → x ∈ [0, 512]'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Seleção:</span>
              {'  roleta (proporcional ao fitness)'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Cruz.:</span>
              {'   1 ponto · troca cauda'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Mut.:</span>
              {'    flip por bit (Pm baixo)'}
            </div>
            <div style={{ color: 'var(--surface-top)', marginTop: 4 }}>
              {'// AG canônico do slide aula 10'}
            </div>
          </div>
        </Card>

        <Card title="Aula 11 — GA v2 (parametrizável)">
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--on-surface)',
              lineHeight: 2,
            }}
          >
            <div style={{ color: 'var(--on-surface)', marginBottom: 4 }}>
              Mesma f(x), mas com upgrades pedidos no slide:
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Seleção:</span>
              {'  torneio (k) ou roleta'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Cruz.:</span>
              {'   1 ou 2 pontos'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Elitismo:</span>
              {' p melhores intactos'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Domínio:</span>
              {' [0, dMax] configurável'}
            </div>
            <div style={{ color: 'var(--surface-top)', marginTop: 4 }}>
              {'// + métrica de diversidade'}
            </div>
          </div>
        </Card>

        <Card title="Aula 12 — GA Horário (matriz)">
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--on-surface)',
              lineHeight: 2,
            }}
          >
            <div style={{ color: 'var(--on-surface)', marginBottom: 4 }}>
              Cromossomo 2D — outro tipo de codificação:
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Cromossomo:</span>
              {' matriz [slot × turma] = id prof'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Cruz.:</span>
              {'   troca de LINHAS entre pais'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Mut.:</span>
              {'    flip por célula (prof aleatório)'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Fitness:</span>
              {' +encad. − choques − faltas'}
            </div>
            <div style={{ color: 'var(--surface-top)', marginTop: 4 }}>
              {'// 29 profs · 3 turmas · 10 slots (slide)'}
            </div>
          </div>
        </Card>

        <Card title="Aula 13 — GA TSP">
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--on-surface)',
              lineHeight: 2,
            }}
          >
            <div style={{ color: 'var(--on-surface)', marginBottom: 4 }}>
              Caixeiro Viajante — encoding novo:
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Cromossomo:</span>
              {' permutação de N cidades'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Cruz.:</span>
              {'   OX · PMX (válido p/ permut.)'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Mut.:</span>
              {'    inversão (≈ 2-opt)'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>Custo:</span>
              {'   d + λ·max + ω·ord + γ·T + μ·OT²'}
            </div>
            <div style={{ color: 'var(--surface-top)', marginTop: 4 }}>
              {'// 5 presets reais · OSRM real-road'}
            </div>
          </div>
        </Card>

        <Card title="Estrutura de Pacotes">
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--on-surface)',
              lineHeight: 1.9,
            }}
          >
            <div style={{ color: 'var(--cyan)' }}>web/server/</div>
            <div>
              {'\u00A0\u251C\u2500 '}
              <span style={{ color: 'var(--primary-glow)' }}>main.go</span>
              {'      \u2190 HTTP + SSE'}
            </div>
            <div>{'\u00A0\u251C\u2500 hebb/'}</div>
            <div>{'\u00A0\u251C\u2500 perceptron_portas/'}</div>
            <div>{'\u00A0\u251C\u2500 perceptron_letras/'}</div>
            <div>{'\u00A0\u251C\u2500 madaline/'}</div>
            <div>{'\u00A0\u251C\u2500 mlp/'}</div>
            <div>{'\u00A0\u251C\u2500 letras/'}</div>
            <div>{'\u00A0\u251C\u2500 mlpfunc/'}</div>
            <div>{'\u00A0\u251C\u2500 mlport/'}</div>
            <div>{'\u00A0\u251C\u2500 imgreg/'}</div>
            <div>{'\u00A0\u251C\u2500 cnn/'}</div>
            <div>{'\u00A0\u251C\u2500 genetico/'}</div>
            <div>{'\u00A0\u251C\u2500 genetico2/'}</div>
            <div>{'\u00A0\u251C\u2500 horario/'}</div>
            <div>{'\u00A0\u2514\u2500 tsp/'}</div>
          </div>
        </Card>
      </div>
    </div>
  );
}
