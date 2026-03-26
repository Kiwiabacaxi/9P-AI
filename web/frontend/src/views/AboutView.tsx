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
            <div>{'\u00A0\u2514\u2500 imgreg/'}</div>
          </div>
        </Card>
      </div>
    </div>
  );
}
