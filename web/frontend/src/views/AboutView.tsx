import Card from '../components/shared/Card';

// =============================================================================
// Guia de navegação — onde encontrar cada trabalho da disciplina.
// (Cada linha diz em qual item do menu à esquerda clicar e qual pacote Go / tela
//  implementa aquele trabalho.)
// =============================================================================

interface Trab {
  n: number;
  nome: string;
  menu: string;   // rótulo exato do item na barra lateral
  pkg: string;    // pacote Go (web/server/) e/ou tela
  aula: string;
  pendente?: boolean;
}

const TRABALHOS: Trab[] = [
  { n: 1, nome: 'Regra de Hebb', menu: 'Hebb', pkg: 'hebb', aula: '02' },
  { n: 2, nome: 'Perceptron (portas + letras)', menu: 'Perceptron', pkg: 'perceptron_portas · perceptron_letras', aula: '03' },
  { n: 3, nome: 'MADALINE', menu: 'MADALINE', pkg: 'madaline', aula: '04' },
  { n: 4, nome: 'MLP — reprodução dos slides', menu: 'MLP Desafio', pkg: 'mlp', aula: '05' },
  { n: 5, nome: 'MLP — reconhecimento de letras (A–Z)', menu: 'MLP Letras', pkg: 'letras', aula: '05' },
  { n: 6, nome: 'Rede Convolucional (CNN)', menu: 'CNN EMNIST', pkg: 'cnn', aula: '07' },
  { n: 7, nome: 'Investimentos (séries temporais)', menu: 'MLP Ações', pkg: 'timeseries', aula: '—' },
  { n: 8, nome: 'AG com função matemática f(x)', menu: 'GA · Aula 10', pkg: 'genetico', aula: '10' },
  { n: 9, nome: 'AG parametrizável', menu: 'GA · Aula 11', pkg: 'genetico2', aula: '11' },
  { n: 10, nome: 'AG com horários (grade escolar)', menu: 'GA · Horário (Aula 12)', pkg: 'horario', aula: '12' },
  { n: 11, nome: 'Caixeiro viajante (10 cidades)', menu: 'GA · TSP', pkg: 'tsp', aula: '13' },
  { n: 12, nome: 'Caixeiro viajante multipopulacional', menu: 'GA · TSP Multi-ilhas', pkg: 'tspmulti', aula: '14' },
  { n: 13, nome: 'AG com cromossomos reais — Rastrigin', menu: 'GA · Rastrigin 3D (reais)', pkg: 'agrastrigin', aula: '15' },
  { n: 14, nome: 'AG com cromossomos Ranking — TSP', menu: 'GA · TSP Ranking', pkg: 'tspranking', aula: '13 + 16' },
  { n: 15, nome: 'AG que descobre a arquitetura de uma RNA', menu: 'GA · Arquitetura RNA', pkg: 'rnaga', aula: '20' },
  { n: 16, nome: 'Qualidade da água — Fuzzy', menu: 'Fuzzy · Água', pkg: 'fuzzy', aula: '17–19' },
];

// Telas extras (bônus) que existem no menu mas não estão na lista numerada.
const EXTRAS: { menu: string; pkg: string; oque: string }[] = [
  { menu: 'MLP Funções', pkg: 'mlpfunc', oque: 'MLP de regressão aproximando funções contínuas' },
  { menu: 'MLP Ortogonal', pkg: 'mlport', oque: 'MLP com vetores bipolares ortogonais (A–Z)' },
  { menu: 'IMG_REGRESSION', pkg: 'imgreg (+ goroutines / matrix / mini-batch / benchmark)', oque: 'MLP que "pinta" uma imagem pixel a pixel' },
  { menu: 'GA · TSP Comparativo', pkg: 'tsp', oque: 'compara execuções/parâmetros do caixeiro viajante' },
];

const COR = {
  head: '#8fa3b8',
  num: '#00e5ff',
  nome: '#f0f2f5',
  menu: '#3ddc84',
  pkg: '#c9a3ff',
  aula: '#9aa4b2',
  muted: '#8a94a3',
};

export default function AboutView() {
  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">
            Mapa dos <span>Trabalhos</span>
          </div>
          <div className="page-sub" style={{ color: '#b9c2cf' }}>
            Onde encontrar cada trabalho da disciplina — clique no item indicado do menu à esquerda.
          </div>
        </div>
      </div>

      <Card title="Onde está cada trabalho" pulse style={{ marginBottom: 16 }}>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table" style={{ fontSize: 13 }}>
            <thead>
              <tr>
                <th style={{ color: COR.head }}>#</th>
                <th style={{ color: COR.head }}>Trabalho</th>
                <th style={{ color: COR.head }}>No menu (clique aqui)</th>
                <th style={{ color: COR.head }}>Pacote Go / tela</th>
                <th style={{ color: COR.head }}>Aula</th>
                <th style={{ color: COR.head }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {TRABALHOS.map((t) => (
                <tr key={t.n} style={{ opacity: t.pendente ? 0.72 : 1 }}>
                  <td style={{ color: COR.num, fontWeight: 700 }}>{t.n}</td>
                  <td style={{ color: COR.nome }}>{t.nome}</td>
                  <td style={{ color: t.pendente ? COR.muted : COR.menu, fontWeight: 600 }}>
                    {t.pendente ? '— (a fazer)' : `▸ ${t.menu}`}
                  </td>
                  <td style={{ color: COR.pkg, fontFamily: 'var(--font-mono)', fontSize: 12 }}>{t.pkg}</td>
                  <td style={{ color: COR.aula, fontFamily: 'var(--font-mono)', fontSize: 12 }}>{t.aula}</td>
                  <td style={{ color: t.pendente ? '#e0a83c' : COR.menu, fontWeight: 600 }}>
                    {t.pendente ? '○ pendente' : '✓ pronto'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ padding: '10px 14px', color: COR.muted, fontSize: 12, lineHeight: 1.6 }}>
          <b style={{ color: COR.menu }}>Todos os 16 trabalhos prontos</b> — o último a entrar foi o
          Trabalho 16 (sistema fuzzy de qualidade da água, menu <b style={{ color: COR.menu }}>Fuzzy · Água</b>).
        </div>
      </Card>

      <div className="grid-2" style={{ marginBottom: 16 }}>
        <Card title="Telas extras (bônus, além dos 16)">
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table" style={{ fontSize: 13 }}>
              <thead>
                <tr>
                  <th style={{ color: COR.head }}>No menu</th>
                  <th style={{ color: COR.head }}>Pacote</th>
                  <th style={{ color: COR.head }}>O que faz</th>
                </tr>
              </thead>
              <tbody>
                {EXTRAS.map((e) => (
                  <tr key={e.menu}>
                    <td style={{ color: COR.menu, fontWeight: 600 }}>{`▸ ${e.menu}`}</td>
                    <td style={{ color: COR.pkg, fontFamily: 'var(--font-mono)', fontSize: 12 }}>{e.pkg}</td>
                    <td style={{ color: COR.nome }}>{e.oque}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Como o projeto está organizado">
          <div style={{ padding: 14, fontSize: 13, color: COR.nome, lineHeight: 1.8 }}>
            <div style={{ marginBottom: 8 }}>
              Cada trabalho é um <b>pacote Go</b> no backend + uma <b>tela React</b> no frontend, ligados por HTTP/SSE.
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: COR.aula, lineHeight: 1.9 }}>
              <div><span style={{ color: COR.num }}>web/server/</span> — backend Go (um pacote por trabalho)</div>
              <div><span style={{ color: COR.num }}>web/server/main.go</span> — rotas HTTP + streaming SSE</div>
              <div><span style={{ color: COR.num }}>web/frontend/src/views/</span> — uma tela por trabalho</div>
              <div><span style={{ color: COR.num }}>slides/</span> — PDFs das aulas</div>
            </div>
            <div style={{ marginTop: 12, color: COR.muted, fontSize: 12 }}>
              Rodar local: <code style={{ color: COR.menu }}>cd web &amp;&amp; make run</code> (compila o frontend e sobe o servidor em <code>localhost:8080</code>).
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
