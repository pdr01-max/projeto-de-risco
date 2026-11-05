/* API calls use relative paths so the dashboard works on any deployment host. */
const $ = (id) => document.getElementById(id);
const pct = (value) => `${(100 * value).toFixed(2)}%`;
const chartColors = { text: '#a4b0b4', grid: '#394349', amber: '#e9ad45', steel: '#7fa7c2', red: '#f17e75' };
const layout = (extra = {}) => ({
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: chartColors.text, family: 'IBM Plex Sans, Segoe UI, Arial' },
  margin: { l: 52, r: 18, t: 24, b: 44 }, autosize: true,
  xaxis: { gridcolor: chartColors.grid, zerolinecolor: chartColors.grid },
  yaxis: { gridcolor: chartColors.grid, zerolinecolor: chartColors.grid },
  legend: { orientation: 'h', y: 1.17, x: 0 }, ...extra
});
const config = { responsive: true, displaylogo: false };

function setAlert(id, measured, limit) {
  const card = $(`card-${id}`);
  const ratio = measured / limit;
  const state = ratio > 1 ? 'red' : ratio >= 0.8 ? 'yellow' : 'green';
  card.classList.remove('red', 'yellow', 'green');
  card.classList.add(state);
  $(`value-${id}`).textContent = pct(measured);
  $(`limit-${id}`).textContent = `Limite: ${pct(limit)}`;
  card.querySelector('.traffic').textContent =
    state === 'red' ? 'Violado' : state === 'yellow' ? 'Alerta' : 'Aprovado';
}

async function post(url, payload) {
  const response = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(detail);
  }
  return data;
}

function renderRisk(data) {
  setAlert('var', data.metrics.var_99, data.limits.var_99);
  setAlert('es', data.metrics.es_99, data.limits.es_99);
  setAlert('vol', data.metrics.volatility_annual, data.limits.volatility_annual);
  setAlert('dd', data.metrics.max_drawdown, data.limits.max_drawdown);
  $('es-value').textContent = pct(data.metrics.es_99);
  Plotly.react('wealth-chart', [{ x: data.dates, y: data.wealth_index, type: 'scatter',
    mode: 'lines', line: { color: chartColors.amber, width: 2.5 },
    fill: 'tozeroy', fillcolor: 'rgba(233,173,69,.10)', hovertemplate: '%{x}<br>Índice: %{y:.3f}<extra></extra>'
  }], layout({ yaxis: { gridcolor: chartColors.grid, title: 'Base 1,0' } }), config);
  Plotly.react('loss-chart', [{ x: data.daily_losses.map(x => x * 100), type: 'histogram',
    marker: { color: '#5b7990', line: { color: chartColors.steel, width: .4 } }, nbinsx: 45,
    hovertemplate: 'Perda: %{x:.2f}%<br>Dias: %{y}<extra></extra>'
  }], layout({ xaxis: { title: 'Perda diária (%)', gridcolor: chartColors.grid },
    yaxis: { title: 'Frequência', gridcolor: chartColors.grid },
    shapes: [{ type: 'line', x0: data.metrics.var_99 * 100, x1: data.metrics.var_99 * 100,
      y0: 0, y1: 1, yref: 'paper', line: { color: chartColors.red, width: 2, dash: 'dash' } }]
  }), config);
}

function renderWeights(data) {
  const names = data.tickers;
  $('allocation-method').textContent = $('method').selectedOptions[0].textContent;
  Plotly.react('weights-chart', [
    { x: names, y: names.map(x => data.weights[x] * 100), type: 'bar', name: 'Peso', marker: { color: chartColors.steel } }
  ], layout({ yaxis: { title: 'Peso (%)', gridcolor: chartColors.grid } }), config);
}

function tickerList() { return $('tickers').value.split(',').map(x => x.trim().toUpperCase()).filter(Boolean); }
function refreshWeights() {
  const names = tickerList();
  const old = Object.fromEntries([...document.querySelectorAll('.weight-input')].map(x => [x.dataset.ticker, x.value]));
  const holder = $('manual-weights');
  holder.replaceChildren();
  names.forEach((name, i) => {
    const row = document.createElement('div');
    row.className = 'd-flex align-items-center gap-2 mb-2';
    const label = document.createElement('label');
    label.className = 'form-label mb-0 flex-grow-1';
    label.textContent = name;
    const input = document.createElement('input');
    input.type = 'number'; input.min = '0'; input.max = '100'; input.step = '0.01';
    input.className = 'form-control weight-input'; input.style.width = '105px';
    input.dataset.ticker = name;
    input.value = old[name] ?? (i === names.length - 1 ? (100 - Math.floor(100 / names.length) * (names.length - 1)) : Math.floor(100 / names.length));
    input.addEventListener('input', updateTotal);
    const sign = document.createElement('span'); sign.textContent = '%';
    row.append(label, input, sign); holder.append(row);
  });
  updateTotal();
}
function updateTotal() {
  const total = [...document.querySelectorAll('.weight-input')].reduce((s, x) => s + Number(x.value), 0);
  $('weight-total').textContent = `Total: ${total.toFixed(2)}%`;
  $('weight-total').style.color = Math.abs(total - 100) < .01 ? '#77c99a' : '#f17e75';
}
function toggleMethod() { $('manual-weights').hidden = $('method').value !== 'manual'; $('weight-total').hidden = $('method').value !== 'manual'; }
let tickerTimer;
$('tickers').addEventListener('input', () => { clearTimeout(tickerTimer); tickerTimer = setTimeout(refreshWeights, 250); });
$('method').addEventListener('change', toggleMethod);
document.querySelectorAll('.preset').forEach(button => button.addEventListener('click', () => { $('tickers').value = button.dataset.tickers; refreshWeights(); }));
refreshWeights(); toggleMethod();

$('risk-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const tickers = tickerList();
  const payload = { tickers, period: $('period').value,
    method: $('method').value,
    var_limit: Number($('var-limit').value) / 100,
    es_limit: Number($('es-limit').value) / 100,
    volatility_limit: Number($('vol-limit').value) / 100,
    drawdown_limit: Number($('dd-limit').value) / 100 };
  const button = $('run-button');
  button.disabled = true;
  $('status').classList.remove('error');
  $('status').textContent = 'Baixando preços e calculando riscos…';
  try {
    if (!tickers.length || !payload.var_limit || !payload.es_limit || !payload.volatility_limit || !payload.drawdown_limit) {
      throw new Error('Informe tickers e limites positivos.');
    }
    if (payload.method === 'manual') {
      payload.weights = [...document.querySelectorAll('.weight-input')].map(x => Number(x.value) / 100);
      if (payload.weights.some(x => !Number.isFinite(x) || x < 0) || Math.abs(payload.weights.reduce((a,b) => a+b, 0) - 1) > 1e-5) {
        throw new Error('Os pesos manuais devem somar exatamente 100%.');
      }
    }
    const risk = await post('/api/portfolio_risk', payload);
    renderRisk(risk);
    renderWeights(risk);
    $('status').textContent = `${risk.observations} pregões analisados · ${tickers.length} ativo(s).`;
  } catch (error) {
    $('status').classList.add('error');
    $('status').textContent = `Não foi possível concluir: ${error.message}`;
  } finally { button.disabled = false; }
});
