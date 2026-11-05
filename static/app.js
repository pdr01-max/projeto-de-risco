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
  $('return-annual').textContent = pct(data.metrics.return_annual);
  $('downside-vol').textContent = pct(data.metrics.downside_volatility_annual);
  $('var-95').textContent = pct(data.metrics.var_95);
  $('worst-day').textContent = pct(data.metrics.worst_day);
  renderRiskBudget(data);
  const purchasePanel = $('purchase-panel');
  purchasePanel.hidden = !data.purchase_analysis?.length;
  const purchaseBody = $('purchase-results');
  purchaseBody.replaceChildren();
  (data.purchase_analysis || []).forEach(item => {
    const row = document.createElement('tr');
    const money = value => new Intl.NumberFormat('pt-BR', { style: 'currency', currency: item.currency }).format(value);
    [item.ticker, item.date, item.holding_days, item.quantity, money(item.purchase_price), money(item.current_price),
      money(item.cost), money(item.market_value), money(item.pnl), pct(item.return),
      item.return_annualized === null ? '—' : pct(item.return_annualized)].forEach(value => {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    });
    [...row.children].slice(-2).forEach(cell => { cell.style.color = item.return >= 0 ? '#77c99a' : '#f17e75'; });
    purchaseBody.append(row);
  });
  Plotly.react('drawdown-chart', [{ x: data.dates, y: data.drawdown.map(x => x * 100),
    type: 'scatter', mode: 'lines', line: { color: chartColors.red, width: 2 },
    fill: 'tozeroy', fillcolor: 'rgba(241,126,117,.12)',
    hovertemplate: '%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
  }], layout({ yaxis: { title: 'Drawdown (%)', gridcolor: chartColors.grid } }), config);
  Plotly.react('rolling-vol-chart', [{ x: data.dates, y: data.rolling_volatility_21d.map(x => x === null ? null : x * 100),
    type: 'scatter', mode: 'lines', line: { color: chartColors.amber, width: 2 },
    hovertemplate: '%{x}<br>Volatilidade: %{y:.2f}%<extra></extra>'
  }], layout({ yaxis: { title: 'Volatilidade (%)', gridcolor: chartColors.grid } }), config);
  Plotly.react('correlation-chart', [{
    x: data.tickers, y: data.tickers, z: data.correlation, type: 'heatmap',
    zmin: -1, zmax: 1, zmid: 0, colorscale: [[0, '#9a524f'], [.5, '#2a3033'], [1, '#e9ad45']],
    hovertemplate: '%{x} × %{y}<br>Correlação: %{z:.2f}<extra></extra>'
  }], layout({ margin: { l: 70, r: 20, t: 24, b: 55 } }), config);
  Plotly.react('contribution-chart', [{
    x: data.tickers, y: data.tickers.map(x => data.risk_contribution[x] * 100),
    type: 'bar', marker: { color: chartColors.steel },
    hovertemplate: '%{x}<br>Contribuição: %{y:.2f} p.p.<extra></extra>'
  }], layout({ yaxis: { title: 'Pontos percentuais', gridcolor: chartColors.grid } }), config);
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

function renderRiskBudget(data) {
  const budget = data.risk_budget;
  $('rebalance-panel').hidden = false;
  $('rebalance-method').textContent = $('method').selectedOptions[0].textContent;
  const status = $('rebalance-status');
  status.textContent = budget.breaches.length
    ? `${budget.breaches.length} limite(s) violado(s). ${budget.compliant ? 'Exposição sugerida' : 'Nenhuma combinação com caixa CDI atende a todos os limites nesta janela'}: ${pct(budget.risk_scale)} em ativos e ${pct(1 - budget.risk_scale)} em caixa.`
    : 'A alocação selecionada respeita todos os limites informados.';
  status.className = budget.breaches.length ? 'status error mt-3 mb-2' : 'status mt-3 mb-2';
  $('cash-note').textContent = budget.cash_return_annual === null
    ? 'Nenhum rebalanceamento necessário.'
    : `Caixa: 100% do CDI histórico em BRL, antes de taxas e impostos${data.tickers.some(x => !x.endsWith('.SA')) ? ', convertido para USD pela série USD/BRL' : ''}. Retorno anualizado do caixa: ${pct(budget.cash_return_annual)}; da alocação sugerida: ${pct(budget.recommended_return_annual)}. Os pesos são simulados com rebalanceamento diário e não representam ordens de negociação.`;
  const names = { var_99: 'VaR 99%', es_99: 'ES 99%', volatility_annual: 'Volatilidade anual', max_drawdown: 'Max drawdown' };
  const breachList = $('breach-list');
  breachList.replaceChildren();
  budget.breaches.forEach(item => {
    const line = document.createElement('div');
    line.className = 'breach-item';
    line.textContent = `${names[item.metric]}: ${pct(item.actual)} acima do limite de ${pct(item.limit)}.`;
    breachList.append(line);
  });
  const metricsBody = $('risk-budget-results');
  metricsBody.replaceChildren();
  Object.entries(names).forEach(([key, name]) => {
    const row = document.createElement('tr');
    [name, pct(budget.selected_metrics[key]), pct(data.limits[key]), pct(budget.recommended_metrics[key])].forEach(value => {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    });
    if (budget.selected_metrics[key] > data.limits[key]) row.children[1].style.color = chartColors.red;
    metricsBody.append(row);
  });
  const assets = [...data.tickers, 'CAIXA'];
  const weightsBody = $('rebalance-weights');
  weightsBody.replaceChildren();
  assets.forEach(asset => {
    const before = data.weights[asset] || 0;
    const after = budget.recommended_weights[asset] || 0;
    const row = document.createElement('tr');
    [asset, pct(before), pct(after), ((after - before) * 100).toFixed(2)].forEach(value => {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    });
    weightsBody.append(row);
  });
  Plotly.react('rebalance-chart', [
    { x: assets, y: assets.map(x => (data.weights[x] || 0) * 100), type: 'bar', name: 'Selecionada', marker: { color: chartColors.steel } },
    { x: assets, y: assets.map(x => (budget.recommended_weights[x] || 0) * 100), type: 'bar', name: 'Sugerida', marker: { color: chartColors.amber } }
  ], layout({ barmode: 'group', yaxis: { title: 'Peso (%)', gridcolor: chartColors.grid } }), config);
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
  refreshPurchases(names);
}
function refreshPurchases(names) {
  const holder = $('purchase-inputs');
  const old = Object.fromEntries([...holder.querySelectorAll('.purchase-input')].map(row =>
    [row.dataset.ticker, [...row.querySelectorAll('input')].map(input => input.value)]));
  holder.replaceChildren();
  names.forEach(name => {
    const row = document.createElement('div');
    row.className = 'purchase-input';
    row.dataset.ticker = name;
    const label = document.createElement('strong');
    label.className = 'ticker-name';
    label.textContent = name;
    row.append(label);
    [['date', 'Data da compra', 'date'], ['price', 'Preço pago', 'number'], ['quantity', 'Quantidade', 'number']].forEach(([field, placeholder, type], i) => {
      const input = document.createElement('input');
      input.type = type; input.className = 'form-control';
      input.dataset.field = field; input.setAttribute('aria-label', name + ' — ' + placeholder);
      input.placeholder = placeholder; input.value = old[name]?.[i] || '';
      if (type === 'number') { input.min = '0.000001'; input.step = 'any'; }
      row.append(input);
    });
    holder.append(row);
  });
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
$('period').addEventListener('change', () => { $('custom-dates').hidden = $('period').value !== 'custom'; });
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
  if (payload.period === 'custom') {
    payload.start_date = $('start-date').value;
    payload.end_date = $('end-date').value;
  }
  payload.purchases = {};
  let purchaseError = '';
  document.querySelectorAll('.purchase-input').forEach(row => {
    const values = Object.fromEntries([...row.querySelectorAll('input')].map(input => [input.dataset.field, input.value]));
    if (Object.values(values).some(Boolean)) {
      if (!values.date || !values.price || !values.quantity || Number(values.price) <= 0 || Number(values.quantity) <= 0) {
        purchaseError = `Complete data, preço e quantidade para ${row.dataset.ticker}.`;
      }
      payload.purchases[row.dataset.ticker] = { date: values.date, price: Number(values.price), quantity: Number(values.quantity) };
    }
  });
  const button = $('run-button');
  button.disabled = true;
  $('status').classList.remove('error');
  $('status').textContent = 'Baixando preços e calculando riscos…';
  try {
    if (purchaseError) throw new Error(purchaseError);
    if (payload.period === 'custom' && (!payload.start_date || !payload.end_date || payload.start_date >= payload.end_date)) {
      throw new Error('Informe um intervalo histórico válido.');
    }
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
