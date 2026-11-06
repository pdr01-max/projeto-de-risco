/* API calls use relative paths so the dashboard works on any deployment host. */
const $ = (id) => document.getElementById(id);
const pct = (value) => `${(100 * value).toFixed(2)}%`;
const money = (value, currency) => new Intl.NumberFormat('pt-BR', { style: 'currency', currency }).format(value);
const ratio = value => value === null || !Number.isFinite(value) ? '—' : value.toFixed(2);
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
  const confidenceLabel = `${(data.confidence*100).toLocaleString('pt-BR')}%`;
  $('card-var').querySelector('.metric-label').textContent = `VaR ${confidenceLabel} / dia`;
  $('card-es').querySelector('.metric-label').textContent = `Expected Shortfall ${confidenceLabel}`;
  setAlert('var', data.metrics.var_99, data.limits.var_99);
  setAlert('es', data.metrics.es_99, data.limits.es_99);
  setAlert('vol', data.metrics.volatility_annual, data.limits.volatility_annual);
  setAlert('dd', data.metrics.max_drawdown, data.limits.max_drawdown);
  $('es-value').textContent = pct(data.metrics.es_99);
  $('es-value').parentElement.firstChild.textContent = `ES ${confidenceLabel}: `;
  $('return-annual').textContent = pct(data.metrics.return_annual);
  $('downside-vol').textContent = pct(data.metrics.downside_volatility_annual);
  $('var-95').textContent = pct(data.metrics.var_95);
  $('worst-day').textContent = pct(data.metrics.worst_day);
  $('money-var').textContent = money(Math.max(0, data.metrics.var_99) * data.capital, data.base_currency);
  $('money-es').textContent = money(Math.max(0, data.metrics.es_99) * data.capital, data.base_currency);
  $('money-var-95').textContent = money(Math.max(0, data.metrics.var_95) * data.capital, data.base_currency);
  $('money-worst-day').textContent = money(Math.max(0, -data.metrics.worst_day) * data.capital, data.base_currency);
  ['sharpe', 'sortino', 'calmar'].forEach(id => { $(id).textContent = ratio(data.metrics[id]); });
  const kupiec = data.backtest;
  $('kupiec-status').textContent = kupiec.status;
  $('kupiec-status').className = kupiec.status === 'Aprovado' ? 'result-approved' : kupiec.status === 'Atenção' ? 'result-warning' : 'result-rejected';
  $('kupiec-details').textContent = `${kupiec.breaches} violações em ${kupiec.observations} observações fora da amostra · esperado: ${kupiec.expected.toFixed(1)} · Kupiec p=${kupiec.p_value.toFixed(4)} · independência p=${kupiec.p_ind.toFixed(4)} · cobertura conjunta p=${kupiec.p_cc.toFixed(4)}. Previsões usam apenas dados anteriores a cada dia.`;
  const diag = data.tail_diagnostics;
  const items = diag?.marginals ? Object.entries(diag.marginals).map(([i, value]) => `${data.tickers[Number(i)]}: ξ=${value.shape.toFixed(3)}, β=${value.scale.toFixed(3)}, Nu=${value.exceedances}, KS p=${value.ks_p_value.toFixed(3)}${value.warning ? ' · amostra de cauda pequena' : ''}`) : diag ? [`ξ=${diag.shape.toFixed(3)}, β=${diag.scale.toFixed(3)}, Nu=${diag.exceedances}, KS p=${diag.ks_p_value.toFixed(3)}${diag.warning ? ' · amostra de cauda pequena' : ''}`] : [];
  const contributions = diag?.tail_es_contribution?.map((value, i) => `${data.tickers[i]}: ${pct(value)}`).join(', ');
  $('tail-diagnostic').textContent = items.length ? `${items.join(' | ')}${diag.copula_nu ? ` · cópula ν=${diag.copula_nu.toFixed(2)}` : ''}${contributions ? ` · Contribuição ao ES conjunto: ${contributions}` : ''}` : 'O modelo selecionado não ajusta uma GPD.';
  const liq = data.liquidity;
  $('liquidity-results').textContent = liq ? `LVaR ${pct(liq.lvar)} (${money(liq.lvar * data.capital, data.base_currency)}); ES ajustado ${pct(liq.les)}. Liquidação estimada: ${liq.days} dia(s) úteis a até 10% do ADV; custo estimado ${money(liq.execution_cost, data.base_currency)}.` : 'Ative a opção de liquidez.';
  const event = data.election_stress;
  $('election-results').textContent = event?.available === false ? 'Sensibilidades indisponíveis para esta janela.' : `Impacto estimado: ${pct(event.return)} (${money(event.pnl, data.base_currency)}). Choques de fatores e contágio conforme os campos selecionados.`;
  const stressBody = $('stress-results');
  stressBody.replaceChildren();
  data.stress_tests.forEach(item => {
    const row = document.createElement('tr');
    [item.scenario + (item.available && !item.complete ? ' · histórico parcial' : ''), `${item.start} a ${item.end}`, item.available ? pct(item.return) : 'Histórico indisponível',
      item.available ? money(item.pnl, data.base_currency) : '—'].forEach(value => {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    });
    stressBody.append(row);
  });
  renderRiskBudget(data);
  const purchasePanel = $('purchase-panel');
  purchasePanel.hidden = !data.purchase_analysis?.length;
  const purchaseBody = $('purchase-results');
  purchaseBody.replaceChildren();
  (data.purchase_analysis || []).forEach(item => {
    const row = document.createElement('tr');
    const localMoney = value => money(value, item.currency);
    const baseMoney = value => money(value, item.base_currency);
    [item.ticker, item.date, item.holding_days, item.quantity, item.split_factor.toFixed(4), item.adjusted_quantity,
      localMoney(item.purchase_price), localMoney(item.current_price), baseMoney(item.cost), baseMoney(item.market_value), baseMoney(item.dividends), baseMoney(item.pnl), pct(item.return),
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
  const wealthTraces = [{ x: data.dates, y: data.wealth_index, type: 'scatter', name: 'Carteira',
    mode: 'lines', line: { color: chartColors.amber, width: 2.5 },
    fill: 'tozeroy', fillcolor: 'rgba(233,173,69,.10)', hovertemplate: '%{x}<br>Índice: %{y:.3f}<extra></extra>'
  }];
  if (data.benchmark) wealthTraces.push({ x: data.benchmark.dates, y: data.benchmark.wealth_index,
    type: 'scatter', mode: 'lines', name: data.benchmark.symbol,
    line: { color: chartColors.steel, width: 2, dash: 'dot' },
    hovertemplate: '%{x}<br>Benchmark: %{y:.3f}<extra></extra>' });
  if (kupiec.breach_dates.length) wealthTraces.push({ x: kupiec.breach_dates, y: kupiec.breach_wealth,
    type: 'scatter', mode: 'markers', name: 'Violação VaR', marker: { color: chartColors.red, size: 8, symbol: 'x' },
    hovertemplate: '%{x}<br>Violação VaR<extra></extra>' });
  if (kupiec.cluster_dates?.length) wealthTraces.push({ x: kupiec.cluster_dates,
    y: kupiec.cluster_dates.map(day => data.wealth_index[data.dates.indexOf(day)]),
    type: 'scatter', mode: 'markers', name: 'Violações consecutivas',
    marker: { color: '#ff4050', size: 12, symbol: 'diamond' } });
  Plotly.react('wealth-chart', wealthTraces, layout({ yaxis: { gridcolor: chartColors.grid, title: 'Base 1,0' } }), config);
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
    ? `${budget.breaches.length} limite(s) violado(s). ${budget.compliant ? 'Exposição sugerida' : 'Nenhuma combinação com caixa CDI atende a todos os limites nesta janela. O cenário abaixo é 100% caixa e também viola os limites'}: ${pct(budget.risk_scale)} em ativos e ${pct(1 - budget.risk_scale)} em caixa.`
    : 'A alocação selecionada respeita todos os limites informados.';
  status.className = budget.breaches.length ? 'status error mt-3 mb-2' : 'status mt-3 mb-2';
  $('cash-note').textContent = budget.cash_return_annual === null
    ? 'Nenhum rebalanceamento necessário.'
    : `Caixa: ${data.cash_rate_source || 'CDI observado (BCB SGS 12)'} em BRL, antes de taxas e impostos${data.base_currency === 'USD' ? ', convertido para USD pela série USD/BRL' : ''}. Retorno anualizado do caixa: ${pct(budget.cash_return_annual)}; do cenário: ${pct(budget.recommended_return_annual)}. Os pesos são simulados com rebalanceamento diário e não representam ordens de negociação.`;
  const names = { var_99: `VaR ${(data.confidence*100).toLocaleString('pt-BR')}%`, es_99: `ES ${(data.confidence*100).toLocaleString('pt-BR')}%`, volatility_annual: 'Volatilidade anual', max_drawdown: 'Max drawdown' };
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
    { x: assets, y: assets.map(x => (budget.recommended_weights[x] || 0) * 100), type: 'bar', name: budget.compliant ? 'Sugerida' : 'Não conforme', marker: { color: chartColors.amber } }
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
const shareKeys = ['tickers', 'period', 'method', 'base-currency', 'capital', 'var-model', 'benchmark',
  'var-limit', 'es-limit', 'vol-limit', 'dd-limit', 'confidence', 'threshold-quantile', 'risk-free',
  'stress-preset', 'stress-ibov', 'stress-sp500', 'stress-fx', 'stress-vol', 'stress-contagion', 'stress-slippage'];
function shareURL() {
  const params = new URLSearchParams();
  shareKeys.forEach(id => params.set(id, $(id).value));
  params.set('liquidity', $('liquidity').checked ? '1' : '0');
  if ($('period').value === 'custom') { params.set('start-date', $('start-date').value); params.set('end-date', $('end-date').value); }
  if ($('method').value === 'manual') params.set('weights', [...document.querySelectorAll('.weight-input')].map(x => x.value).join(','));
  return `${location.origin}${location.pathname}?${params.toString()}`;
}
function syncURL() { history.replaceState(null, '', shareURL()); }
function restoreURL() {
  const params = new URLSearchParams(location.search);
  if (!params.size) {
    $('tickers').value = 'PETR4.SA, VALE3.SA, ITUB4.SA, AAPL, SPY, TLT';
    $('method').value = 'risk_parity';
  } else {
    shareKeys.forEach(id => { if (params.has(id)) $(id).value = params.get(id); });
    $('liquidity').checked = params.get('liquidity') === '1';
    if (!params.has('risk-free')) $('risk-free').value = $('base-currency').value === 'BRL' ? '11.25' : '4.50';
    $('start-date').value = params.get('start-date') || '';
    $('end-date').value = params.get('end-date') || '';
  }
  refreshWeights(); toggleMethod();
  $('custom-dates').hidden = $('period').value !== 'custom';
  if (params.has('weights') && $('method').value === 'manual') {
    params.get('weights').split(',').forEach((weight, i) => {
      const input = document.querySelectorAll('.weight-input')[i];
      if (input) input.value = weight;
    });
    updateTotal();
  }
}
let tickerTimer;
$('tickers').addEventListener('input', () => { clearTimeout(tickerTimer); tickerTimer = setTimeout(() => { refreshWeights(); syncURL(); }, 250); });
$('method').addEventListener('change', () => { toggleMethod(); syncURL(); });
$('period').addEventListener('change', () => { $('custom-dates').hidden = $('period').value !== 'custom'; syncURL(); });
$('base-currency').addEventListener('change', () => { $('risk-free').value = $('base-currency').value === 'BRL' ? '11.25' : '4.50'; syncURL(); });
const stressPresets = { election: [-20,-5,15,3,.8,3], fx: [-10,-3,25,2,.6,4], global: [-15,-25,10,3,.9,3] };
$('stress-preset').addEventListener('change', () => { const values = stressPresets[$('stress-preset').value]; if (values) ['stress-ibov','stress-sp500','stress-fx','stress-vol','stress-contagion','stress-slippage'].forEach((id,i) => { $(id).value = values[i]; }); syncURL(); });
document.querySelectorAll('.preset').forEach(button => button.addEventListener('click', () => { $('tickers').value = button.dataset.tickers; refreshWeights(); syncURL(); }));
$('risk-form').addEventListener('change', syncURL);
$('risk-form').addEventListener('input', event => { if (event.target.id !== 'tickers') syncURL(); });
$('copy-link').addEventListener('click', async () => {
  try {
    const url = shareURL();
    await navigator.clipboard.writeText(url);
    $('copy-link').textContent = 'Link copiado!';
    setTimeout(() => { $('copy-link').textContent = 'Copiar link da carteira'; }, 2500);
  } catch { $('copy-link').textContent = 'Não foi possível copiar'; }
});
restoreURL();
window.addEventListener('popstate', () => { restoreURL(); $('risk-form').requestSubmit(); });

$('risk-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const tickers = tickerList();
  const payload = { tickers, period: $('period').value,
    method: $('method').value,
    base_currency: $('base-currency').value,
    capital: Number($('capital').value),
    var_model: $('var-model').value,
    confidence: Number($('confidence').value), threshold_quantile: Number($('threshold-quantile').value),
    risk_free_annual: Number($('risk-free').value)/100, liquidity: $('liquidity').checked,
    stress_ibov: Number($('stress-ibov').value)/100, stress_sp500: Number($('stress-sp500').value)/100,
    stress_fx: Number($('stress-fx').value)/100, stress_vol_multiplier: Number($('stress-vol').value),
    stress_contagion: Number($('stress-contagion').value), stress_slippage_multiplier: Number($('stress-slippage').value),
    benchmark: $('benchmark').value,
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
  $('run-spinner').hidden = false;
  $('run-label').textContent = 'Calculando…';
  $('results-main').classList.add('loading-results');
  $('status').classList.remove('error');
  $('status').textContent = 'Baixando preços e calculando riscos…';
  try {
    if (purchaseError) throw new Error(purchaseError);
    if (payload.period === 'custom' && (!payload.start_date || !payload.end_date || payload.start_date >= payload.end_date)) {
      throw new Error('Informe um intervalo histórico válido.');
    }
    if (!tickers.length || !payload.var_limit || !payload.es_limit || !payload.volatility_limit || !payload.drawdown_limit || !Number.isFinite(payload.capital) || payload.capital <= 0) {
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
    syncURL();
  } catch (error) {
    $('status').classList.add('error');
    $('status').textContent = `Não foi possível concluir: ${error.message}`;
  } finally { button.disabled = false; $('run-spinner').hidden = true; $('run-label').textContent = 'Calcular risco e otimizar'; $('results-main').classList.remove('loading-results'); }
});
$('risk-form').requestSubmit();
