function indicatorColor(indicator, value) {
  const v = parseFloat(value);
  if (isNaN(v)) return '#888888';
  switch(indicator) {
    case 'rsi':
      if (v >= 70) return '#ff1744';
      if (v >= 60) return '#ff6d00';
      if (v >= 45) return '#ffd600';
      if (v >= 30) return '#76ff03';
      return '#00c853';
    case 'macd':
      if (v > 0.5)  return '#00c853';
      if (v > 0)    return '#76ff03';
      if (v > -0.5) return '#ffd600';
      if (v > -1)   return '#ff6d00';
      return '#ff1744';
    case 'atr':
      if (v <= 1.5) return '#00c853';
      if (v <= 2.5) return '#76ff03';
      if (v <= 3.5) return '#ffd600';
      if (v <= 5)   return '#ff6d00';
      return '#ff1744';
    case 'bb':
      if (v <= 20)  return '#00c853';
      if (v <= 40)  return '#76ff03';
      if (v <= 60)  return '#ffd600';
      if (v <= 80)  return '#ff6d00';
      return '#ff1744';
    case 'rr':
      if (v >= 2.5) return '#00c853';
      if (v >= 2.0) return '#76ff03';
      if (v >= 1.5) return '#ffd600';
      if (v >= 1.0) return '#ff6d00';
      return '#ff1744';
    case 'score':
      if (v >= 80) return '#00c853';
      if (v >= 65) return '#76ff03';
      if (v >= 50) return '#ffd600';
      if (v >= 35) return '#ff6d00';
      return '#ff1744';
  }
  return '#888888';
}

function buildSparkline(prices) {
  if (!prices || prices.length < 2) return '';
  const w = 400, h = 55;
  const mn = Math.min(...prices);
  const mx = Math.max(...prices);
  const rng = mx - mn || 1;
  const pts = prices.map((p, i) => {
    const x = (i / (prices.length - 1)) * w;
    const y = h - ((p - mn) / rng * (h - 10)) - 5;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const isUp = prices[prices.length - 1] >= prices[0];
  const color = isUp ? '#00c853' : '#ff1744';
  const uid = Math.random().toString(36).slice(2, 7);
  const fillPts = `0,${h} ${pts} ${w},${h}`;
  return `
    <div style="margin:8px 0 4px 0;">
      <svg width="100%" viewBox="0 0 ${w} ${h}"
           preserveAspectRatio="none" style="display:block;height:55px;">
        <defs>
          <linearGradient id="sg${uid}" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stop-color="${color}" stop-opacity="0.35"/>
            <stop offset="100%" stop-color="${color}" stop-opacity="0.0"/>
          </linearGradient>
        </defs>
        <polygon points="${fillPts}" fill="url(#sg${uid})"/>
        <polyline points="${pts}" fill="none"
                  stroke="${color}" stroke-width="2"
                  stroke-linejoin="round" stroke-linecap="round"/>
      </svg>
    </div>`;
}
async function loadData() {
  try {
    const res = await fetch(`data/selections.json?t=${Date.now()}`);
    const data = await res.json();
    render(data);
  } catch(e) {
    document.getElementById('loading').textContent = 
      '❌ Erreur de chargement des données.';
  }
}

function render(data) {
  document.getElementById('loading').classList.add('hidden');

  document.getElementById('update-date').textContent =
    `Mise à jour : ${data.generated_at || data.date}`;

  const statsBar = document.getElementById('stats-bar');
  if (data.meta) {
    statsBar.innerHTML = `
      <div class="stat-item">
        <div class="val">${data.stocks.length}</div>
        <div class="lbl">Actions sélectionnées</div>
      </div>
      <div class="stat-item">
        <div class="val">${data.meta.universe_size || '—'}</div>
        <div class="lbl">Univers analysé</div>
      </div>
      <div class="stat-item">
        <div class="val">${data.meta.api_calls || 0}</div>
        <div class="lbl">Appels IA utilisés</div>
      </div>
    `;
    statsBar.classList.remove('hidden');
  }

  const grid = document.getElementById('stocks-grid');
  grid.innerHTML = data.stocks.map((s, i) => buildCard(s, i)).join('');
}

function buildCard(s, rank) {
  const chgClass = s.chg1d > 0 ? 'positive' : s.chg1d < 0 ? 'negative' : 'neutral';
  const rsiClass = s.rsi > 70 ? 'negative' : s.rsi < 30 ? 'positive' : 'neutral';
  const conf     = Math.round(s.score || s.score_total || 50);

  const badgeClass = s.signal === 'ACHETER' ? 'badge-buy'
                   : s.signal === 'SURVEILLER' ? 'badge-wait'
                   : s.signal === 'EVITER' ? 'badge-avoid'
                   : 'badge-wait';

  const macdIcon  = s.macd_hist > 0 ? '▲' : '▼';
  const macdClass = s.macd_hist > 0 ? 'positive' : 'negative';

  // Gain net : priorité au champ Python, fallback calcul JS
  const netGain = (s.net_gain !== undefined && s.net_gain !== null)
    ? s.net_gain
    : (s.net_gain_pct !== undefined && s.net_gain_pct !== null)
      ? s.net_gain_pct
      : (s.target_1m && s.entry)
        ? parseFloat((((s.target_1m - s.entry) / s.entry * 100) - 2).toFixed(2))
        : null;

  const netGainHTML = netGain !== null
    ? `<span class="${netGain > 0 ? 'positive' : 'negative'}">
         ${netGain > 0 ? '+' : ''}${netGain.toFixed(2)}%
       </span>`
    : `<span style="color:#888">N/A</span>`;

  // Fondamentaux : champs Python nouveaux + fallback ancien champ
  const resumeLine = s.resume && s.resume !== 'Donnees fondamentales en cours de chargement.'
    ? `<p style="margin:0 0 6px 0;color:#cbd5e1;font-size:0.82rem;line-height:1.5;">${s.resume}</p>`
    : s.ai_fundamentals
      ? `<p style="margin:0 0 6px 0;color:#cbd5e1;font-size:0.82rem;line-height:1.5;">${s.ai_fundamentals}</p>`
      : `<p style="margin:0 0 6px 0;color:#888;font-size:0.82rem;">Analyse en cours...</p>`;

  const bullLine = s.bull_case
    ? `<p style="margin:0 0 4px 0;font-size:0.80rem;">
         <span style="color:#4ade80;">▲ Optimiste :</span>
         <span style="color:#94a3b8;"> ${s.bull_case}</span>
       </p>`
    : '';

  const bearLine = s.bear_case
    ? `<p style="margin:0;font-size:0.80rem;">
         <span style="color:#f87171;">▼ Pessimiste :</span>
         <span style="color:#94a3b8;"> ${s.bear_case}</span>
       </p>`
    : '';

  // Analyse chartiste : champ Python nouveau + fallback ancien champ
  const chartisteLine = s.chartiste && s.chartiste !== ''
    ? s.chartiste
    : s.ai_chartist && s.ai_chartist !== '—'
      ? s.ai_chartist
      : s.entry_tip || 'Analyse chartiste en cours...';

  return `
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <div style="font-size:.75rem;color:#888">#${rank+1}</div>
          <h3>${s.name}</h3>
          <div class="ticker">${s.ticker}</div>
          <div class="sector">${s.sector}</div>
        </div>
        <div class="price-block">
          <div class="price">${s.price?.toFixed(2)} €</div>
          <div class="chg ${chgClass}">
            ${s.chg1d > 0 ? '+' : ''}${s.chg1d?.toFixed(2)}% aujourd'hui
          </div>
          <div class="chg ${s.chg1m > 0 ? 'positive' : 'negative'}" style="font-size:.75rem">
            1M : ${s.chg1m > 0 ? '+' : ''}${s.chg1m?.toFixed(1)}%
          </div>
        </div>
      </div>

 <div class="indicators">
        <div class="ind">
          <div class="ind-val" style="color:${indicatorColor('rsi', s.rsi)}">${s.rsi?.toFixed(0)}</div>
          <div class="ind-lbl">RSI 14</div>
        </div>
        <div class="ind">
          <div class="ind-val" style="color:${indicatorColor('macd', s.macd_hist)}">${macdIcon} MACD</div>
          <div class="ind-lbl">${s.macd_hist?.toFixed(3)}</div>
        </div>
        <div class="ind">
          <div class="ind-val" style="color:${indicatorColor('atr', s.atr_pct)}">${s.atr_pct?.toFixed(1)}%</div>
          <div class="ind-lbl">ATR %</div>
        </div>
        <div class="ind">
          <div class="ind-val" style="color:${indicatorColor('bb', s.bb_pos * 100)}">${s.bb_pos?.toFixed(0)}%</div>
          <div class="ind-lbl">Bandes Boll.</div>
        </div>
        <div class="ind">
          <div class="ind-val" style="color:${indicatorColor('rr', s.rr)}">${s.rr?.toFixed(1)}</div>
          <div class="ind-lbl">Risk/Reward</div>
        </div>
        <div class="ind">
          <div class="ind-val" style="color:${indicatorColor('score', conf)}">${conf}/100</div>
          <div class="ind-lbl">Score global</div>
        </div>
      </div>

      ${buildSparkline(s.prices_6m)}

      <div class="section-title">📊 Fondamentaux</div>
      <div class="analysis-block">
        ${resumeLine}
        ${bullLine}
        ${bearLine}
      </div>

      <div class="section-title">📉 Analyse chartiste</div>
      <div class="analysis-block">
        <p style="margin:0;color:#cbd5e1;font-size:0.82rem;line-height:1.6;">
          ${chartisteLine}
        </p>
      </div>

      <div class="section-title">🎯 Points d'entrée / sortie</div>
      <div class="trade-block">
        <div class="trade-item">
          <div class="t-val positive">${s.entry?.toFixed(2)} €</div>
          <div class="t-lbl">Entrée</div>
        </div>
        <div class="trade-item">
          <div class="t-val negative">${s.stop_loss?.toFixed(2)} €</div>
          <div class="t-lbl">Stop Loss</div>
        </div>
        <div class="trade-item">
          <div class="t-val" style="color:#4da6ff">${s.target_1m?.toFixed(2)} €</div>
          <div class="t-lbl">Objectif 1M</div>
        </div>
      </div>

      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:.8rem">
        <span class="badge ${badgeClass}">${s.signal || '—'}</span>
        <span style="font-size:.75rem;color:#888">
          Gain net estimé : ${netGainHTML}
        </span>
      </div>

      <div class="confidence-bar" title="Score de confiance : ${conf}/100">
        <div class="confidence-fill" style="width:${conf}%"></div>
      </div>
    </div>
  `;
}

loadData();
