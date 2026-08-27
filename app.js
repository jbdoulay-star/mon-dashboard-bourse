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
  // Filtre les null/NaN qui viendraient du JSON
  const clean = prices.filter(p => p !== null && p !== undefined && !isNaN(p));
  if (clean.length < 2) return '';
  const w = 400, h = 55;
  const mn = Math.min(...clean);
  const mx = Math.max(...clean);
  const rng = mx - mn || 1;
  const pts = clean.map((p, i) => {
    const x = (i / (clean.length - 1)) * w;
    const y = h - ((p - mn) / rng * (h - 10)) - 5;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const isUp = clean[clean.length - 1] >= clean[0];
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
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    // Nettoie les NaN/Infinity/booléens malformés que Python peut générer
    const cleaned = text
      .replace(/:\s*NaN/g, ': null')
      .replace(/:\s*Infinity/g, ': null')
      .replace(/:\s*-Infinity/g, ': null');
    const data = JSON.parse(cleaned);
    if (!data.stocks || data.stocks.length === 0) throw new Error('Données vides');
    render(data);
  } catch(e) {
    console.error('Erreur chargement:', e);
    document.getElementById('loading').innerHTML =
      '❌ Erreur de chargement des données.<br><small style="color:#666">Réessayez dans quelques instants ou consultez la console pour plus de détails.</small>';
  }
}

function render(data) {
  document.getElementById('loading').classList.add('hidden');

  document.getElementById('update-date').textContent =
    `Mise à jour : ${data.generated_at || data.date || '—'}`;

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

// Helpers sécurisés pour éviter les crashes sur valeurs null/undefined
function fmt(val, decimals = 2) {
  const v = parseFloat(val);
  return isNaN(v) ? '—' : v.toFixed(decimals);
}

function fmtSign(val, decimals = 2) {
  const v = parseFloat(val);
  if (isNaN(v)) return '—';
  return (v > 0 ? '+' : '') + v.toFixed(decimals);
}

function buildCard(s, rank) {
  const chg1d  = parseFloat(s.chg1d);
  const chg1m  = parseFloat(s.chg1m);
  const rsi    = parseFloat(s.rsi);
  const macdH  = parseFloat(s.macd_hist);
  const atrPct = parseFloat(s.atr_pct);
  const bbPos  = parseFloat(s.bb_pos);
  const rr     = parseFloat(s.rr);

  const chgClass = chg1d > 0 ? 'positive' : chg1d < 0 ? 'negative' : 'neutral';
  const conf     = Math.round(s.score || s.score_total || 50);

  const badgeClass = s.signal === 'ACHETER'    ? 'badge-buy'
                   : s.signal === 'SURVEILLER' ? 'badge-wait'
                   : s.signal === 'EVITER'     ? 'badge-avoid'
                   : 'badge-wait';

  const macdIcon  = macdH > 0 ? '▲' : '▼';

  // Gain net
  const netGain = (s.net_gain !== undefined && s.net_gain !== null && !isNaN(s.net_gain))
    ? s.net_gain
    : (s.net_gain_pct !== undefined && s.net_gain_pct !== null && !isNaN(s.net_gain_pct))
      ? s.net_gain_pct
      : (s.target_1m && s.entry)
        ? parseFloat((((s.target_1m - s.entry) / s.entry * 100) - 2).toFixed(2))
        : null;

  const netGainHTML = netGain !== null && !isNaN(netGain)
    ? `<span class="${netGain > 0 ? 'positive' : 'negative'}">
         ${netGain > 0 ? '+' : ''}${netGain.toFixed(2)}%
       </span>`
    : `<span style="color:#888">N/A</span>`;

  // Fondamentaux
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

  const chartisteLine = s.chartiste && s.chartiste !== ''
    ? s.chartiste
    : s.ai_chartist && s.ai_chartist !== '—'
      ? s.ai_chartist
      : s.entry_tip || 'Analyse chartiste en cours...';

  return `
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <div style="font-size:.75rem;color:#888">#${rank + 1}</div>
          <h3>${s.name || '—'}</h3>
          <div class="ticker">${s.ticker}</div>
          <div class="sector">${s.sector || '—'}</div>
        </div>
        <div class="price-block">
          <div class="price">${fmt(s.price)} €</div>
          <div class="chg ${chgClass}">
            ${fmtSign(chg1d)}% aujourd'hui
          </div>
          <div class="chg ${chg1m > 0 ? 'positive' : 'negative'}" style="font-size:.75rem">
            1M : ${fmtSign(chg1m, 1)}%
          </div>
        </div>
      </div>

      <div class="indicators">
        <div class="ind">
          <div class="ind-val" style="color:${indicatorColor('rsi', rsi)}">${fmt(rsi, 0)}</div>
          <div class="ind-lbl">RSI 14</div>
        </div>
        <div class="ind">
          <div class="ind-val" style="color:${indicatorColor('macd', macdH)}">${isNaN(macdH) ? '—' : macdIcon} MACD</div>
          <div class="ind-lbl">${fmt(macdH, 3)}</div>
        </div>
        <div class="ind">
          <div class="ind-val" style="color:${indicatorColor('atr', atrPct)}">${fmt(atrPct, 1)}%</div>
          <div class="ind-lbl">ATR %</div>
        </div>
        <div class="ind">
          <div class="ind-val" style="color:${indicatorColor('bb', bbPos * 100)}">${isNaN(bbPos) ? '—' : fmt(bbPos * 100, 0)}%</div>
          <div class="ind-lbl">Bandes Boll.</div>
        </div>
        <div class="ind">
          <div class="ind-val" style="color:${indicatorColor('rr', rr)}">${fmt(rr, 1)}</div>
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
          <div class="t-val positive">${fmt(s.entry)} €</div>
          <div class="t-lbl">Entrée</div>
        </div>
        <div class="trade-item">
          <div class="t-val negative">${fmt(s.stop_loss)} €</div>
          <div class="t-lbl">Stop Loss</div>
        </div>
        <div class="trade-item">
          <div class="t-val" style="color:#4da6ff">${fmt(s.target_1m)} €</div>
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
