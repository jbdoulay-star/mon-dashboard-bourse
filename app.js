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

  // Date
  document.getElementById('update-date').textContent =
    `Mise à jour : ${data.generated_at || data.date}`;

  // Stats bar
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

  // Cards
  const grid = document.getElementById('stocks-grid');
  grid.innerHTML = data.stocks.map((s, i) => buildCard(s, i)).join('');
}

function buildCard(s, rank) {
  const chgClass = s.chg1d > 0 ? 'positive' : s.chg1d < 0 ? 'negative' : 'neutral';
  const rsiClass = s.rsi > 70 ? 'negative' : s.rsi < 30 ? 'positive' : 'neutral';
  const conf     = Math.round((s.score_total || 50));
  
  const badgeClass = s.signal === 'ACHAT'   ? 'badge-buy'
                   : s.signal === 'ATTENTE' ? 'badge-wait'
                   : 'badge-avoid';

  const macdIcon = s.macd_hist > 0 ? '▲' : '▼';
  const macdClass = s.macd_hist > 0 ? 'positive' : 'negative';

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
          <div class="ind-val ${rsiClass}">${s.rsi?.toFixed(0)}</div>
          <div class="ind-lbl">RSI 14</div>
        </div>
        <div class="ind">
          <div class="ind-val ${macdClass}">${macdIcon} MACD</div>
          <div class="ind-lbl">${s.macd_hist?.toFixed(3)}</div>
        </div>
        <div class="ind">
          <div class="ind-val">${s.atr_pct?.toFixed(1)}%</div>
          <div class="ind-lbl">ATR %</div>
        </div>
        <div class="ind">
          <div class="ind-val">${s.bb_pos?.toFixed(0)}%</div>
          <div class="ind-lbl">Bandes Boll.</div>
        </div>
        <div class="ind">
          <div class="ind-val ${s.rr >= 2 ? 'positive' : s.rr >= 1.5 ? 'neutral' : 'negative'}">${s.rr?.toFixed(1)}</div>
          <div class="ind-lbl">Risk/Reward</div>
        </div>
        <div class="ind">
          <div class="ind-val ${conf >= 70 ? 'positive' : conf >= 50 ? 'neutral' : 'negative'}">${conf}/100</div>
          <div class="ind-lbl">Score global</div>
        </div>
      </div>

      <div class="section-title">📊 Fondamentaux</div>
      <div class="analysis-block">${s.ai_fundamentals || '—'}</div>

      <div class="section-title">📉 Analyse chartiste</div>
      <div class="analysis-block">${s.ai_chartist || '—'}</div>

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
          Gain net estimé : 
          <span class="${s.net_gain_pct > 0 ? 'positive' : 'negative'}">
            ${s.net_gain_pct > 0 ? '+' : ''}${s.net_gain_pct?.toFixed(2)}%
          </span>
        </span>
      </div>

      <div class="confidence-bar" title="Score de confiance : ${conf}/100">
        <div class="confidence-fill" style="width:${conf}%"></div>
      </div>
    </div>
  `;
}

loadData();
