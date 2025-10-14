// Steam EDA (CSV) — all in browser, no server.
// Robust CSV parsing via Papa Parse; charts drawn with plain Canvas2D (no external chart libs).

const els = {
  maxRows: document.getElementById('maxRows'),
  maxItems: document.getElementById('maxItems'),
  debugMode: document.getElementById('debugMode'),
  runBtn: document.getElementById('runBtn'),
  resetBtn: document.getElementById('resetBtn'),
  exportBtn: document.getElementById('exportBtn'),
  reviewsDrop: document.getElementById('reviewsDrop'),
  itemsDrop: document.getElementById('itemsDrop'),
  reviewsInfo: document.getElementById('reviewsInfo'),
  itemsInfo: document.getElementById('itemsInfo'),
  status: document.getElementById('status'),
  log: document.getElementById('log'),
  headlineTbl: document.getElementById('headlineTbl').querySelector('tbody'),
  genresChart: document.getElementById('genresChart'),
  timeChart: document.getElementById('timeChart'),
  topGamesTbl: document.getElementById('topGamesTbl').querySelector('tbody'),
};

let reviewsFile = null;
let itemsFile = null;

let aggregates = null; // will hold all computed summaries
let genresOrder = [];  // for chart hover
let genresValues = [];

function logLine(msg){
  if(!els.debugMode.checked) return;
  els.log.textContent += msg + '\n';
}

// --- Drag & drop helpers ---
['dragenter','dragover'].forEach(evt=>{
  els.reviewsDrop.addEventListener(evt, e=>{e.preventDefault(); e.dataTransfer.dropEffect='copy'; }, false);
  els.itemsDrop.addEventListener(evt, e=>{e.preventDefault(); e.dataTransfer.dropEffect='copy'; }, false);
});
els.reviewsDrop.addEventListener('drop', e=>{
  e.preventDefault();
  if(e.dataTransfer.files && e.dataTransfer.files.length) {
    reviewsFile = e.dataTransfer.files[0];
    els.reviewsInfo.textContent = filePretty(reviewsFile);
  }
});
els.itemsDrop.addEventListener('drop', e=>{
  e.preventDefault();
  if(e.dataTransfer.files && e.dataTransfer.files.length) {
    itemsFile = e.dataTransfer.files[0];
    els.itemsInfo.textContent = filePretty(itemsFile);
  }
});
// Click to open
els.reviewsDrop.addEventListener('click', ()=> pickFile(f=>{ reviewsFile=f; els.reviewsInfo.textContent=filePretty(f); }));
els.itemsDrop.addEventListener('click', ()=> pickFile(f=>{ itemsFile=f; els.itemsInfo.textContent=filePretty(f); }));

function pickFile(cb){
  const inp = document.createElement('input');
  inp.type='file';
  inp.accept='.csv,text/csv';
  inp.onchange=()=>{ if(inp.files && inp.files[0]) cb(inp.files[0]); };
  inp.click();
}
function filePretty(f){
  const mb = (f.size/1024/1024).toFixed(1)+' MB';
  return `${f.name} — ${mb}`;
}

// --- Main buttons ---
els.resetBtn.addEventListener('click', ()=>{
  reviewsFile = null; itemsFile = null; aggregates=null;
  els.reviewsInfo.textContent = 'No file.';
  els.itemsInfo.textContent = 'No file.';
  els.status.textContent = 'Waiting for files...';
  els.log.textContent='';
  clearHeadline(); clearTopGames(); clearCanvas(els.genresChart); clearCanvas(els.timeChart);
  els.exportBtn.disabled = true;
});
els.runBtn.addEventListener('click', runEDA);
els.exportBtn.addEventListener('click', exportCSV);

// --- EDA core ---
async function runEDA(){
  if(!reviewsFile || !itemsFile){
    els.status.textContent = 'Please select both CSV files first.';
    return;
  }
  els.status.textContent = 'Parsing CSV files...';
  logLine('Parsing items metadata…');

  const maxItems = Math.max(100, +els.maxItems.value|0);
  const itemsMeta = await parseItemsCSV(itemsFile, maxItems);
  logLine(`Items parsed (metadata): ${itemsMeta.size}`);

  logLine('Parsing reviews (interactions)…');
  const maxRows = Math.max(1000, +els.maxRows.value|0);
  const interactions = await parseReviewsCSV(reviewsFile, maxRows);
  logLine(`Interactions parsed: ${interactions.length}`);

  els.status.textContent = 'Aggregating…';
  aggregates = computeAggregates(interactions, itemsMeta);

  // Render
  renderHeadline(aggregates);
  renderTopGames(aggregates);
  drawGenresBar(aggregates);
  drawTimeHistogram(aggregates);

  els.status.textContent = 'Done. You can export CSV.';
  els.exportBtn.disabled = false;
}

function exportCSV(){
  if(!aggregates){ return; }
  // Build two CSVs: genres and games
  const gRows = [['genre','count','positive','positive_share','avg_hours','unique_games']];
  for(const g of aggregates.genresTopOrder){
    const s = aggregates.genreStats[g];
    gRows.push([g, s.count, s.pos, (s.pos/s.count||0).toFixed(3), (s.hours/s.count||0).toFixed(2), s.games.size]);
  }
  const gCsv = toCSV(gRows);

  const tRows = [['title','count','positive_share','avg_hours']];
  for(const t of aggregates.topGamesOrder){
    const s = aggregates.gameStats[t];
    tRows.push([t, s.count, (s.pos/s.count||0).toFixed(3), (s.hours/s.count||0).toFixed(2)]);
  }
  const tCsv = toCSV(tRows);

  downloadBlob(new Blob([gCsv], {type:'text/csv'}), 'aggregated_genres.csv');
  downloadBlob(new Blob([tCsv], {type:'text/csv'}), 'aggregated_games.csv');
}

function toCSV(rows){
  return rows.map(r => r.map(x => {
    if(x==null) return '';
    const s = String(x);
    return (s.includes(',') || s.includes('"') || s.includes('\n')) ? '"' + s.replace(/"/g,'""') + '"' : s;
  }).join(',')).join('\n');
}

function downloadBlob(blob, filename){
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

// --- CSV parsers (Papa) ---
async function parseItemsCSV(file, maxItems){
  return new Promise((resolve, reject)=>{
    const meta = new Map(); // nameLower -> {genres:Set, developer, publisher, release_date}
    let count=0;
    Papa.parse(file, {
      header:true,
      dynamicTyping:false,
      worker:true,
      step: (results, parser)=>{
        const row = results.data;
        if(!row || Object.keys(row).length===0) return;
        const name = (row.name || row.title || row.game_name || '').trim();
        if(!name){ return; }
        const key = name.toLowerCase();
        const genresRaw = (row.genres || '').toString();
        const genres = parseGenresCell(genresRaw);
        meta.set(key, {
          genres: new Set(genres),
          developer: row.developer || '',
          publisher: row.publisher || '',
          release_date: row.release_date || ''
        });
        count++; 
        if(count>=maxItems){ parser.abort(); }
      },
      complete: ()=> resolve(meta),
      error: err=> reject(err)
    });
  });
}

async function parseReviewsCSV(file, maxRows){
  return new Promise((resolve, reject)=>{
    const rows = [];
    Papa.parse(file, {
      header:true,
      dynamicTyping:false,
      worker:true,
      step: (results, parser)=>{
        const r = results.data;
        if(!r || Object.keys(r).length===0) return;
        const user = (r.username || r.user || r.user_id || '').toString().trim();
        const title = (r.game_name || r.game || r.title || '').toString().trim();
        if(!user || !title){ return; }

        const recRaw = (r.recommendation || r.recommended || r.voted_up || '').toString().toLowerCase();
        const recommended = (recRaw==='recommended' || recRaw==='true' || recRaw==='1');
        const hours = parseFloat(r.hours_played || r.playtime || r.hours || r.value || '0') || 0;
        const d = (r.date || r.posted || '').toString().trim();
        const month = d ? normalizeMonth(d) : null;

        rows.push({user, title, recommended, hours, month});
        if(rows.length>=maxRows){ parser.abort(); }
      },
      complete: ()=> resolve(rows),
      error: err=> reject(err)
    });
  });
}

function parseGenresCell(s){
  if(!s) return [];
  let txt = String(s).trim();
  // Try JSON-like list: ['Action','RPG']
  if((txt.startsWith('[') && txt.endsWith(']')) || (txt.includes("'") && txt.includes(","))){
    try {
      txt = txt.replace(/'/g,'"');
      const arr = JSON.parse(txt);
      return Array.isArray(arr) ? arr.map(v=>String(v).trim()).filter(Boolean) : [];
    } catch(e){ /* fallthrough */ }
  }
  // Fallback: split by comma
  return txt.split(/[,;|]/g).map(v=>v.trim()).filter(Boolean);
}

// Month normalize: '14 September'/'2013-08-01' → 'YYYY-MM'
function normalizeMonth(s){
  // if already iso-like
  const m = s.match(/(\d{4})[-/](\d{1,2})/);
  if(m){ return `${m[1]}-${String(m[2]).padStart(2,'0')}`; }
  // handle '14 September 2011.' or '14 September'
  const d = new Date(s.replace(/\.$/, ''));
  if(!isNaN(d.valueOf())){
    return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}`;
  }
  return null;
}

// --- Aggregations ---
function computeAggregates(interactions, itemsMeta){
  const users = new Set();
  const games = new Set();

  const genreStats = {}; // name -> {count,pos,hours,games:Set}
  const gameStats = {};  // title -> {count,pos,hours}
  const monthHist = {};  // 'YYYY-MM' -> count

  let pos=0, sumHours=0, withGenres=0;

  for(const r of interactions){
    users.add(r.user);
    games.add(r.title);
    if(r.recommended) pos++;
    sumHours += r.hours;

    // per-month
    if(r.month){
      monthHist[r.month] = (monthHist[r.month]||0)+1;
    }

    // per-game
    gameStats[r.title] = gameStats[r.title] || {count:0,pos:0,hours:0};
    const gs = gameStats[r.title];
    gs.count++; gs.hours += r.hours; if(r.recommended) gs.pos++;

    // genres (via metadata)
    const meta = itemsMeta.get(r.title.toLowerCase());
    if(meta && meta.genres && meta.genres.size){
      withGenres++;
      for(const g of meta.genres){
        if(!genreStats[g]) genreStats[g]={count:0,pos:0,hours:0,games:new Set()};
        const s = genreStats[g];
        s.count++; s.hours+=r.hours; if(r.recommended) s.pos++; s.games.add(r.title);
      }
    }
  }

  // orderings
  const genresTopOrder = Object.keys(genreStats).sort((a,b)=> genreStats[b].count - genreStats[a].count).slice(0,20);
  const topGamesOrder = Object.keys(gameStats).sort((a,b)=> gameStats[b].count - gameStats[a].count).slice(0,20);
  const monthsOrder = Object.keys(monthHist).sort();

  return {
    total: interactions.length,
    uniqueUsers: users.size,
    uniqueItems: games.size,
    positiveShare: (pos/Math.max(1,interactions.length)),
    avgHours: (sumHours/Math.max(1,interactions.length)),
    withGenresShare: (withGenres/Math.max(1,interactions.length)),
    itemsParsed: itemsMeta.size,
    genreStats, gameStats, monthHist,
    genresTopOrder, topGamesOrder, monthsOrder
  };
}

// --- Renderers ---
function clearHeadline(){ els.headlineTbl.innerHTML=''; }
function renderHeadline(a){
  const rows = [
    ['Total interactions', a.total.toLocaleString()],
    ['Unique users', a.uniqueUsers.toLocaleString()],
    ['Unique games', a.uniqueItems.toLocaleString()],
    ['Positive share (Recommended)', (a.positiveShare*100).toFixed(1)+'%'],
    ['Avg playtime (hours, where present)', a.avgHours.toFixed(2)],
    ['Coverage with genres (by interactions)', (a.withGenresShare*100).toFixed(1)+'%'],
    ['Items parsed (metadata)', a.itemsParsed.toLocaleString()],
  ];
  els.headlineTbl.innerHTML = rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');
}
function clearTopGames(){ els.topGamesTbl.innerHTML=''; }
function renderTopGames(a){
  const rows = a.topGamesOrder.map((t,idx)=>{
    const s = a.gameStats[t];
    const posShare = (s.pos/s.count)||0;
    return `<tr><td>${idx+1}</td><td>${escapeHtml(t)}</td><td>${s.count.toLocaleString()}</td><td>${(posShare*100).toFixed(1)}%</td></tr>`;
  }).join('');
  els.topGamesTbl.innerHTML = rows;
}

// --- Canvas Charts ---
function clearCanvas(c){ const ctx=c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height); }

function drawGenresBar(a){
  const c = els.genresChart;
  const ctx = c.getContext('2d');
  clearCanvas(c);

  const labels = a.genresTopOrder;
  const values = labels.map(g=> a.genreStats[g].count);
  genresOrder = labels; genresValues = values;

  const W=c.width, H=c.height, pad=40;
  const max = Math.max(...values,1);
  // axes
  ctx.strokeStyle='#cbd5e1'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,10); ctx.lineTo(pad,H-pad); ctx.lineTo(W-10,H-pad); ctx.stroke();

  const barW = (W-pad-20)/values.length;
  ctx.fillStyle='#60a5fa';
  values.forEach((v,i)=>{
    const h = (H-pad-20)* (v/max);
    const x = pad + i*barW + 4;
    const y = H-pad-h;
    ctx.fillRect(x, y, barW-8, h);
  });

  // x labels (rotated)
  ctx.save();
  ctx.translate(0,0);
  ctx.fillStyle='#475569'; ctx.font='11px System-ui,Segoe UI,Arial';
  values.forEach((_,i)=>{
    const x = pad + i*barW + 4 + (barW-8)/2;
    const y = H-6;
    ctx.save();
    ctx.translate(x,y);
    ctx.rotate(-Math.PI/3);
    const txt = labels[i].length>12 ? labels[i].slice(0,12)+'…' : labels[i];
    ctx.fillText(txt,0,0);
    ctx.restore();
  });
  ctx.restore();
}

function drawTimeHistogram(a){
  const c = els.timeChart;
  const ctx = c.getContext('2d');
  clearCanvas(c);

  const labels = a.monthsOrder;
  const values = labels.map(m=> a.monthHist[m]);
  const W=c.width, H=c.height, pad=40;
  const max = Math.max(...values,1);
  ctx.strokeStyle='#cbd5e1'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,10); ctx.lineTo(pad,H-pad); ctx.lineTo(W-10,H-pad); ctx.stroke();

  const step = (W-pad-20)/Math.max(values.length,1);
  ctx.fillStyle='#34d399';
  values.forEach((v,i)=>{
    const h = (H-pad-20)*(v/max);
    const x = pad + i*step + 1;
    const y = H-pad-h;
    ctx.fillRect(x, y, Math.max(1, step-2), h);
  });

  // sparse x labels
  ctx.fillStyle='#475569'; ctx.font='11px System-ui,Segoe UI,Arial';
  const every = Math.ceil(values.length/10);
  labels.forEach((m,i)=>{
    if(i%every!==0) return;
    const x = pad + i*step + 2;
    const y = H-6;
    ctx.fillText(m, x, y);
  });
}

// --- Small helpers ---
function escapeHtml(s){ return s.replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
