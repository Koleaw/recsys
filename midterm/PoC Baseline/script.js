// Minimal CSV parser (no external libs). Assumes first row is header.
function parseCSV(text) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (!lines.length) return {header:[], rows:[]};
  const header = splitCSVLine(lines[0]);
  const rows = [];
  for (let i=1;i<lines.length;i++){
    const parts = splitCSVLine(lines[i]);
    if (!parts.length) continue;
    const obj = {};
    for (let j=0;j<header.length;j++){
      obj[header[j]] = parts[j] ?? "";
    }
    rows.push(obj);
  }
  return {header, rows};
}
// handle simple quoted CSV fields
function splitCSVLine(line){
  const out=[], re = /,(?=(?:(?:[^"]*"){2})*[^"]*$)/g; // split by commas not inside quotes
  const parts = line.split(re);
  for(let p of parts){
    p = p.trim();
    if (p.startsWith('"') && p.endsWith('"')) p = p.slice(1,-1).replace(/""/g,'"');
    out.push(p);
  }
  return out;
}
function log(msg){ const el=document.getElementById('log'); el.textContent += msg + "\n"; }

// Data holders
let reviews = []; // {username, game_name, recommendation, hours_played, date}
let items = new Map(); // name -> {genres:[], raw:{}}
let users = new Map(); // username -> interactions []
let gameStats = new Map(); // name -> {pos,total,avgHours}
let userList = [];

const els = {
  fileRevs: document.getElementById('file-revs'),
  fileMeta: document.getElementById('file-meta'),
  dropRevs: document.getElementById('drop-revs'),
  dropMeta: document.getElementById('drop-meta'),
  revsStatus: document.getElementById('revs-status'),
  metaStatus: document.getElementById('meta-status'),
  minInteractions: document.getElementById('min-interactions'),
  alpha: document.getElementById('alpha'),
  beta: document.getElementById('beta'),
  buildBtn: document.getElementById('build'),
  buildStatus: document.getElementById('build-status'),
  userSelect: document.getElementById('user-select'),
  recommendBtn: document.getElementById('recommend'),
  histList: document.getElementById('hist-list'),
  recList: document.getElementById('rec-list'),
};

function hookupDrop(zone, input){
  zone.addEventListener('click', ()=> input.click());
  zone.addEventListener('dragover', e=>{ e.preventDefault(); zone.classList.add('drag'); });
  zone.addEventListener('dragleave', ()=> zone.classList.remove('drag'));
  zone.addEventListener('drop', e=>{
    e.preventDefault(); zone.classList.remove('drag');
    if (e.dataTransfer.files?.length) input.files = e.dataTransfer.files;
    input.dispatchEvent(new Event('change'));
  });
}
hookupDrop(els.dropRevs, els.fileRevs);
hookupDrop(els.dropMeta, els.fileMeta);

els.fileRevs.addEventListener('change', async (e)=>{
  if (!e.target.files?.length) return;
  const file = e.target.files[0];
  const text = await file.text();
  const {header, rows} = parseCSV(text);
  log("Reviews header: " + header.join(", "));
  // normalize keys
  const norm = (s)=> s.toLowerCase().replace(/[^a-z0-9_]/g,'').trim();
  const keyMap = {};
  for (const h of header){
    keyMap[norm(h)] = h;
  }
  function get(row, k){
    const h = keyMap[norm(k)];
    return h ? row[h] : "";
  }
  reviews = rows.map(r=> ({
    username: String(get(r,'username')||get(r,'user_name')||get(r,'user')).trim(),
    game_name: String(get(r,'game_name')||get(r,'name')||get(r,'title')).trim(),
    recommendation: String(get(r,'recommendation')||get(r,'recommended')).toLowerCase().includes('recommend'),
    hours_played: parseFloat(get(r,'hours_played')||get(r,'hours')||"0") || 0,
    date: get(r,'date')||""
  })).filter(r=> r.username && r.game_name);

  els.revsStatus.textContent = `Loaded ${reviews.length.toLocaleString()} interactions`;
});

els.fileMeta.addEventListener('change', async (e)=>{
  if (!e.target.files?.length) return;
  const file = e.target.files[0];
  const text = await file.text();
  const {header, rows} = parseCSV(text);
  log("Items header: " + header.join(", "));
  const norm = (s)=> s.toLowerCase().replace(/[^a-z0-9_]/g,'').trim();
  const keyMap = {};
  for (const h of header){ keyMap[norm(h)] = h; }
  function get(row, k){
    const h = keyMap[norm(k)];
    return h ? row[h] : "";
  }
  items.clear();
  for (const r of rows){
    const name = String(get(r,'name')||get(r,'game_name')||get(r,'title')).trim();
    if (!name) continue;
    let genresRaw = get(r,'genres')||"";
    let genres = [];
    try{
      // try JSON array format like "['Action','RPG']" or '["Action","RPG"]'
      const fix = genresRaw.replace(/'/g,'"');
      const arr = JSON.parse(fix);
      if (Array.isArray(arr)) genres = arr.map(s=> String(s).trim());
    }catch(e){
      // fallback: split by comma
      genres = genresRaw.split(/[,|]/).map(s=> s.trim()).filter(Boolean);
    }
    items.set(name, {genres, raw: r});
  }
  els.metaStatus.textContent = `Loaded ${items.size.toLocaleString()} items with genres`;
});

els.buildBtn.addEventListener('click', ()=>{
  // Build users map and game stats
  users.clear();
  gameStats.clear();

  for (const r of reviews){
    if (!users.has(r.username)) users.set(r.username, []);
    users.get(r.username).push(r);

    if (!gameStats.has(r.game_name)) gameStats.set(r.game_name, {pos:0,total:0,hoursSum:0});
    const gs = gameStats.get(r.game_name);
    gs.total += 1;
    if (r.recommendation) gs.pos += 1;
    gs.hoursSum += (r.hours_played||0);
  }
  // avg hours
  for (const [g, s] of gameStats){
    s.avgHours = s.total ? s.hoursSum/s.total : 0;
  }

  // populate user list
  const minInt = parseInt(els.minInteractions.value||"20");
  userList = Array.from(users.entries()).filter(([u,arr])=> arr.length>=minInt).map(([u])=>u).sort();
  els.userSelect.innerHTML = userList.map(u=> `<option value="${u}">${u}</option>`).join("");
  els.buildStatus.textContent = `Ready. Qualified users: ${userList.length.toLocaleString()}`;
});

els.recommendBtn.addEventListener('click', ()=>{
  const uname = els.userSelect.value;
  if (!uname){ alert("Pick a user"); return; }
  const arr = users.get(uname)||[];
  // Build user genre profile
  const genreCounts = new Map();
  for (const r of arr){
    const it = items.get(r.game_name);
    if (!it || !it.genres?.length) continue;
    const w = 1 + Math.log1p(Math.max(0, r.hours_played||0)) + (r.recommendation?0.5:0);
    for (const g of it.genres){
      genreCounts.set(g, (genreCounts.get(g)||0)+w);
    }
  }
  // normalize to vector
  const totalW = Array.from(genreCounts.values()).reduce((a,b)=>a+b,0) || 1;
  const profile = new Map();
  for (const [g,c] of genreCounts) profile.set(g, c/totalW);
  const seen = new Set(arr.map(r=> r.game_name));

  // scoring
  const alpha = parseFloat(els.alpha.value||"0.7");
  const beta = parseFloat(els.beta.value||"0.3");

  const scored = [];
  for (const [name, meta] of items){
    if (seen.has(name)) continue;
    if (!meta.genres?.length) continue;
    // cosine-like with profile (since profile sums to 1, dot product suffices)
    let genreScore = 0;
    for (const g of meta.genres){
      genreScore += profile.get(g)||0;
    }
    // popularity prior
    const s = gameStats.get(name) || {pos:0,total:0,avgHours:0};
    const posRate = s.total ? (s.pos/s.total) : 0;
    const pop = Math.log1p(s.total);
    const prior = posRate * pop;
    const score = alpha * genreScore + beta * prior;
    scored.push({name, score, genres: meta.genres, stats: s});
  }
  scored.sort((a,b)=> b.score - a.score);
  const top = scored.slice(0,10);

  // render history
  const sortedHist = [...arr].sort((a,b)=>{
    if (a.recommendation !== b.recommendation) return (b.recommendation?1:0) - (a.recommendation?1:0);
    return (b.hours_played||0) - (a.hours_played||0);
  }).slice(0,10);
  renderList(els.histList, sortedHist.map(h=> ({
    title: h.game_name,
    badges: [
      h.recommendation ? "Recommended" : "Not recommended",
      (h.hours_played||0) ? `Hours: ${Math.round(h.hours_played)}` : null
    ].filter(Boolean)
  })));

  // render recommendations + reasons
  const topGenres = Array.from(profile.entries()).sort((a,b)=> b[1]-a[1]).slice(0,3).map(([g])=>g);
  renderList(els.recList, top.map(t=> ({
    title: t.name,
    badges: [
      ...t.genres.filter(g=> topGenres.includes(g)).map(g=> `Because you like: ${g}`),
      `Positive share: ${t.stats.total? Math.round((t.stats.pos/t.stats.total)*100):0}%`,
      t.stats.avgHours>120 ? "Long sessions cohort" : null
    ].filter(Boolean)
  })));
});

function renderList(root, items){
  root.innerHTML = items.map(it=> `
    <div class="card-item">
      <h4>${escapeHTML(it.title)}</h4>
      <div class="badges">${it.badges.map(b=> `<span class="badge">${escapeHTML(b)}</span>`).join(" ")}</div>
    </div>
  `).join("");
}
function escapeHTML(s){ return String(s).replace(/[&<>"']/g, ch=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[ch])); }
