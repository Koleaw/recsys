/* app.js — Hybrid demo: Two-Tower (Deep + genres) and MF baseline, client-side TF.js. */

const state = {
  interactions: [],
  items: new Map(), // itemId -> {title, year, genres: Float32Array(18)}
  usersSet: new Set(),
  itemsSet: new Set(),
  u2idx: new Map(), i2idx: new Map(),
  idx2u: [], idx2i: [],
  userRated: new Map(), userTopRated: new Map(),
  model: null, modelType: 'two',
  optimizer: null,
  config: { maxInteractions: 80000, embDim: 32, batchSize: 256, epochs: 8, lr: 0.005, lossKind: 'softmax' },
  lossHistory: [], projPoints: [], itemFeatMat: null,
};

const $ = (id) => document.getElementById(id);
function setStatus(msg){ $('status').textContent = msg; }
function setKPI(entries){ $('kpi').innerHTML = (entries||[]).map(([k,v])=>`<span class="pill">${k}: ${v}</span>`).join(''); }
function getCfg(){
  state.modelType = $('cfg-mode').value;
  state.config.maxInteractions = +$('cfg-max').value || 80000;
  state.config.embDim = +$('cfg-emb').value || 32;
  state.config.batchSize = +$('cfg-batch').value || 256;
  state.config.epochs = +$('cfg-epochs').value || 8;
  state.config.lr = +$('cfg-lr').value || 0.005;
  state.config.lossKind = $('cfg-loss').value;
}

function drawLossChart(){
  const cvs = $('lossChart'); const ctx = cvs.getContext('2d');
  ctx.clearRect(0,0,cvs.width,cvs.height);
  const data = state.lossHistory; if (!data.length) return;
  const pad=30, W=cvs.width-pad*2, H=cvs.height-pad*2;
  const min = Math.min(...data), max = Math.max(...data);
  ctx.strokeStyle='#94a3b8'; ctx.strokeRect(pad,pad,W,H);
  ctx.beginPath();
  data.forEach((v,i)=>{ const x=pad+(i/(data.length-1))*W; const y=pad+(1-(v-min)/(max-min+1e-9))*H; if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y); });
  ctx.strokeStyle='#2563eb'; ctx.lineWidth=2; ctx.stroke();
  ctx.fillStyle='#475569'; ctx.font='12px system-ui';
  ctx.fillText(`loss: min=${min.toFixed(4)}, max=${max.toFixed(4)}`, pad, pad-8);
}

function drawProjection(){
  const cvs=$('projChart'); const ctx=cvs.getContext('2d');
  ctx.clearRect(0,0,cvs.width,cvs.height);
  const pts=state.projPoints; if(!pts.length) return;
  const pad=20; let minX=1e9,maxX=-1e9,minY=1e9,maxY=-1e9;
  for(const p of pts){minX=Math.min(minX,p.x);maxX=Math.max(maxX,p.x);minY=Math.min(minY,p.y);maxY=Math.max(maxY,p.y);}
  const sx=(x)=>pad+(x-minX)/(maxX-minX+1e-9)*(cvs.width-pad*2);
  const sy=(y)=>pad+(1-(y-minY)/(maxY-minY+1e-9))*(cvs.height-pad*2);
  ctx.fillStyle='#0ea5e9'; for(const p of pts){ctx.beginPath();ctx.arc(sx(p.x),sy(p.y),2.5,0,Math.PI*2);ctx.fill();}
  cvs.onmousemove=(e)=>{const r=cvs.getBoundingClientRect();const mx=(e.clientX-r.left)*cvs.width/r.width;const my=(e.clientY-r.top)*cvs.height/r.height;let best=null,d=12;for(const p of pts){const dd=Math.hypot(mx-sx(p.x),my-sy(p.y));if(dd<d){d=dd;best=p;}}cvs.title=best?best.title:'';};
}

async function loadData(){
  setStatus('Loading data…');
  const [uDataResp, uItemResp] = await Promise.all([
    fetch('data/u.data',{cache:'no-store'}), fetch('data/u.item',{cache:'no-store'})
  ]);
  if(!uDataResp.ok) throw new Error('Failed to load data/u.data');
  if(!uItemResp.ok) throw new Error('Failed to load data/u.item');
  const [uDataText,uItemText]=await Promise.all([uDataResp.text(),uItemResp.text()]);
  parseItem(uItemText); parseData(uDataText); indexEntities(); buildItemFeatureMatrix(); precomputeUserLists();
  setStatus(`Loaded: interactions=${state.interactions.length}, users=${state.usersSet.size}, items=${state.itemsSet.size}`);
  $('btn-train').disabled=false; $('btn-test').disabled=false;
}

function parseItem(text){
  const lines=text.split(/\r?\n/);
  for(const line of lines){
    if(!line.trim()) continue;
    const parts=line.split('|'); const itemId=parseInt(parts[0],10); const title=parts[1]||`Item ${itemId}`;
    let year=null; const m=title.match(/\((\d{4})\)\s*$/); if(m) year=parseInt(m[1],10);
    const start=parts.length-19; const genres=new Float32Array(18);
    if(start>=2){ for(let j=1;j<19;j++){ const flag=parts[start+j]; if(flag==='1') genres[j-1]=1.0; } }
    state.items.set(itemId,{title,year,genres}); state.itemsSet.add(itemId);
  }
}

function parseData(text){
  const lines=text.split(/\r?\n/); const maxN=state.config.maxInteractions;
  for(const line of lines){
    if(!line.trim()) continue;
    const [u,i,r,ts]=line.split('\t'); const userId=+u; const itemId=+i; const rating=+r; const tsNum=+ts;
    state.interactions.push({userId,itemId,rating,ts:tsNum}); state.usersSet.add(userId); state.itemsSet.add(itemId);
    if(state.interactions.length>=maxN) break;
  }
}

function indexEntities(){
  state.idx2u=[]; state.idx2i=[]; let idx=0;
  for(const u of state.usersSet){ state.u2idx.set(u,idx++); state.idx2u.push(u); }
  idx=0; for(const i of state.itemsSet){ state.i2idx.set(i,idx++); state.idx2i.push(i); }
}

function buildItemFeatureMatrix(){
  const I=state.itemsSet.size, F=18; const mat=new Float32Array(I*F);
  for(let i=0;i<I;i++){ const raw=state.idx2i[i]; const g=state.items.get(raw)?.genres||new Float32Array(F); mat.set(g,i*F); }
  state.itemFeatMat=mat;
}

function precomputeUserLists(){
  state.userRated.clear(); state.userTopRated.clear();
  const byUser=new Map();
  for(const it of state.interactions){ if(!byUser.has(it.userId)) byUser.set(it.userId,[]); byUser.get(it.userId).push(it); }
  for(const [u,arr] of byUser.entries()){
    const s=new Set(arr.map(x=>x.itemId)); state.userRated.set(u,s);
    const sorted=arr.slice().sort((a,b)=>(b.rating-a.rating)||(b.ts-a.ts)); state.userTopRated.set(u,sorted.map(x=>({itemId:x.itemId,rating:x.rating,ts:x.ts})));
  }
}

async function train(){
  getCfg(); state.lossHistory=[]; drawLossChart();
  const U=state.usersSet.size, I=state.itemsSet.size, K=state.config.embDim, mode=state.modelType;
  if(mode==='mf'){
    setStatus('Training MF (baseline)…'); const mf=new MFModel(U,I,K,state.config.lr); await trainMF(mf); state.model=mf; await computeAndDrawProjectionFromMF(); setKPI([['Model','MF']]); setStatus('MF trained. Use “Test”.'); return;
  }
  if(mode==='two'){
    setStatus(`Training Two-Tower (Deep+Genres, ${state.config.lossKind})…`);
    const two=new TwoTowerModel(U,I,K,state.config.lossKind,18,true,state.itemFeatMat);
    await trainTwoTower(two); state.model=two; await computeAndDrawProjectionFromTwoTower(); setKPI([['Model','Two-Tower (Deep+Genres)']]); setStatus('Two-Tower trained. Use “Test”.'); return;
  }
  // compare
  setStatus('Training MF (baseline) for comparison…'); const mf=new MFModel(U,I,K,state.config.lr); await trainMF(mf); const mfHR=await evalHitRateAt10(mf,'mf',60); setKPI([['MF HR@10',mfHR.toFixed(3)]]);
  setStatus(`Training Two-Tower (Deep+Genres, ${state.config.lossKind}) for comparison…`); const two=new TwoTowerModel(U,I,K,state.config.lossKind,18,true,state.itemFeatMat); await trainTwoTower(two); const twoHR=await evalHitRateAt10(two,'two',60); setKPI([['MF HR@10',mfHR.toFixed(3)],['Two HR@10',twoHR.toFixed(3)]]); state.model=two; state.modelType='two'; await computeAndDrawProjectionFromTwoTower(); setStatus('Comparison done. Current: Two-Tower. Use “Test”.');
}

// ----- MF baseline -----
class MFModel{
  constructor(numUsers,numItems,embDim=32,lr=0.005){
    this.numUsers=numUsers; this.numItems=numItems; this.embDim=embDim;
    this.userEmbedding=tf.variable(tf.randomNormal([numUsers,embDim],0,0.05));
    this.itemEmbedding=tf.variable(tf.randomNormal([numItems,embDim],0,0.05));
    this.userBias=tf.variable(tf.zeros([numUsers,1]));
    this.itemBias=tf.variable(tf.zeros([numItems,1]));
    this.globalBias=tf.scalar(0);
    this.optimizer=tf.train.adam(lr);
  }
  predict(uIdx,iIdx){
    const U=tf.gather(this.userEmbedding,uIdx); const I=tf.gather(this.itemEmbedding,iIdx);
    const dot=tf.sum(tf.mul(U,I),-1,true); const ub=tf.gather(this.userBias,uIdx); const ib=tf.gather(this.itemBias,iIdx);
    return tf.addN([dot,ub,ib,this.globalBias]);
  }
  trainStep(uIdx,iIdx,y){
    return this.optimizer.minimize(()=>{
      const pred=this.predict(uIdx,iIdx); const diff=tf.sub(pred,y); return tf.mean(tf.square(diff));
    }, true);
  }
}

async function trainMF(model){
  const B=state.config.batchSize,E=state.config.epochs;
  const pairs=[], targets=[];
  for(const it of state.interactions){
    const u=state.u2idx.get(it.userId), i=state.i2idx.get(it.itemId); if(u===undefined||i===undefined) continue;
    pairs.push([u,i]); targets.push(it.rating);
  }
  function shuffle(a,b){ for(let i=a.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; [a[i],a[j]]=[a[j],a[i]]; [b[i],b[j]]=[b[j],b[i]]; } }
  for(let ep=0; ep<E; ep++){
    shuffle(pairs,targets); let sum=0,cnt=0;
    for(let k=0;k<pairs.length;k+=B){
      const s=k,e=Math.min(pairs.length,k+B);
      const uIdx=tf.tensor1d(pairs.slice(s,e).map(p=>p[0]),'int32');
      const iIdx=tf.tensor1d(pairs.slice(s,e).map(p=>p[1]),'int32');
      const y=tf.tensor2d(targets.slice(s,e).map(v=>[v]),[e-s,1],'float32');
      const lossTensor=model.trainStep(uIdx,iIdx,y); const l=lossTensor.dataSync()[0];
      lossTensor.dispose(); uIdx.dispose(); iIdx.dispose(); y.dispose();
      state.lossHistory.push(Number.isFinite(l)?l:(state.lossHistory.at(-1)||0)); if(Number.isFinite(l)) { sum+=l; } cnt++;
      if(cnt%4===0) drawLossChart(); if(cnt%30===0) await tf.nextFrame();
    }
    drawLossChart(); setStatus(`MF Epoch ${ep+1}/${E} — avg loss ${(sum/Math.max(1,cnt)).toFixed(4)}`); await tf.nextFrame();
  }
}

async function computeAndDrawProjectionFromMF(){
  const I=state.itemsSet.size, K=state.config.embDim; const emb=state.model.itemEmbedding; await computeAndDrawProjectionGeneric(emb,I,K);
}

// ----- Two-Tower -----
async function trainTwoTower(model){
  const B=state.config.batchSize,E=state.config.epochs;
  const pairs=[]; for(const it of state.interactions){ const u=state.u2idx.get(it.userId), i=state.i2idx.get(it.itemId); if(u===undefined||i===undefined) continue; pairs.push([u,i]); }
  function shuffle(a){ for(let i=a.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; [a[i],a[j]]=[a[j],a[i]]; } }
  for(let ep=0; ep<E; ep++){
    shuffle(pairs); let sum=0,cnt=0;
    for(let k=0;k<pairs.length;k+=B){
      const s=k,e=Math.min(pairs.length,k+B);
      const uIdx=tf.tensor1d(pairs.slice(s,e).map(p=>p[0]),'int32');
      const posIdx=tf.tensor1d(pairs.slice(s,e).map(p=>p[1]),'int32');
      let negIdx=null;
      if(state.config.lossKind==='bpr'){
        const itemCount=state.itemsSet.size; const neg=new Int32Array(e-s); for(let t=0;t<neg.length;t++){ neg[t]=(Math.random()*itemCount)|0; } negIdx=tf.tensor1d(neg,'int32');
      }
      const lossTensor=model.trainStep(uIdx,posIdx,model.optimizer,negIdx); const l=lossTensor.dataSync()[0];
      lossTensor.dispose(); uIdx.dispose(); posIdx.dispose(); if(negIdx) negIdx.dispose();
      state.lossHistory.push(l); sum+=l; cnt++;
      if(cnt%4===0) drawLossChart(); if(cnt%30===0) await tf.nextFrame();
    }
    drawLossChart(); setStatus(`Two-Tower Epoch ${ep+1}/${E} — avg loss ${(sum/Math.max(1,cnt)).toFixed(4)}`); await tf.nextFrame();
  }
}

async function computeAndDrawProjectionFromTwoTower(){
  const I=state.itemsSet.size, K=state.config.embDim; const all=await state.model.computeAllItemVectors(); await computeAndDrawProjectionGeneric(all,I,K,800); all.dispose();
}

async function computeAndDrawProjectionGeneric(embTensor,I,K,maxItems=800){
  const take=Math.min(I,maxItems); const idxs=Array.from({length:I},(_,i)=>i); for(let i=idxs.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; [idxs[i],idxs[j]]=[idxs[j],idxs[i]]; }
  const pick=idxs.slice(0,take); const emb=tf.gather(embTensor, tf.tensor1d(pick,'int32')); const mean=emb.mean(0); const X=emb.sub(mean); const C=tf.matMul(X.transpose(),X);
  function power(mat,dim,iters=30){ let v=tf.randomNormal([dim,1]); for(let t=0;t<iters;t++){ v=tf.tidy(()=>{ const mv=mat.matMul(v); const n=tf.norm(mv).add(1e-8); return mv.div(n); }); } return v; }
  const v1=power(C,K,40); const v2t=power(C,K,45); const proj=tf.matMul(v1.transpose(),v2t).div(tf.matMul(v1.transpose(),v1).add(1e-8)); const v2=tf.tidy(()=>v2t.sub(v1.mul(proj)));
  const comp1=tf.matMul(X,v1).reshape([take]); const comp2=tf.matMul(X,v2).reshape([take]); const xs=comp1.arraySync(), ys=comp2.arraySync(); const pts=[];
  for(let n=0;n<take;n++){ const raw=state.idx2i[pick[n]]; const title=state.items.get(raw)?.title||`Item ${raw}`; pts.push({x:xs[n],y:ys[n],title}); }
  state.projPoints=pts; drawProjection(); emb.dispose(); mean.dispose(); X.dispose(); C.dispose(); v1.dispose(); v2t.dispose(); v2.dispose(); comp1.dispose(); comp2.dispose(); await tf.nextFrame();
}

async function evalHitRateAt10(model,type,sampleUsers=60){
  const users=[]; for(const u of state.usersSet){ const arr=state.userTopRated.get(u)||[]; if(arr.length>=20) users.push(u); }
  if(!users.length) return 0; for(let i=users.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; [users[i],users[j]]=[users[j],users[i]]; }
  const take=Math.min(sampleUsers,users.length); let hits=0;
  for(let k=0;k<take;k++){ const u=users[k]; const arr=state.userTopRated.get(u); const hold=arr[0]; const seen=new Set(arr.map(x=>x.itemId)); seen.delete(hold.itemId); const recs=await recommendTopN(model,type,u,10,seen); if(recs.includes(hold.itemId)) hits++; if(k%8===0) await tf.nextFrame(); }
  return hits/take;
}

async function recommendTopN(model,type,userId,N=10,excludeSet=new Set()){
  const uIdx=state.u2idx.get(userId); if(uIdx===undefined) return [];
  if(type==='mf'){
    const userEmb=tf.gather(model.userEmbedding, tf.tensor1d([uIdx],'int32'));
    const I=state.itemsSet.size, chunk=4096; let scores=[];
    for(let s=0;s<I;s+=chunk){
      const e=Math.min(I,s+chunk);
      const slice=tf.slice(model.itemEmbedding,[s,0],[e-s,model.embDim]);
      const sc=tf.matMul(slice,userEmb.transpose()); const ub=tf.gather(model.userBias, tf.tensor1d([uIdx],'int32')); const ib=tf.slice(model.itemBias,[s,0],[e-s,1]); const sum=tf.addN([sc,ib,ub]);
      const arr=Array.from(await sum.data()); scores.push(...arr.map((v,idx)=>({idx:s+idx,v}))); slice.dispose(); sc.dispose(); ub.dispose(); ib.dispose(); sum.dispose(); await tf.nextFrame();
    }
    userEmb.dispose(); scores.sort((a,b)=>b.v-a.v); const out=[]; for(const o of scores){ const raw=state.idx2i[o.idx]; if(!excludeSet.has(raw)){ out.push(raw); if(out.length>=N) break; } } return out;
  } else {
    const uEmb=state.model.getUserEmbedding(tf.tensor1d([uIdx],'int32')); const topIdx=await state.model.getTopKForUser(uEmb,Math.max(200,N*20)); uEmb.dispose(); const out=[]; for(const idx of topIdx){ const raw=state.idx2i[idx]; if(!excludeSet.has(raw)){ out.push(raw); if(out.length>=N) break; } } return out;
  }
}

function pickQualifiedUser(minCount=20){ const c=[]; for(const u of state.usersSet){ const r=state.userTopRated.get(u)||[]; if(r.length>=minCount) c.push(u);} if(!c.length) return null; return c[(Math.random()*c.length)|0]; }

async function testOnce(){
  if(!state.model){ setStatus('Train first.'); return; }
  const userId=pickQualifiedUser(20); if(!userId){ setStatus('No user with ≥20 ratings in subset. Increase max interactions.'); return; }
  const ratedList=state.userTopRated.get(userId); const topRated=ratedList.slice(0,10);
  const seen=new Set(ratedList.map(x=>x.itemId)); const recs=await recommendTopN(state.model,state.modelType,userId,10,seen);
  renderSideBySide(userId, topRated, recs);
}

function renderSideBySide(userId, topRated, recItemIds){
  const leftRows=topRated.map((r,i)=>{ const title=state.items.get(r.itemId)?.title||`Item ${r.itemId}`; return `<tr><td>${i+1}</td><td>${title}</td><td>${r.rating}</td></tr>`; }).join('');
  const rightRows=recItemIds.map((iid,i)=>{ const t=state.items.get(iid)?.title||`Item ${iid}`; return `<tr><td>${i+1}</td><td>${t}</td></tr>`; }).join('');
  $('results').innerHTML=`<div class="two-col"><div><table><thead><tr><th colspan="3">User ${userId} — Top-10 Rated (by rating, then recency)</th></tr><tr><th>#</th><th>Title</th><th>Rating</th></tr></thead><tbody>${leftRows}</tbody></table></div><div><table><thead><tr><th colspan="2">Model Top-10 Recommended (seen excluded)</th></tr><tr><th>#</th><th>Title</th></tr></thead><tbody>${rightRows}</tbody></table></div></div>`;
}

window.addEventListener('load', ()=>{
  $('btn-load').addEventListener('click', async ()=>{
    try{ getCfg(); $('btn-load').disabled=true; await loadData(); setStatus('Data ready. You can Train or Test.'); setKPI([]); }catch(e){ console.error(e); setStatus(`Load failed: ${e && e.message ? e.message : String(e)}`);} finally{$('btn-load').disabled=false;}
  });
  $('btn-train').addEventListener('click', async ()=>{
    try{ getCfg(); $('btn-train').disabled=true; $('btn-test').disabled=true; await train(); }catch(e){ console.error(e); setStatus(`Training failed: ${e && e.message ? e.message : String(e)}`);} finally{$('btn-train').disabled=false; $('btn-test').disabled=false;}
  });
  $('btn-test').addEventListener('click', async ()=>{
    try{ getCfg(); await testOnce(); }catch(e){ console.error(e); setStatus(`Test failed: ${e && e.message ? e.message : String(e)}`);} 
  });
});
