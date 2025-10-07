/* two-tower.js — Two-Tower retrieval model (Deep + genres support) */

class TwoTowerModel {
  constructor(numUsers, numItems, embDim=32, lossKind='softmax', itemFeatDim=0, deep=true, itemFeatMatFloat32=null){
    this.numUsers = numUsers; this.numItems = numItems; this.embDim = embDim;
    this.lossKind = lossKind; this.itemFeatDim = itemFeatDim; this.deep = deep;
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05), true, 'userEmbedding');
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05), true, 'itemEmbedding');
    this.itemFeat = null;
    if (itemFeatDim > 0 && itemFeatMatFloat32){
      this.itemFeat = tf.tensor2d(itemFeatMatFloat32, [numItems, itemFeatDim], 'float32');
    }
    if (this.deep && this.itemFeatDim > 0){
      const inDim = this.embDim + this.itemFeatDim, hid = this.embDim;
      this.W1 = tf.variable(tf.randomNormal([inDim, hid], 0, 0.05), true, 'W1');
      this.b1 = tf.variable(tf.zeros([hid]), true, 'b1');
      this.Wo = tf.variable(tf.randomNormal([hid, this.embDim], 0, 0.05), true, 'Wo');
      this.bo = tf.variable(tf.zeros([this.embDim]), true, 'bo');
    }
    this.optimizer = tf.train.adam(0.005);
  }
  getUserEmbedding(userIdxTensor){ const flat=userIdxTensor.reshape([-1]); return tf.gather(this.userEmbedding, flat); }
  getItemBaseEmbedding(itemIdxTensor){ const flat=itemIdxTensor.reshape([-1]); return tf.gather(this.itemEmbedding, flat); }
  getItemFeatBatch(itemIdxTensor){ if(!this.itemFeat) return null; const flat=itemIdxTensor.reshape([-1]); return tf.gather(this.itemFeat, flat); }
  itemForward(itemIdxTensor){
    const E=this.getItemBaseEmbedding(itemIdxTensor);
    if(!(this.deep && this.itemFeat)) return E;
    const G=this.getItemFeatBatch(itemIdxTensor);
    const X=tf.concat([E,G],-1);
    const H=tf.relu(tf.add(tf.matMul(X,this.W1),this.b1));
    const O=tf.add(tf.matMul(H,this.Wo),this.bo);
    return O;
  }
  score(uEmb,iEmb){ return tf.sum(tf.mul(uEmb,iEmb), -1, true); }
  softmaxLoss(uIdx,posIdx){
    const U=this.getUserEmbedding(uIdx), Ipos=this.itemForward(posIdx);
    const logits=tf.matMul(U,Ipos,false,true); // [B,B]
    const B=logits.shape[0]; const targetIdx=tf.tensor1d([...Array(B).keys()],'int32'); const y=tf.oneHot(targetIdx,B);
    const logSoft=tf.logSoftmax(logits,1); const mul=tf.mul(y,logSoft); const perEx=tf.neg(tf.sum(mul,1));
    return tf.mean(perEx);
  }
  bprLoss(uIdx,posIdx,negIdx){
    const U=this.getUserEmbedding(uIdx), Ipos=this.itemForward(posIdx), Ineg=this.itemForward(negIdx);
    const diff=tf.sub(this.score(U,Ipos), this.score(U,Ineg)); const prob=tf.sigmoid(diff);
    return tf.neg(tf.mean(tf.log(tf.add(prob,1e-8))));
  }
  trainStep(uIdx,posIdx,optimizer,negIdx=null){
    const opt=optimizer||this.optimizer;
    return opt.minimize(()=>{
      if(this.lossKind==='bpr'){ if(!negIdx) throw new Error('BPR requires negIdx'); return this.bprLoss(uIdx,posIdx,negIdx); }
      return this.softmaxLoss(uIdx,posIdx);
    }, true);
  }
  async getTopKForUser(userEmb_1xK,K=200){
    const I=this.numItems, chunk=2048; let scores=[];
    for(let s=0;s<I;s+=chunk){
      const e=Math.min(I,s+chunk); const idx=tf.range(s,e,1,'int32');
      const sliceVecs=this.itemForward(idx); const sc=tf.matMul(sliceVecs, userEmb_1xK.transpose());
      const arr=Array.from(await sc.data()); scores.push(...arr.map((v,ii)=>({idx:s+ii,v}))); idx.dispose(); sliceVecs.dispose(); sc.dispose(); await tf.nextFrame();
    }
    scores.sort((a,b)=>b.v-a.v); return scores.slice(0,K).map(x=>x.idx);
  }
  async computeAllItemVectors(){
    const I=this.numItems, out=[], chunk=2048;
    for(let s=0;s<I;s+=chunk){ const e=Math.min(I,s+chunk); const idx=tf.range(s,e,1,'int32'); const vec=this.itemForward(idx); out.push(vec); idx.dispose(); await tf.nextFrame(); }
    const all=tf.concat(out,0); out.forEach(t=>t.dispose()); return all;
  }
}
window.TwoTowerModel = TwoTowerModel;
