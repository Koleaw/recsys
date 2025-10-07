/* two-tower.js — Two-Tower retrieval model in TensorFlow.js.
   Supports:
   - User tower: user_id -> embedding
   - Item tower: item_id -> embedding, optional Deep MLP with genres features (18-dim)
   - Scoring: dot product
   - Loss: in-batch softmax (default) or BPR pairwise
   - Compute all item vectors (for projection) and Top-K for a user
*/

class TwoTowerModel {
  constructor(numUsers, numItems, embDim=32, lossKind='softmax', itemFeatDim=0, deep=true, itemFeatMatFloat32=null){
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;
    this.lossKind = lossKind;
    this.itemFeatDim = itemFeatDim;
    this.deep = deep;

    // Trainable embeddings
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05), true, 'userEmbedding');
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05), true, 'itemEmbedding');

    // Item features matrix [I, F] (non-trainable tensor) — multi-hot genres
    this.itemFeat = null;
    if (itemFeatDim > 0 && itemFeatMatFloat32){
      const I = numItems;
      this.itemFeat = tf.tensor2d(itemFeatMatFloat32, [I, itemFeatDim], 'float32');
    }

    // If deep tower enabled and features exist — define MLP weights
    if (this.deep && this.itemFeatDim > 0){
      const inDim = this.embDim + this.itemFeatDim;
      const hid = this.embDim; // 1 hidden layer of size K
      this.W1 = tf.variable(tf.randomNormal([inDim, hid], 0, 0.05), true, 'W1');
      this.b1 = tf.variable(tf.zeros([hid]), true, 'b1');
      this.Wo = tf.variable(tf.randomNormal([hid, this.embDim], 0, 0.05), true, 'Wo');
      this.bo = tf.variable(tf.zeros([this.embDim]), true, 'bo');
    }

    this.optimizer = tf.train.adam(0.005);
  }

  // Gather rows by integer indices
  getUserEmbedding(userIdxTensor){ // [B] or [B,1] -> [B,K]
    const flat = userIdxTensor.reshape([-1]);
    return tf.gather(this.userEmbedding, flat);
  }
  getItemBaseEmbedding(itemIdxTensor){ // [B] or [B,1] -> [B,K]
    const flat = itemIdxTensor.reshape([-1]);
    return tf.gather(this.itemEmbedding, flat);
  }
  getItemFeatBatch(itemIdxTensor){ // -> [B,F]
    if (!this.itemFeat) return null;
    const flat = itemIdxTensor.reshape([-1]);
    return tf.gather(this.itemFeat, flat);
  }

  // Item tower forward: either plain embedding or deep with genres features
  itemForward(itemIdxTensor){
    const E = this.getItemBaseEmbedding(itemIdxTensor); // [B,K]
    if (!(this.deep && this.itemFeat)){
      return E;
    }
    const G = this.getItemFeatBatch(itemIdxTensor); // [B,F]
    const X = tf.concat([E, G], -1);                // [B,K+F]
    const H = tf.relu(tf.add(tf.matMul(X, this.W1), this.b1));    // [B,K]
    const O = tf.add(tf.matMul(H, this.Wo), this.bo);             // [B,K]
    return O;
  }

  // Dot product score: sum over K -> [B,1]
  score(uEmb, iEmb){
    return tf.sum(tf.mul(uEmb, iEmb), -1, true);
  }

  // In-batch sampled softmax loss
  softmaxLoss(uIdx, posIdx){
    const U = this.getUserEmbedding(uIdx);     // [B,K]
    const Ipos = this.itemForward(posIdx);     // [B,K] (deep or not)
    const logits = tf.matMul(U, Ipos, false, true); // [B,B]
    // labels: diagonal
    const labels = tf.tensor1d(Array(U.shape[0]).fill(0).map((_,i)=>i), 'int32');
    const ce = tf.losses.softmaxCrossEntropy(tf.oneHot(labels, U.shape[0]), logits);
    return ce;
  }

  // BPR pairwise loss
  bprLoss(uIdx, posIdx, negIdx){
    const U = this.getUserEmbedding(uIdx);         // [B,K]
    const Ipos = this.itemForward(posIdx);         // [B,K]
    const Ineg = this.itemForward(negIdx);         // [B,K]
    const sPos = this.score(U, Ipos);              // [B,1]
    const sNeg = this.score(U, Ineg);              // [B,1]
    const diff = tf.sub(sPos, sNeg);
    const prob = tf.sigmoid(diff);
    const loglik = tf.log(tf.add(prob, 1e-8));
    const loss = tf.neg(tf.mean(loglik));
    return loss;
  }

  // One training step; returns scalar loss tensor
  trainStep(uIdx, posIdx, optimizer, negIdx=null){
    const opt = optimizer || this.optimizer;
    return opt.minimize(()=>{
      if (this.lossKind === 'bpr'){
        if (!negIdx) throw new Error('BPR requires negIdx');
        return this.bprLoss(uIdx, posIdx, negIdx);
      } else {
        return this.softmaxLoss(uIdx, posIdx);
      }
    }, true);
  }

  // Compute Top-K items for a user embedding [1,K] (batched matmul). Returns indices.
  async getTopKForUser(userEmb_1xK, K=200){
    const I = this.numItems;
    const chunk = 2048;
    let scores = [];
    for (let s=0; s<I; s+=chunk){
      const e = Math.min(I, s+chunk);
      const idx = tf.range(s, e, 1, 'int32');              // [(e-s)]
      const sliceVecs = this.itemForward(idx);             // [(e-s),K] deep forward
      const sc = tf.matMul(sliceVecs, userEmb_1xK.transpose()); // [(e-s),1]
      const arr = Array.from((await sc.data()));
      scores.push(...arr.map((v, ii)=> ({idx: s+ii, v}) ));
      idx.dispose(); sliceVecs.dispose(); sc.dispose();
      await tf.nextFrame();
    }
    scores.sort((a,b)=> b.v - a.v);
    return scores.slice(0, K).map(x=>x.idx);
  }

  // Build full item vectors through itemForward for all items (for projection)
  async computeAllItemVectors(){
    const I = this.numItems;
    const out = [];
    const chunk = 2048;
    for (let s=0; s<I; s+=chunk){
      const e = Math.min(I, s+chunk);
      const idx = tf.range(s, e, 1, 'int32');
      const vec = this.itemForward(idx); // [(e-s),K]
      out.push(vec);
      idx.dispose();
      await tf.nextFrame();
    }
    const all = tf.concat(out, 0); // [I,K]
    out.forEach(t=>t.dispose());
    return all;
  }
}

window.TwoTowerModel = TwoTowerModel;
