/* Playable world models — the actual trained network, in your browser.
 *
 * Nothing here is a hand-written game loop. You press a key, the exported
 * transformer autoregressively emits the next authoritative state, and that
 * state is rendered. If the model has the physics wrong, the demo has it
 * wrong — which is the honest behaviour for a world-model demo.
 *
 * The architecture folds to `tok[x] + pos[p] + kind0` plus six pre-norm
 * blocks (LayerNorm, biasless QKV, causal attention, biasless proj,
 * LayerNorm, fc1, GELU, fc2), a final LayerNorm and a biasless head. No RoPE,
 * no biases in the projections. Weights ship as fp16 and every accumulation
 * here is fp32; the exporter verified 0/2640, 0/1680 and 0/2160 predicted
 * nibbles change under that round trip, so this is the same function as the
 * checkpoint, not an approximation of it.
 */
'use strict';

// ---------------------------------------------------------------- linalg
function matvecT(W, x, out, R, C) {
  // W is (R, C) row-major; computes out[r] = sum_c W[r][c] * x[c]
  for (let r = 0; r < R; r++) {
    const base = r * C;
    let s = 0;
    for (let c = 0; c < C; c++) s += W[base + c] * x[c];
    out[r] = s;
  }
  return out;
}
function layernorm(x, g, b, out, D) {
  let m = 0;
  for (let i = 0; i < D; i++) m += x[i];
  m /= D;
  let v = 0;
  for (let i = 0; i < D; i++) { const d = x[i] - m; v += d * d; }
  v = 1 / Math.sqrt(v / D + 1e-5);
  for (let i = 0; i < D; i++) out[i] = (x[i] - m) * v * g[i] + b[i];
  return out;
}
const gelu = (t) => 0.5 * t * (1 + Math.tanh(0.7978845608 * (t + 0.044715 * t * t * t)));

// ---------------------------------------------------------------- model
class WorldModel {
  constructor(meta, buf) {
    this.m = meta;
    const D = meta.d_model, L = meta.n_layers;
    this.D = D; this.L = L; this.H = meta.n_heads; this.hd = D / meta.n_heads;
    const raw = new Uint16Array(buf);
    const T = {};
    for (const t of meta.tensors) {
      const f = new Float32Array(t.count);
      for (let i = 0; i < t.count; i++) f[i] = f16(raw[t.offset + i]);
      T[t.name] = f;
    }
    this.T = T;
    // preallocated scratch — no allocation inside the decode loop
    this.x = new Float32Array(D); this.h = new Float32Array(D);
    this.q = new Float32Array(D); this.k = new Float32Array(D);
    this.v = new Float32Array(D); this.o = new Float32Array(D);
    this.qkv = new Float32Array(3 * D);
    this.ff = new Float32Array(4 * D);
    this.logits = new Float32Array(meta.vocab);
    this.att = new Float32Array(meta.length);
    // KV cache: per layer, (length, D)
    this.ck = []; this.cv = [];
    for (let l = 0; l < L; l++) {
      this.ck.push(new Float32Array(meta.length * D));
      this.cv.push(new Float32Array(meta.length * D));
    }
    this.n = 0;                      // cached positions
  }
  reset() { this.n = 0; }

  /** One token through all layers; appends to the cache. Returns logits. */
  step(tokenId, pos) {
    const { D, L, H, hd, T, m } = this;
    const x = this.x;
    for (let i = 0; i < D; i++) {
      x[i] = T.tok[tokenId * D + i] + T.pos[pos * D + i] + T.emb_bias[i];
    }
    const t = this.n;                // this token's cache slot
    for (let l = 0; l < L; l++) {
      layernorm(x, T[`l${l}.ln1_g`], T[`l${l}.ln1_b`], this.h, D);
      matvecT(T[`l${l}.qkv`], this.h, this.qkv, 3 * D, D);
      const ck = this.ck[l], cv = this.cv[l];
      for (let i = 0; i < D; i++) {
        this.q[i] = this.qkv[i];
        ck[t * D + i] = this.qkv[D + i];
        cv[t * D + i] = this.qkv[2 * D + i];
      }
      // multi-head causal attention over positions 0..t
      const scale = 1 / Math.sqrt(hd);
      for (let hI = 0; hI < H; hI++) {
        const off = hI * hd;
        let mx = -Infinity;
        for (let j = 0; j <= t; j++) {
          let s = 0;
          const jb = j * D + off;
          for (let d = 0; d < hd; d++) s += this.q[off + d] * ck[jb + d];
          s *= scale; this.att[j] = s; if (s > mx) mx = s;
        }
        let sum = 0;
        for (let j = 0; j <= t; j++) { const e = Math.exp(this.att[j] - mx); this.att[j] = e; sum += e; }
        const inv = 1 / sum;
        for (let d = 0; d < hd; d++) this.o[off + d] = 0;
        for (let j = 0; j <= t; j++) {
          const w = this.att[j] * inv, jb = j * D + off;
          for (let d = 0; d < hd; d++) this.o[off + d] += w * cv[jb + d];
        }
      }
      matvecT(T[`l${l}.proj`], this.o, this.h, D, D);
      for (let i = 0; i < D; i++) x[i] += this.h[i];
      layernorm(x, T[`l${l}.ln2_g`], T[`l${l}.ln2_b`], this.h, D);
      matvecT(T[`l${l}.fc1`], this.h, this.ff, 4 * D, D);
      for (let i = 0; i < 4 * D; i++) this.ff[i] = gelu(this.ff[i]);
      matvecT(T[`l${l}.fc2`], this.ff, this.h, D, 4 * D);
      for (let i = 0; i < D; i++) x[i] += this.h[i];
    }
    this.n++;
    layernorm(x, T.lnf_g, T.lnf_b, this.h, D);
    return matvecT(T.head, this.h, this.logits, m.vocab, D);
  }
}

function f16(h) {                       // IEEE half -> float
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

// ---------------------------------------------------------------- codec
class Codec {
  constructor(meta) { this.m = meta; }
  /** field block (Int32Array of n_fields nibbles) -> {leaf: Float64Array} */
  assemble(blk) {
    const out = {}; let i = 0;
    for (const r of this.m.plan) {
      const vals = new Float64Array(r.n);
      for (let e = 0; e < r.n; e++) {
        let q = 0;
        for (let b = 0; b < r.nibs; b++) q = (q << 4) | blk[i++];
        vals[e] = r.kind === 'float'
          ? r.lo + q * (r.hi - r.lo) / (Math.pow(16, r.nibs) - 1)
          : q + r.lo;
      }
      out[r.leaf] = vals;
    }
    return out;
  }
  /** context tokens for (state block, action): BOS ACT act-nibs state OUT */
  context(blk, action) {
    const m = this.m, toks = new Int32Array(m.out_pos + 1);
    let i = 0;
    toks[i++] = m.BOS; toks[i++] = m.ACT;
    for (let b = m.ACT_NIBS - 1; b >= 0; b--)
      toks[i++] = m.NIB0 + ((action >> (4 * b)) & 0xf);
    for (let f = 0; f < m.n_fields; f++) toks[i++] = m.NIB0 + blk[f];
    toks[i++] = m.OUT;
    return toks;
  }
}

/** One transition: prefix pass then n_fields greedy decode steps. */
function predict(net, codec, blk, action) {
  const m = net.m;
  net.reset();
  const ctx = codec.context(blk, action);
  let lg = null;
  for (let p = 0; p < ctx.length; p++) lg = net.step(ctx[p], p);
  const out = new Int32Array(m.n_fields);
  let pos = ctx.length;
  for (let f = 0; f < m.n_fields; f++) {
    let best = -1, bv = -Infinity;
    for (let t = 0; t < m.vocab; t++) {
      if (!m.nib_mask[t]) continue;          // representation mask only
      if (lg[t] > bv) { bv = lg[t]; best = t; }
    }
    // COPY resolves against the block the model was CONDITIONED on — during
    // a rollout that is its own previous prediction, never ground truth.
    out[f] = (m.delta && best === m.COPY) ? blk[f]
             : Math.max(0, Math.min(15, best - m.NIB0));
    if (f + 1 < m.n_fields) lg = net.step(best, pos++);
  }
  return out;
}

// ---------------------------------------------------------------- render
const PAL = ['#0b0d13', '#ffc478', '#55aaff', '#ff5566', '#78fa96',
             '#b18cff', '#f0a202', '#ffffff'];

/** MinAtar states are entity lists over a 10x10 board; draw what we can
 *  name and fall back to a field strip for the rest, so nothing is hidden. */
function drawMinAtar(ctx, env, s, W, H) {
  const N = 10, cell = Math.floor(Math.min(W, H) / N);
  const ox = Math.floor((W - cell * N) / 2), oy = Math.floor((H - cell * N) / 2);
  ctx.fillStyle = '#0b0d13'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#1c2334';
  for (let i = 0; i <= N; i++) {
    ctx.beginPath(); ctx.moveTo(ox + i * cell, oy); ctx.lineTo(ox + i * cell, oy + cell * N); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ox, oy + i * cell); ctx.lineTo(ox + cell * N, oy + i * cell); ctx.stroke();
  }
  const put = (x, y, col, pad = 1) => {
    if (!(x >= 0 && x < N && y >= 0 && y < N)) return;
    ctx.fillStyle = col;
    ctx.fillRect(ox + x * cell + pad, oy + y * cell + pad, cell - 2 * pad, cell - 2 * pad);
  };
  const g = (k) => s[k];
  if (env.startsWith('Breakout')) {
    const br = g('brick_map');
    if (br) for (let i = 0; i < br.length; i++)
      if (br[i] > 0.5) put(i % N, Math.floor(i / N), '#ffc478', 2);
    if (g('pos')) put(g('pos')[0], N - 1, '#78fa96', 1);
    if (g('ball_x')) put(g('ball_x')[0], g('ball_y')[0], '#ff5566', 3);
  } else if (env.startsWith('Freeway')) {
    const cars = g('cars');
    if (cars) for (let i = 0; i + 3 < cars.length; i += 4)
      put(cars[i], cars[i + 1], '#55aaff', 1);
    if (g('pos')) put(4, g('pos')[0], '#78fa96', 1);
  } else if (env.startsWith('Asterix')) {
    // entities is (8, 5): x, y, ..., is_gold, active — stride 5, and the
    // last element gates whether the slot holds anything at all.
    const ent = g('entities');
    if (ent) for (let i = 0; i + 4 < ent.length; i += 5)
      if (ent[i + 4] > 0.5) put(ent[i], ent[i + 1], ent[i + 3] > 0.5 ? '#ffc478' : '#ff5566', 1);
    if (g('player_x')) put(g('player_x')[0], g('player_y')[0], '#78fa96', 1);
  }
  return { ox, oy, cell, N };
}

/** The raw authoritative state, one cell per nibble, grouped by field.
 *  This is the ground truth of what the model actually emitted, so it stays
 *  correct even where the game overlay above is a best-effort guess about
 *  entity layout. Changed nibbles are highlighted against the previous
 *  state, which is also what makes the identity floor visible: on most
 *  environments almost nothing moves per tick. */
function drawFields(ctx, meta, blk, prev, W, H) {
  const n = meta.n_fields;
  const cols = Math.ceil(Math.sqrt(n * W / H));
  const cw = Math.max(2, Math.floor(W / cols));
  const rows = Math.ceil(n / cols);
  const chh = Math.max(2, Math.min(cw, Math.floor(H / rows)));
  ctx.fillStyle = '#0b0d13'; ctx.fillRect(0, 0, W, H);
  for (let i = 0; i < n; i++) {
    const x = (i % cols) * cw, y = Math.floor(i / cols) * chh;
    const v = blk[i] / 15;
    const changed = prev && prev[i] !== blk[i];
    ctx.fillStyle = changed ? '#ff5566'
      : `rgb(${Math.round(40 + 150 * v)},${Math.round(50 + 140 * v)},${Math.round(70 + 165 * v)})`;
    ctx.fillRect(x, y, cw - 1, chh - 1);
  }
  return { changed: prev ? blk.reduce((a, b, i) => a + (prev[i] !== b ? 1 : 0), 0) : 0 };
}
