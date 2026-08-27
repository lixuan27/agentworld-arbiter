/* VoxelWorld — the SIMULATOR, ported to JavaScript, playable in first person.
 *
 * ⚠️ THIS IS NOT THE WORLD MODEL. It is the authoritative engine the world
 * model is trained to imitate: the same block types, the same 8 yaws and 5
 * pitches, the same reach-4 raycast, the same gravity-and-step-up rule, the
 * same distance shading. Keeping the two demos clearly separated matters —
 * `play.js` runs the learned network, this file runs the ground truth, and
 * conflating them would make the impressive one meaningless.
 *
 * It is here because the view-changing substrate is the one part of this
 * project that has to be FELT to be understood. Occlusion is real: a ray
 * stops at the first solid block, so what you see is a genuinely many-to-one
 * function of the world state, and turning changes which 30% of the world's
 * randomness reaches you (the `visible_chance` number on the page).
 *
 * Ported from src/agentworld/env/voxel/engine.py. Constants copied verbatim.
 */
'use strict';

const VX = (() => {
  const AIR = 0, GRASS = 1, DIRT = 2, STONE = 3, WOOD = 4, LEAF = 5,
        ORE = 6, BEDROCK = 7, MOB = 8;
  const SOLID = [0, 1, 1, 1, 1, 0, 1, 1, 1];          // LEAF is walk-through
  const BREAKABLE = [0, 1, 1, 1, 1, 1, 1, 0, 0];      // BEDROCK never
  const N_YAW = 8, PITCH_MIN = -2, PITCH_MAX = 2, REACH = 4;
  const YAW_DX = [1, 1, 0, -1, -1, -1, 0, 1];
  const YAW_DY = [0, 1, 1, 1, 0, -1, -1, -1];
  const PAL = [[135, 180, 235], [92, 168, 74], [134, 96, 67], [128, 128, 132],
               [122, 92, 52], [76, 140, 66], [206, 176, 84], [60, 60, 66],
               [212, 84, 96]];

  // deterministic PRNG so a seed reproduces a world exactly, as in the engine
  function rng(seed) {
    let s = seed >>> 0;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }

  class World {
    constructor(W, H, D, seed) {
      this.W = W; this.H = H; this.D = D;
      this.b = new Uint8Array(W * H * D);
      const r = rng(seed);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const h = 2 + Math.floor(r() * 3);
        for (let z = 0; z < D; z++) {
          const i = this.idx(x, y, z);
          if (z === 0) this.b[i] = BEDROCK;
          else if (z < h - 1) this.b[i] = r() < 0.12 ? STONE : DIRT;
          else if (z < h) this.b[i] = GRASS;
          else this.b[i] = AIR;
        }
        if (r() < 0.06) {                    // a tree
          const h2 = h + 2 + Math.floor(r() * 2);
          for (let z = h; z < Math.min(h2, D); z++) this.b[this.idx(x, y, z)] = WOOD;
          for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              const X = x + dx, Y = y + dy, Z = h2 + dz;
              if (X >= 0 && X < W && Y >= 0 && Y < H && Z >= 0 && Z < D
                  && this.b[this.idx(X, Y, Z)] === AIR)
                this.b[this.idx(X, Y, Z)] = LEAF;
            }
        }
      }
      // agent
      this.ax = Math.floor(W / 2); this.ay = Math.floor(H / 2);
      this.az = 1; this.yaw = 0; this.pitch = 0;
      this.inv = new Int32Array(9);
      while (this.az + 1 < D && SOLID[this.b[this.idx(this.ax, this.ay, this.az)]]) this.az++;
      // mobs
      this.mobs = [];
      for (let m = 0; m < 4; m++) {
        const mx = Math.floor(r() * W), my = Math.floor(r() * H);
        let mz = 1; while (mz + 1 < D && SOLID[this.b[this.idx(mx, my, mz)]]) mz++;
        this.mobs.push({ x: mx, y: my, z: mz, active: 1 });
      }
      this.tick = 0; this.r = r;
      this.broke = 0; this.placed = 0;
    }
    idx(x, y, z) { return (z * this.H + y) * this.W + x; }
    get(x, y, z) {
      if (x < 0 || x >= this.W || y < 0 || y >= this.H || z < 0 || z >= this.D) return BEDROCK;
      return this.b[this.idx(x, y, z)];
    }

    /** engine.step, in the same order: yaw/pitch, move, jump+gravity,
     *  break/place, then mobs. Actions match A_* in engine.py. */
    step(a, mobMove) {
      if (a === 5) this.yaw = (this.yaw + N_YAW - 1) % N_YAW;       // TURN_L
      if (a === 6) this.yaw = (this.yaw + 1) % N_YAW;               // TURN_R
      if (a === 7) this.pitch = Math.min(PITCH_MAX, this.pitch + 1);
      if (a === 8) this.pitch = Math.max(PITCH_MIN, this.pitch - 1);
      let dx = 0, dy = 0;
      const f = this.yaw, rt = (this.yaw + 2) % N_YAW;
      if (a === 1) { dx = YAW_DX[f]; dy = YAW_DY[f]; }              // FWD
      if (a === 2) { dx = -YAW_DX[f]; dy = -YAW_DY[f]; }            // BACK
      if (a === 3) { dx = YAW_DX[rt]; dy = YAW_DY[rt]; }            // LEFT
      if (a === 4) { dx = -YAW_DX[rt]; dy = -YAW_DY[rt]; }          // RIGHT
      if (dx || dy) {
        const nx = Math.max(0, Math.min(this.W - 1, this.ax + dx));
        const ny = Math.max(0, Math.min(this.H - 1, this.ay + dy));
        let nz = this.az;
        if (SOLID[this.get(nx, ny, nz)]) {
          if (nz + 1 < this.D && !SOLID[this.get(nx, ny, nz + 1)]) nz += 1;
          else { dx = 0; }                    // blocked: no move at all
        }
        if (dx || dy) { this.ax = nx; this.ay = ny; this.az = nz; }
      }
      if (a === 9 && this.az + 1 < this.D
          && !SOLID[this.get(this.ax, this.ay, this.az + 1)]) this.az += 1;
      while (this.az > 1 && !SOLID[this.get(this.ax, this.ay, this.az - 1)]) this.az -= 1;
      if (a === 10 || a === 11) {             // BREAK / PLACE
        const hit = this.raycast();
        if (a === 10 && hit.hit && BREAKABLE[this.get(hit.bx, hit.by, hit.bz)]) {
          const t = this.get(hit.bx, hit.by, hit.bz);
          this.b[this.idx(hit.bx, hit.by, hit.bz)] = AIR;
          this.inv[t] += 1; this.broke += 1;
          if (t === STONE && this.r() < 0.15) this.inv[ORE] += 1;
        } else if (a === 11 && hit.px >= 0) {
          let bt = -1;
          for (let t = 1; t < 9; t++) if (this.inv[t] > 0) { bt = t; break; }
          if (bt > 0 && !SOLID[this.get(hit.px, hit.py, hit.pz)]) {
            this.b[this.idx(hit.px, hit.py, hit.pz)] = bt;
            this.inv[bt] -= 1; this.placed += 1;
          }
        }
      }
      this.tick += 1;
      if (mobMove) for (const m of this.mobs) {
        if (!m.active) continue;
        const d = Math.floor(this.r() * N_YAW);
        const nx = Math.max(0, Math.min(this.W - 1, m.x + YAW_DX[d]));
        const ny = Math.max(0, Math.min(this.H - 1, m.y + YAW_DY[d]));
        if (!SOLID[this.get(nx, ny, m.z)]) { m.x = nx; m.y = ny; }
        while (m.z > 1 && !SOLID[this.get(m.x, m.y, m.z - 1)]) m.z -= 1;
      }
    }

    /** The block the agent is aiming at, and the empty cell in front of it. */
    raycast() {
      const yaw = this.yaw * (2 * Math.PI / N_YAW), pit = this.pitch * (Math.PI / 8);
      const dx = Math.cos(yaw) * Math.cos(pit), dy = Math.sin(yaw) * Math.cos(pit),
            dz = Math.sin(pit);
      let px = -1, py = -1, pz = -1;
      const ox = this.ax + 0.5, oy = this.ay + 0.5, oz = this.az + 0.5;
      for (let s = 0.25; s <= REACH; s += 0.25) {
        const x = Math.floor(ox + dx * s), y = Math.floor(oy + dy * s),
              z = Math.floor(oz + dz * s);
        if (x < 0 || x >= this.W || y < 0 || y >= this.H || z < 0 || z >= this.D) break;
        if (this.get(x, y, z) !== AIR)
          return { hit: true, bx: x, by: y, bz: z, px, py, pz };
        px = x; py = y; pz = z;
      }
      return { hit: false, bx: -1, by: -1, bz: -1, px, py, pz };
    }

    /** engine.render_first_person: mobs are overlaid INTO the occupancy grid
     *  before marching, not drawn afterwards, so occlusion is correct — a mob
     *  behind a wall is hidden because the wall is hit first. The first
     *  version of the Python renderer marched against blocks alone, which
     *  made the substrate's only stochastic entity invisible BY CONSTRUCTION
     *  and produced a vacuous p_stoch of exactly 0. */
    render(img, w, h, fovDeg) {
      const occ = (x, y, z) => {
        for (const m of this.mobs)
          if (m.active && m.x === x && m.y === y && m.z === z) return MOB;
        return this.get(x, y, z);
      };
      const fov = fovDeg * Math.PI / 180;
      const yaw = this.yaw * (2 * Math.PI / N_YAW), pit = this.pitch * (Math.PI / 8);
      const fx = Math.cos(yaw) * Math.cos(pit), fy = Math.sin(yaw) * Math.cos(pit),
            fz = Math.sin(pit);
      const rx = -Math.sin(yaw), ry = Math.cos(yaw), rz = 0;
      const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
      const ox = this.ax + 0.5, oy = this.ay + 0.5, oz = this.az + 0.5;
      const tanH = Math.tan(fov / 2), maxD = 18, stp = 0.3;
      let p = 0;
      for (let j = 0; j < h; j++) {
        const V = tanH * (h / w) * ((h - 1) / 2 - j) / ((h - 1) / 2);
        for (let i = 0; i < w; i++) {
          const U = tanH * (i - (w - 1) / 2) / ((w - 1) / 2);
          let dx = fx + U * rx + V * ux, dy = fy + U * ry + V * uy,
              dz = fz + U * rz + V * uz;
          const n = Math.hypot(dx, dy, dz); dx /= n; dy /= n; dz /= n;
          let t = AIR, dist = maxD;
          for (let s = stp; s <= maxD; s += stp) {
            const x = Math.floor(ox + dx * s), y = Math.floor(oy + dy * s),
                  z = Math.floor(oz + dz * s);
            if (x < 0 || x >= this.W || y < 0 || y >= this.H || z < 0 || z >= this.D) break;
            const b = occ(x, y, z);
            if (b !== AIR) { t = b; dist = s; break; }
          }
          const c = PAL[t];
          const sh = t === AIR ? 1 : Math.max(0.25, 1 - dist / maxD);
          img[p++] = c[0] * sh; img[p++] = c[1] * sh; img[p++] = c[2] * sh; img[p++] = 255;
        }
      }
    }
  }
  return { World, N_YAW, PITCH_MIN, PITCH_MAX,
           NAMES: ['air', 'grass', 'dirt', 'stone', 'wood', 'leaf', 'ore',
                   'bedrock', 'mob'] };
})();
