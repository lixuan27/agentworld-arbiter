# ARBITER — typed-state world models across 21 environments

Demo page: **https://lixuan27.github.io/agentworld-arbiter/**

One 5–6M causal Transformer, one 20-token nibble vocabulary, 21 environments
(self-built SkirmishCTF multiplayer arena, Atari RAM ×12, SMAX, Craftax
Classic + Full, MinAtar ×5, jumanji Snake). The model predicts the engine's
authoritative typed state — never pixels — and every predicted state is
scored raw: no repair, with do-nothing floors and paired tests everywhere.

Highlights: 95.3% full-state EXACT 64-step autoregression on Freeway;
model-internal planning at 5.4× the matched random floor with deep plans no
random search finds (p≈0.003); per-view rendering flat 5–7 ms measured to
N=1024 population; truth-vs-model GIFs drawn by the official Craftax
renderer.

See `index.html` (the page) and `STAGE_REPORT_2026-08-25.md` (full technical
report, Chinese).
