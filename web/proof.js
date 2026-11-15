// Proof page: shows only what scripts/evaluate.py (or /api/evaluate) measured.
import { $, api } from "/common.js";

const pct = (x) => (x == null ? "–" : `${(100 * x).toFixed(1)}%`);
function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

function render(res) {
  const el = $("#content");
  if (!res || !res.overall) {
    el.innerHTML = "No measurements yet. Calibrate at least 2 phrases with 2+ samples each, then press <b>Re-run evaluation</b>.";
    return;
  }
  const o = res.overall;
  let html = `<div class="stats">
    <div class="stat"><div class="v">${pct(o.top1)}</div><div class="k">Top-1 accuracy, personal model (own recordings only)</div></div>
    <div class="stat"><div class="v">${pct(o.top3)}</div><div class="k">Top-3 accuracy: correct sentence among the 3 choices</div></div>
    <div class="stat"><div class="v">${o.samples}</div><div class="k">Held-out samples tested</div></div>
    <div class="stat"><div class="v">${Object.keys(res.users).length}</div><div class="k">People</div></div>
  </div>`;
  html += `<h2>Per person</h2><table><thead><tr><th>Person</th><th>Samples</th><th>Phrases</th><th>Top-1</th><th>Top-3</th><th>Chance top-1 / top-3</th>${res.with_gemini ? "<th>With Gemini top-1 / top-3</th>" : ""}</tr></thead><tbody>`;
  for (const [u, r] of Object.entries(res.users)) {
    html += `<tr><td>${esc(u)}</td><td>${r.samples}</td><td>${r.phrases}</td><td><b>${pct(r.top1)}</b></td><td><b>${pct(r.top3)}</b></td><td class="muted">${pct(r.chance_top1)} / ${pct(r.chance_top3)}</td>${res.with_gemini ? `<td>${pct(r.gemini_top1)} / ${pct(r.gemini_top3)}${r.gemini_answered != null ? ` <span class="muted small">(Gemini answered ${r.gemini_answered}/${r.samples}; the rest fell back offline)</span>` : ""}</td>` : ""}</tr>`;
  }
  html += "</tbody></table>";

  // As deployed: one pool of everyone's recordings. And with no calibration at all: other people's recordings only.
  const pooled = res.pooled && res.pooled.users ? Object.entries(res.pooled.users) : [];
  const starter = res.starter && res.starter.users ? Object.entries(res.starter.users) : [];
  if (pooled.length || starter.length) {
    html += `<h2 style="margin-top:24px">Shared starter model</h2>
      <p class="muted">Unmute matches against everyone's recordings. "As deployed" holds each sample out of that pool.
      "No calibration" matches a person's samples only against <i>other</i> people's recordings: what a brand-new patient gets.</p>
      <table><thead><tr><th>Person</th><th>As deployed top-1 / top-3</th><th>No calibration top-1 / top-3</th><th>Chance top-1 / top-3</th></tr></thead><tbody>`;
    const names = [...new Set([...pooled.map(([u]) => u), ...starter.map(([u]) => u)])];
    for (const u of names) {
      const p = res.pooled.users[u], s = res.starter.users[u];
      html += `<tr><td>${esc(u)}</td><td>${p ? `<b>${pct(p.top1)}</b> / <b>${pct(p.top3)}</b> <span class="muted small">(${p.samples} samples)</span>` : "–"}</td>`
        + `<td>${s ? `<b>${pct(s.top1)}</b> / <b>${pct(s.top3)}</b> <span class="muted small">(${s.samples} samples, ${s.phrases} phrases)</span>` : "–"}</td>`
        + `<td class="muted">${s ? `${pct(s.chance_top1)} / ${pct(s.chance_top3)}` : "–"}</td></tr>`;
    }
    html += "</tbody></table>";
  }

  // Learning from use: start with no own recordings, replay them in order, learn each after predicting it.
  const lr = res.learning && res.learning.overall ? Object.entries(res.learning.overall) : [];
  if (lr.length) {
    html += `<h2 style="margin-top:24px">Learning from use</h2>
      <p class="muted">A person starts with none of their own recordings, only other people's. Their recordings are replayed in
      the order they were made: each is predicted first, then learned under its true phrase (what confirming a card or
      fixing it on the touch board does). Grouped by how many of their own samples of that phrase Unmute already had.</p>
      <div class="curve">`;
    for (const [b, v] of lr) {
      const label = +b === res.learning.max_bucket ? `${b}+` : b;
      html += `<div class="curve-col"><div class="curve-bars">
          <div class="bar t3" style="height:${(100 * v.top3).toFixed(0)}%" title="top-3 ${pct(v.top3)}"></div>
          <div class="bar t1" style="height:${(100 * v.top1).toFixed(0)}%" title="top-1 ${pct(v.top1)}"></div>
        </div><div class="curve-v"><b>${pct(v.top1)}</b> / ${pct(v.top3)}</div>
        <div class="curve-k">${label} own sample${label === "1" ? "" : "s"}<br><span class="muted small">${v.n} recordings</span></div></div>`;
    }
    html += `</div><p class="muted small">Top-1 (solid) / top-3 (light). Caveat: our samples of one phrase were recorded one after
      another, so this is a little easier than samples made hours apart.</p>`;
  }

  // Most-confused pairs across users.
  const conf = {};
  for (const r of Object.values(res.users)) for (const c of r.confusions) {
    const k = c.said + "→" + c.predicted; conf[k] = conf[k] || { ...c, count: 0 }; conf[k].count += c.count;
  }
  const top = Object.values(conf).sort((a, b) => b.count - a.count).slice(0, 10);
  html += `<h2 style="margin-top:24px">Most confused phrases (top-1 mistakes)</h2>`;
  if (!top.length) html += `<p class="muted">No top-1 mistakes.</p>`;
  else {
    html += "<table><thead><tr><th>Mouthed</th><th>Read as</th><th>Times</th></tr></thead><tbody>";
    for (const c of top) html += `<tr><td>${esc(c.said)}</td><td>${esc(c.predicted)}</td><td>${c.count}</td></tr>`;
    html += "</tbody></table>";
  }

  // Per-phrase table (all users pooled).
  const pp = {};
  for (const r of Object.values(res.users)) for (const [p, v] of Object.entries(r.per_phrase)) {
    pp[p] = pp[p] || { n: 0, t1: 0, t3: 0 };
    pp[p].n += v.n; pp[p].t1 += Math.round(v.top1 * v.n); pp[p].t3 += Math.round(v.top3 * v.n);
  }
  html += `<h2 style="margin-top:24px">Per phrase</h2><table><thead><tr><th>Phrase</th><th>Samples</th><th>Top-1</th><th>Top-3</th></tr></thead><tbody>`;
  for (const [p, v] of Object.entries(pp).sort((a, b) => a[1].t1 / a[1].n - b[1].t1 / b[1].n)) {
    html += `<tr><td>${esc(p)}</td><td>${v.n}</td><td>${pct(v.t1 / v.n)}</td><td>${pct(v.t3 / v.n)}</td></tr>`;
  }
  html += "</tbody></table>";
  html += `<p class="muted small" style="margin-top:16px">Measured ${esc(res.generated_at)} · ${esc(res.method)} · settings ${esc(JSON.stringify(res.config))}</p>`;
  el.innerHTML = html;
}

async function loadUsage() {
  try {
    const u = await api("/api/usage");
    const el = $("#usage");
    if (!u.n) { el.innerHTML = `<p class="muted">No confirmed sentences with a recorded card position yet. Use the Patient page, then come back.</p>`; return; }
    el.innerHTML = `<div class="stats">
      <div class="stat"><div class="v">${pct(u.top1)}</div><div class="k">Right on the first card</div></div>
      <div class="stat"><div class="v">${pct(u.top3)}</div><div class="k">Right sentence among the 3 cards</div></div>
      <div class="stat"><div class="v">${u.corrections}</div><div class="k">Fixed on the touch board (and learned)</div></div>
      <div class="stat"><div class="v">${u.n}</div><div class="k">Confirmed sentences</div></div></div>`;
  } catch {}
}
async function load() { loadUsage(); try { render((await api("/api/results")).results); } catch (e) { $("#content").textContent = e.message; } }
$("#runBtn").addEventListener("click", async () => {
  $("#runBtn").disabled = true; $("#runBtn").textContent = "Running…";
  try { render((await api("/api/evaluate", { method: "POST" })).results); }
  catch (e) { $("#content").textContent = "Evaluation failed: " + e.message; }
  $("#runBtn").disabled = false; $("#runBtn").textContent = "Re-run evaluation";
});
load();
