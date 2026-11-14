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
    <div class="stat"><div class="v">${pct(o.top1)}</div><div class="k">Top-1 accuracy (overall)</div></div>
    <div class="stat"><div class="v">${pct(o.top3)}</div><div class="k">Top-3 accuracy: correct sentence among the 3 choices</div></div>
    <div class="stat"><div class="v">${o.samples}</div><div class="k">Held-out samples tested</div></div>
    <div class="stat"><div class="v">${Object.keys(res.users).length}</div><div class="k">People</div></div>
  </div>`;
  html += `<h2>Per person</h2><table><thead><tr><th>Person</th><th>Samples</th><th>Phrases</th><th>Top-1</th><th>Top-3</th><th>Chance top-1 / top-3</th>${res.with_gemini ? "<th>With Gemini top-1 / top-3</th>" : ""}</tr></thead><tbody>`;
  for (const [u, r] of Object.entries(res.users)) {
    html += `<tr><td>${esc(u)}</td><td>${r.samples}</td><td>${r.phrases}</td><td><b>${pct(r.top1)}</b></td><td><b>${pct(r.top3)}</b></td><td class="muted">${pct(r.chance_top1)} / ${pct(r.chance_top3)}</td>${res.with_gemini ? `<td>${pct(r.gemini_top1)} / ${pct(r.gemini_top3)}${r.gemini_answered != null ? ` <span class="muted small">(Gemini answered ${r.gemini_answered}/${r.samples}; the rest fell back offline)</span>` : ""}</td>` : ""}</tr>`;
  }
  html += "</tbody></table>";

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

async function load() { try { render((await api("/api/results")).results); } catch (e) { $("#content").textContent = e.message; } }
$("#runBtn").addEventListener("click", async () => {
  $("#runBtn").disabled = true; $("#runBtn").textContent = "Running…";
  try { render((await api("/api/evaluate", { method: "POST" })).results); }
  catch (e) { $("#content").textContent = "Evaluation failed: " + e.message; }
  $("#runBtn").disabled = false; $("#runBtn").textContent = "Re-run evaluation";
});
load();
