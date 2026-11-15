// Nurse view: live request log, refreshed every 2 seconds, with acknowledge.
import { $, api, alarm, unlockAudio } from "/common.js";

let soundOn = false;
$("#soundBtn").addEventListener("click", () => {
  unlockAudio(); soundOn = true;
  $("#soundBtn").textContent = "Alarm sound on"; $("#soundBtn").classList.add("primary");
});

async function ack(id) {
  try { await api("/api/ack", { body: { id } }); } catch {}
  refresh();
}
$("#ackAll").addEventListener("click", () => ack("all"));

function cell(text, cls) { const td = document.createElement("td"); td.textContent = text; if (cls) td.className = cls; return td; }

async function refresh() {
  let data;
  try { data = await api("/api/requests?limit=50"); } catch { return; }
  const tbody = $("#rows"); tbody.innerHTML = "";
  if (!data.requests.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="muted">No requests yet.</td></tr>';
  }
  for (const r of data.requests) {
    const tr = document.createElement("tr");
    if (r.urgent) tr.className = "urgent-row" + (r.acknowledged ? "" : " open");
    const t = new Date(r.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    const possible = r.confirmed === false;
    tr.append(cell(`${t} (${r.ago})`), cell(`${r.bed} · ${r.user}`), cell((possible ? "Possible: " : "") + r.sentence, "sentence"),
      cell(possible ? "best guess, not confirmed yet" : `${r.method === "auto-urgent" ? "raised automatically" : r.method}${r.mode ? ", " + r.mode : ""}`));
    const st = document.createElement("td");
    if (r.urgent && !r.acknowledged) {
      const b = document.createElement("button"); b.className = "btn danger"; b.textContent = "Acknowledge";
      b.onclick = () => ack(r.id); st.appendChild(b);
    } else st.textContent = r.urgent ? (r.cancelled ? "cancelled by the patient: misread" : r.superseded_by ? "possible · patient then confirmed below" : "urgent · acknowledged") : "logged";
    tr.appendChild(st);
    tbody.appendChild(tr);
  }
  const open = data.requests.filter((r) => r.urgent && !r.acknowledged);
  $("#alarmBox").classList.toggle("hidden", !open.length);
  if (open.length) $("#alarmText").textContent = `${open[0].confirmed === false ? "POSSIBLE URGENT" : "URGENT"} · ${open[0].bed}: “${open[0].sentence}”` + (open.length > 1 ? ` (+${open.length - 1} more)` : "");
  alarm(soundOn && open.length > 0);
}

async function boot() {
  try {
    const s = await api("/api/status");
    const el = $("#pushInfo");
    el.innerHTML = "Phone alerts: install the free <b>ntfy</b> app and subscribe to topic <code></code> (server ntfy.sh).";
    el.querySelector("code").textContent = s.ntfy_topic;
  } catch {}
  refresh();
  setInterval(refresh, 2000);
}
boot();
