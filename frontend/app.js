import { renderMeshes, disposeViewer } from "./viewer3d.js";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;
const md = (text) => DOMPurify.sanitize(marked.parse(text || ""), { ADD_ATTR: ["target"] });

const SEQUENCES = ["", "flair", "t1", "t1ce", "t2"];
const LAYER_HELP = {
  original: "The scan as uploaded (normalized for display).",
  overlay: "Segmentation: the region the AI outlined as tumor, colored by sub-region.",
  heatmap: "Grad-CAM: brighter areas influenced the AI's decision most. It explains the model, not the disease.",
  uncertainty: "Where the AI is unsure about the tumor boundary (Monte-Carlo dropout). Bright = less certain.",
};
const SUGGESTIONS = {
  patient: ["What does this result mean?", "Where is the tumor and how big is it?", "Has it changed since my last scan?", "What questions should I ask my doctor?"],
  doctor: ["Summarize the findings.", "Show the slice with the largest tumor.", "Compare with the prior study.", "How reliable is this result?"],
};

const state = { files: [], study: null, slice: 0, layer: "overlay", sequence: "", audience: "patient", language: "English", profiles: [], busy: false };

// ------------------------------------------------------------------ API helpers
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

// ------------------------------------------------------------------ status bar
async function loadStatus() {
  const [health, profiles, languages] = await Promise.all([api("/api/health"), api("/api/profiles"), api("/api/languages")]);
  state.profiles = profiles;
  const pills = [];
  pills.push(health.demo_mode
    ? `<span class="pill warn" title="No trained weights found in weights/. Using the heuristic detector.">Demo mode: no trained models</span>`
    : `<span class="pill good">Trained models loaded</span>`);
  pills.push(health.llm_configured
    ? `<span class="pill good">LLM ready</span>`
    : `<span class="pill warn" title="Set ANTHROPIC_API_KEY in .env to enable the assistant and AI-written reports.">LLM key not set</span>`);
  pills.push(`<span class="pill">${esc(health.models.device)}</span>`);
  $("#status").innerHTML = pills.join("");

  $("#scan-type").innerHTML = `<option value="auto">Auto-detect</option>` + profiles.map((p) => `<option value="${esc(p.key)}">${esc(p.display_name)}</option>`).join("");
  $("#language").innerHTML = languages.map((l) => `<option>${esc(l)}</option>`).join("");
}

// ------------------------------------------------------------------ upload
function guessSequence(name) {
  const tokens = name.toLowerCase().split(/[^a-z0-9]+/);
  for (const [seq, aliases] of [["t1ce", ["t1ce", "t1c", "t1gd", "t1post"]], ["flair", ["flair"]], ["t2", ["t2"]], ["t1", ["t1"]]]) {
    if (aliases.some((a) => tokens.includes(a))) return seq;
  }
  return "";
}

function renderFileList() {
  $("#file-list").innerHTML = state.files.map((f, i) => `
    <li><span title="${esc(f.file.name)}">${esc(f.file.name)}</span>
      <select data-i="${i}" title="MRI sequence">${SEQUENCES.map((s) => `<option value="${s}" ${s === f.seq ? "selected" : ""}>${s || "sequence?"}</option>`).join("")}</select>
      <button type="button" data-remove="${i}" title="Remove">×</button></li>`).join("");
  $("#analyze-btn").disabled = state.files.length === 0 || state.busy;
}

function addFiles(list) {
  for (const file of list) state.files.push({ file, seq: guessSequence(file.name) });
  renderFileList();
}

function setupUpload() {
  const drop = $("#drop");
  $("#file-input").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
  $("#file-list").addEventListener("change", (e) => { if (e.target.dataset.i) state.files[+e.target.dataset.i].seq = e.target.value; });
  $("#file-list").addEventListener("click", (e) => {
    if (e.target.dataset.remove) { state.files.splice(+e.target.dataset.remove, 1); renderFileList(); }
  });
  $("#scan-date").value = new Date().toISOString().slice(0, 10);

  $("#upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData();
    state.files.forEach((f) => fd.append("files", f.file));
    fd.append("sequences", state.files.map((f) => f.seq).join(","));
    fd.append("scan_type", $("#scan-type").value);
    fd.append("patient_label", $("#patient-label").value.trim() || "Unlabeled");
    fd.append("scan_date", $("#scan-date").value);
    if ($("#pixel-spacing").value) fd.append("pixel_spacing_mm", $("#pixel-spacing").value);
    state.busy = true;
    renderFileList();
    $("#analyze-btn").textContent = "Analyzing…";
    $("#upload-msg").textContent = "Running detection, segmentation and uncertainty estimation. Volumes can take a minute on CPU.";
    try {
      const study = await api("/api/analyze", { method: "POST", body: fd });
      state.files = [];
      $("#upload-msg").textContent = "";
      await loadPatients();
      openStudy(study);
    } catch (err) {
      $("#upload-msg").textContent = `Error: ${err.message}`;
    } finally {
      state.busy = false;
      $("#analyze-btn").textContent = "Analyze scan";
      renderFileList();
    }
  });
}

// ------------------------------------------------------------------ patients list
async function loadPatients() {
  const studies = await api("/api/studies");
  const byPatient = new Map();
  for (const s of studies) {
    if (!byPatient.has(s.patient.id)) byPatient.set(s.patient.id, { label: s.patient.label, studies: [] });
    byPatient.get(s.patient.id).studies.push(s);
  }
  $("#patient-options").innerHTML = [...byPatient.values()].map((p) => `<option value="${esc(p.label)}">`).join("");
  if (!studies.length) { $("#patients").innerHTML = `<p class="muted">No studies yet.</p>`; return; }
  $("#patients").innerHTML = [...byPatient.values()].map((p) => `
    <div class="patient"><div class="patient-name">${esc(p.label)}</div>
      ${p.studies.slice().reverse().map((s) => `
        <button class="study-link ${state.study?.id === s.id ? "active" : ""}" data-study="${esc(s.id)}">
          <span>${esc(s.scan_date)} · ${esc(s.profile.display_name)}</span>
          <span class="tag">${esc(s.classification.label)}</span>
        </button>`).join("")}
    </div>`).join("");
}

// ------------------------------------------------------------------ study view
function openStudy(study) {
  state.study = study;
  state.slice = study.measurements.key_slice ?? Math.floor(study.scan.shape[0] / 2);
  state.sequence = "";
  $("#empty").hidden = true;
  $("#study").hidden = false;
  $("#study-title").textContent = `${study.profile.display_name}: ${study.classification.display}`;
  $("#study-sub").textContent = `${study.patient.label} · scan date ${study.scan_date} · study ${study.id}`;
  $("#report-doctor").href = `/api/studies/${study.id}/report.pdf?audience=doctor`;
  $("#report-patient").href = `/api/studies/${study.id}/report.pdf?audience=patient`;
  document.querySelectorAll(".study-link").forEach((b) => b.classList.toggle("active", b.dataset.study === study.id));

  renderBanners(study);
  renderClassification(study);
  renderMeasurements(study);
  renderScanInfo(study);
  renderLegend(study);

  const z = study.scan.shape[0];
  $("#slider-row").style.display = z > 1 ? "flex" : "none";
  $("#slice-slider").max = z - 1;
  $("#slice-slider").value = state.slice;
  $("#sequence-select").innerHTML = study.scan.sequences.map((s) => `<option value="${esc(s)}">${esc(s.toUpperCase())}</option>`).join("");
  $("#sequence-select").style.display = study.scan.sequences.length > 1 ? "" : "none";
  const pref = ["t1ce", "t1", "flair", "t2"].find((s) => study.scan.sequences.includes(s));
  if (pref) { $("#sequence-select").value = pref; state.sequence = pref; }
  updateSlice();
  switchTab("viewer");
  loadChat();
}

function renderBanners(s) {
  const parts = [];
  if (s.mode !== "model") {
    parts.push(`<div class="banner warn"><b>${s.mode === "demo" ? "Demo mode" : "Partly demo mode"}:</b> some results come from a simple image-processing detector, not a trained AI model, so the tumor type cannot be determined. Train models with <code>oncovision.training</code> (see README).</div>`);
  }
  if (s.review.required) {
    parts.push(`<div class="banner bad"><b>Flagged for expert review</b><ul>${s.review.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>`);
  }
  parts.push(`<div class="banner warn" style="opacity:.8">${esc(s.disclaimer)}</div>`);
  $("#banners").innerHTML = parts.join("");
}

function renderClassification(s) {
  const c = s.classification;
  const probs = Object.entries(c.probabilities).sort((a, b) => b[1] - a[1]);
  $("#classification-card").innerHTML = `
    <h3>AI result</h3>
    <div class="big-label">${esc(c.display)}</div>
    <span class="method">${c.method === "model" ? "trained classifier" : "heuristic (demo)"}</span>
    <div class="bars">${probs.map(([k, v], i) => `
      <div class="bar ${i === 0 ? "top" : ""}"><span>${esc(k.replace("_", " "))}</span>
        <div class="track"><div class="fill" style="width:${(v * 100).toFixed(1)}%"></div></div><span>${pct(v)}</span></div>`).join("")}
    </div>
    <div class="gauges">
      <div class="gauge"><b>${pct(c.confidence)}</b><span>confidence</span></div>
      <div class="gauge"><b>${(s.segmentation.mean_uncertainty_in_tumor ?? 0).toFixed(2)}</b><span>boundary uncertainty (0-1)</span></div>
    </div>
    ${c.mc_std != null ? `<p class="hint">Monte-Carlo dropout spread: ±${(c.mc_std * 100).toFixed(1)}%</p>` : ""}`;
}

function sizeText(m) {
  const t = m.total || {};
  if (t.volume_cm3 != null) return `${t.volume_cm3} cm³`;
  if (t.area_mm2 != null) return `${t.area_mm2} mm² (${t.area_cm2} cm²)`;
  return `${t.voxels ?? t.pixels} ${m.is_3d ? "voxels" : "pixels"}`;
}

function renderMeasurements(s) {
  const m = s.measurements;
  if (!m.detected) {
    $("#measurements-card").innerHTML = `<h3>Measurements</h3><p class="muted">No tumor region was segmented.</p>`;
    return;
  }
  const u = m.units;
  const rows = [
    ["Tumor size", sizeText(m)],
    ["Longest diameter", `${m.longest_diameter} ${u}`],
    ["Perpendicular diameter", `${m.perpendicular_diameter} ${u}`],
    [s.profile.organ === "brain" ? "RANO product" : "Diameter product", `${m.bidimensional_product} ${u}²`],
  ];
  if (m.craniocaudal_extent) rows.push(["Top-to-bottom extent", `${m.craniocaudal_extent} ${u}`]);
  const loc = m.location || {};
  if (loc.side) rows.push(["Side", loc.side]);
  if (loc.approximate_region) rows.push(["Approximate region", loc.approximate_region]);
  if (loc.depth_from_skin != null) rows.push(["Depth from skin", `${loc.depth_from_skin} ${u}`]);
  if (loc.distance_from_midline != null) rows.push(["Distance from midline", `${loc.distance_from_midline} ${u}`]);
  for (const r of m.regions || []) {
    const v = r.volume_cm3 != null ? `${r.volume_cm3} cm³` : r.area_mm2 != null ? `${r.area_mm2} mm²` : r.voxels ?? r.pixels;
    rows.push([r.description, `${v} · ${r.percent_of_tumor}%`]);
  }
  $("#measurements-card").innerHTML = `<h3>Measurements</h3>
    <table>${rows.map(([a, b]) => `<tr><td>${esc(a)}</td><td>${esc(b)}</td></tr>`).join("")}</table>
    ${s.scan.spacing_estimated ? `<p class="hint">Pixel size was estimated from typical head size, so sizes are approximate. Enter the real pixel spacing under Advanced for exact values.</p>` : ""}
    ${u === "px" ? `<p class="hint">Pixel size unknown, so sizes are in pixels. Enter the pixel spacing under Advanced.</p>` : ""}`;
}

function renderScanInfo(s) {
  const sc = s.scan;
  const rows = [
    ["Format", sc.source_format],
    ["Dimensions", sc.shape.join(" × ")],
    ["Sequences", sc.sequences.join(", ")],
    ["Scan type detection", `${s.route.method} (${pct(s.route.confidence)}): ${s.route.reason}`],
  ];
  if (sc.spacing_mm) rows.push(["Voxel size (mm)", sc.spacing_mm.map((v) => v.toFixed(2)).join(" × ") + (sc.spacing_estimated ? " (estimated)" : "")]);
  if (sc.removed_phi?.length) rows.push(["Privacy", `Removed ${sc.removed_phi.length} identifying DICOM field(s)`]);
  $("#scan-card").innerHTML = `<h3>Scan</h3><table>${rows.map(([a, b]) => `<tr><td>${esc(a)}</td><td>${esc(b)}</td></tr>`).join("")}</table>`;
}

function renderLegend(s) {
  $("#legend").innerHTML = s.profile.seg_labels.map((l) => `<span><i style="background:rgb(${l.color.join(",")})"></i>${esc(l.description)}</span>`).join("");
}

function updateSlice() {
  const s = state.study;
  if (!s) return;
  const seq = state.sequence ? `&sequence=${encodeURIComponent(state.sequence)}` : "";
  $("#slice-img").src = `/api/studies/${s.id}/slice/${state.slice}.png?layer=${state.layer}${seq}`;
  $("#slice-label").textContent = `${state.slice + 1} / ${s.scan.shape[0]}`;
  $("#layer-help").textContent = LAYER_HELP[state.layer];
  $("#legend").style.display = state.layer === "overlay" ? "flex" : "none";
}

function setupViewer() {
  $("#layer-buttons").addEventListener("click", (e) => {
    const layer = e.target.dataset.layer;
    if (!layer) return;
    state.layer = layer;
    document.querySelectorAll("#layer-buttons button").forEach((b) => b.classList.toggle("active", b === e.target));
    updateSlice();
  });
  $("#slice-slider").addEventListener("input", (e) => { state.slice = +e.target.value; updateSlice(); });
  $("#sequence-select").addEventListener("change", (e) => { state.sequence = e.target.value; updateSlice(); });
  $("#jump-key").addEventListener("click", () => {
    state.slice = state.study.measurements.key_slice ?? state.slice;
    $("#slice-slider").value = state.slice;
    updateSlice();
  });
  $("#patients").addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-study]");
    if (btn) openStudy(await api(`/api/studies/${btn.dataset.study}`));
  });
  $("#delete-study").addEventListener("click", async () => {
    if (!state.study || !confirm("Delete this study and its chat? This cannot be undone.")) return;
    await api(`/api/studies/${state.study.id}`, { method: "DELETE" });
    state.study = null;
    $("#study").hidden = true;
    $("#empty").hidden = false;
    $("#messages").innerHTML = `<p class="muted">Analyze or open a study to start chatting.</p>`;
    setChatEnabled(false);
    loadPatients();
  });
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  for (const pane of ["viewer", "three", "timeline"]) $(`#tab-${pane}`).hidden = pane !== name;
  if (name === "three") load3D();
  else disposeViewer();
  if (name === "timeline") loadTimeline();
}

// ------------------------------------------------------------------ 3D
async function load3D() {
  const s = state.study;
  $("#mesh-toggles").innerHTML = "";
  if (!s.scan.is_volume) {
    $("#three-msg").textContent = "3D view needs a volume (NIfTI file or DICOM series). This study is a single 2D image.";
    $("#three-container").innerHTML = "";
    return;
  }
  $("#three-msg").textContent = "Building 3D surfaces…";
  try {
    const data = await api(`/api/studies/${s.id}/mesh`);
    const handles = await renderMeshes($("#three-container"), data);
    $("#three-msg").textContent = data.meshes.length > 1 ? "" : "No tumor surface to show.";
    $("#mesh-toggles").innerHTML = data.meshes.map((m, i) => `
      <label><input type="checkbox" data-mesh="${i}" checked><i style="display:inline-block;width:10px;height:10px;border-radius:2px;background:rgb(${m.color.join(",")})"></i>${esc(m.description || m.name)}</label>`).join("");
    $("#mesh-toggles").onchange = (e) => { if (e.target.dataset.mesh) handles[+e.target.dataset.mesh].visible = e.target.checked; };
  } catch (err) {
    $("#three-msg").textContent = `Could not load 3D view: ${err.message}`;
  }
}

// ------------------------------------------------------------------ timeline
async function loadTimeline() {
  const s = state.study;
  const points = (await api(`/api/patients/${s.patient.id}/timeline`)).filter((p) => p.profile === s.profile.key);
  const chart = $("#timeline-chart");
  if (points.length < 2) {
    chart.innerHTML = `<p class="muted">Only one ${esc(s.profile.display_name)} study for ${esc(s.patient.label)}. Upload another scan with the same patient label and a different date to track changes over time.</p>`;
  } else {
    const W = 640, H = 240, P = 40;
    const max = Math.max(...points.map((p) => p.size || 0)) * 1.15 || 1;
    const t0 = new Date(points[0].scan_date), t1 = new Date(points.at(-1).scan_date);
    const span = Math.max(1, t1 - t0);
    const x = (p) => P + ((new Date(p.scan_date) - t0) / span) * (W - 2 * P);
    const y = (p) => H - P + 10 - ((p.size || 0) / max) * (H - 2 * P);
    const path = points.map((p, i) => `${i ? "L" : "M"}${x(p).toFixed(1)},${y(p).toFixed(1)}`).join(" ");
    const m = s.measurements;
    const unit = m.units === "px" ? (m.is_3d ? "voxels" : "pixels") : m.is_3d ? "cm³" : "mm²";
    chart.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Tumor size over time">
      <text x="${P - 30}" y="16" fill="#8b97b0" font-size="11">Total tumor size (${unit})</text>
      <line x1="${P}" y1="${H - P + 10}" x2="${W - P}" y2="${H - P + 10}" stroke="#22304d"/>
      <path d="${path}" fill="none" stroke="#38bdf8" stroke-width="2.5"/>
      ${points.map((p) => `<g><circle cx="${x(p)}" cy="${y(p)}" r="${p.study_id === s.id ? 6 : 4.5}" fill="${p.study_id === s.id ? "#34d399" : "#38bdf8"}"><title>${esc(p.scan_date)}: ${p.size}</title></circle>
        <text x="${x(p)}" y="${y(p) - 12}" fill="#e5e9f2" font-size="11" text-anchor="middle">${p.size}</text>
        <text x="${x(p)}" y="${H - 8}" fill="#8b97b0" font-size="11" text-anchor="middle">${esc(p.scan_date)}</text></g>`).join("")}
    </svg>`;
  }
  const others = points.filter((p) => p.study_id !== s.id && p.scan_date <= s.scan_date);
  const card = $("#comparison-card");
  if (!others.length) { card.innerHTML = `<h3>Comparison</h3><p class="muted">No earlier study of this type to compare with.</p>`; return; }
  const c = await api(`/api/studies/${s.id}/compare/${others.at(-1).study_id}`);
  const cls = c.response.category.split(" ")[0];
  const fmt = (v) => (v == null ? "n/a" : `${v > 0 ? "+" : ""}${v}%`);
  card.innerHTML = `<h3>Compared with ${esc(c.prior_date)} (${c.interval_days} days earlier)</h3>
    <p><span class="response ${esc(cls)}">${esc(c.response.category)}</span> <span class="muted">by ${esc(c.response.criteria)}, ${esc(c.response.metric)} ${fmt(c.response.change_percent)}</span></p>
    <table>
      <tr><td>Total size</td><td>${c.size_before} → ${c.size_after} ${esc(c.size_unit)} (${fmt(c.size_change_percent)})</td></tr>
      <tr><td>Longest diameter</td><td>${c.longest_diameter_before ?? 0} → ${c.longest_diameter_after ?? 0}</td></tr>
      ${Object.entries(c.regions).map(([k, r]) => `<tr><td>${esc(k.replace("_", " "))}</td><td>${r.before} → ${r.after} (${fmt(r.change_percent)})</td></tr>`).join("")}
    </table>
    <p class="hint">Simplified, imaging-only criteria. Real response assessment also considers new lesions, steroid use and clinical status.</p>`;
}

// ------------------------------------------------------------------ chat
function setChatEnabled(on) {
  $("#chat-text").disabled = !on;
  $("#chat-send").disabled = !on;
}

function renderSuggestions() {
  $("#suggestions").innerHTML = state.study ? SUGGESTIONS[state.audience].map((q) => `<button type="button">${esc(q)}</button>`).join("") : "";
}

function addMessage(role, html, cls = "") {
  const div = document.createElement("div");
  div.className = `msg ${role} ${cls}`;
  div.innerHTML = html;
  $("#messages").appendChild(div);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return div;
}

async function loadChat() {
  $("#messages").innerHTML = "";
  setChatEnabled(true);
  renderSuggestions();
  const history = await api(`/api/studies/${state.study.id}/chat`);
  if (!history.length) {
    addMessage("assistant", md(`Hi! I can explain this ${state.study.profile.display_name.toLowerCase()} result, show specific slices, compare with earlier scans and create reports. What would you like to know?`));
  }
  for (const m of history) {
    if (m.role === "user") addMessage("user", esc(m.text));
    else addMessage("assistant", md(m.text) + (m.images || []).map((u) => `<img src="${esc(u)}" alt="Slice the assistant looked at">`).join(""));
  }
}

const TOOL_LABELS = {
  get_findings: () => "Reading the analysis results",
  view_slice: (i) => `Looking at slice ${i.slice_index ?? "with the largest tumor"}`,
  list_patient_studies: () => "Listing this patient's studies",
  compare_studies: () => "Comparing with an earlier scan",
  search_medical_knowledge: (i) => `Searching medical sources: “${i.query ?? ""}”`,
  create_report: () => "Preparing a PDF report",
};

async function sendChat(text) {
  if (!state.study || !text.trim() || state.busy) return;
  state.busy = true;
  setChatEnabled(false);
  addMessage("user", esc(text));
  const bubble = addMessage("assistant", "", "typing");
  let raw = "", chips = "", images = "";
  const paint = () => {
    bubble.innerHTML = chips + md(raw) + images;
    $("#messages").scrollTop = $("#messages").scrollHeight;
  };
  try {
    const res = await fetch(`/api/studies/${state.study.id}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, audience: state.audience, language: state.language }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 2);
        if (!line.startsWith("data:")) continue;
        const ev = JSON.parse(line.slice(5));
        if (ev.type === "text") raw += ev.text;
        else if (ev.type === "tool") chips += `<span class="tool-chip">${esc((TOOL_LABELS[ev.name] || (() => ev.name))(ev.input || {}))}</span>`;
        else if (ev.type === "image") images += `<img src="${esc(ev.url)}" alt="Slice the assistant looked at">`;
        else if (ev.type === "error") { bubble.classList.add("error"); raw += `\n\n${ev.message}`; }
        paint();
      }
    }
  } catch (err) {
    bubble.classList.add("error");
    raw += `\n\nError: ${err.message}`;
    paint();
  } finally {
    bubble.classList.remove("typing");
    state.busy = false;
    setChatEnabled(true);
    $("#chat-text").focus();
  }
}

function setupChat() {
  $("#chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("#chat-text").value;
    $("#chat-text").value = "";
    sendChat(text);
  });
  $("#chat-text").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chat-form").requestSubmit(); }
  });
  $("#suggestions").addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") sendChat(e.target.textContent); });
  $("#audience").addEventListener("click", (e) => {
    if (!e.target.dataset.audience) return;
    state.audience = e.target.dataset.audience;
    document.querySelectorAll("#audience button").forEach((b) => b.classList.toggle("active", b === e.target));
    renderSuggestions();
  });
  $("#language").addEventListener("change", (e) => { state.language = e.target.value; });
  $("#messages").addEventListener("click", (e) => {
    // Links from the assistant (sources, report links) open in a new tab.
    const a = e.target.closest("a");
    if (a) { e.preventDefault(); window.open(a.href, "_blank", "noopener"); }
  });
}

// ------------------------------------------------------------------ boot
setupUpload();
setupViewer();
setupChat();
loadStatus().catch((err) => { $("#status").innerHTML = `<span class="pill warn">Backend unreachable: ${esc(err.message)}</span>`; });
loadPatients().catch(() => {});
