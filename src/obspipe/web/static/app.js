"use strict";

// obspipe trace explorer. No framework, no build step: the server serves this
// file as-is and every view is rendered from the JSON API.

const state = {
  meta: null,
  traces: [],
  selectedId: null,
  detail: null,
  step: null,
  subtab: "overview",
  newCaseId: null,
};

const HEALTH = {
  healthy: { label: "Healthy", icon: "✓", chip: "good" },
  low_confidence: { label: "Low confidence", icon: "!", chip: "warn" },
  affected: { label: "Affected", icon: "!", chip: "warn" },
  root_cause: { label: "Root cause", icon: "✕", chip: "crit" },
  skipped: { label: "Skipped", icon: "–", chip: "skip" },
};
const STEP_NO = { intake: 1, extraction: 2, classification: 3, summarization: 4 };

// -- utilities -----------------------------------------------------------------

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const json = (value) => esc(JSON.stringify(value, null, 2));
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : "");
const pct = (v, digits = 0) => (v == null ? "–" : `${(v * 100).toFixed(digits)}%`);
const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "–");
const categoryLabel = (value) => state.meta?.categories.find((c) => c.value === value)?.label || value || "–";

async function api(path, options = {}) {
  const init = { headers: { "Content-Type": "application/json" }, ...options };
  if (init.body && typeof init.body !== "string") init.body = JSON.stringify(init.body);
  const response = await fetch(path, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

function toast(html, ms = 5200) {
  const el = $("#toast");
  el.innerHTML = html;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), ms);
}

function showTip(event, html) {
  const tip = $("#tooltip");
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14;
  const { innerWidth: w, innerHeight: h } = window;
  const rect = tip.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + rect.width > w - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > h - 8) y = event.clientY - rect.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
const hideTip = () => ($("#tooltip").hidden = true);

function bindTips(root) {
  root.querySelectorAll("[data-tip]").forEach((el) => {
    el.addEventListener("mousemove", (e) => showTip(e, el.dataset.tip));
    el.addEventListener("mouseleave", hideTip);
  });
}

function healthChip(health) {
  const h = HEALTH[health] || HEALTH.healthy;
  return `<span class="chip ${h.chip}"><span aria-hidden="true">${h.icon}</span>${h.label}</span>`;
}

function verdictChip(verdict) {
  if (verdict === "failure") return `<span class="chip crit"><span aria-hidden="true">✕</span>Failure</span>`;
  if (verdict === "healthy") return `<span class="chip good"><span aria-hidden="true">✓</span>Healthy</span>`;
  return `<span class="chip neutral">Not diagnosed</span>`;
}

// -- routing ---------------------------------------------------------------------

function route() {
  const [, tab = "traces", id] = location.hash.split("/");
  document.querySelectorAll(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.tab === tab));
  ["traces", "evals", "analytics"].forEach((name) => ($(`#view-${name}`).hidden = name !== tab));
  if (tab === "traces") {
    if (id && id !== state.selectedId) selectTrace(id);
  } else if (tab === "evals") {
    renderEvals();
  } else if (tab === "analytics") {
    renderAnalytics();
  }
}

// -- trace list --------------------------------------------------------------------

async function loadTraces() {
  const params = new URLSearchParams();
  const q = $("#filter-q").value.trim();
  if (q) params.set("q", q);
  if ($("#filter-batch").value) params.set("batch", $("#filter-batch").value);
  if ($("#filter-verdict").value) params.set("verdict", $("#filter-verdict").value);
  if ($("#filter-evals").checked) params.set("include_evals", "1");
  const data = await api(`/api/traces?${params}`);
  state.traces = data.traces;

  const batchSelect = $("#filter-batch");
  const current = batchSelect.value;
  batchSelect.innerHTML = `<option value="">All batches</option>` +
    data.batches.map((b) => `<option ${b === current ? "selected" : ""}>${esc(b)}</option>`).join("");
  renderTraceList();
}

function renderTraceList() {
  const failures = state.traces.filter((t) => t.verdict === "failure").length;
  $("#trace-count").textContent = `${state.traces.length} traces · ${failures} failures`;
  $("#trace-list").innerHTML = state.traces
    .map(
      (t) => `<li data-id="${t.trace_id}" class="${t.verdict || ""} ${t.trace_id === state.selectedId ? "selected" : ""}">
        <div class="row1"><span class="doc">${esc(t.doc_id)}</span>${t.flag ? `<span class="chip neutral" title="flag ${esc(t.flag)}">⚑ ${esc(t.flag)}</span>` : ""}</div>
        <div class="row2"><span>${t.verdict === "failure" ? esc(categoryLabel(t.category)) + " · " + esc(t.root_step) : esc(t.verdict || t.status)}</span>
        <span>${esc(t.batch || "")}${t.pipeline_version ? " · " + esc(t.pipeline_version) : ""}</span></div>
      </li>`
    )
    .join("") || `<li class="muted">No traces match. Run <code>obspipe demo seed</code> to create some.</li>`;
}

// -- trace detail -----------------------------------------------------------------

async function selectTrace(id, keepStep = false) {
  state.selectedId = id;
  if (location.hash !== `#/traces/${id}`) history.replaceState(null, "", `#/traces/${id}`);
  renderTraceList();
  $("#trace-detail").innerHTML = `<p class="muted"><span class="spinner"></span> Loading trace…</p>`;
  try {
    state.detail = await api(`/api/traces/${id}`);
  } catch (err) {
    $("#trace-detail").innerHTML = `<div class="banner">${esc(err.message)}</div>`;
    return;
  }
  const diagnosis = state.detail.diagnosis;
  if (!keepStep || !state.step) {
    state.step = diagnosis.root_cause_step || state.detail.spans[0]?.step;
    state.subtab = diagnosis.root_cause_step && diagnosis.method === "ground_truth" ? "diff" : "overview";
  }
  renderDetail();
}

function renderDetail() {
  const { trace, diagnosis, flags, final_output: final } = state.detail;
  const openFlag = flags.find((f) => f.status === "open");
  const resolved = flags.find((f) => f.status === "confirmed" || f.status === "overridden");
  const tags = trace.tags || {};

  $("#trace-detail").innerHTML = `
    <div class="detail-head">
      <div>
        <h2>${esc(trace.doc_id)}</h2>
        <div class="meta">
          <span class="mono">${esc(trace.trace_id)}</span>
          <span>${esc(trace.source_name)}</span>
          ${tags.batch ? `<span>batch <b>${esc(tags.batch)}</b></span>` : ""}
          ${tags.pipeline_version ? `<span>pipeline <b>${esc(tags.pipeline_version)}</b></span>` : ""}
          <span>${esc(trace.backend)} · ${esc(trace.model || "-")}</span>
          <span>${trace.duration_ms.toFixed(0)} ms</span>
          <span>${fmtTime(trace.started_at)}</span>
          ${trace.has_golden ? `<span class="chip neutral">ground truth available</span>` : ""}
        </div>
      </div>
      <div class="button-row">
        ${verdictChip(diagnosis.verdict)}
        ${resolved ? `<span class="chip neutral">⚑ ${esc(resolved.status)} · <a href="#/evals">${esc(resolved.eval_case_id)}</a></span>` : ""}
        ${openFlag ? `<button class="danger" id="resume-flag">⚑ Resume open flag</button>` :
          resolved ? "" : `<button class="danger" id="flag-btn">⚑ Flag bad output</button>`}
      </div>
    </div>

    ${final ? `<div class="card">
      <div class="card-title"><h4>Final output</h4><span class="muted small">what the user saw · ${esc(final.doc_type)} summary</span></div>
      <div style="font-weight:600;margin-bottom:4px">${esc(final.headline)}</div>
      <ul style="margin:4px 0 6px;padding-left:18px">${(final.bullets || []).map((b) => `<li>${esc(b)}</li>`).join("")}</ul>
      ${(final.unresolved_questions || []).map((q) => `<div class="small" style="color:var(--warn)">? ${esc(q)}</div>`).join("")}
      <details><summary class="small muted">type-specific fields</summary><pre>${json(final.type_specific)}</pre></details>
    </div>` : `<div class="card"><h4>Final output</h4><p class="muted">No summary was produced - the run stopped early.</p></div>`}

    <div class="card">
      <div class="card-title"><h4>Pipeline</h4><span class="muted small">click a step for its span</span></div>
      <div class="pipeline">${renderNodes(diagnosis)}</div>
      <div class="legend"><span class="l-good">healthy</span><span class="l-warn">low confidence / affected</span><span class="l-crit">identified root cause</span><span class="l-skip">skipped</span></div>
    </div>

    ${renderDiagnosis(diagnosis)}

    <div class="card" id="span-panel"></div>
  `;

  document.querySelectorAll(".node").forEach((node) =>
    node.addEventListener("click", () => {
      state.step = node.dataset.step;
      renderDetail();
      $("#span-panel").scrollIntoView({ behavior: "smooth", block: "start" });
    })
  );
  $("#flag-btn")?.addEventListener("click", () => openFlagModal());
  $("#resume-flag")?.addEventListener("click", () => openFlagModal(openFlag));
  renderSpanPanel();
}

function renderNodes(diagnosis) {
  const byStep = Object.fromEntries(diagnosis.steps.map((s) => [s.step, s]));
  const spans = state.detail.spans;
  return spans
    .map((span, i) => {
      const a = byStep[span.step] || { health: "healthy" };
      const hot = a.health === "root_cause" || a.health === "affected";
      const arrow = i < spans.length - 1
        ? `<div class="arrow ${hot && (byStep[spans[i + 1].step]?.health === "affected" || byStep[spans[i + 1].step]?.health === "skipped") ? "hot" : ""}" aria-hidden="true">
             <svg width="26" height="14" viewBox="0 0 26 14"><path d="M1 7h21M17 2l5 5-5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div>`
        : "";
      const conf = span.self_confidence != null ? `${span.self_confidence}/5` : "–";
      const measured = span.computed_confidence != null ? span.computed_confidence.toFixed(2) : "–";
      return `<button class="node ${a.health} ${span.step === state.step ? "selected" : ""}" data-step="${span.step}" aria-label="${span.step}: ${HEALTH[a.health]?.label}">
          <div class="n">Step ${STEP_NO[span.step]}</div>
          <div class="name">${cap(span.step)}</div>
          ${healthChip(a.health)}
          <div class="stats">
            <span>self-rated</span><b>${conf}</b>
            <span>measured</span><b>${measured}</b>
            <span>suspicion</span><b>${span.suspicion ? span.suspicion.toFixed(2) : "–"}</b>
            <span>warnings</span><b>${span.warnings.length}</b>
          </div>
        </button>${arrow}`;
    })
    .join("");
}

function renderDiagnosis(d) {
  if (d.verdict === "healthy") {
    return `<div class="card diagnosis healthy">
      <div class="card-title"><h3>Diagnosis: no failure found</h3><span class="muted small">${esc(d.method.replace("_", " "))} · ${d.computed_ms.toFixed(1)} ms</span></div>
      <p class="narrative">${esc(d.narrative)}</p></div>`;
  }
  const links = d.chain
    .map(
      (link) => `<li class="link ${link.role}">
        <div class="link-head">
          ${link.role === "root_cause" ? `<span class="chip crit">✕ Root cause</span>` : `<span class="chip warn">↳ Propagation</span>`}
          <b>Step ${STEP_NO[link.step]} · ${cap(link.step)}</b>
          <span class="codes">${link.warning_codes.map((c) => `<span class="code-tag">${esc(c)}</span>`).join("")}</span>
        </div>
        <div>${esc(link.finding)}</div>
        ${link.received || link.produced || link.expected ? `<details ${link.role === "root_cause" ? "open" : ""}><summary class="small muted">evidence: input / output pair</summary>
          <div class="evidence">
            ${link.received ? `<div><h5>Received</h5><pre>${json(link.received)}</pre></div>` : ""}
            ${link.produced ? `<div><h5>Produced</h5><pre>${json(link.produced)}</pre></div>` : ""}
            ${link.expected ? `<div><h5>Expected</h5><pre>${json(link.expected)}</pre></div>` : ""}
          </div></details>` : ""}
      </li>`
    )
    .join("");
  return `<div class="card diagnosis failure">
    <div class="card-title"><h3>${esc(d.headline)}</h3>
      <span class="muted small">confidence ${pct(d.confidence)} · ${esc(d.method.replace("_", " "))} · diagnosed in ${d.computed_ms.toFixed(1)} ms</span></div>
    <p class="narrative">${esc(d.narrative)}</p>
    <h4>Evidence chain</h4>
    <ol class="chain">${links}</ol>
  </div>`;
}

// -- span panel --------------------------------------------------------------------

const SUBTABS = [
  ["overview", "Overview"],
  ["diff", "Diff"],
  ["input", "Input"],
  ["output", "Output"],
  ["prompt", "Prompt"],
  ["raw", "Raw response"],
];

function renderSpanPanel() {
  const span = state.detail.spans.find((s) => s.step === state.step);
  if (!span) return;
  const panel = $("#span-panel");
  panel.innerHTML = `
    <div class="card-title"><h3>Step ${STEP_NO[span.step]} · ${cap(span.step)} <span class="muted small">span ${esc(span.span_id)}</span></h3>
      <span class="chip ${span.status === "ok" ? "good" : span.status === "degraded" ? "warn" : span.status === "failed" ? "crit" : "skip"}">${esc(span.status)}</span></div>
    <div class="subtabs" role="tablist">${SUBTABS.map(([k, label]) => `<button data-sub="${k}" class="${state.subtab === k ? "active" : ""}">${label}</button>`).join("")}</div>
    <div id="sub-body"></div>`;
  panel.querySelectorAll("[data-sub]").forEach((b) =>
    b.addEventListener("click", () => {
      state.subtab = b.dataset.sub;
      renderSpanPanel();
    })
  );
  $("#sub-body").innerHTML = renderSubtab(span);
}

function renderSubtab(span) {
  const call = span.llm_calls[0];
  switch (state.subtab) {
    case "overview":
      return `<dl class="kv">
          <dt>status</dt><dd>${esc(span.status)}${span.error_type ? ` · <b style="color:var(--crit)">${esc(span.error_type)}</b>` : ""}</dd>
          <dt>duration</dt><dd>${span.duration_ms.toFixed(2)} ms</dd>
          <dt>self-reported confidence</dt><dd>${span.self_confidence != null ? `${span.self_confidence}/5` : "–"} ${span.confidence_reason ? `<span class="muted">“${esc(span.confidence_reason)}”</span>` : ""}</dd>
          <dt>measured confidence</dt><dd>${span.computed_confidence != null ? span.computed_confidence.toFixed(3) : "–"}</dd>
          <dt>suspicion</dt><dd>${span.suspicion.toFixed(3)}</dd>
          <dt>LLM calls</dt><dd>${span.llm_calls.length ? span.llm_calls.map((c) => `${esc(c.schema_name)} · ${c.duration_ms.toFixed(1)} ms · ${c.tokens.total_tokens ?? "–"} tokens`).join("<br>") : "none (deterministic step)"}</dd>
          ${Object.keys(span.attributes || {}).length ? `<dt>attributes</dt><dd class="mono">${esc(Object.entries(span.attributes).map(([k, v]) => `${k}=${v}`).join("  "))}</dd>` : ""}
        </dl>
        ${span.error_message ? `<div class="banner">${esc(span.error_message)}</div>` : ""}
        <h4>Warnings</h4>
        ${span.warnings.length ? `<ul class="warnings">${span.warnings.map((w) => `<li class="${w.severity}"><span class="code-tag">${esc(w.code)}</span> <span class="muted small">${esc(w.severity)}${w.field_path ? " · " + esc(w.field_path) : ""}</span><div>${esc(w.message)}</div></li>`).join("")}</ul>` : `<p class="muted">None.</p>`}`;
    case "diff":
      return renderDiff(span.step);
    case "input":
      return Object.keys(span.inputs || {}).length ? `<pre>${json(span.inputs)}</pre>` : `<p class="muted">No input recorded${span.status === "skipped" ? " - the step never ran" : ""}.</p>`;
    case "output":
      return span.output != null ? `<pre>${json(span.output)}</pre>` : `<p class="muted">No output.</p>`;
    case "prompt":
      if (!call) return `<p class="muted">This step makes no model call.</p>`;
      return (call.prompt || []).flat().map((m) => `<div class="msg"><div class="role">${esc(m.role)}</div><pre>${esc(m.content)}</pre></div>`).join("");
    case "raw":
      if (!call) return `<p class="muted">This step makes no model call.</p>`;
      return `${call.parsing_error ? `<div class="banner">${esc(call.parsing_error)}</div>` : ""}
        <h4>Raw response</h4><pre>${esc(typeof call.raw_response === "string" ? call.raw_response : JSON.stringify(call.raw_response, null, 2))}</pre>
        <h4 style="margin-top:12px">Parsed</h4><pre>${json(call.parsed)}</pre>`;
  }
  return "";
}

// -- diff view ---------------------------------------------------------------------

function renderDiff(step) {
  const row = state.detail.diffs.find((d) => d.step === step);
  if (!row) return `<p class="muted">No diff available.</p>`;
  if (row.status === "skipped") return `<p class="muted">The step never ran, so there is nothing to compare.</p>`;
  const divs = row.divergences;
  const source = row.expected_source === "correction" ? "human correction" : row.expected_source === "golden" ? "golden dataset" : null;
  return `
    <p class="small muted">What the step received, what it produced, and what it should have produced${source ? ` (from the ${source})` : ""}. Divergent fields are highlighted.</p>
    <div class="diff">
      <div class="diff-col"><h5>Received</h5><div class="body">${renderPlain(row.received)}</div></div>
      <div class="diff-col"><h5>Produced</h5><div class="body">${renderSide(row.produced, row.expected, divs, "produced")}</div></div>
      <div class="diff-col"><h5>Expected</h5><div class="body">${row.expected ? renderSide(row.expected, row.expected, divs, "expected") : `<span class="muted">No ground truth for this step. Flag the trace and enter a corrected output to create one.</span>`}</div></div>
    </div>
    ${divs.length ? `<ul class="divergence-list">${divs.map((d) => `<li>${esc(describe(d))}</li>`).join("")}</ul>` : row.expected ? `<p class="small" style="color:var(--good)">✓ Output matches expectation on every checked field.</p>` : ""}`;
}

function describe(d) {
  if (d.kind === "missing") return `${d.path}: expected ${JSON.stringify(d.expected)}, which is missing`;
  if (d.kind === "unexpected") return `${d.path}: ${JSON.stringify(d.actual)} should not be there`;
  return `${d.path}: expected ${JSON.stringify(d.expected)}, got ${JSON.stringify(d.actual)}`;
}

function renderPlain(obj) {
  if (obj == null) return `<span class="muted">–</span>`;
  return Object.entries(obj)
    .map(([k, v]) => `<div class="drow"><span class="k">${esc(k)}:</span> ${renderValue(v)}</div>`)
    .join("");
}

function renderValue(v) {
  if (v && typeof v === "object" && !Array.isArray(v)) return `<pre style="max-height:180px;margin-top:3px">${json(v)}</pre>`;
  if (Array.isArray(v)) return v.length ? v.map((i) => `<span class="item">${esc(typeof i === "string" ? i : JSON.stringify(i))}</span>`).join("") : "[]";
  return esc(JSON.stringify(v));
}

function renderSide(obj, expected, divergences, side) {
  if (!obj) return `<span class="muted">–</span>`;
  const keys = side === "produced" ? Object.keys(obj) : Object.keys(expected || {});
  return keys
    .map((k) => {
      const v = obj[k];
      const here = divergences.filter((d) => d.path === k);
      const checked = expected && k in expected;
      const mismatch = here.some((d) => d.kind === "mismatch");
      const cls = mismatch ? (side === "produced" ? "bad" : "good") : "";
      let valueHtml;
      if (Array.isArray(v)) {
        const flagged = new Set(here.filter((d) => (side === "produced" ? d.kind === "unexpected" : d.kind === "missing")).map((d) => JSON.stringify(side === "produced" ? d.actual : d.expected)));
        const extras = side === "produced" ? here.filter((d) => d.kind === "missing").map((d) => d.expected) : [];
        valueHtml = (v.length ? v : []).map((i) => `<span class="item ${flagged.has(JSON.stringify(i)) ? (side === "produced" ? "extra" : "want") : ""}">${esc(typeof i === "string" ? i : JSON.stringify(i))}</span>`).join("") +
          extras.map((i) => `<span class="item bad" title="missing">${esc(i)}</span>`).join("") || "[]";
      } else if (v && typeof v === "object") {
        valueHtml = v.any_of ? `any of ${esc(v.any_of.join(" | "))}` : `<pre style="max-height:160px;margin-top:3px">${json(v)}</pre>`;
      } else {
        valueHtml = esc(JSON.stringify(v));
      }
      return `<div class="drow ${cls}" ${checked ? "" : 'style="opacity:.6"'}><span class="k">${esc(k)}:</span> ${valueHtml}</div>`;
    })
    .join("");
}

// -- flagging ------------------------------------------------------------------------

function closeModal() {
  $("#modal").hidden = true;
  $("#modal-card").innerHTML = "";
}

async function openFlagModal(existing) {
  const modal = $("#modal");
  const card = $("#modal-card");
  modal.hidden = false;
  let flag = existing;
  let diagnosis = state.detail.diagnosis;
  if (!flag) {
    card.innerHTML = `<h2 id="modal-title">Flag bad output</h2>
      <label class="form-grid"><span class="full">What looked wrong? <span class="muted">(optional)</span></span></label>
      <textarea id="flag-reason" style="min-height:70px" placeholder="e.g. the summary names a signatory who is not in the contract"></textarea>
      <div class="button-row" style="margin-top:12px"><button class="danger" id="flag-go">⚑ Flag and diagnose</button><button id="flag-cancel">Cancel</button></div>`;
    $("#flag-cancel").onclick = closeModal;
    $("#flag-reason").focus();
    await new Promise((resolve) => ($("#flag-go").onclick = resolve));
    const reason = $("#flag-reason").value;
    card.innerHTML = `<p><span class="spinner"></span> Running backward trace analysis…</p>`;
    try {
      const res = await api(`/api/traces/${state.selectedId}/flag`, { method: "POST", body: { reason } });
      flag = res.flag;
      diagnosis = res.diagnosis;
    } catch (err) {
      card.innerHTML = `<div class="banner">${esc(err.message)}</div><button onclick="closeModal()">Close</button>`;
      return;
    }
  }
  renderResolveForm(flag, diagnosis);
}

function correctionFor(step) {
  const row = state.detail.diffs.find((d) => d.step === step);
  if (!row) return "";
  return JSON.stringify(row.expected || row.produced || {}, null, 2);
}

function renderResolveForm(flag, diagnosis) {
  const card = $("#modal-card");
  const found = diagnosis.verdict === "failure";
  const steps = state.meta.steps;
  const cats = state.meta.categories;
  const initialStep = diagnosis.root_cause_step || "";
  card.innerHTML = `
    <h2 id="modal-title">${found ? "Root cause diagnosis" : "No automated root cause"}</h2>
    <div class="small muted">flag <span class="mono">${esc(flag.flag_id)}</span> · analysis took ${(diagnosis.computed_ms ?? 0).toFixed(1)} ms</div>
    ${found
      ? `<div class="card diagnosis failure" style="margin:12px 0;box-shadow:none"><b>${esc(diagnosis.headline)}</b> <span class="muted small">confidence ${pct(diagnosis.confidence)} · ${esc(diagnosis.method.replace("_", " "))}</span><p class="narrative" style="margin-bottom:0">${esc(diagnosis.narrative)}</p></div>`
      : `<div class="banner">Every check passed${state.detail.trace.has_golden ? " and the output matches ground truth" : ""}. This is a failure the system cannot see yet - name the step and category below, and the eval case you create will make it visible from now on.</div>`}
    <h3>Confirm or override</h3>
    <div class="form-grid">
      <label>Failing step<select id="r-step"><option value="">– choose –</option>${steps.map((s) => `<option value="${s}" ${s === initialStep ? "selected" : ""}>Step ${STEP_NO[s]} · ${cap(s)}</option>`).join("")}</select></label>
      <label>Failure category<select id="r-cat"><option value="">– choose –</option>${cats.map((c) => `<option value="${c.value}" ${c.value === diagnosis.category ? "selected" : ""}>${esc(c.label)}</option>`).join("")}</select></label>
      <label class="full">Corrected output for this step <span class="muted">(projection JSON - becomes the eval case's expected output)</span>
        <textarea id="r-corrected">${esc(correctionFor(initialStep))}</textarea></label>
      <label class="full">Notes<input type="text" id="r-notes" value="${esc(flag.reason || "")}"></label>
    </div>
    <div id="r-error"></div>
    <div class="button-row">
      <button class="primary" id="r-save">Confirm diagnosis</button>
      <button id="r-dismiss">Dismiss flag</button>
      <button id="r-later">Decide later</button>
    </div>`;

  const sync = () => {
    const same = $("#r-step").value === (diagnosis.root_cause_step || "") && $("#r-cat").value === (diagnosis.category || "");
    $("#r-save").textContent = same && found ? "Confirm diagnosis" : "Save override";
    $("#r-save").disabled = !$("#r-step").value || !$("#r-cat").value;
  };
  $("#r-step").addEventListener("change", () => {
    $("#r-corrected").value = correctionFor($("#r-step").value);
    sync();
  });
  $("#r-cat").addEventListener("change", sync);
  sync();
  $("#r-later").onclick = () => {
    closeModal();
    selectTrace(state.selectedId, true);
  };
  $("#r-dismiss").onclick = async () => {
    await api(`/api/flags/${flag.flag_id}/dismiss`, { method: "POST", body: { notes: $("#r-notes").value } });
    closeModal();
    toast("Flag dismissed.");
    selectTrace(state.selectedId, true);
    loadTraces();
  };
  $("#r-save").onclick = async () => {
    let corrected = $("#r-corrected").value.trim();
    try {
      corrected = corrected ? JSON.parse(corrected) : null;
    } catch {
      $("#r-error").innerHTML = `<div class="banner">Corrected output is not valid JSON.</div>`;
      return;
    }
    $("#r-save").disabled = true;
    try {
      const res = await api(`/api/flags/${flag.flag_id}/resolve`, {
        method: "POST",
        body: { step: $("#r-step").value, category: $("#r-cat").value, corrected_output: corrected, notes: $("#r-notes").value },
      });
      state.newCaseId = res.case.case_id;
      closeModal();
      toast(`Eval case <b>${esc(res.case.case_id)}</b> added (${esc(res.flag.status)}) - dataset now ${res.dataset_size} cases. <a href="#/evals">View eval dataset →</a>`, 8000);
      selectTrace(state.selectedId, true);
      loadTraces();
    } catch (err) {
      $("#r-error").innerHTML = `<div class="banner">${esc(err.message)}</div>`;
      $("#r-save").disabled = false;
    }
  };
}

// -- charts --------------------------------------------------------------------------

function hbarChart(rows, { width = 560, labelWidth = 180, format = (v) => v, tip } = {}) {
  if (!rows.length) return `<p class="muted">No data yet.</p>`;
  const bar = 18, gap = 9, top = 4;
  const height = top + rows.length * (bar + gap);
  const max = Math.max(...rows.map((r) => r.value), 1);
  const plot = width - labelWidth - 48;
  const body = rows
    .map((r, i) => {
      const y = top + i * (bar + gap);
      const w = Math.max((r.value / max) * plot, r.value ? 3 : 0);
      const x0 = labelWidth;
      const rr = Math.min(4, w / 2);
      const path = w ? `M${x0},${y} h${w - rr} a${rr},${rr} 0 0 1 ${rr},${rr} v${bar - 2 * rr} a${rr},${rr} 0 0 1 -${rr},${rr} h-${w - rr} z` : "";
      const t = esc(tip ? tip(r) : `${r.label}: ${format(r.value)}`);
      return `<g>
        <text x="${labelWidth - 10}" y="${y + bar / 2 + 4}" text-anchor="end">${esc(r.label)}</text>
        ${path ? `<path class="bar" d="${path}"/>` : ""}
        <text class="value-label" x="${x0 + w + 6}" y="${y + bar / 2 + 4}">${esc(format(r.value))}</text>
        <rect class="hit" x="0" y="${y - gap / 2}" width="${width}" height="${bar + gap}" data-tip="${t}"/>
      </g>`;
    })
    .join("");
  return `<svg class="chart" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img">${body}</svg>`;
}

function lineChart(points, { width = 560, height = 220, yMax = 1, format = (v) => pct(v), label } = {}) {
  if (!points.length) return `<p class="muted">No data yet.</p>`;
  const m = { l: 44, r: 56, t: 14, b: 40 };
  const pw = width - m.l - m.r, ph = height - m.t - m.b;
  const x = (i) => m.l + (points.length === 1 ? pw / 2 : (i / (points.length - 1)) * pw);
  const y = (v) => m.t + ph - (v / yMax) * ph;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax);
  const grid = ticks.map((t) => `<g class="grid"><line x1="${m.l}" x2="${m.l + pw}" y1="${y(t)}" y2="${y(t)}"/></g><text x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end">${esc(format(t))}</text>`).join("");
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.y)}`).join(" ");
  const area = `${d} L${x(points.length - 1)},${y(0)} L${x(0)},${y(0)} Z`;
  const step = Math.max(1, Math.ceil(points.length / 8));
  const xlabels = points.map((p, i) => (i % step === 0 || i === points.length - 1 ? `<text x="${x(i)}" y="${height - m.b + 18}" text-anchor="middle">${esc(p.x)}</text>` : "")).join("");
  const band = points.length > 1 ? pw / (points.length - 1) : pw;
  const dots = points.map((p, i) => `<circle class="dot" cx="${x(i)}" cy="${y(p.y)}" r="5"/>
      <rect class="hit" x="${x(i) - band / 2}" y="${m.t}" width="${band}" height="${ph}" data-tip="${esc(p.tip || `${p.x}: ${format(p.y)}`)}"/>`).join("");
  const last = points[points.length - 1];
  return `<svg class="chart" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(label || "")}">
    ${grid}<path class="area" d="${area}"/><path class="line" d="${d}"/>${dots}${xlabels}
    <text class="value-label" x="${x(points.length - 1) + 9}" y="${y(last.y) + 4}">${esc(format(last.y))}</text></svg>`;
}

// Charts are drawn at the container's real pixel width so text stays 12px at
// any card size; a viewBox that scales would shrink or balloon the labels.
const chartSpecs = new Map();
let chartSeq = 0;

function chartSlot(render, height) {
  const id = `chart-${++chartSeq}`;
  chartSpecs.set(id, render);
  return `<div class="chart-slot" data-chart="${id}" style="min-height:${height || 40}px"></div>`;
}

function mountCharts(root) {
  root.querySelectorAll(".chart-slot").forEach((slot) => {
    const render = chartSpecs.get(slot.dataset.chart);
    if (!render) return;
    slot.innerHTML = render(Math.max(280, slot.clientWidth));
    chartSpecs.delete(slot.dataset.chart);
  });
  bindTips(root);
}

function trendPoints(trend) {
  return trend.map((r, i) => ({
    x: `${r.pipeline_version} #${i + 1}`,
    y: r.resolved_rate,
    tip: `<b>${esc(r.pipeline_version)}</b> · ${fmtTime(r.started_at)}<br>${r.passed} of ${r.total} known issues resolved (${pct(r.resolved_rate)})${r.label ? `<br><span style="opacity:.75">${esc(r.label)}</span>` : ""}`,
  }));
}

// -- eval dataset view -----------------------------------------------------------------

async function renderEvals() {
  const view = $("#view-evals");
  view.innerHTML = `<p class="muted"><span class="spinner"></span> Loading…</p>`;
  const [data, analytics] = await Promise.all([api("/api/evals"), api("/api/analytics")]);
  const versions = state.meta.versions;
  const fixed = data.cases.filter((c) => c.latest?.passed).length;
  view.innerHTML = `
    <div class="detail-head">
      <div><h2>Eval dataset</h2><div class="meta"><span>${data.cases.length} cases</span><span class="mono">${esc(data.dataset_path)}</span><span>grown from confirmed flags</span></div></div>
      <div class="button-row">
        <select id="eval-version" aria-label="Pipeline version">${versions.map((v) => `<option value="${v.version}" ${v.version === state.meta.current_version ? "selected" : ""}>${v.version} - ${esc(v.notes.slice(0, 48))}</option>`).join("")}</select>
        <button class="primary" id="eval-run" ${data.cases.length ? "" : "disabled"}>Run regression</button>
      </div>
    </div>
    <div class="kpis">
      <div class="kpi"><div class="label">Known issues</div><div class="value">${data.cases.length}</div><div class="sub">one per confirmed flag</div></div>
      <div class="kpi"><div class="label">Resolved (latest run)</div><div class="value">${fixed}</div><div class="sub">${pct(data.cases.length ? fixed / data.cases.length : 0)} of known issues</div></div>
      <div class="kpi"><div class="label">Still failing</div><div class="value">${data.cases.length - fixed}</div><div class="sub">where to invest next</div></div>
      <div class="kpi"><div class="label">Regression runs</div><div class="value">${data.runs.length}</div><div class="sub">${data.runs.length ? "last " + fmtTime(data.runs[data.runs.length - 1].started_at) : "none yet"}</div></div>
    </div>
    <div class="card">
      <div class="card-title"><h3>Known issues resolved over time</h3><span class="muted small">share of the eval dataset passing, per regression run</span></div>
      ${chartSlot((w) => lineChart(trendPoints(analytics.eval_trend), { width: w, height: 260, label: "Known issues resolved" }), 260)}
    </div>
    <div class="card">
      <div class="card-title"><h3>Cases</h3><span class="muted small">click a row to open the trace it came from</span></div>
      <div class="table-wrap"><table>
        <thead><tr><th>case</th><th>document</th><th>failing step</th><th>category</th><th>failure codes</th><th>corrected output</th><th>latest result</th></tr></thead>
        <tbody>${data.cases.slice().reverse().map((c) => `<tr class="clickable ${c.case_id === state.newCaseId ? "new" : ""}" data-trace="${esc(c.source_trace_id)}">
            <td class="mono">${esc(c.case_id)}</td><td>${esc(c.doc_id)}</td><td>${esc(c.failing_step)}</td><td>${esc(categoryLabel(c.failure_category))}</td>
            <td>${c.failure_codes.map((x) => `<span class="code-tag">${esc(x)}</span>`).join(" ") || '<span class="muted">silent</span>'}</td>
            <td>${c.corrected_output ? `<span class="code-tag" data-tip="${esc(JSON.stringify(c.corrected_output))}">${Object.keys(c.corrected_output).length} field(s)</span>` : "–"}</td>
            <td>${c.latest ? (c.latest.passed ? `<span class="chip good">✓ Fixed</span>` : `<span class="chip crit">✕ Failing</span>`) + `<div class="small muted">${esc(c.latest.detail)}</div>` : '<span class="muted">not run yet</span>'}</td>
          </tr>`).join("") || `<tr><td colspan="7" class="muted">No cases yet. Flag a bad output in the Traces tab and confirm its root cause.</td></tr>`}</tbody>
      </table></div>
    </div>`;
  mountCharts(view);
  view.querySelectorAll("tr[data-trace]").forEach((tr) => tr.addEventListener("click", () => (location.hash = `#/traces/${tr.dataset.trace}`)));
  $("#eval-run").onclick = async () => {
    const button = $("#eval-run");
    button.disabled = true;
    button.innerHTML = `<span class="spinner"></span> Replaying…`;
    try {
      const res = await api("/api/evals/run", { method: "POST", body: { version: $("#eval-version").value } });
      toast(`Regression on <b>${esc(res.version)}</b>: ${res.passed} of ${res.total} known issues resolved.`);
    } catch (err) {
      toast(esc(err.message));
    }
    renderEvals();
  };
}

// -- analytics view ----------------------------------------------------------------------

async function renderAnalytics() {
  const view = $("#view-analytics");
  view.innerHTML = `<p class="muted"><span class="spinner"></span> Loading…</p>`;
  const a = await api("/api/analytics");
  const t = a.totals, tt = a.time_to_root_cause;
  const humanValue = tt.human_seconds_median == null ? "–" : tt.human_seconds_median < 120 ? `${tt.human_seconds_median.toFixed(0)} s` : `${(tt.human_seconds_median / 60).toFixed(1)} min`;
  const steps = state.meta.steps;
  const cats = [...new Set(Object.values(a.step_category_matrix).flatMap((m) => Object.keys(m)))];
  const maxCell = Math.max(1, ...steps.flatMap((s) => cats.map((c) => a.step_category_matrix[s]?.[c] || 0)));
  const heat = (n) => (n ? `background: color-mix(in srgb, var(--accent) ${Math.round(12 + (n / maxCell) * 60)}%, transparent)` : "");

  view.innerHTML = `
    <div class="detail-head"><div><h2>Failure analytics</h2><div class="meta"><span>production traces only - eval replays excluded</span></div></div></div>
    <div class="kpis">
      <div class="kpi"><div class="label">Traces analysed</div><div class="value">${t.traces}</div><div class="sub">${t.diagnosed} diagnosed</div></div>
      <div class="kpi"><div class="label">Failure rate</div><div class="value">${pct(t.failure_rate, 1)}</div><div class="sub">${t.failures} diagnosed failures</div></div>
      <div class="kpi"><div class="label">Time to root cause (automated)</div><div class="value">${tt.automated_ms_median == null ? "–" : tt.automated_ms_median.toFixed(1) + " ms"}</div><div class="sub">median · mean ${tt.automated_ms_mean ?? "–"} ms</div></div>
      <div class="kpi"><div class="label">Flag → confirmed root cause</div><div class="value">${humanValue}</div><div class="sub">${tt.human_samples ? `median over ${tt.human_samples} human flag(s)` : "no human flags yet"}</div></div>
      <div class="kpi"><div class="label">Diagnosis agreement</div><div class="value">${pct(tt.diagnosis_agreement)}</div><div class="sub">${tt.resolved_flags} resolved flags confirmed as diagnosed</div></div>
      <div class="kpi"><div class="label">Eval cases</div><div class="value">${t.eval_cases}</div><div class="sub">${t.flags} flags raised</div></div>
    </div>
    <div class="grid-2">
      <div class="card"><div class="card-title"><h3>Most common failure types</h3><span class="muted small">diagnosed failures</span></div>
        ${chartSlot((w) => hbarChart(a.by_category.map((r) => ({ label: r.label, value: r.count })), { width: w, labelWidth: 230, tip: (r) => `${r.label}: ${r.value} failure(s)` }))}</div>
      <div class="card"><div class="card-title"><h3>Which step fails most</h3><span class="muted small">root-cause step</span></div>
        ${chartSlot((w) => hbarChart(a.by_root_step.map((r) => ({ label: `Step ${STEP_NO[r.step]} · ${cap(r.step)}`, value: r.count })), { width: w, labelWidth: 170, tip: (r) => `${r.label}: root cause of ${r.value} failure(s)` }))}</div>
      <div class="card"><div class="card-title"><h3>Failure rate over time</h3><span class="muted small">per batch</span></div>
        ${chartSlot((w) => lineChart(a.failure_rate_over_time.map((b) => ({ x: b.batch, y: b.failure_rate, tip: `<b>${esc(b.batch)}</b>${b.pipeline_version ? " · " + esc(b.pipeline_version) : ""}<br>${b.failures} of ${b.traces} traces failed (${pct(b.failure_rate, 1)})` })), { width: w, yMax: Math.max(0.2, Math.ceil(Math.max(...a.failure_rate_over_time.map((b) => b.failure_rate), 0) * 5) / 5), label: "Failure rate per batch" }), 220)}</div>
      <div class="card"><div class="card-title"><h3>Known issues resolved</h3><span class="muted small">eval dataset, per regression run</span></div>
        ${chartSlot((w) => lineChart(trendPoints(a.eval_trend), { width: w, label: "Known issues resolved" }), 220)}</div>
    </div>
    <div class="card">
      <div class="card-title"><h3>Where to invest</h3><span class="muted small">root-cause step × failure category</span></div>
      <div class="table-wrap"><table class="heat"><thead><tr><th>step</th>${cats.map((c) => `<th class="num">${esc(categoryLabel(c))}</th>`).join("")}<th class="num">total</th></tr></thead>
      <tbody>${steps.map((s) => {
        const rowData = a.step_category_matrix[s] || {};
        const total = Object.values(rowData).reduce((x, y) => x + y, 0);
        return `<tr><td>Step ${STEP_NO[s]} · ${cap(s)}</td>${cats.map((c) => `<td class="cell" style="${heat(rowData[c] || 0)}">${rowData[c] || ""}</td>`).join("")}<td class="num"><b>${total}</b></td></tr>`;
      }).join("")}</tbody></table></div>
    </div>`;
  mountCharts(view);
}

// -- boot ------------------------------------------------------------------------------

async function boot() {
  state.meta = await api("/api/meta");
  let debounce;
  $("#filter-q").addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(loadTraces, 200);
  });
  ["#filter-batch", "#filter-verdict", "#filter-evals"].forEach((s) => $(s).addEventListener("change", loadTraces));
  $("#trace-list").addEventListener("click", (e) => {
    const li = e.target.closest("li[data-id]");
    if (li) location.hash = `#/traces/${li.dataset.id}`;
  });
  $("#modal").addEventListener("click", (e) => {
    if (e.target.id === "modal") closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#modal").hidden) closeModal();
  });
  window.addEventListener("hashchange", route);
  await loadTraces();
  if (!location.hash || location.hash === "#/traces") {
    const first = state.traces.find((t) => t.verdict === "failure") || state.traces[0];
    if (first) location.hash = `#/traces/${first.trace_id}`;
  }
  route();
}

boot().catch((err) => {
  document.body.insertAdjacentHTML("beforeend", `<div class="banner" style="margin:20px">Explorer failed to start: ${esc(err.message)}</div>`);
});
