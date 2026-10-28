// snow label: the page. Plain JS, no build step; served by src/snow/label/server.py.
"use strict";

const $ = (sel) => document.querySelector(sel);
const video = $("#video");
const overlay = $("#overlay");
const timeline = $("#timeline");

// Values allowed per trick field; must match snow/learn/labels.py TRICK_VALUES.
const FIELDS = [
  { key: "spin", label: "spin", guide: "counting-the-spin",
    values: ["0", "180", "360", "540", "720", "900", "1080", "1260", "1440", "1620", "1800", "unsure"] },
  { key: "direction", label: "direction", guide: "frontside-or-backside",
    values: ["frontside", "backside", "unsure"] },
  { key: "axis", label: "axis", guide: "flat-spin-or-cork",
    values: ["flat", "cork", "double_cork", "triple_cork", "unsure"] },
  { key: "grab", label: "grab", guide: "grabs",
    values: ["none", "indy", "mute", "melon", "stalefish", "nose", "tail", "method", "japan", "other", "unsure"] },
  { key: "stance", label: "stance", guide: "stance-regular-or-goofy",
    values: ["regular", "goofy", "unsure"] },
  { key: "switch", label: "took off switch", guide: "stance-regular-or-goofy",
    values: ["n", "y", "unsure"], names: { n: "no", y: "yes" } },
];
const pretty = (v) => String(v).replace("_", " ");

const S = {
  clips: [], clip: null, info: null, airs: [], noAir: false, sel: null, frame: 0,
  poses: null, prefill: null, prefillStatus: "", events: [], labeller: "",
  playing: false, saveTimer: null, readingTimers: {}, readings: {}, nextId: 1, pollTimer: null,
};

// ---- small helpers -----------------------------------------------------------

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json();
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const int = (v) => (v === "" || v === null || v === undefined || isNaN(+v) ? null : Math.round(+v));
const fps = () => (S.info ? S.info.fps : 30);
const lastFrame = () => Math.max(0, (S.info ? S.info.frames : 1) - 1);
const clamp = (f) => Math.max(0, Math.min(lastFrame(), f));
const pct = (p) => (p === null || p === undefined ? "-" : p.toFixed(2));
const selected = () => S.airs.find((a) => a._id === S.sel) || null;

function status(text, kind) {
  const el = $("#save-status");
  el.textContent = text;
  el.className = "save-status " + (kind || "");
}

// ---- clips -------------------------------------------------------------------

async function loadClips() {
  const got = await api("/api/clips");
  S.clips = got.clips;
  S.labeller = got.labeller;
  renderClipList();
  status(`labels save to ${got.csv}`);
  const wanted = decodeURIComponent(location.hash.slice(1));
  const first = S.clips.find((c) => c.name === wanted) || S.clips[0];
  if (first) openClip(first.name);
}

function renderClipList() {
  $("#clip-list").innerHTML = S.clips.map((c) => {
    const done = c.no_air ? "no air" : c.labelled_airs ? `${c.labelled_airs} air(s) labelled` : "not labelled";
    const slow = c.slowmo_factor ? `${c.slowmo_factor}x slow-mo` : "slow-mo unknown";
    return `<li data-name="${esc(c.name)}" class="${S.clip === c.name ? "active" : ""}">
      <div class="name">${esc(c.name)}</div>
      <div class="meta">${c.frames} frames, ${c.fps} fps, ${slow}</div>
      <div class="meta">${done}${c.has_poses ? "" : ", no poses"}</div></li>`;
  }).join("");
  document.querySelectorAll("#clip-list li").forEach((li) =>
    li.addEventListener("click", () => openClip(li.dataset.name)));
}

async function openClip(name) {
  if (S.saveTimer) await flushSave();
  const got = await api(`/api/clip/${encodeURIComponent(name)}`);
  S.clip = name;
  S.info = got.info;
  S.events = got.events;
  S.labeller = got.labeller;
  S.prefill = got.prefill;
  S.prefillStatus = got.prefill_status;
  S.readings = {};
  S.noAir = got.rows.some((r) => String(r.air).trim() === "0");
  S.airs = got.rows.filter((r) => String(r.air).trim() !== "0").map((r) => ({ ...r, _id: S.nextId++ }));
  S.sel = S.airs.length ? S.airs[0]._id : null;
  S.poses = null;
  location.hash = encodeURIComponent(name);
  renderClipList();
  renderFacts();
  sizeStage();
  video.src = `/video/${encodeURIComponent(name)}`;
  S.frame = 0;
  updateScrub();
  if (S.info.has_poses) {
    api(`/api/poses/${encodeURIComponent(name)}`).then((p) => { if (S.clip === name) { S.poses = p; drawOverlay(); } });
  }
  renderMachine();
  renderAirs();
  renderProblems(got.problems);
  S.airs.forEach(requestReading);
  pollPrefill();
}

function renderFacts() {
  const i = S.info;
  const slow = i.slowmo_factor
    ? `<b>${i.slowmo_factor}x</b> slow motion (from meta.json)`
    : `slow motion <b>unknown</b> (no meta.json${i.has_meta ? " factor" : ""}); label in frames anyway`;
  $("#clip-title").textContent = i.name;
  $("#clip-facts").innerHTML = `<b>${i.frames}</b> frames at <b>${i.fps}</b> fps, ${i.width}x${i.height}, ${slow}` +
    (i.has_poses ? "" : ", <b>no poses</b>: no skeleton and no machine guesses");
}

function sizeStage() {
  const w = S.info.width || 16, h = S.info.height || 9;
  const stage = $("#stage");
  stage.style.aspectRatio = `${w} / ${h}`;
  stage.style.width = `min(100%, calc(52vh * ${w / h}))`;
  overlay.width = w;
  overlay.height = h;
}

function pollPrefill() {
  clearTimeout(S.pollTimer);
  if (!S.info.has_poses || S.prefillStatus === "ready" || S.prefillStatus === "failed") return;
  const name = S.clip;
  S.pollTimer = setTimeout(async () => {
    const got = await api(`/api/clip/${encodeURIComponent(name)}`);
    if (S.clip !== name) return;
    S.prefill = got.prefill;
    S.prefillStatus = got.prefill_status;
    renderMachine();
    renderAirs();
    drawTimeline();
    pollPrefill();
  }, 1500);
}

// ---- the player ----------------------------------------------------------------

function seek(f) {
  S.frame = clamp(f);
  if (S.playing) video.pause();
  // The middle of the frame's display interval, so rounding never lands on a neighbour.
  if (video.readyState >= 1) video.currentTime = (S.frame + 0.5) / fps();
  renderFrame();
}

function renderFrame() {
  const t = S.frame / fps();
  $("#frame-badge").textContent = `frame ${S.frame}`;
  const real = S.info && S.info.slowmo_factor ? ` | ${(t / S.info.slowmo_factor).toFixed(2)} s real` : "";
  $("#frame-readout").textContent = `frame ${S.frame} / ${lastFrame()} | ${t.toFixed(3)} s clip${real}`;
  $("#scrub").value = S.frame;
  drawOverlay();
  drawTimeline();
}

function updateScrub() {
  $("#scrub").max = lastFrame();
  renderFrame();
}

function onVideoFrame(_now, meta) {
  if (S.playing) {
    S.frame = clamp(Math.round(meta.mediaTime * fps()));
    renderFrame();
  }
  video.requestVideoFrameCallback(onVideoFrame);
}

function togglePlay() {
  if (S.playing) {
    video.pause();
  } else {
    if (S.frame >= lastFrame()) seek(0);
    video.playbackRate = +$("#speed").value;
    video.play();
  }
}

video.addEventListener("loadedmetadata", () => {
  if (!S.info.width) { S.info.width = video.videoWidth; S.info.height = video.videoHeight; sizeStage(); }
  seek(S.frame);
});
video.addEventListener("play", () => { S.playing = true; $("#play").textContent = "pause"; });
video.addEventListener("pause", () => {
  S.playing = false;
  $("#play").textContent = "play";
  // Snap to the frame on screen so the next step starts from it exactly.
  video.currentTime = (S.frame + 0.5) / fps();
});
video.addEventListener("ended", () => { S.playing = false; $("#play").textContent = "play"; });
if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
  video.requestVideoFrameCallback(onVideoFrame);
} else {
  video.addEventListener("timeupdate", () => {
    if (S.playing) { S.frame = clamp(Math.floor(video.currentTime * fps())); renderFrame(); }
  });
}

function drawOverlay() {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!$("#skeleton").checked || !S.poses || !S.info) return;
  const pts = S.poses.frames[S.frame];
  if (!pts) return;
  const scale = overlay.width / 1280;
  ctx.lineWidth = Math.max(2, 3 * scale);
  ctx.strokeStyle = "rgba(80, 255, 160, 0.9)";
  ctx.beginPath();
  for (const [a, b] of S.poses.links) {
    const p = pts[a], q = pts[b];
    if (!p || !q || p[2] < 0.2 || q[2] < 0.2) continue;
    ctx.moveTo(p[0], p[1]);
    ctx.lineTo(q[0], q[1]);
  }
  ctx.stroke();
  // Head and feet marked, since where they point is what the guide asks about.
  // 71 is the left toe and 76 the right, as inferred in snow/synth/nova.py.
  for (const [j, colour] of [[6, "#ff5d8f"], [71, "#ffd84d"], [76, "#4dc3ff"]]) {
    const p = pts[j];
    if (!p) continue;
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(p[0], p[1], Math.max(4, 7 * scale), 0, Math.PI * 2);
    ctx.fill();
  }
}

// ---- the timeline ----------------------------------------------------------------

const ROWS = { you: [4, 22], rate: [26, 40], seg: [44, 58], prob: [60, 74] };

function drawTimeline() {
  if (!S.info) return;
  const dpr = window.devicePixelRatio || 1;
  const w = timeline.clientWidth;
  timeline.width = w * dpr;
  timeline.height = 76 * dpr;
  const ctx = timeline.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, 76);
  const n = Math.max(1, S.info.frames);
  const x = (f) => (f / n) * w;
  const band = (row, a, b, colour, alpha) => {
    ctx.globalAlpha = alpha;
    ctx.fillStyle = colour;
    ctx.fillRect(x(a), ROWS[row][0], Math.max(2, x(b) - x(a)), ROWS[row][1] - ROWS[row][0]);
    ctx.globalAlpha = 1;
  };
  const css = getComputedStyle(document.documentElement);
  for (const a of S.airs) {
    const t = int(a.takeoff_frame), l = int(a.landing_frame);
    if (t !== null && l !== null && l > t) band("you", t, l, css.getPropertyValue("--you"), a._id === S.sel ? 1 : 0.55);
    else if (t !== null) band("you", t, t + 1, css.getPropertyValue("--you"), 1);
  }
  const airs = (S.prefill && S.prefill.airs) || {};
  for (const a of airs.rate || []) band("rate", a.takeoff, a.landing, css.getPropertyValue("--rate"), 0.85);
  for (const a of airs.segmenter || []) band("seg", a.takeoff, a.landing, css.getPropertyValue("--seg"), 0.85);
  const prob = S.prefill && S.prefill.airborne_prob;
  if (prob) {
    ctx.strokeStyle = "#a78bfa";
    ctx.beginPath();
    let started = false;
    for (const [f, p] of prob) {
      if (p === null) { started = false; continue; }
      const y = ROWS.prob[1] - p * (ROWS.prob[1] - ROWS.prob[0]);
      if (!started) { ctx.moveTo(x(f), y); started = true; } else ctx.lineTo(x(f), y);
    }
    ctx.stroke();
  }
  if (S.noAir) {
    ctx.fillStyle = "#5d6875";
    ctx.font = "12px sans-serif";
    ctx.fillText("marked: no air in this clip", 6, 17);
  }
  ctx.strokeStyle = "#e11d48";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x(S.frame + 0.5), 0);
  ctx.lineTo(x(S.frame + 0.5), 76);
  ctx.stroke();
}

timeline.addEventListener("click", (e) => {
  const r = timeline.getBoundingClientRect();
  seek(Math.floor(((e.clientX - r.left) / r.width) * S.info.frames));
});

// ---- machine guesses ---------------------------------------------------------------

function geoText(g) {
  if (!g) return '<span class="empty">no reading</span>';
  if (g.error) return `<span class="empty">${esc(g.error)}</span>`;
  if (g.spin_deg === undefined) return '<span class="empty">too few posed frames</span>';
  return `<b>${g.spin_deg} ${esc(g.sense)}</b>, <b>${pretty(g.axis)}</b>
    <br><small>path ${g.spin_path_deg} deg, margin ${pct(g.spin_margin)} (not a probability),
    max tilt ${g.max_tilt_deg ?? "-"} deg, posed ${Math.round((g.posed_fraction || 0) * 100)}%</small>`;
}

function clfText(c) {
  if (!c) return '<span class="empty">no classifier</span>';
  if (c.error) return `<span class="empty">${esc(c.error)}</span>`;
  const head = (h) => `${pretty(c[h].label)} <small>p=${pct(c[h].p)}</small>`;
  return `<b>${esc(c.name)}</b><br><small>spin ${head("rotation")}, sense ${head("sense")},
    stance ${head("stance")}, axis ${head("axis")}, grab ${head("grab")}</small>`;
}

function renderMachine() {
  const el = $("#machine");
  if (!S.info.has_poses) {
    el.innerHTML = `<h3>Machine guesses</h3><p class="warn">This clip has no pose output, so there are
      no machine guesses and no skeleton. Label it by eye; the guide on the right is all you need.</p>`;
    return;
  }
  const pf = S.prefill;
  let body = "";
  if (!pf) {
    body = `<p class="empty">computing (${esc(S.prefillStatus)}); the first clip takes a few seconds while the models load</p>`;
  } else {
    const rows = [];
    for (const [key, tag] of [["rate", "rate"], ["segmenter", "seg"]]) {
      const list = (pf.airs || {})[key];
      if (!list) {
        rows.push(`<tr><td><span class="tag ${tag}">${key}</span></td><td colspan="4" class="empty">${esc(pf.sources[key] || "unavailable")}</td></tr>`);
        continue;
      }
      if (!list.length) rows.push(`<tr><td><span class="tag ${tag}">${key}</span></td><td colspan="4" class="empty">found no air</td></tr>`);
      list.forEach((a, k) => rows.push(`<tr>
        <td><span class="tag ${tag}">${key}</span> ${k + 1}</td>
        <td>${a.takeoff} to ${a.landing}<br><small>${a.landing - a.takeoff} frames</small></td>
        <td><span class="tag geo">geometry</span> ${geoText(a.geometry)}</td>
        <td><span class="tag clf">classifier</span> ${clfText(a.classifier)}</td>
        <td><button data-go="${a.takeoff}">go</button>
            <button data-draft="${a.takeoff},${a.landing}" class="primary">use as draft</button></td></tr>`));
    }
    body = `<table><tr><th>source</th><th>frames [takeoff, landing)</th><th>geometry</th><th>classifier</th><th></th></tr>${rows.join("")}</table>`;
    if (pf.sources && pf.sources.classifier && !pf.sources.classifier.startsWith("classifier ("))
      body += `<p class="empty">${esc(pf.sources.classifier)}</p>`;
  }
  el.innerHTML = `<h3>Machine guesses</h3>
    <p class="warn"><b>Unvalidated on real footage.</b> These are starting points, not answers, and are
    never saved unless you press "use as draft" and correct them. On synthetic tests the classifier
    beat the geometry only on turn sense, stance (so frontside/backside) and grab; the geometry was
    as good or better on spin count and axis (docs/classifier.md).</p>${body}`;
  el.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => seek(+b.dataset.go)));
  el.querySelectorAll("[data-draft]").forEach((b) => b.addEventListener("click", () => {
    const [t, l] = b.dataset.draft.split(",").map(Number);
    addAir(t, l);
    seek(t);
  }));
}

// ---- your labels -----------------------------------------------------------------

function defaultEvent() {
  return `${S.clip}/air${S.airs.length + 1}`;
}

function addAir(takeoff = null, landing = null) {
  if (S.noAir) { S.noAir = false; }
  const air = { _id: S.nextId++, air: String(S.airs.length + 1), takeoff_frame: takeoff ?? "",
    landing_frame: landing ?? "", event: defaultEvent(), sure: "y", labeller: S.labeller, notes: "" };
  for (const f of FIELDS) air[f.key] = "";
  S.airs.push(air);
  S.sel = air._id;
  changed(air);
  return air;
}

function trickName(a) {
  const q = (v) => (!v || v === "unsure" ? "?" : v);
  const parts = [];
  let dir = q(a.direction);
  if (a.switch === "y" && a.direction === "frontside") dir = "cab";
  else if (a.switch === "y") parts.push("switch");
  else if (a.switch === "unsure") parts.push("(switch?)");
  if (a.spin !== "0") parts.push(dir);
  if (a.axis && a.axis !== "flat") parts.push(q(a.axis).replace("_", " "));
  if (a.spin !== "0") parts.push(q(a.spin));
  if (a.grab && a.grab !== "none") parts.push(q(a.grab));
  if (a.spin === "0" && (!a.grab || a.grab === "none")) parts.push("straight air");
  return parts.join(" ");
}

function suggestions(a) {
  // What the machines read over YOUR window: shown as marks on the chips.
  const r = S.readings[a._id];
  const out = { g: {}, c: {} };
  if (!r || !r.data) return out;
  const g = r.data.geometry, c = r.data.classifier;
  if (g && g.spin_deg !== undefined) { out.g.spin = String(g.spin_deg); out.g.axis = g.axis; }
  if (c && !c.error) {
    out.c.spin = String(c.rotation.label);
    out.c.direction = c.direction;
    out.c.axis = c.axis.label;
    out.c.grab = c.grab.label;
    out.c.stance = c.stance.label;
  }
  return out;
}

function renderAirs() {
  const el = $("#airs");
  $("#no-air").textContent = S.noAir ? "clip marked: no air (undo)" : "no air in this clip";
  $("#no-air").className = S.noAir ? "primary" : "secondary";
  if (S.noAir) {
    el.innerHTML = '<p class="empty">You marked this clip as holding no air. That is saved as a label (air 0).</p>';
    drawTimeline();
    return;
  }
  if (!S.airs.length) {
    el.innerHTML = `<p class="empty">No airs labelled yet. Step to the first frame the board leaves the snow and
      press <b>t</b>, then the first frame it touches again and press <b>l</b>; or press "use as draft" on a
      machine guess above and correct it.</p>`;
    drawTimeline();
    return;
  }
  el.innerHTML = S.airs.map((a, k) => {
    const t = int(a.takeoff_frame), l = int(a.landing_frame);
    const len = t !== null && l !== null ? `${l - t} frames` : "";
    const sug = suggestions(a);
    const fieldRows = FIELDS.map((f) => {
      const chips = f.values.map((v) => {
        const on = a[f.key] === v ? " on" : "";
        const g = sug.g[f.key] === v ? '<span class="m g" title="geometry reads this">G</span>' : "";
        const c = sug.c[f.key] === v ? '<span class="m c" title="classifier reads this">C</span>' : "";
        const cls = `chip${on}${v === "unsure" ? " unsure" : ""}${g || c ? " suggested" : ""}`;
        return `<button class="${cls}" data-air="${a._id}" data-field="${f.key}" data-value="${v}">${esc((f.names || {})[v] || pretty(v))}${g}${c}</button>`;
      }).join("");
      return `<div class="row"><span class="lbl">${f.label}<a href="#" data-guide="${f.guide}" title="how to tell">?</a></span><div class="chips">${chips}</div></div>`;
    }).join("");
    const r = S.readings[a._id];
    const reading = !S.info.has_poses ? "" : !r ? "" : r.pending ? '<div class="reading">reading your window</div>' :
      `<div class="reading">Machines on <b>your</b> frames: <span class="tag geo">geometry</span> ${geoText(r.data.geometry)}
       <br><span class="tag clf">classifier</span> ${clfText(r.data.classifier)}</div>`;
    return `<div class="air${a._id === S.sel ? " selected" : ""}" data-air="${a._id}">
      <div class="air-head"><span class="title">Air ${k + 1}</span><span class="name">${esc(trickName(a))}</span>
        <small>${len}</small>
        <button class="danger" data-delete="${a._id}" style="margin-left:auto">delete</button></div>
      <div class="row"><span class="lbl">takeoff<a href="#" data-guide="takeoff-and-landing">?</a></span>
        <input type="number" min="0" data-air="${a._id}" data-input="takeoff_frame" value="${esc(a.takeoff_frame)}">
        <button data-set="takeoff_frame" data-air="${a._id}">set here (t)</button>
        <button data-jump="takeoff_frame" data-air="${a._id}">go ([)</button>
        <span class="lbl" style="width:auto;margin-left:14px">landing</span>
        <input type="number" min="0" data-air="${a._id}" data-input="landing_frame" value="${esc(a.landing_frame)}">
        <button data-set="landing_frame" data-air="${a._id}">set here (l)</button>
        <button data-jump="landing_frame" data-air="${a._id}">go (])</button></div>
      <div class="row"><span class="lbl">event<a href="#" data-guide="replays-the-same-jump-twice">?</a></span>
        <input type="text" list="event-list" data-air="${a._id}" data-input="event" value="${esc(a.event)}"
          placeholder="contest/round/rider/run/hit; same id for a replay">
        <label class="toggle"><input type="checkbox" data-air="${a._id}" data-sure ${a.sure === "n" ? "checked" : ""}> not sure (cut, hidden, blurred)</label></div>
      ${fieldRows}
      <div class="row"><span class="lbl">notes</span>
        <input type="text" data-air="${a._id}" data-input="notes" value="${esc(a.notes)}" placeholder="anything: commentator's name for it, other grab, crash"></div>
      ${reading}</div>`;
  }).join("");
  $("#event-list").innerHTML = [...new Set([...S.events, ...S.airs.map((a) => a.event)])]
    .filter(Boolean).map((e) => `<option value="${esc(e)}">`).join("");
  bindAirs();
  drawTimeline();
}

function bindAirs() {
  const el = $("#airs");
  const find = (node) => S.airs.find((a) => a._id === +node.dataset.air);
  el.querySelectorAll(".air").forEach((card) => card.addEventListener("click", () => {
    if (S.sel !== +card.dataset.air) {
      S.sel = +card.dataset.air;
      el.querySelectorAll(".air").forEach((c) => c.classList.toggle("selected", +c.dataset.air === S.sel));
      drawTimeline();
    }
  }));
  el.querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => {
    const a = find(b);
    a[b.dataset.field] = a[b.dataset.field] === b.dataset.value ? "" : b.dataset.value;
    changed(a, true);
  }));
  el.querySelectorAll("[data-input]").forEach((inp) => inp.addEventListener("input", () => {
    const a = find(inp);
    a[inp.dataset.input] = inp.value;
    changed(a, false);
  }));
  el.querySelectorAll("[data-input]").forEach((inp) => inp.addEventListener("change", () => renderAirs()));
  el.querySelectorAll("[data-sure]").forEach((inp) => inp.addEventListener("change", () => {
    const a = find(inp);
    a.sure = inp.checked ? "n" : "y";
    changed(a, false);
  }));
  el.querySelectorAll("[data-set]").forEach((b) => b.addEventListener("click", () => setFrame(find(b), b.dataset.set)));
  el.querySelectorAll("[data-jump]").forEach((b) => b.addEventListener("click", () => {
    const v = int(find(b)[b.dataset.jump]);
    if (v !== null) seek(v);
  }));
  el.querySelectorAll("[data-delete]").forEach((b) => b.addEventListener("click", (e) => {
    e.stopPropagation();
    const a = find(b);
    if (!confirm(`Delete this air (frames ${a.takeoff_frame || "?"} to ${a.landing_frame || "?"})?`)) return;
    S.airs = S.airs.filter((x) => x !== a);
    S.sel = S.airs.length ? S.airs[S.airs.length - 1]._id : null;
    changed(null, true);
  }));
  el.querySelectorAll("[data-guide]").forEach((l) => l.addEventListener("click", (e) => {
    e.preventDefault();
    showGuide(l.dataset.guide);
  }));
}

function setFrame(air, field) {
  if (!air) air = addAir();
  air[field] = String(S.frame);
  S.sel = air._id;
  changed(air, true);
}

function changed(air, rerender = true) {
  if (air) requestReading(air);
  if (rerender) renderAirs(); else drawTimeline();
  scheduleSave();
}

function requestReading(air) {
  if (!S.info.has_poses) return;
  const t = int(air.takeoff_frame), l = int(air.landing_frame);
  if (t === null || l === null || l <= t) { delete S.readings[air._id]; return; }
  const key = `${t}-${l}`;
  const have = S.readings[air._id];
  if (have && have.key === key) return;
  S.readings[air._id] = { key, pending: true };
  clearTimeout(S.readingTimers[air._id]);
  S.readingTimers[air._id] = setTimeout(async () => {
    try {
      const data = await api(`/api/reading/${encodeURIComponent(S.clip)}`, { takeoff: t, landing: l });
      if (S.readings[air._id] && S.readings[air._id].key === key) S.readings[air._id] = { key, data };
    } catch (err) {
      S.readings[air._id] = { key, data: { geometry: { error: String(err) }, classifier: null } };
    }
    const focused = document.activeElement && document.activeElement.dataset && document.activeElement.dataset.input;
    if (!focused) renderAirs();
  }, 400);
}

// ---- saving ----------------------------------------------------------------------

function rowsForSave() {
  if (S.noAir) return [{ air: "0", sure: "y", labeller: S.labeller, notes: "" }];
  const order = [...S.airs].sort((a, b) => (int(a.takeoff_frame) ?? 1e9) - (int(b.takeoff_frame) ?? 1e9));
  return order.map((a, k) => {
    const row = { ...a, air: String(k + 1) };
    delete row._id;
    return row;
  });
}

function scheduleSave() {
  status("unsaved changes", "");
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(flushSave, 500);
}

async function flushSave() {
  clearTimeout(S.saveTimer);
  S.saveTimer = null;
  const name = S.clip;
  try {
    const got = await api(`/api/save/${encodeURIComponent(name)}`, { rows: rowsForSave() });
    const time = new Date().toLocaleTimeString();
    const mine = got.problems.filter((p) => p.includes(` ${name} `) || p.startsWith(`${name}:`));
    status(mine.length ? `saved ${time}, ${mine.length} problem(s) below` : `saved ${time} to ${got.csv}`,
      mine.length ? "bad" : "ok");
    renderProblems(got.problems);
    const c = S.clips.find((x) => x.name === name);
    if (c) {
      c.no_air = S.noAir;
      c.labelled_airs = S.noAir ? 0 : S.airs.length;
      renderClipList();
    }
  } catch (err) {
    status(`NOT saved: ${err.message}`, "bad");
  }
}

function renderProblems(problems) {
  const name = S.clip;
  const mine = (problems || []).filter((p) => p.includes(` ${name} `) || p.startsWith(`${name}:`));
  const others = (problems || []).length - mine.length;
  $("#problems").innerHTML = mine.map((p) => `<div>${esc(p)}</div>`).join("") +
    (others ? `<div>${others} problem(s) in other clips' rows</div>` : "");
}

// ---- guide -----------------------------------------------------------------------

async function loadGuide() {
  const res = await fetch("/api/guide");
  $("#guide").innerHTML = await res.text();
  // In-guide links scroll the panel; the page's own hash names the clip.
  $("#guide").querySelectorAll("a[data-anchor]").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault();
    showGuide(a.dataset.anchor);
  }));
}

function showGuide(id) {
  const target = document.getElementById(id);
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "start" });
  target.classList.add("flash");
  setTimeout(() => target.classList.remove("flash"), 1500);
}

// ---- wiring ----------------------------------------------------------------------

document.querySelectorAll("[data-step]").forEach((b) =>
  b.addEventListener("click", () => seek(S.frame + +b.dataset.step)));
$("#play").addEventListener("click", togglePlay);
$("#speed").addEventListener("change", () => { video.playbackRate = +$("#speed").value; });
$("#skeleton").addEventListener("change", drawOverlay);
$("#scrub").addEventListener("input", () => seek(+$("#scrub").value));
$("#new-air").addEventListener("click", () => addAir());
$("#no-air").addEventListener("click", () => {
  if (!S.noAir && S.airs.some((a) => a.takeoff_frame || a.landing_frame) &&
      !confirm("Mark this clip as holding no air? Its labelled airs will be removed.")) return;
  S.noAir = !S.noAir;
  if (S.noAir) S.airs = [];
  changed(null, true);
});
// Buttons never keep focus, so Space always means play and never re-clicks a button.
document.addEventListener("mousedown", (e) => { if (e.target.closest("button")) e.preventDefault(); });
window.addEventListener("resize", () => { drawTimeline(); });

document.addEventListener("keydown", (e) => {
  const tag = (e.target.tagName || "").toLowerCase();
  const typing = (tag === "input" && !["checkbox", "range"].includes(e.target.type)) || tag === "textarea" || tag === "select";
  if (typing || e.metaKey || e.ctrlKey || e.altKey || !S.info) return;
  const step = e.shiftKey ? 10 : 1;
  const handled = {
    ArrowLeft: () => seek(S.frame - step),
    ArrowRight: () => seek(S.frame + step),
    ArrowDown: () => seek(S.frame - 10),
    ArrowUp: () => seek(S.frame + 10),
    " ": togglePlay,
    t: () => setFrame(selected(), "takeoff_frame"),
    l: () => setFrame(selected(), "landing_frame"),
    n: () => addAir(),
    s: () => { $("#skeleton").checked = !$("#skeleton").checked; drawOverlay(); },
    "[": () => { const a = selected(); if (a && int(a.takeoff_frame) !== null) seek(int(a.takeoff_frame)); },
    "]": () => { const a = selected(); if (a && int(a.landing_frame) !== null) seek(int(a.landing_frame)); },
    1: () => setSpeed("0.125"), 2: () => setSpeed("0.25"), 3: () => setSpeed("0.5"), 4: () => setSpeed("1"),
  }[e.key.length === 1 ? e.key.toLowerCase() : e.key];
  if (handled) {
    e.preventDefault();
    handled();
  }
});

function setSpeed(v) {
  $("#speed").value = v;
  video.playbackRate = +v;
}

window.addEventListener("beforeunload", (e) => {
  if (S.saveTimer) { flushSave(); e.preventDefault(); }
});

loadGuide();
loadClips().catch((err) => status(`could not load clips: ${err.message}`, "bad"));
