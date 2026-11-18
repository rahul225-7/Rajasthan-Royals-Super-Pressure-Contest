/* Under Pressure - dashboard logic. No framework, no chart library: every chart is plain SVG. */
(() => {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const WEIGHT_COLORS = ["#2c4baa", "#c41560", "#1f7a55", "#6b4fa8", "#b07a1a", "#5b6380"];

  const state = { data: null, role: "bat", selected: null, xrole: "bat", sort: { key: "model_rank", dir: 1 }, query: "" };

  // ---------- helpers ----------
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const fmtInt = (n) => n.toLocaleString("en-GB");
  const fmt = (n, d = 1) => (n === null || n === undefined ? "–" : Number(n).toFixed(d));
  const signed = (n, d = 2) => (n === null || n === undefined ? "–" : (n > 0 ? "+" : n < 0 ? "−" : "") + Math.abs(n).toFixed(d));
  const pct = (n) => (n === null || n === undefined ? "–" : Math.round(n * 100) + "%");
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function svg(tag, attrs = {}, children = []) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, v);
    for (const c of [].concat(children)) if (c) el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    return el;
  }
  const text = (x, y, s, cls = "ax", extra = {}) => svg("text", { x, y, class: cls, ...extra }, [s]);

  const tip = $("#tooltip");
  function showTip(html, evt) {
    tip.innerHTML = html;
    tip.hidden = false;
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }
  const hideTip = () => { tip.hidden = true; };

  const shortName = (n) => { const w = n.split(" "); return w.length > 1 ? `${w[0][0]}. ${w[w.length - 1]}` : n; };
  const players = (role) => state.data[role].players;
  const byId = (role, id) => players(role).find((p) => p.id === id);
  const submittedIds = (role) => new Set(state.data[role].submitted.map((s) => s.id));

  // ---------- hero ----------
  function renderFacts() {
    const m = state.data.meta;
    const items = [
      ["Matches with ball-by-ball data", fmtInt(m.matches)],
      ["Balls scored", fmtInt(m.balls)],
      ["Pressure balls", fmtInt(m.pressure_balls)],
      ["Batters and bowlers ranked", `${m.eligible_bat} + ${m.eligible_bowl}`],
    ];
    $("#facts").innerHTML = items.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  }

  function renderWorm() {
    const h = state.data.hero;
    const pts = h.points;
    const winner = h.chase_won ? h.chase : h.first;
    const tied = h.first_score.split("/")[0] === h.chase_score.split("/")[0];
    const outcome = tied ? `Scores finished level; ${esc(winner)} won the super over.` : `${esc(winner)} won.`;
    $("#worm-caption").innerHTML =
      `<b>The tightest knockout chase in the data.</b> ${esc(h.first)} ${esc(h.first_score)} against ${esc(h.chase)} ${esc(h.chase_score)}, ` +
      `${esc(h.competition)} ${esc(h.stage.toLowerCase())}. ${outcome} The line is ${esc(h.chase)}’s chance of winning before each ball.`;

    const W = Math.max($("#worm-chart").clientWidth, 340), H = W < 700 ? 250 : 300, L = 46, R = 10, T = 30, B = 34;
    const n = pts.length;
    const x = (i) => L + (i / (n - 1)) * (W - L - R);
    const y = (p) => T + (1 - p) * (H - T - B);
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" });

    [0, 0.25, 0.5, 0.75, 1].forEach((g) => {
      root.appendChild(svg("line", { x1: L, x2: W - R, y1: y(g), y2: y(g), class: g === 0.5 ? "even-line" : "grid-line" }));
      root.appendChild(text(L - 8, y(g) + 4, g === 0.5 ? "Even" : Math.round(g * 100) + "%", "ax", { "text-anchor": "end" }));
    });

    const split = pts.findIndex((p) => p.inn === 2);
    root.appendChild(svg("line", { x1: x(split) - 0.5, x2: x(split) - 0.5, y1: T - 8, y2: H - B, class: "grid-line" }));
    const short = W < 560;
    root.appendChild(text((L + x(split)) / 2, T - 12, short ? h.first : `${h.first} batting`, "ax-strong", { "text-anchor": "middle" }));
    root.appendChild(text((x(split) + W - R) / 2, T - 12, short ? `${h.chase} chase` : `${h.chase} chasing`, "ax-strong", { "text-anchor": "middle" }));

    let lastOver = null;
    pts.forEach((p, i) => {
      if (p.over !== lastOver && p.over % 5 === 0) {
        if (!short || p.over % 10 === 0) root.appendChild(text(x(i), H - B + 18, short ? String(p.over) : `Over ${p.over}`, "ax", { "text-anchor": "middle" }));
      }
      lastOver = p.over;
    });

    const g = svg("g", { "clip-path": "url(#worm-clip)" });
    const animate = !REDUCED && !state.wormDrawn;
    const clipRect = svg("rect", { x: 0, y: 0, width: animate ? 0 : W, height: H });
    root.appendChild(svg("defs", {}, [svg("clipPath", { id: "worm-clip" }, [clipRect])]));

    // pressure ticks along the baseline
    pts.forEach((p, i) => {
      if (p.p) g.appendChild(svg("rect", { x: x(i) - 1.2, y: H - B + 1, width: 2.4, height: 6, fill: "var(--pressure)" }));
    });
    const d = pts.map((p, i) => `${i && i !== split ? "L" : "M"}${x(i).toFixed(1)},${y(p.wp).toFixed(1)}`).join("");
    g.appendChild(svg("path", { d, class: "worm-line" }));
    let hot = "";
    pts.forEach((p, i) => {
      if (!p.p || i === 0 || i === split) return;
      hot += `M${x(i - 1).toFixed(1)},${y(pts[i - 1].wp).toFixed(1)}L${x(i).toFixed(1)},${y(p.wp).toFixed(1)}`;
    });
    g.appendChild(svg("path", { d: hot, class: "worm-hot" }));
    pts.forEach((p, i) => {
      if (p.w) g.appendChild(svg("circle", { cx: x(i), cy: y(p.wp), r: 4.5, fill: "var(--surface)", stroke: "var(--ink)", "stroke-width": 2 }));
    });
    root.appendChild(g);

    if (animate) {
      state.wormDrawn = true;
      clipRect.appendChild(svg("animate", { attributeName: "width", from: 0, to: W, dur: "2.4s", begin: "0.25s", fill: "freeze",
        calcMode: "spline", keySplines: "0.4 0 0.2 1", keyTimes: "0;1" }));
    }

    const guide = svg("line", { y1: T, y2: H - B, stroke: "var(--ink-3)", "stroke-width": 1, visibility: "hidden" });
    root.appendChild(guide);
    const hit = svg("rect", { x: L, y: 0, width: W - L - R, height: H, fill: "transparent" });
    root.appendChild(hit);
    hit.addEventListener("mousemove", (evt) => {
      const box = root.getBoundingClientRect();
      const vx = ((evt.clientX - box.left) / box.width) * W;
      const i = clamp(Math.round(((vx - L) / (W - L - R)) * (n - 1)), 0, n - 1);
      const p = pts[i];
      guide.setAttribute("x1", x(i)); guide.setAttribute("x2", x(i)); guide.setAttribute("visibility", "visible");
      const who = h.chase;
      showTip(`<b>${p.inn === 1 ? "First innings" : "Chase"}, over ${p.over}</b><br>${esc(who)} ${Math.round(p.wp * 100)}% to win` +
        `${p.p ? "<br>Pressure ball" : ""}${p.w ? "<br>Wicket" : ""}`, evt);
    });
    hit.addEventListener("mouseleave", () => { guide.setAttribute("visibility", "hidden"); hideTip(); });

    $("#worm-chart").replaceChildren(root);
  }

  // ---------- rankings ----------
  const STAT_DEF = {
    bat: [
      { k: "sr", label: "Strike rate", max: 280, d: 1 },
      { k: "avg", label: "Average", max: 70, d: 1 },
      { k: "dot_pct", label: "Dot balls %", max: 55, d: 1, low: true },
      { k: "bdry_pct", label: "Boundary balls %", max: 50, d: 1 },
    ],
    bowl: [
      { k: "econ", label: "Economy", max: 12, d: 2, low: true },
      { k: "bpw", label: "Balls per wicket", max: 40, d: 1, low: true },
      { k: "dot_pct", label: "Dot balls %", max: 55, d: 1 },
      { k: "extras_pct", label: "Wides and no-balls %", max: 8, d: 1, low: true },
    ],
  };

  function renderRankNote() {
    const m = state.data.meta;
    const [b, mt] = m.eligibility[state.role];
    const n = state.role === "bat" ? m.eligible_bat : m.eligible_bowl;
    const who = state.role === "bat" ? "batters" : "bowlers";
    $("#rank-note").textContent =
      `Chosen from ${n} eligible ${who}: at least ${b} pressure balls across ${mt} matches. ` +
      `The figure on the right of each row is how often the player stays in the top 10 when the matches are resampled 150 times.`;
  }

  function renderRanklist() {
    const role = state.role;
    const list = state.data[role].submitted;
    const maxR = Math.max(...list.map((s) => byId(role, s.id).rating));
    $("#ranklist").innerHTML = list.map((s) => {
      const p = byId(role, s.id);
      const tag = s.note ? `<span class="tag">Selector’s call</span>` : "";
      return `<li><button class="rankrow" data-id="${p.id}" aria-current="${p.id === state.selected}">
        <span class="rankrow__n">${s.rank}</span>
        <span><span class="rankrow__name">${esc(p.name)}${tag}</span>
          <span class="rankrow__bar" aria-hidden="true"><i style="width:${(p.rating / maxR) * 100}%"></i></span></span>
        <span class="rankrow__chance"><b>${pct(p.p_top10)}</b>top 10</span>
      </button></li>`;
    }).join("");
    $$(".rankrow").forEach((b) => b.addEventListener("click", () => select(Number(b.dataset.id))));
  }

  function select(id) {
    state.selected = id;
    $$(".rankrow").forEach((b) => b.setAttribute("aria-current", String(Number(b.dataset.id) === id)));
    renderDetail();
  }

  function renderDetail() {
    const role = state.role;
    const p = byId(role, state.selected);
    const s = state.data[role].submitted.find((x) => x.id === p.id);
    const n = players(role).length;
    const comps = state.data[role].components;
    const weights = state.data.weights[role];

    const pairs = STAT_DEF[role].map((def) => {
      const pv = p.stats.pressure ? p.stats.pressure[def.k] : null;
      const ov = p.stats.ordinary ? p.stats.ordinary[def.k] : null;
      const w = (v) => (v === null ? 0 : clamp(v / def.max, 0, 1) * 100);
      return `<div class="pair">
        <div class="pair__label"><span>${def.label}</span>${def.low ? "<span>lower is better</span>" : ""}</div>
        <div class="pair__row pair__row--p"><div class="pair__track"><i style="width:${w(pv)}%"></i></div><span>${fmt(pv, def.d)}</span></div>
        <div class="pair__row pair__row--o"><div class="pair__track"><i style="width:${w(ov)}%"></i></div><span>${fmt(ov, def.d)}</span></div>
      </div>`;
    }).join("");

    const ZMIN = -3, ZMAX = 5, zero = ((0 - ZMIN) / (ZMAX - ZMIN)) * 100;
    const compRows = comps.map((c) => {
      const v = p.z[c.key];
      const pos = ((clamp(v, ZMIN, ZMAX) - ZMIN) / (ZMAX - ZMIN)) * 100;
      const left = Math.min(pos, zero), width = Math.abs(pos - zero);
      return `<div class="comp"><span class="comp__label">${c.label} <small>(${Math.round(weights[c.key] * 100)}%)</small></span>
        <span class="comp__track" style="--zero:${zero}%"><i class="${v < 0 ? "neg" : ""}" style="left:${left}%;width:${width}%"></i></span>
        <span class="comp__v">${signed(v, 1)}</span></div>`;
    }).join("");

    const note = s && s.note ? `<p class="callout">${esc(s.note)}</p>` : "";
    const ps = p.stats.pressure || {}, os = p.stats.ordinary || {};
    const balls = role === "bat" ? `${fmtInt(ps.balls || 0)} pressure balls faced, ${fmtInt(os.balls || 0)} ordinary` :
      `${fmtInt(ps.balls || 0)} pressure balls bowled, ${fmtInt(os.balls || 0)} ordinary`;

    $("#detail").innerHTML = `
      <div class="detail__top">
        <div>
          <h3>${esc(p.name)}</h3>
          <p class="detail__sub">Model rank ${p.model_rank} of ${n}. ${p.p_matches} matches with pressure balls, at ${p.n_venues} venues.</p>
        </div>
        <div class="detail__rating"><b>${fmt(p.rating, 2)}</b><span>rating, where 0 is an average eligible player</span></div>
      </div>
      ${note}
      <h4>Under pressure and on ordinary balls <small>${balls}</small></h4>
      <p class="worm__key" style="margin-top:-.4rem;margin-bottom:.9rem"><span class="key key--pressure"></span>Pressure <span class="key" style="background:var(--ordinary);border-radius:2px;margin-left:.6rem"></span>Ordinary</p>
      <div class="pairs">${pairs}</div>
      <h4>What drives the rating <small>z-score among eligible players, weight in brackets</small></h4>
      <div class="comps">${compRows}</div>
      <h4>How settled the rank is <small>middle 90% of ranks across 150 resampled datasets</small></h4>
      <div class="range" id="range"></div>`;
    renderRange(p);
  }

  function renderRange(p) {
    const W = Math.max($("#range").clientWidth, 260), H = 70, L = 8, R = 8, Y = 31;
    const maxRank = Math.max(40, Math.ceil(p.rank_hi / 10) * 10);
    const x = (r) => L + ((r - 1) / (maxRank - 1)) * (W - L - R);
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": `Rank between ${Math.round(p.rank_lo)} and ${Math.round(p.rank_hi)} in 90% of resamples` });
    root.appendChild(text(x(1), 11, "Top 10", "ax"));
    root.appendChild(svg("rect", { x: x(1) - 4, y: Y - 12, width: x(10.5) - x(1) + 4, height: 24, fill: "var(--pressure-soft)", rx: 3 }));
    root.appendChild(svg("line", { x1: x(1), x2: x(maxRank), y1: Y, y2: Y, class: "grid-line" }));
    root.appendChild(svg("line", { x1: x(p.rank_lo), x2: x(p.rank_hi), y1: Y, y2: Y, stroke: "var(--ink)", "stroke-width": 4, "stroke-linecap": "round" }));
    root.appendChild(svg("circle", { cx: x(p.model_rank), cy: Y, r: 6.5, fill: "var(--pressure)", stroke: "var(--surface)", "stroke-width": 2 }));
    for (let r = 10; r <= maxRank; r += 10) root.appendChild(text(x(r), 64, String(r), "ax", { "text-anchor": "middle" }));
    root.appendChild(text(x(1), 64, "1", "ax", { "text-anchor": "middle" }));
    root.appendChild(text(x(maxRank), 11, `Model rank ${p.model_rank}, range ${Math.round(p.rank_lo)} to ${Math.round(p.rank_hi)}`, "ax", { "text-anchor": "end" }));
    $("#range").replaceChildren(root);
  }

  function setRole(role) {
    state.role = role;
    $$("[data-role]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.role === role)));
    state.selected = state.data[role].submitted[0].id;
    renderRankNote();
    renderRanklist();
    renderDetail();
  }

  // ---------- pressure ----------
  function renderHeatmap() {
    const g = state.data.grid;
    const W = Math.max($("#heatmap").clientWidth, 300), H = 190, L = W < 420 ? 96 : 112, T = 26, cw = (W - L) / g.phases.length, ch = 62;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Share of balls flagged as pressure by innings and phase" });
    g.phases.forEach((ph, j) => root.appendChild(text(L + j * cw + cw / 2, 16, (W < 420 ? ph.replace("Overs ", "").replace("Powerplay", "PP") : ph).replace(/ \(.*\)/, ""), "ax", { "text-anchor": "middle" })));
    g.innings.forEach((row, i) => {
      const yy = T + i * (ch + 8);
      root.appendChild(text(L - 12, yy + ch / 2 - 2, i === 0 ? "First innings" : "Chase", "ax-strong", { "text-anchor": "end" }));
      root.appendChild(text(L - 12, yy + ch / 2 + 14, i === 0 ? "setting a total" : "second innings", "ax", { "text-anchor": "end" }));
      row.share.forEach((v, j) => {
        const op = clamp(v / 40, 0.12, 1);
        root.appendChild(svg("rect", { x: L + j * cw + 3, y: yy, width: cw - 6, height: ch, rx: 4, fill: "var(--pressure)", "fill-opacity": op }));
        root.appendChild(text(L + j * cw + cw / 2, yy + ch / 2 + 6, `${Math.round(v)}%`, "lab",
          { "text-anchor": "middle", style: op > 0.6 ? "fill:#fff" : "" }));
      });
    });
    $("#heatmap").replaceChildren(root);
  }

  function renderDeciles() {
    const d = state.data.deciles;
    const W = Math.max($("#deciles").clientWidth, 300), H = 200, L = 40, R = 6, T = 14, B = 34;
    const max = Math.max(...d.map((x) => x.swing)) * 1.12;
    const bw = (W - L - R) / d.length;
    const y = (v) => T + (1 - v / max) * (H - T - B);
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Average swing rises with leverage decile" });
    [0, 0.02, 0.04, 0.06].forEach((g) => {
      if (g > max) return;
      root.appendChild(svg("line", { x1: L, x2: W - R, y1: y(g), y2: y(g), class: "grid-line" }));
      root.appendChild(text(L - 6, y(g) + 4, `${Math.round(g * 100)}%`, "ax", { "text-anchor": "end" }));
    });
    d.forEach((v, i) => {
      root.appendChild(svg("rect", { x: L + i * bw + 5, y: y(v.swing), width: bw - 10, height: y(0) - y(v.swing), rx: 2,
        fill: "var(--ordinary)" }));
      root.appendChild(text(L + i * bw + bw / 2, H - B + 16, String(v.decile), "ax", { "text-anchor": "middle" }));
    });
    root.appendChild(text(L, H - 2, "Lowest leverage", "ax"));
    root.appendChild(text(W - R, H - 2, "Highest leverage", "ax", { "text-anchor": "end" }));
    $("#deciles").replaceChildren(root);
  }

  function renderCompare() {
    const c = state.data.pressure_vs_ordinary;
    const rows = [
      ["Runs per over", "run_rate", 2], ["Wickets per 100 balls", "wkts_per_100", 2],
      ["Dot balls %", "dot_pct", 1], ["Average leverage", "mean_li", 2],
    ];
    $("#compare").innerHTML = rows.map(([label, k, dp]) => `<div><h4>${label}</h4>
      <p class="is-p"><span>Pressure</span><b>${fmt(c.pressure[k], dp)}</b></p>
      <p class="is-o"><span>Ordinary</span><b>${fmt(c.ordinary[k], dp)}</b></p></div>`).join("");
  }

  // ---------- explorer ----------
  function renderScatter() {
    const role = state.xrole;
    const ps = players(role).filter((p) => p.usual !== null && p.pressure_raw !== null);
    const top = submittedIds(role);
    const q = state.query.trim().toLowerCase();
    const W = Math.max($("#scatter").clientWidth, 320), H = Math.round(clamp(W * 0.46, 320, 460)), L = 52, R = 16, T = 18, B = 46;
    const xs = ps.map((p) => p.usual), ys = ps.map((p) => p.pressure_raw);
    const lo = Math.min(...xs, ...ys), hi = Math.max(...xs, ...ys);
    const pad = (hi - lo) * 0.06;
    const xmin = Math.min(...xs) - pad, xmax = Math.max(...xs) + pad;
    const ymin = Math.min(...ys) - pad, ymax = Math.max(...ys) + pad;
    const x = (v) => L + ((v - xmin) / (xmax - xmin)) * (W - L - R);
    const y = (v) => T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B);
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Usual level against pressure level for every eligible player" });

    const ticks = (a, b) => { const step = (b - a) > 0.8 ? 0.2 : 0.1; const out = []; for (let v = Math.ceil(a / step) * step; v <= b; v += step) out.push(+v.toFixed(2)); return out; };
    ticks(ymin, ymax).forEach((v) => {
      root.appendChild(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: v === 0 ? "even-line" : "grid-line" }));
      root.appendChild(text(L - 8, y(v) + 4, signed(v, 1), "ax", { "text-anchor": "end" }));
    });
    ticks(xmin, xmax).forEach((v) => {
      root.appendChild(svg("line", { x1: x(v), x2: x(v), y1: T, y2: H - B, class: v === 0 ? "even-line" : "grid-line" }));
      root.appendChild(text(x(v), H - B + 18, signed(v, 1), "ax", { "text-anchor": "middle" }));
    });
    const dlo = Math.max(xmin, ymin), dhi = Math.min(xmax, ymax);
    root.appendChild(svg("line", { x1: x(dlo), y1: y(dlo), x2: x(dhi), y2: y(dhi), stroke: "var(--ink-2)", "stroke-width": 1.5 }));
    root.appendChild(text(x(dhi) - 6, y(dhi) + 16, "Same level under pressure", "ax", { "text-anchor": "end" }));
    root.appendChild(text(W - R, H - 6, "Usual level (ordinary balls)", "ax-strong", { "text-anchor": "end" }));
    root.appendChild(text(L, T - 2, "Pressure level", "ax-strong"));

    const sorted = [...ps].sort((a, b) => (top.has(a.id) - top.has(b.id)) || (b.p_balls - a.p_balls));
    sorted.forEach((p) => {
      const isTop = top.has(p.id);
      const hit = q && p.name.toLowerCase().includes(q);
      let cls = "dot " + (isTop ? "dot--top" : "dot--base");
      if (q) cls += hit ? " dot--match" : " dot--dim";
      const c = svg("circle", { cx: x(p.usual), cy: y(p.pressure_raw), r: clamp(Math.sqrt(p.p_balls) / 3, 3, 9.5), class: cls, tabindex: isTop || hit ? 0 : null });
      const html = `<b>${esc(p.name)}</b><br>Model rank ${p.model_rank}${isTop ? ", in the top 10" : ""}<br>` +
        `Usual ${signed(p.usual)}, under pressure ${signed(p.pressure_raw)}<br>${p.p_balls} pressure balls`;
      c.addEventListener("mousemove", (e) => showTip(html, e));
      c.addEventListener("mouseleave", hideTip);
      c.addEventListener("focus", () => { const r = c.getBoundingClientRect(); showTip(html, { clientX: r.right, clientY: r.top }); });
      c.addEventListener("blur", hideTip);
      root.appendChild(c);
      if (isTop) root.appendChild(text(x(p.usual) + 10, y(p.pressure_raw) + 4, shortName(p.name), "ax-strong"));
    });
    $("#scatter").replaceChildren(root);
  }

  const COLS = [
    { k: "model_rank", label: "Model rank", num: true, f: (p) => p.model_rank },
    { k: "name", label: "Player", f: (p) => esc(p.name) },
    { k: "rating", label: "Rating", num: true, f: (p) => fmt(p.rating, 2) },
    { k: "p_balls", label: "Pressure balls", num: true, f: (p) => p.p_balls },
    { k: "p_matches", label: "Matches", num: true, f: (p) => p.p_matches },
    { k: "usual", label: "Usual level", num: true, f: (p) => signed(p.usual) },
    { k: "pressure_raw", label: "Under pressure", num: true, f: (p) => signed(p.pressure_raw) },
    { k: "delta", label: "Rise vs usual", num: true, f: (p) => `<span class="${p.delta > 0 ? "pos" : ""}">${signed(p.delta, 3)}</span>` },
    { k: "consistency", label: "Consistency", num: true, f: (p) => pct(p.consistency) },
    { k: "p_top10", label: "Top-10 chance", num: true, f: (p) => pct(p.p_top10) },
  ];

  function renderTable() {
    const role = state.xrole;
    const top = submittedIds(role);
    const { key, dir } = state.sort;
    const q = state.query.trim().toLowerCase();
    const rows = players(role).filter((p) => !q || p.name.toLowerCase().includes(q))
      .sort((a, b) => {
        const va = a[key], vb = b[key];
        if (typeof va === "string") return va.localeCompare(vb) * dir;
        return ((va ?? -Infinity) - (vb ?? -Infinity)) * dir;
      });
    $("#ptable thead").innerHTML = "<tr>" + COLS.map((c) => {
      const sort = c.k === key ? (dir === 1 ? "ascending" : "descending") : "none";
      return `<th scope="col" class="${c.num ? "num" : ""}" aria-sort="${sort}"><button data-k="${c.k}">${c.label}</button></th>`;
    }).join("") + "</tr>";
    $("#ptable tbody").innerHTML = rows.length ? rows.map((p) =>
      `<tr class="${top.has(p.id) ? "is-top" : ""}">${COLS.map((c) => `<td class="${c.num ? "num" : ""}">${c.f(p)}</td>`).join("")}</tr>`).join("")
      : `<tr><td colspan="${COLS.length}">No eligible player matches “${esc(state.query)}”. Check the spelling, or switch between batters and bowlers.</td></tr>`;
    $$("#ptable th button").forEach((b) => b.addEventListener("click", () => {
      const k = b.dataset.k;
      state.sort = { key: k, dir: state.sort.key === k ? -state.sort.dir : (k === "model_rank" || k === "name" ? 1 : -1) };
      renderTable();
    }));
  }

  function setXRole(role) {
    state.xrole = role;
    $$("[data-xrole]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.xrole === role)));
    renderScatter();
    renderTable();
  }

  // ---------- method ----------
  function renderWeights() {
    $("#weights").innerHTML = ["bat", "bowl"].map((role) => {
      const comps = state.data[role].components;
      const w = state.data.weights[role];
      const bar = comps.map((c, i) => `<span style="width:${w[c.key] * 100}%;background:${WEIGHT_COLORS[i]}" title="${esc(c.label)}">${Math.round(w[c.key] * 100)}%</span>`).join("");
      const list = comps.map((c, i) => `<li><i style="background:${WEIGHT_COLORS[i]}"></i>${c.label}<b>${Math.round(w[c.key] * 100)}%</b></li>`).join("");
      return `<div><h4>${role === "bat" ? "Batting rating" : "Bowling rating"}</h4><div class="wbar" aria-hidden="true">${bar}</div><ul class="wlist">${list}</ul></div>`;
    }).join("");
  }

  function renderSensitivity() {
    const s = state.data.sensitivity;
    const label = (t) => t.replace(/ LI$/, " of leverage").replace(/^\w/, (c) => c.toUpperCase());
    const bowl = Object.fromEntries(s.bowl.map((r) => [r.scenario, r]));
    const body = s.bat.map((r) => {
      const b = bowl[r.scenario] || {};
      return `<tr><td>${esc(label(r.scenario))}</td><td class="num">${r.top10_overlap} of 10</td><td class="num">${b.top10_overlap ?? "–"} of 10</td>
        <td class="num">${fmt(r.rank_rho_all, 2)}</td><td class="num">${fmt(b.rank_rho_all, 2)}</td></tr>`;
    }).join("");
    $("#stable").innerHTML = `<thead><tr><th scope="col">If we change this</th><th scope="col" class="num">Batters kept</th><th scope="col" class="num">Bowlers kept</th>
      <th scope="col" class="num">Rank correlation, batters</th><th scope="col" class="num">Rank correlation, bowlers</th></tr></thead><tbody>${body}</tbody>`;
  }

  function fillSpans() {
    const m = state.data.meta;
    const vals = { wicket_runs: fmt(m.wicket_runs, 1), split_bat: fmt(m.split_half.bat, 2), split_bowl: fmt(m.split_half.bowl, 2) };
    $$("[data-fill]").forEach((el) => { el.textContent = vals[el.dataset.fill] ?? ""; });
  }

  // ---------- boot ----------
  async function boot() {
    try {
      const res = await fetch("data/dashboard.json", { cache: "no-cache" });
      if (!res.ok) throw new Error(`data/dashboard.json returned ${res.status}`);
      state.data = await res.json();
    } catch (err) {
      $("#worm-caption").textContent =
        `The data file did not load (${err.message}). Serve the site folder over HTTP, for example with “python -m http.server” inside site/, rather than opening index.html directly.`;
      return;
    }
    renderFacts();
    renderWorm();
    $$("[data-role]").forEach((b) => b.addEventListener("click", () => setRole(b.dataset.role)));
    $$("[data-xrole]").forEach((b) => b.addEventListener("click", () => setXRole(b.dataset.xrole)));
    $("#search").addEventListener("input", (e) => { state.query = e.target.value; renderScatter(); renderTable(); });
    setRole("bat");
    renderHeatmap();
    renderDeciles();
    renderCompare();
    setXRole("bat");
    renderWeights();
    renderSensitivity();
    fillSpans();
    const repo = $("#repo-link");
    if (repo && repo.href.includes("YOUR-USERNAME")) repo.closest("p").hidden = true;
    let t = null;
    let lastW = window.innerWidth;
    window.addEventListener("resize", () => {
      if (window.innerWidth === lastW) return;
      lastW = window.innerWidth;
      clearTimeout(t);
      t = setTimeout(() => {
        renderWorm(); renderHeatmap(); renderDeciles(); renderScatter();
        if (state.selected) renderRange(byId(state.role, state.selected));
      }, 150);
    });
  }

  boot();
})();
