// Which way round? Fit a Gaussian q to a bimodal p and watch the two KL directions disagree.
import { CSS, injectStyles, makePanel, makeReadout, el as h, fmt, opt, animate } from "./_common.js";

const GRID = []; for (let i = 0; i <= 1200; i++) GRID.push(-6 + 12 * i / 1200);
const DX = 12 / 1200;
const normal = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
const P = GRID.map((x) => 0.5 * normal(x, -2, 0.6) + 0.5 * normal(x, 2, 0.6));
const LOGP = P.map((v) => Math.log(v + 1e-300));

function kls(mu, sigma) {
  let qp = 0, pq = 0;
  for (let i = 0; i < GRID.length; i++) {
    const q = normal(GRID[i], mu, sigma), lq = Math.log(q + 1e-300);
    qp += q * (lq - LOGP[i]) * DX;  // KL(q || p)
    pq += P[i] * (LOGP[i] - lq) * DX; // KL(p || q)
  }
  return [qp, pq];
}
function optimum(which) {
  let best = null, bv = Infinity;
  for (let mu = -4; mu <= 4.0001; mu += 0.05) for (let s = 0.2; s <= 4.0001; s += 0.02) {
    const v = kls(mu, s)[which]; if (v < bv) { bv = v; best = [mu, s]; }
  }
  return best;
}

export function mount(el, opts = {}) {
  injectStyles();
  el.classList.add("bi-widget");
  let mu = opt(el, opts, "mu", 0.0), sigma = opt(el, opts, "sigma", 1.0);
  const optQP = optimum(0), optPQ = optimum(1);

  const wrap = h("div", { class: "bi-canvas-wrap" });
  const canvas = h("canvas"); wrap.appendChild(canvas); el.appendChild(wrap);
  const ctx = canvas.getContext("2d");
  function resize() {
    const w = Math.max(300, wrap.clientWidth), hh = Math.round(w / 1.6);
    wrap.style.height = hh + "px";
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = w * dpr; canvas.height = hh * dpr; canvas.style.width = w + "px"; canvas.style.height = hh + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  new ResizeObserver(resize).observe(wrap); resize();

  let target = null; // animation target [mu, sigma]
  const panel = makePanel(el);
  const sMu = panel.slider("mu", -4, 4, 0.01, mu, (v) => { mu = v; target = null; }, (v) => fmt(v, 2));
  const sSg = panel.slider("sigma", 0.2, 4, 0.01, sigma, (v) => { sigma = v; target = null; }, (v) => fmt(v, 2));
  panel.button("minimise KL(q || p)", () => (target = optQP));
  panel.button("minimise KL(p || q)", () => (target = optPQ));
  panel.button("Reset", () => { mu = 0; sigma = 1; target = null; sMu.set(mu); sSg.set(sigma); });
  const readout = makeReadout(el);

  // Drag: horizontal moves mu, vertical changes sigma.
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY, mu, sigma]; canvas.setPointerCapture(e.pointerId); target = null; });
  canvas.addEventListener("pointerup", () => (drag = null));
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const w = canvas.clientWidth;
    mu = Math.max(-4, Math.min(4, drag[2] + (e.clientX - drag[0]) / (w / 12)));
    sigma = Math.max(0.2, Math.min(4, drag[3] * Math.exp((e.clientY - drag[1]) / 120)));
    sMu.set(mu); sSg.set(sigma);
  });
  canvas.style.cursor = "ew-resize";

  function draw() {
    const w = canvas.clientWidth, hh = canvas.clientHeight;
    ctx.fillStyle = "#f8f9fa"; ctx.fillRect(0, 0, w, hh);
    const top = { y0: 10, y1: hh * 0.58 }, bot = { y0: hh * 0.66, y1: hh - 18 };
    const xpx = (x) => (x + 6) / 12 * w;
    const [qp, pq] = kls(mu, sigma);
    const Q = GRID.map((x) => normal(x, mu, sigma));
    const ymax = Math.max(0.7, ...Q) * 1.08;
    const ypx = (y) => top.y1 - (y / ymax) * (top.y1 - top.y0);
    // main curves
    ctx.lineWidth = 2;
    const curve = (arr, color, fill) => {
      ctx.beginPath(); GRID.forEach((x, i) => i ? ctx.lineTo(xpx(x), ypx(arr[i])) : ctx.moveTo(xpx(x), ypx(arr[i])));
      ctx.strokeStyle = color; ctx.stroke();
      if (fill) { ctx.lineTo(xpx(6), top.y1); ctx.lineTo(xpx(-6), top.y1); ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); }
    };
    curve(P, CSS.reference, "rgba(31,41,55,0.08)");
    curve(Q, CSS.primary, "rgba(194,65,12,0.12)");
    ctx.strokeStyle = "#d1d5db"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, top.y1); ctx.lineTo(w, top.y1); ctx.stroke();
    ctx.fillStyle = CSS.reference; ctx.font = "13px Inter, system-ui, sans-serif";
    ctx.fillText("p(x) = 0.5 N(-2, 0.6^2) + 0.5 N(2, 0.6^2)", 12, 24);
    ctx.fillStyle = CSS.primary; ctx.fillText(`q(x) = N(${fmt(mu, 2)}, ${fmt(sigma, 2)}^2)   (drag: left-right sets mu, up-down sets sigma)`, 12, 42);
    // integrand panels
    const half = w / 2 - 8;
    const panelPlot = (x0, arr, color, label, value) => {
      let m = 1e-9; arr.forEach((v) => (m = Math.max(m, Math.abs(v))));
      const yz = (bot.y0 + bot.y1) / 2;
      const yp = (v) => yz - (v / m) * (bot.y1 - bot.y0) / 2 * 0.9;
      ctx.strokeStyle = "#d1d5db"; ctx.beginPath(); ctx.moveTo(x0, yz); ctx.lineTo(x0 + half, yz); ctx.stroke();
      ctx.beginPath();
      GRID.forEach((x, i) => { const px = x0 + (x + 6) / 12 * half; i ? ctx.lineTo(px, yp(arr[i])) : ctx.moveTo(px, yp(arr[i])); });
      ctx.lineTo(x0 + half, yz); ctx.lineTo(x0, yz); ctx.closePath();
      ctx.fillStyle = color.replace(")", ",0.25)").replace("rgb", "rgba"); ctx.fill();
      ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.fillStyle = "#111827"; ctx.font = "12px Inter, system-ui, sans-serif";
      ctx.fillText(`${label} integrand, area = ${fmt(value, 3)} nats`, x0 + 4, bot.y0 + 12);
    };
    const intQP = GRID.map((x, i) => { const q = Q[i]; return q * (Math.log(q + 1e-300) - LOGP[i]); });
    const intPQ = GRID.map((x, i) => P[i] * (LOGP[i] - Math.log(Q[i] + 1e-300)));
    panelPlot(4, intQP, "rgb(194,65,12)", "q log(q/p)", qp);
    panelPlot(w / 2 + 4, intPQ, "rgb(37,99,235)", "p log(p/q)", pq);
    // bars in readout
    const bar = (v) => "#".repeat(Math.min(40, Math.round(v * 10))).padEnd(40, ".");
    readout(`KL(q || p) = ${fmt(qp, 3)}  ${bar(qp)}   (variational inference minimises this one; the optimum sits in one mode)\n` +
      `KL(p || q) = ${fmt(pq, 3)}  ${bar(pq)}   (moment matching / maximum likelihood minimises this one; the optimum covers both modes)`);
  }

  animate(el, () => {
    if (target) {
      mu += 0.08 * (target[0] - mu); sigma += 0.08 * (target[1] - sigma);
      if (Math.abs(mu - target[0]) < 1e-3 && Math.abs(sigma - target[1]) < 1e-3) { mu = target[0]; sigma = target[1]; target = null; }
      sMu.set(mu); sSg.set(sigma);
    }
    draw();
  });
}
