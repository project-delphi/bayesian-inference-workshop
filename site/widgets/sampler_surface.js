// Random-walk Metropolis and HMC walking on a 3-D density surface.
import { THREE, COLORS, injectStyles, makeRng, makePanel, makeReadout, playControls, makeScene, makeSurface, makeAxes, makePolyline, sphere, fmt, opt, animate, reducedMotion } from "./_common.js";

const QPCR_X = [1.844, 1.754, 2.024, 1.837, 1.613, 1.927, 2.256, 2.131, 1.554, 1.357, 1.582, 1.814];

const TARGETS = {
  gauss2d: {
    name: "correlated Gaussian, cov [[1, 0.9], [0.9, 2]]",
    xr: [-3.5, 3.5], yr: [-4.5, 4.5], labels: ["z1", "z2"], start: [-2.5, 3.0],
    // precision of [[1,.9],[.9,2]]: det = 2 - 0.81 = 1.19
    logp(x, y) { const d = 1.19; return -0.5 * (2 * x * x - 1.8 * x * y + y * y) / d; },
    grad(x, y) { const d = 1.19; return [-(2 * x - 0.9 * y) / d, -(y - 0.9 * x) / d]; },
    rwmStep: 1.0, eps: 0.5, L: 10,
  },
  qpcr: {
    name: "qPCR posterior over (mu, log lambda)",
    xr: [1.2, 2.4], yr: [0.0, 4.5], labels: ["mu", "log lambda"], start: [1.4, 0.6],
    logp(mu, phi) {
      const lam = Math.exp(phi);
      let lp = -0.5 * (mu * mu) / 100;            // N(0, 10^2)
      lp += (2 - 1) * phi - 0.5 * lam + phi;      // Gamma(2, 0.5) in lambda + Jacobian
      let ss = 0; for (const x of QPCR_X) ss += (x - mu) * (x - mu);
      lp += 0.5 * QPCR_X.length * phi - 0.5 * lam * ss;
      return lp;
    },
    grad(mu, phi) {
      const lam = Math.exp(phi);
      let s1 = 0, ss = 0; for (const x of QPCR_X) { s1 += (x - mu); ss += (x - mu) * (x - mu); }
      return [-mu / 100 + lam * s1, 1 - 0.5 * lam + 1 + 0.5 * QPCR_X.length - 0.5 * lam * ss];
    },
    rwmStep: 0.35, eps: 0.08, L: 12,
  },
  banana: {
    name: "banana: log p = -x^2/2 - (y - x^2 + 2)^2 / (2 * 0.5^2)",
    xr: [-3, 3], yr: [-3, 6], labels: ["x", "y"], start: [-2.2, 2.5],
    logp(x, y) { const r = y - x * x + 2; return -0.5 * x * x - r * r / 0.5; },
    grad(x, y) { const r = y - x * x + 2; return [-x + 8 * r * x, -4 * r]; },
    rwmStep: 0.6, eps: 0.12, L: 20,
  },
};

export function mount(el, opts = {}) {
  injectStyles();
  el.classList.add("bi-widget");
  const tKey = opt(el, opts, "target", "gauss2d");
  const T = TARGETS[tKey] || TARGETS.gauss2d;
  let mode = opt(el, opts, "mode", "hmc");
  const seed = opt(el, opts, "seed", 7);
  const ZS = 4.0; // surface height scale

  // Normalise surface height by the grid maximum of log p.
  let lpMax = -Infinity;
  for (let i = 0; i <= 120; i++) for (let j = 0; j <= 120; j++) {
    const x = T.xr[0] + (T.xr[1] - T.xr[0]) * i / 120, y = T.yr[0] + (T.yr[1] - T.yr[0]) * j / 120;
    lpMax = Math.max(lpMax, T.logp(x, y));
  }
  const h = (x, y) => ZS * Math.exp(Math.min(0, T.logp(x, y) - lpMax));
  const onSurf = (x, y, dz = 0.08) => [x, y, h(x, y) + dz];

  const cx = (T.xr[0] + T.xr[1]) / 2, cy = (T.yr[0] + T.yr[1]) / 2;
  const scn = makeScene(el, { cameraPos: [cx + 8.5, cy - 9.5, ZS * 1.25], target: [cx, cy, ZS * 0.35] });
  // Scale data coordinates so that both axes span 8 scene units whatever the target's range.
  const sx = 8 / (T.xr[1] - T.xr[0]), sy = 8 / (T.yr[1] - T.yr[0]);
  const world = new THREE.Group(); world.scale.set(sx, sy, 1); world.position.set(cx - cx * sx, cy - cy * sy, 0);
  scn.scene.add(world);

  world.add(makeSurface(h, T.xr, T.yr, 90, 90, undefined, { wire: true, transparent: true, opacity: 0.88 }));
  scn.scene.add(makeAxes({ x: [cx - 4, cx + 4.3], y: [cy - 4, cy + 4.3], z: [0, ZS * 1.05] },
    { x: T.labels[0], y: T.labels[1], z: "density" }, { origin: [cx - 4, cy - 4, 0], labelSize: 0.4 }));

  const trail = makePolyline(400, COLORS.reference, { opacity: 0.9 }); world.add(trail.line);
  const path = makePolyline(600, COLORS.compare); world.add(path.line);
  const state = sphere(0.17, COLORS.reference); state.scale.set(1 / sx, 1 / sy, 1); world.add(state);
  const prop = sphere(0.13, COLORS.accept); prop.scale.set(1 / sx, 1 / sy, 1); prop.visible = false; world.add(prop);

  // ---- sampler state
  let rng = makeRng(seed);
  let q = [...T.start], nAcc = 0, nTot = 0, lastDH = NaN, lastAlpha = NaN, flash = 0;
  let rwmStep = T.rwmStep, eps = T.eps, L = T.L, speed = 1;

  function place(mesh, p) { const s = onSurf(p[0], p[1]); mesh.position.set(s[0], s[1], s[2]); }
  function reset() {
    rng = makeRng(seed); q = [...T.start]; nAcc = 0; nTot = 0; lastDH = NaN; lastAlpha = NaN;
    trail.clear(); path.clear(); trail.push(onSurf(q[0], q[1])); place(state, q); prop.visible = false;
  }

  function rwmIter() {
    const prop_ = [q[0] + rwmStep * rng.normal(), q[1] + rwmStep * rng.normal()];
    const logr = T.logp(prop_[0], prop_[1]) - T.logp(q[0], q[1]);
    lastAlpha = Math.min(1, Math.exp(logr));
    const acc = Math.log(rng.uniform()) < logr;
    nTot++; showProposal(prop_, acc);
    if (acc) { nAcc++; q = prop_; trail.push(onSurf(q[0], q[1])); place(state, q); }
  }

  function hmcIter() {
    let p = [rng.normal(), rng.normal()];
    const q0 = [...q];
    const H0 = -T.logp(q0[0], q0[1]) + 0.5 * (p[0] * p[0] + p[1] * p[1]);
    let x = [...q0], g = T.grad(x[0], x[1]);
    const pts = [onSurf(x[0], x[1], 0.12)];
    for (let i = 0; i < L; i++) {
      p = [p[0] + 0.5 * eps * g[0], p[1] + 0.5 * eps * g[1]];
      x = [x[0] + eps * p[0], x[1] + eps * p[1]];
      g = T.grad(x[0], x[1]);
      p = [p[0] + 0.5 * eps * g[0], p[1] + 0.5 * eps * g[1]];
      pts.push(onSurf(x[0], x[1], 0.12));
    }
    const H1 = -T.logp(x[0], x[1]) + 0.5 * (p[0] * p[0] + p[1] * p[1]);
    lastDH = Number.isFinite(H1) ? H1 - H0 : Infinity;
    lastAlpha = Math.min(1, Math.exp(-lastDH));
    const acc = Number.isFinite(lastDH) && Math.log(rng.uniform()) < -lastDH;
    nTot++; path.set(pts); showProposal(x, acc);
    if (acc) { nAcc++; q = x; trail.push(onSurf(q[0], q[1])); place(state, q); }
  }

  function showProposal(pt, acc) {
    const inside = pt[0] > T.xr[0] && pt[0] < T.xr[1] && pt[1] > T.yr[0] && pt[1] < T.yr[1];
    prop.visible = inside;
    if (inside) { place(prop, pt); prop.material.color.setHex(acc ? COLORS.accept : COLORS.danger); }
    path.line.material.color.setHex(acc ? COLORS.compare : COLORS.danger);
    flash = 12;
  }

  // ---- controls
  const panel = makePanel(el);
  const modeT = panel.toggle(["rwm", "hmc"], mode, (m) => { mode = m; syncSliders(); path.clear(); });
  const sStep = panel.slider("RWM step", 0.02, 3, 0.01, rwmStep, (v) => (rwmStep = v), (v) => fmt(v, 2));
  const sEps = panel.slider("eps", 0.01, 2.5, 0.01, eps, (v) => (eps = v), (v) => fmt(v, 2));
  const sL = panel.slider("L", 1, 50, 1, L, (v) => (L = v), (v) => String(v));
  panel.slider("speed", 1, 40, 1, speed, (v) => (speed = v), (v) => v + "/frame");
  const play = playControls(panel, reset, !reducedMotion());
  const readout = makeReadout(el);
  function syncSliders() {
    const hmc = mode === "hmc";
    sStep.set(rwmStep); sEps.set(eps); sL.set(L);
    [...panel.panel.querySelectorAll(".bi-ctrl")].forEach((c) => {
      const lbl = c.querySelector("label")?.textContent;
      if (lbl === "RWM step") c.style.display = hmc ? "none" : "";
      if (lbl === "eps" || lbl === "L") c.style.display = hmc ? "" : "none";
    });
  }
  syncSliders();
  reset();

  animate(el, () => {
    if (play.playing) for (let i = 0; i < speed; i++) (mode === "hmc" ? hmcIter : rwmIter)();
    if (flash > 0) { flash--; if (flash === 0) { prop.visible = false; } }
    const rate = nTot ? nAcc / nTot : 0;
    readout(`${T.name}\n${mode.toUpperCase()}  samples ${nTot}  accepted ${nAcc}  acceptance ${fmt(rate, 2)}` +
      (mode === "hmc" ? `   last dH ${fmt(lastDH, 3)}   alpha = min(1, e^-dH) = ${fmt(lastAlpha, 2)}` : `   last alpha ${fmt(lastAlpha, 2)}`) +
      `   state (${fmt(q[0], 2)}, ${fmt(q[1], 2)})`);
    scn.render();
  });
  return { reset, setMode: (m) => modeT.set(m) };
}
