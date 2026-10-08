// Forward and reverse VP-SDE on a three-mode mixture, drawn in (x, y, t) with t vertical.
import { THREE, COLORS, injectStyles, makeRng, makePanel, makeReadout, playControls, makeScene, makeAxes, makePolyline, fmt, opt, animate, reducedMotion } from "./_common.js";

const CENTRES = [[-1.1, -0.7], [-2.1, 2.3], [1.0, 0.8]];
const SCALES = [[0.25, 0.3], [0.35, 0.35], [0.2, 0.2]];
const W = [0.55, 0.35, 0.10];
const MODE_COLORS = [COLORS.primary, COLORS.compare, COLORS.accept];
const beta = (t) => 0.1 + 19.9 * t;
const alphaBar = (t) => Math.exp(-(0.1 * t + 9.95 * t * t));
const T_EPS = 1e-3, TZ = 5; // t is drawn scaled by TZ

// Exact score of the perturbed mixture at time t.
function score(x, t) {
  const ab = alphaBar(t), sab = Math.sqrt(ab);
  const lw = [], mu = [], v = [];
  for (let k = 0; k < 3; k++) {
    mu.push([sab * CENTRES[k][0], sab * CENTRES[k][1]]);
    v.push([ab * SCALES[k][0] ** 2 + 1 - ab, ab * SCALES[k][1] ** 2 + 1 - ab]);
    const d0 = x[0] - mu[k][0], d1 = x[1] - mu[k][1];
    lw.push(Math.log(W[k]) - 0.5 * (d0 * d0 / v[k][0] + d1 * d1 / v[k][1]) - 0.5 * Math.log(v[k][0] * v[k][1]));
  }
  const m = Math.max(...lw); const r = lw.map((l) => Math.exp(l - m)); const z = r[0] + r[1] + r[2];
  let s0 = 0, s1 = 0;
  for (let k = 0; k < 3; k++) { s0 += (r[k] / z) * (-(x[0] - mu[k][0]) / v[k][0]); s1 += (r[k] / z) * (-(x[1] - mu[k][1]) / v[k][1]); }
  return [s0, s1];
}

export function mount(el, opts = {}) {
  injectStyles();
  el.classList.add("bi-widget");
  const N = opt(el, opts, "points", 400);
  const seed = opt(el, opts, "seed", 11);
  let nSteps = opt(el, opts, "steps", 200);
  let ode = opt(el, opts, "ode", false);

  const scn = makeScene(el, { cameraPos: [11, -13, 7], target: [0, 0, TZ / 2] });
  scn.scene.add(makeAxes({ x: [-4, 4.4], y: [-4, 4.4], z: [0, TZ * 1.08] }, { x: "x", y: "y", z: "t" }, { labelSize: 0.6 }));
  // t = 0 and t = 1 reference squares
  const sq = (z, color) => { const g = new THREE.BufferGeometry().setFromPoints([[-4, -4], [4, -4], [4, 4], [-4, 4], [-4, -4]].map((p) => new THREE.Vector3(p[0], p[1], z))); scn.scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.4 }))); };
  sq(0, 0x6b7280); sq(TZ, 0x6b7280);

  let rng = makeRng(seed);
  const x0 = [], mode = [];
  const lines = [], cloudPos = new Float32Array(N * 3), cloudCol = new Float32Array(N * 3);
  const cloudGeo = new THREE.BufferGeometry();
  cloudGeo.setAttribute("position", new THREE.BufferAttribute(cloudPos, 3));
  cloudGeo.setAttribute("color", new THREE.BufferAttribute(cloudCol, 3));
  const cloud = new THREE.Points(cloudGeo, new THREE.PointsMaterial({ size: 0.14, vertexColors: true }));
  scn.scene.add(cloud);
  for (let i = 0; i < N; i++) {
    const pl = makePolyline(420, COLORS.reference, { opacity: 0.35 }); scn.scene.add(pl.line); lines.push(pl);
  }

  let phase = "forward", t = 0, x = [], done = false;
  function sampleData() {
    rng = makeRng(seed); x0.length = 0; mode.length = 0;
    for (let i = 0; i < N; i++) {
      const u = rng.uniform(); const k = u < W[0] ? 0 : u < W[0] + W[1] ? 1 : 2;
      mode.push(k); x0.push([CENTRES[k][0] + SCALES[k][0] * rng.normal(), CENTRES[k][1] + SCALES[k][1] * rng.normal()]);
    }
  }
  function colorBy(byMode) {
    for (let i = 0; i < N; i++) {
      const c = new THREE.Color(byMode ? MODE_COLORS[mode[i]] : COLORS.reference);
      cloudCol[3 * i] = c.r; cloudCol[3 * i + 1] = c.g; cloudCol[3 * i + 2] = c.b;
      lines[i].line.material.color.set(c);
    }
    cloudGeo.attributes.color.needsUpdate = true;
  }
  function startForward() {
    sampleData(); phase = "forward"; t = 0; done = false;
    x = x0.map((p) => [...p]);
    lines.forEach((l, i) => { l.clear(); l.push([x[i][0], x[i][1], 0]); });
    colorBy(true); writeCloud();
  }
  function startReverse() {
    rng = makeRng(seed + 1); phase = "reverse"; t = 1; done = false;
    x = []; for (let i = 0; i < N; i++) x.push([rng.normal(), rng.normal()]);
    mode.length = 0; for (let i = 0; i < N; i++) mode.push(0);
    lines.forEach((l, i) => { l.clear(); l.push([x[i][0], x[i][1], TZ]); });
    colorBy(false); writeCloud();
  }
  function writeCloud() {
    for (let i = 0; i < N; i++) { cloudPos[3 * i] = x[i][0]; cloudPos[3 * i + 1] = x[i][1]; cloudPos[3 * i + 2] = TZ * t; }
    cloudGeo.attributes.position.needsUpdate = true;
  }
  function stepForward(dt) {
    const b = beta(t), sq = Math.sqrt(b * dt);
    for (let i = 0; i < N; i++) {
      x[i][0] += -0.5 * b * x[i][0] * dt + sq * rng.normal();
      x[i][1] += -0.5 * b * x[i][1] * dt + sq * rng.normal();
    }
    t = Math.min(1, t + dt);
    for (let i = 0; i < N; i++) lines[i].push([x[i][0], x[i][1], TZ * t]);
    writeCloud(); if (t >= 1) done = true;
  }
  function stepReverse(dt) {
    const b = beta(t), sq = Math.sqrt(b * dt), last = t - dt <= T_EPS;
    for (let i = 0; i < N; i++) {
      const s = score(x[i], t);
      if (ode) { // probability-flow ODE: dx = (f - g^2 s / 2) dt, run backwards
        x[i][0] -= (-0.5 * b * x[i][0] - 0.5 * b * s[0]) * dt;
        x[i][1] -= (-0.5 * b * x[i][1] - 0.5 * b * s[1]) * dt;
      } else {    // reverse SDE: dx = (f - g^2 s) dt + g dW, run backwards; no noise on the final step
        x[i][0] -= (-0.5 * b * x[i][0] - b * s[0]) * dt; x[i][1] -= (-0.5 * b * x[i][1] - b * s[1]) * dt;
        if (!last) { x[i][0] += sq * rng.normal(); x[i][1] += sq * rng.normal(); }
      }
    }
    t = Math.max(T_EPS, t - dt);
    for (let i = 0; i < N; i++) lines[i].push([x[i][0], x[i][1], TZ * t]);
    writeCloud();
    if (last) {
      done = true;
      // colour final samples by nearest mode so the three blobs can be read
      for (let i = 0; i < N; i++) { let best = 0, bd = Infinity; for (let k = 0; k < 3; k++) { const d = (x[i][0] - CENTRES[k][0]) ** 2 + (x[i][1] - CENTRES[k][1]) ** 2; if (d < bd) { bd = d; best = k; } } mode[i] = best; }
      colorBy(true);
    }
  }

  const panel = makePanel(el);
  panel.button("Forward", startForward);
  panel.button("Reverse", startReverse);
  panel.slider("reverse steps", 50, 400, 10, nSteps, (v) => (nSteps = v), (v) => String(v));
  panel.checkbox("probability-flow ODE (deterministic reverse)", ode, (v) => { ode = v; if (phase === "reverse") startReverse(); });
  const play = playControls(panel, startForward, !reducedMotion());
  const readout = makeReadout(el);
  startForward();

  animate(el, () => {
    if (play.playing && !done) {
      if (phase === "forward") stepForward(1 / 200);
      else stepReverse((1 - T_EPS) / nSteps);
    }
    readout(`${phase === "forward" ? "forward SDE: dx = -beta x/2 dt + sqrt(beta) dW, data (bottom) to noise (top)" : (ode ? "probability-flow ODE: dx = -beta (x + s(x,t))/2 dt, noise (top) to data (bottom)" : "reverse SDE: dx = [-beta x/2 - beta s(x,t)] dt + sqrt(beta) dW, noise (top) to data (bottom)")}` +
      `\nt = ${fmt(t, 3)}   sqrt(alpha_bar) = ${fmt(Math.sqrt(alphaBar(t)), 3)}   sigma_t = ${fmt(Math.sqrt(1 - alphaBar(t)), 3)}   ${N} paths${done ? "   (finished; press Forward or Reverse)" : ""}`);
    scn.render();
  });
}
