// Prior draws of the hierarchical uplift model in 3-D: the funnel exists before any data.
import { THREE, COLORS, injectStyles, makeRng, makePanel, makeReadout, playControls, makeScene, makeAxes, viridis, fmt, opt, animate, reducedMotion } from "./_common.js";

export function mount(el, opts = {}) {
  injectStyles();
  el.classList.add("bi-widget");
  const seed = opt(el, opts, "seed", 3);
  let n = opt(el, opts, "points", 6000);
  let view = opt(el, opts, "view", "centred"); // centred | non-centred
  let eps = opt(el, opts, "eps", 0.3);
  const MAXN = 20000;

  // Scene: x = theta_1 / eta_1, y = theta_2 / eta_2, z = log tau. Scene units: z is log tau, x,y scaled.
  const RX = 30; // visible half-range of theta
  const ZSC = 3;  // log tau is drawn multiplied by 3 so the funnel is tall
  const scn = makeScene(el, { cameraPos: [58, -68, 6], target: [0, 0, -4] });
  const axesC = makeAxes({ x: [-RX, RX], y: [-RX, RX], z: [-5 * ZSC, 3 * ZSC] }, { x: "theta_1", y: "theta_2", z: "log tau" }, { origin: [-RX, -RX, -5 * ZSC], labelSize: 2.4 });
  const axesN = makeAxes({ x: [-RX, RX], y: [-RX, RX], z: [-5 * ZSC, 3 * ZSC] }, { x: "eta_1 (x8)", y: "eta_2 (x8)", z: "log tau" }, { origin: [-RX, -RX, -5 * ZSC], labelSize: 2.4 });
  scn.scene.add(axesC); scn.scene.add(axesN); axesN.visible = false;

  // Buffers
  const posC = new Float32Array(MAXN * 3), posN = new Float32Array(MAXN * 3), cur = new Float32Array(MAXN * 3);
  const colors = new Float32Array(MAXN * 3), tau = new Float32Array(MAXN);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(cur, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.45, vertexColors: true, transparent: true, opacity: 0.85 }));
  scn.scene.add(pts);

  const ETA_SCALE = 8; // eta is shown multiplied by 8 so the cloud fills the same box
  let rng;
  function draw() {
    rng = makeRng(seed);
    for (let i = 0; i < n; i++) {
      const mu = 5 * rng.normal();
      const t = Math.abs(5 * rng.normal());
      const e1 = rng.normal(), e2 = rng.normal();
      tau[i] = t;
      const lt = ZSC * Math.log(t);
      posC[3 * i] = mu + t * e1; posC[3 * i + 1] = mu + t * e2; posC[3 * i + 2] = lt;
      posN[3 * i] = ETA_SCALE * e1; posN[3 * i + 1] = ETA_SCALE * e2; posN[3 * i + 2] = lt;
    }
    geo.setDrawRange(0, n);
    morphFrom = view === "centred" ? posC : posN; morphTo = morphFrom; morphT = 1;
    cur.set(morphFrom.subarray(0, 3 * n));
    geo.attributes.position.needsUpdate = true;
    recolor();
  }

  function recolor() {
    let unreachable = 0;
    for (let i = 0; i < n; i++) {
      const lt = Math.log(tau[i]);
      const red = view === "centred" && tau[i] < eps / 2;
      if (red) unreachable++;
      const c = red ? new THREE.Color(COLORS.danger) : viridis((lt + 4) / 7);
      colors[3 * i] = c.r; colors[3 * i + 1] = c.g; colors[3 * i + 2] = c.b;
    }
    geo.attributes.color.needsUpdate = true;
    fracRed = unreachable / n;
  }

  let morphFrom = posC, morphTo = posC, morphT = 1, fracRed = 0, spin = 0;
  function setView(v) {
    view = v;
    morphFrom = cur.slice(0, 3 * n); morphTo = v === "centred" ? posC : posN; morphT = 0;
    axesC.visible = v === "centred"; axesN.visible = v !== "centred";
    recolor();
  }

  const panel = makePanel(el);
  panel.toggle(["centred", "non-centred"], view, setView);
  panel.slider("points", 2000, MAXN, 1000, n, (v) => { n = v; draw(); }, (v) => String(v));
  panel.slider("step size eps", 0.02, 2.0, 0.01, eps, (v) => { eps = v; recolor(); }, (v) => fmt(v, 2));
  const play = playControls(panel, () => { setView("centred"); draw(); }, !reducedMotion());
  const readout = makeReadout(el);
  draw();

  animate(el, () => {
    if (morphT < 1) {
      morphT = Math.min(1, morphT + 1 / 60);
      const s = morphT * morphT * (3 - 2 * morphT);
      for (let i = 0; i < 3 * n; i++) cur[i] = morphFrom[i] + s * (morphTo[i] - morphFrom[i]);
      geo.attributes.position.needsUpdate = true;
    }
    if (play.playing) { spin += 0.0025; scn.orbit.spin(0.0025); }
    readout(`${view}: ${n} prior draws, coloured by log tau` +
      (view === "centred"
        ? `\nleapfrog on theta_j | tau ~ N(mu, tau^2) is unstable when eps > 2 tau: with eps = ${fmt(eps, 2)}, ${fmt(100 * fracRed, 1)}% of prior mass (red, tau < eps/2) is unreachable`
        : `\neta_j ~ N(0, 1) whatever tau is: no step size excludes any region (nothing turns red)`));
    scn.render();
  });
  return { setView };
}
