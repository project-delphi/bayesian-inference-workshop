// KL as a Bregman divergence: the gap between the log-partition surface and a tangent plane.
import { THREE, COLORS, injectStyles, makePanel, makeReadout, playControls, makeScene, makeSurface, makeAxes, sphere, fmt, opt, animate, reducedMotion } from "./_common.js";

const A = (e1, e2) => Math.log(1 + Math.exp(e1) + Math.exp(e2));
const probs = (e1, e2) => { const z = 1 + Math.exp(e1) + Math.exp(e2); return [Math.exp(e1) / z, Math.exp(e2) / z, 1 / z]; };
const gradA = (e1, e2) => probs(e1, e2).slice(0, 2); // mean parameters

export function mount(el, opts = {}) {
  injectStyles();
  el.classList.add("bi-widget");
  let ep = [opt(el, opts, "p1", -1.0), opt(el, opts, "p2", 0.5)];
  let eq = [opt(el, opts, "q1", 1.5), opt(el, opts, "q2", -1.0)];
  const R = [-4, 4];

  const scn = makeScene(el, { cameraPos: [9, -11, 7], target: [0, 0, 2.2] });
  scn.scene.add(makeSurface(A, R, R, 70, 70, undefined, { transparent: true, opacity: 0.55, wire: true }));
  scn.scene.add(makeAxes({ x: [-4, 4.4], y: [-4, 4.4], z: [0, 5] }, { x: "eta_1", y: "eta_2", z: "A(eta)" }, { labelSize: 0.6 }));

  // Tangent plane at eta_p: z = A(p) + grad A(p) . (eta - p)
  const planeGeo = new THREE.PlaneGeometry(8, 8, 1, 1);
  const plane = new THREE.Mesh(planeGeo, new THREE.MeshStandardMaterial({ color: COLORS.compare, transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false }));
  scn.scene.add(plane);
  const pBall = sphere(0.14, COLORS.compare); scn.scene.add(pBall);
  const qBall = sphere(0.14, COLORS.primary); scn.scene.add(qBall);
  const qFoot = sphere(0.1, COLORS.reference); scn.scene.add(qFoot);
  const segGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const seg = new THREE.Line(segGeo, new THREE.LineBasicMaterial({ color: COLORS.danger, linewidth: 2 }));
  scn.scene.add(seg);

  let kl = 0, klFormula = 0;
  function update() {
    const Ap = A(ep[0], ep[1]), g = gradA(ep[0], ep[1]);
    const Aq = A(eq[0], eq[1]);
    const tangentAtQ = Ap + g[0] * (eq[0] - ep[0]) + g[1] * (eq[1] - ep[1]);
    kl = Aq - tangentAtQ;
    const p = probs(ep[0], ep[1]), q = probs(eq[0], eq[1]);
    klFormula = p.reduce((s, pk, k) => s + pk * Math.log(pk / q[k]), 0);
    // plane vertices: the geometry covers [-4, 4]^2 in (eta_1, eta_2); lift each corner onto the tangent plane
    const pos = planeGeo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i);
      pos.setZ(i, Ap + g[0] * (x - ep[0]) + g[1] * (y - ep[1]));
    }
    pos.needsUpdate = true; planeGeo.computeVertexNormals();
    pBall.position.set(ep[0], ep[1], Ap);
    qBall.position.set(eq[0], eq[1], Aq);
    qFoot.position.set(eq[0], eq[1], tangentAtQ);
    segGeo.setFromPoints([new THREE.Vector3(eq[0], eq[1], tangentAtQ), new THREE.Vector3(eq[0], eq[1], Aq)]);
    bars(p, q);
  }

  const panel = makePanel(el);
  panel.slider("eta_p,1", -4, 4, 0.05, ep[0], (v) => { ep[0] = v; update(); }, (v) => fmt(v, 2));
  panel.slider("eta_p,2", -4, 4, 0.05, ep[1], (v) => { ep[1] = v; update(); }, (v) => fmt(v, 2));
  panel.slider("eta_q,1", -4, 4, 0.05, eq[0], (v) => { eq[0] = v; update(); }, (v) => fmt(v, 2));
  panel.slider("eta_q,2", -4, 4, 0.05, eq[1], (v) => { eq[1] = v; update(); }, (v) => fmt(v, 2));
  const play = playControls(panel, () => { ep = [-1.0, 0.5]; eq = [1.5, -1.0]; sync(); update(); }, !reducedMotion());
  const sliders = [...panel.panel.querySelectorAll("input[type=range]")];
  function sync() { [ep[0], ep[1], eq[0], eq[1]].forEach((v, i) => { sliders[i].value = v; sliders[i].nextSibling.textContent = fmt(v, 2); }); }
  const readout = makeReadout(el);
  const barWrap = document.createElement("div"); barWrap.className = "bi-bars"; el.appendChild(barWrap);
  function bars(p, q) {
    barWrap.innerHTML = "";
    const mk = (label, v, color) => { const b = document.createElement("div"); b.className = "bi-bar"; const d = document.createElement("div"); d.style.height = (4 + 60 * v) + "px"; d.style.background = color; b.append(d, label + " " + v.toFixed(2)); return b; };
    ["p_1", "p_2", "p_3"].forEach((l, k) => barWrap.appendChild(mk(l, p[k], "#2563eb")));
    const gap = document.createElement("div"); gap.style.width = "12px"; barWrap.appendChild(gap);
    ["q_1", "q_2", "q_3"].forEach((l, k) => barWrap.appendChild(mk(l, q[k], "#c2410c")));
  }
  update();
  let t = 0;
  animate(el, () => {
    if (play.playing) { // slow drift of eta_q along a loop so the gap is seen changing
      t += 0.008; eq = [1.5 * Math.cos(t) + 0.3, 1.5 * Math.sin(1.3 * t) - 0.5]; sync(); update();
    }
    readout(`blue plane: tangent to A at eta_p.  red segment at eta_q: A(eta_q) - tangent = D_A(eta_q, eta_p) = ${fmt(kl, 4)}` +
      `\nKL(p || q) = sum_k p_k log(p_k / q_k) = ${fmt(klFormula, 4)}   (equal to the gap; the surface never dips below its tangent plane because A is convex)`);
    scn.render();
  });
}
