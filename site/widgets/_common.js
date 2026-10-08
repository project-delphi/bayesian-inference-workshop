// Shared helpers for the workshop's interactive widgets.
// Plain ES module. three.js core only, loaded from cdnjs.
import * as THREE from "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.module.min.js";
export { THREE };

export const COLORS = {
  primary: 0xc2410c,
  reference: 0x1f2937,
  compare: 0x2563eb,
  danger: 0xdc2626,
  accept: 0x15803d,
  bg: 0xffffff,
};
export const CSS = {
  primary: "#c2410c",
  reference: "#1f2937",
  compare: "#2563eb",
  danger: "#dc2626",
  accept: "#15803d",
  grey: "#9ca3af",
};

// ---------------------------------------------------------------- styles (once)
let stylesInjected = false;
export function injectStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const s = document.createElement("style");
  s.textContent = `
.bi-widget{font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#1f2937;background:#fff;border:1px solid #e5e7eb;border-radius:6px;padding:10px 12px 12px;box-sizing:border-box}
.bi-canvas-wrap{position:relative;width:100%;background:#f8f9fa;border-radius:4px;overflow:hidden}
.bi-canvas-wrap canvas{display:block;width:100%;height:100%}
.bi-panel{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;margin-top:8px;font-size:0.85rem}
.bi-ctrl{display:flex;align-items:center;gap:6px;white-space:nowrap}
.bi-ctrl label{color:#374151}
.bi-ctrl input[type=range]{width:120px;accent-color:#c2410c}
.bi-ctrl .bi-val{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;min-width:3.2em;text-align:right;color:#111827}
.bi-btn{border:1px solid #c2410c;background:#fff;color:#c2410c;border-radius:4px;padding:3px 10px;font-size:0.85rem;cursor:pointer}
.bi-btn:hover{background:#fff7ed}
.bi-btn.bi-on{background:#c2410c;color:#fff}
.bi-toggle{display:inline-flex;border:1px solid #d1d5db;border-radius:4px;overflow:hidden}
.bi-toggle button{border:0;background:#fff;color:#374151;padding:3px 10px;font-size:0.85rem;cursor:pointer}
.bi-toggle button.bi-on{background:#1f2937;color:#fff}
.bi-readout{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.82rem;color:#111827;margin-top:8px;min-height:1.2em;white-space:pre-wrap}
.bi-bars{display:flex;gap:14px;align-items:flex-end;margin-top:6px;font-size:0.75rem}
.bi-bar{display:flex;flex-direction:column;align-items:center;gap:2px}
.bi-bar div{width:18px;background:#2563eb;border-radius:2px 2px 0 0}
.bi-note{font-size:0.78rem;color:#6b7280;margin-top:4px}
`;
  document.head.appendChild(s);
}

// ---------------------------------------------------------------- PRNG
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function makeRng(seed = 1) {
  const u = mulberry32(seed);
  let spare = null;
  const normal = () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let a = 0, b = 0;
    do { a = u(); } while (a <= 1e-12);
    b = u();
    const r = Math.sqrt(-2 * Math.log(a));
    spare = r * Math.sin(2 * Math.PI * b);
    return r * Math.cos(2 * Math.PI * b);
  };
  return { uniform: u, normal };
}

export const reducedMotion = () =>
  window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---------------------------------------------------------------- DOM panel
export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) e.append(c);
  return e;
}

export function makePanel(parent) {
  const panel = el("div", { class: "bi-panel" });
  parent.appendChild(panel);
  const api = {
    panel,
    slider(label, min, max, step, value, onChange, fmt = (v) => String(v)) {
      const val = el("span", { class: "bi-val" }, fmt(value));
      const input = el("input", { type: "range", min, max, step, value });
      input.addEventListener("input", () => {
        const v = parseFloat(input.value);
        val.textContent = fmt(v);
        onChange(v);
      });
      panel.appendChild(el("div", { class: "bi-ctrl" }, el("label", {}, label), input, val));
      return {
        get: () => parseFloat(input.value),
        set: (v) => { input.value = v; val.textContent = fmt(v); },
      };
    },
    button(label, onClick) {
      const b = el("button", { class: "bi-btn", onclick: onClick }, label);
      panel.appendChild(b);
      return b;
    },
    toggle(options, value, onChange) {
      const wrap = el("div", { class: "bi-toggle" });
      const btns = options.map((o) => {
        const b = el("button", { onclick: () => set(o) }, o);
        wrap.appendChild(b);
        return b;
      });
      function set(o, fire = true) {
        value = o;
        btns.forEach((b, i) => b.classList.toggle("bi-on", options[i] === o));
        if (fire) onChange(o);
      }
      set(value, false);
      panel.appendChild(wrap);
      return { get: () => value, set };
    },
    checkbox(label, value, onChange) {
      const input = el("input", { type: "checkbox" });
      input.checked = value;
      input.addEventListener("change", () => onChange(input.checked));
      panel.appendChild(el("div", { class: "bi-ctrl" }, input, el("label", {}, label)));
      return { get: () => input.checked, set: (v) => { input.checked = v; } };
    },
  };
  return api;
}

export function makeReadout(parent) {
  const r = el("div", { class: "bi-readout" });
  parent.appendChild(r);
  return (text) => { r.textContent = text; };
}

// Play/pause + reset pair. Returns state object {playing}.
export function playControls(panel, onReset, startPlaying) {
  const st = { playing: startPlaying };
  const btn = panel.button(startPlaying ? "Pause" : "Play", () => {
    st.playing = !st.playing;
    btn.textContent = st.playing ? "Pause" : "Play";
  });
  panel.button("Reset", () => onReset());
  st.setPlaying = (p) => { st.playing = p; btn.textContent = p ? "Pause" : "Play"; };
  return st;
}

// ---------------------------------------------------------------- three.js scene
export function makeScene(container, { cameraPos = [6, -8, 6], target = [0, 0, 0.5], up = [0, 0, 1] } = {}) {
  const wrap = el("div", { class: "bi-canvas-wrap" });
  container.appendChild(wrap);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0xf8f9fa, 1);
  wrap.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1.6, 0.05, 500);
  camera.up.set(...up);
  camera.position.set(...cameraPos);
  const tgt = new THREE.Vector3(...target);
  camera.lookAt(tgt);

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const dir = new THREE.DirectionalLight(0xffffff, 1.1);
  dir.position.set(5, -6, 10);
  scene.add(dir);

  function resize() {
    const w = Math.max(200, wrap.clientWidth);
    const h = Math.round(w / 1.6);
    wrap.style.height = h + "px";
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(wrap);
  resize();

  const orbit = makeOrbit(renderer.domElement, camera, tgt);
  const render = () => { orbit.update(); renderer.render(scene, camera); };
  return { wrap, renderer, scene, camera, render, orbit, target: tgt };
}

// Minimal orbit: left-drag rotates around target, wheel zooms, with damping.
export function makeOrbit(dom, camera, target) {
  const sph = new THREE.Spherical().setFromVector3(camera.position.clone().sub(target));
  // Convert so that polar is measured from +Z (camera.up = z): use a rotated frame.
  const off = camera.position.clone().sub(target);
  let radius = off.length();
  let theta = Math.atan2(off.y, off.x);             // azimuth in xy plane
  let phi = Math.acos(Math.min(1, Math.max(-1, off.z / radius))); // from +z
  let dTheta = 0, dPhi = 0, dRadius = 0;
  let dragging = false, lx = 0, ly = 0;
  dom.style.cursor = "grab";
  dom.addEventListener("pointerdown", (e) => { dragging = true; lx = e.clientX; ly = e.clientY; dom.setPointerCapture(e.pointerId); dom.style.cursor = "grabbing"; });
  dom.addEventListener("pointerup", (e) => { dragging = false; dom.style.cursor = "grab"; });
  dom.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    dTheta -= (e.clientX - lx) * 0.006;
    dPhi -= (e.clientY - ly) * 0.006;
    lx = e.clientX; ly = e.clientY;
  });
  dom.addEventListener("wheel", (e) => { e.preventDefault(); dRadius += Math.sign(e.deltaY) * radius * 0.08; }, { passive: false });
  void sph;
  return {
    spin(d) { theta += d; },
    update() {
      theta += dTheta; phi += dPhi; radius += dRadius;
      dTheta *= 0.6; dPhi *= 0.6; dRadius *= 0.5;
      phi = Math.min(Math.PI - 0.05, Math.max(0.05, phi));
      radius = Math.min(200, Math.max(0.5, radius));
      camera.position.set(
        target.x + radius * Math.sin(phi) * Math.cos(theta),
        target.y + radius * Math.sin(phi) * Math.sin(theta),
        target.z + radius * Math.cos(phi));
      camera.lookAt(target);
    },
  };
}

// Viridis-like colormap (t in [0,1]) -> THREE.Color
const VIRIDIS = [[0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530], [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551], [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318], [0.741, 0.873, 0.150], [0.993, 0.906, 0.144]];
export function viridis(t) {
  t = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.floor(t), f = t - i, a = VIRIDIS[i], b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return new THREE.Color(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]));
}
// Warm colormap (white -> primary) for density surfaces.
export function warm(t) {
  t = Math.min(1, Math.max(0, t));
  const c0 = new THREE.Color(0xfff7ed), c1 = new THREE.Color(0xc2410c);
  return c0.lerp(c1, Math.pow(t, 0.7));
}

// Surface mesh z = f(x,y) with vertex colours from colormap(normalised z).
export function makeSurface(f, [x0, x1], [y0, y1], nx = 80, ny = 80, colormap = warm, opts = {}) {
  const geo = new THREE.PlaneGeometry(x1 - x0, y1 - y0, nx - 1, ny - 1);
  geo.translate((x0 + x1) / 2, (y0 + y1) / 2, 0);
  const pos = geo.attributes.position;
  const zs = new Float32Array(pos.count);
  let zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i < pos.count; i++) {
    const z = f(pos.getX(i), pos.getY(i));
    zs[i] = z; pos.setZ(i, z);
    if (z < zmin) zmin = z; if (z > zmax) zmax = z;
  }
  const colors = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const c = colormap((zs[i] - zmin) / (zmax - zmin + 1e-12));
    colors[3 * i] = c.r; colors[3 * i + 1] = c.g; colors[3 * i + 2] = c.b;
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.9, metalness: 0.0, transparent: !!opts.transparent, opacity: opts.opacity ?? 1 });
  const mesh = new THREE.Mesh(geo, mat);
  if (opts.wire) {
    const wire = new THREE.LineSegments(new THREE.WireframeGeometry(geo), new THREE.LineBasicMaterial({ color: 0x1f2937, transparent: true, opacity: 0.08 }));
    mesh.add(wire);
  }
  return mesh;
}

// Text sprite from a canvas texture.
export function textSprite(text, { color = "#1f2937", size = 0.5, font = 44 } = {}) {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 64;
  const ctx = c.getContext("2d");
  ctx.font = `${font}px Inter, system-ui, sans-serif`;
  ctx.fillStyle = color; ctx.textBaseline = "middle"; ctx.textAlign = "center";
  ctx.fillText(text, 128, 32);
  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sp.scale.set(size * 4, size, 1);
  return sp;
}

// Axes lines with labels. ranges = {x:[a,b], y:[a,b], z:[a,b]}, labels = {x,y,z}
export function makeAxes(ranges, labels = { x: "x", y: "y", z: "z" }, { origin = null, color = 0x6b7280, labelSize = 0.5 } = {}) {
  const g = new THREE.Group();
  const o = origin || [ranges.x[0], ranges.y[0], ranges.z[0]];
  const mat = new THREE.LineBasicMaterial({ color });
  const line = (a, b) => g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]), mat));
  line(o, [ranges.x[1], o[1], o[2]]);
  line(o, [o[0], ranges.y[1], o[2]]);
  line(o, [o[0], o[1], ranges.z[1]]);
  const lx = textSprite(labels.x, { size: labelSize }); lx.position.set(ranges.x[1], o[1], o[2]); g.add(lx);
  const ly = textSprite(labels.y, { size: labelSize }); ly.position.set(o[0], ranges.y[1], o[2]); g.add(ly);
  const lz = textSprite(labels.z, { size: labelSize }); lz.position.set(o[0], o[1], ranges.z[1]); g.add(lz);
  return g;
}

// Dynamic polyline with a fixed capacity.
export function makePolyline(capacity, color, { opacity = 1, width = 1 } = {}) {
  const geo = new THREE.BufferGeometry();
  const arr = new Float32Array(capacity * 3);
  geo.setAttribute("position", new THREE.BufferAttribute(arr, 3));
  geo.setDrawRange(0, 0);
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, linewidth: width }));
  let n = 0;
  return {
    line,
    clear() { n = 0; geo.setDrawRange(0, 0); },
    set(points) {
      n = Math.min(points.length, capacity);
      for (let i = 0; i < n; i++) { arr[3 * i] = points[i][0]; arr[3 * i + 1] = points[i][1]; arr[3 * i + 2] = points[i][2]; }
      geo.attributes.position.needsUpdate = true; geo.setDrawRange(0, n);
    },
    push(p) {
      if (n < capacity) { arr[3 * n] = p[0]; arr[3 * n + 1] = p[1]; arr[3 * n + 2] = p[2]; n++; }
      else { arr.copyWithin(0, 3); arr[3 * (n - 1)] = p[0]; arr[3 * (n - 1) + 1] = p[1]; arr[3 * (n - 1) + 2] = p[2]; }
      geo.attributes.position.needsUpdate = true; geo.setDrawRange(0, n);
    },
  };
}

export function sphere(radius, color) {
  return new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 12), new THREE.MeshStandardMaterial({ color, roughness: 0.5 }));
}

export function fmt(v, d = 3) { return Number.isFinite(v) ? v.toFixed(d) : String(v); }

// Read a dataset option with a fallback.
export function opt(el, opts, key, fallback) {
  if (opts && opts[key] !== undefined) return opts[key];
  const v = el.dataset[key];
  if (v === undefined) return fallback;
  if (v === "true") return true; if (v === "false") return false;
  const n = parseFloat(v);
  return Number.isFinite(n) && String(n) === v ? n : v;
}

// Animation loop helper that only runs while the element is visible.
export function animate(el, fn) {
  let visible = true, raf = 0;
  const io = new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible && !raf) loop(); });
  io.observe(el);
  function loop() { raf = 0; if (!visible) return; fn(); raf = requestAnimationFrame(loop); }
  loop();
}
