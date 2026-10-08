# Interactive widgets

Plain ES modules under `site/widgets/`, no bundler. The four 3-D widgets import three.js
core from cdnjs (`three.js/0.160.0/three.module.min.js`); `kl_direction.js` is a 2-D
canvas and loads nothing external. `_common.js` holds the shared orbit control, panel
builder, surface and axes helpers, seeded PRNG and the injected `bi-` CSS. Every widget
exports `mount(el, opts)`; options can also be given as `data-*` attributes on the
container. All widgets respect `prefers-reduced-motion` (they start paused), have
Play/Pause (except `kl_direction`, which has no free-running animation) and Reset, and
show a one-line readout of the key quantity. They only animate while scrolled into view.

`_test.html` is a development page that mounts every widget with default options. Serve
`site/` over HTTP to use it (`python -m http.server 8765 --directory site`, then open
`/widgets/_test.html`); ES modules do not load from `file://`. It is not part of the
Quarto site.

`_quarto.yml` needs `project: resources: - widgets/**` so the `.js` files are copied to
`_site/` (the parent build handles this). Snippets below assume a page in `site/dayN/`,
hence `../widgets/`.

Colours: `#c2410c` sampler / learner quantities, `#1f2937` reference, `#2563eb` second
comparison, `#dc2626` rejections and divergences, `#15803d` accepted proposals.

---

## 1. `sampler_surface.js` — RWM and HMC on a density surface

```{=html}
<div class="bi-widget" id="hmc-surface" data-target="gauss2d" data-mode="hmc" style="width:100%;max-width:860px;margin:1.5rem auto;"></div>
<script type="module">
import { mount } from "../widgets/sampler_surface.js";
mount(document.getElementById("hmc-surface"));
</script>
```

Options: `data-target` = `gauss2d` (zero-mean Gaussian, covariance [[1, 0.9], [0.9, 2]]),
`qpcr` (the Module 2b posterior over (mu, log lambda) with the twelve replicates and
the Jacobian term), `banana` (log p = -x^2/2 - (y - x^2 + 2)^2 / (2 * 0.5^2));
`data-mode` = `rwm` | `hmc` (the panel toggle switches at run time); `data-seed` (int).
Per-target default step sizes and L are built in.

Page paragraph: The surface is the unnormalised density; the dark sphere is the current
state and the grey trail is the last few hundred accepted states. In RWM mode each
proposal flashes green (accepted) or red (rejected); raise the step size until most
proposals land in the tails and are refused, then lower it until the chain crawls. In
HMC mode the blue polyline is the leapfrog trajectory: one proposal moves across the
whole mode, and the readout gives the energy error dH and the acceptance probability
min(1, e^-dH). Push eps past 2 (the stability limit for a unit-variance Gaussian) and
watch dH explode and every proposal turn red. Drag to rotate, scroll to zoom.

For Module 2b use `data-target="qpcr" data-mode="rwm"`: the surface is the exact
posterior of Step 2 and the walker is Step 5's sampler. The `banana` target is for the
Challenge: a fixed step size that works along the ridge fails across it.

## 2. `funnel.js` — the funnel is in the prior

```{=html}
<div class="bi-widget" id="funnel-prior" data-points="6000" data-eps="0.3" style="width:100%;max-width:860px;margin:1.5rem auto;"></div>
<script type="module">
import { mount } from "../widgets/funnel.js";
mount(document.getElementById("funnel-prior"));
</script>
```

Options: `data-points` (2000–20000, default 6000), `data-view` = `centred` |
`non-centred`, `data-eps` (step size for the red colouring, default 0.3), `data-seed`.

Page paragraph: Every point is one draw from the prior of the regional-uplift model, no
data involved: x and y are theta_1 and theta_2, height is log tau, colour is log tau.
Where tau is small the two theta coordinates collapse onto mu, which is the neck of the
funnel. The step-size slider colours red every draw with tau < eps / 2, where leapfrog
on theta_j | tau ~ N(mu, tau^2) is unstable; the readout gives the prior mass that a
sampler with that step size cannot enter. Switch to non-centred and the same draws are
re-plotted in (eta_1, eta_2, log tau): eta has unit scale at every tau, the cloud is a
cylinder, and nothing turns red. Same joint distribution, different coordinates.

## 3. `bregman.js` — KL as the gap above a tangent plane

```{=html}
<div class="bi-widget" id="bregman-surface" data-p1="-1" data-p2="0.5" data-q1="1.5" data-q2="-1" style="width:100%;max-width:860px;margin:1.5rem auto;"></div>
<script type="module">
import { mount } from "../widgets/bregman.js";
mount(document.getElementById("bregman-surface"));
</script>
```

Options: `data-p1`, `data-p2` (eta_p), `data-q1`, `data-q2` (eta_q), all in [-4, 4].

Page paragraph: The surface is the log-partition function
A(eta) = log(1 + e^eta_1 + e^eta_2) of a three-category categorical. The blue plane is
tangent to it at eta_p; the red segment at eta_q runs from the plane up to the surface
and its length is the Bregman divergence D_A(eta_q, eta_p). The readout computes the
same number as sum_k p_k log(p_k / q_k): the two agree to four decimals for every
position of the sliders, and the segment never points downward because a convex surface
lies above all its tangent planes. Press Play to let eta_q wander and watch the gap
grow as q moves away from p. The bars show the probability vectors p (blue) and q
(orange) that the two natural-parameter points represent.

## 4. `diffusion_paths.js` — forward and reverse SDE in (x, y, t)

```{=html}
<div class="bi-widget" id="diffusion-paths" data-points="400" data-steps="200" style="width:100%;max-width:860px;margin:1.5rem auto;"></div>
<script type="module">
import { mount } from "../widgets/diffusion_paths.js";
mount(document.getElementById("diffusion-paths"));
</script>
```

Options: `data-points` (number of paths, default 400), `data-steps` (reverse Euler–Maruyama
steps, 50–400, default 200), `data-ode` = `true` to start on the probability-flow ODE,
`data-seed`.

Page paragraph: Time runs upward. Forward: 400 points sampled from the three-mode
torsion mixture (coloured by mode) follow dx = -beta x / 2 dt + sqrt(beta) dW from
t = 0 at the bottom to t = 1 at the top, where the cloud is a standard Gaussian and the
colours are thoroughly mixed. Reverse: a fresh Gaussian cloud at the top follows the
reverse SDE driven by the exact score of the perturbed mixture and arrives at the three
modes; the readout shows t and the signal and noise factors sqrt(alpha_bar) and
sigma_t. Reduce the number of reverse steps to 50 and the small mode loses mass to
discretisation bias. Tick the probability-flow ODE box: the same marginals are reached
along deterministic paths, which is what makes the likelihood computation of Step 5
possible.

## 5. `kl_direction.js` — which way round?

```{=html}
<div class="bi-widget" id="kl-direction" data-mu="0" data-sigma="1" style="width:100%;max-width:860px;margin:1.5rem auto;"></div>
<script type="module">
import { mount } from "../widgets/kl_direction.js";
mount(document.getElementById("kl-direction"));
</script>
```

Options: `data-mu`, `data-sigma` (initial Gaussian).

Page paragraph: The black curve is a bimodal target p; the orange curve is a Gaussian q
you control by dragging (left–right moves the mean, up–down changes the width) or with
the sliders. The two readouts are KL(q || p), which variational inference minimises, and
KL(p || q), which moment matching and maximum likelihood minimise. The lower panels draw
the two integrands: q log(q / p) is large wherever q puts mass that p lacks, so the
KL(q || p) optimum sits inside one mode and is narrow; p log(p / q) is large wherever p
has mass that q lacks, so the KL(p || q) optimum straddles both modes and is wide. The
two buttons animate q to each optimum; compare the widths.

---

Deviations from the brief: `kl_direction.js` has Reset but no Play/Pause (its only
motion is the optimiser animation, which the two buttons start). The funnel widget
draws eta scaled by 8 in the non-centred view so the cylinder fills the same box as the
centred cloud; the axis label says so.
