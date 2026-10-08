# Facilitator guide

This guide is for the person running the room. It assumes you have worked through every
module yourself against `solutions/` and can run the full suite.

## Format

Participants work alone or in pairs through the module pages in order. There are no
lectures. Every module page carries figures generated from the reference code, several
carry short animations, and six carry interactive widgets (a sampler walking on a
density surface, the funnel in three dimensions, KL in both directions, the Bregman
tangent plane, the diffusion process in space and time). Point people at the widgets
when they are stuck on the concept rather than the code; the page says what to look
for. Each hard result is preceded by a "Predict before reading on" prompt with the
answer collapsed; ask pairs to commit to a prediction out loud before they open it.
Every Checkpoint ends by asking which expectation the module computed; this is the
thread that ties the days together, and it is worth asking the room at each boundary. The audience is engineers with no prior Bayesian background: every module's
Background section builds its concepts from first principles with a worked numerical
example, and Module 2b on Day 0 is the conceptual anchor the rest refers back to.
Insist that participants read Module 2b before Day 1; it is where the words prior,
posterior, evidence and expectation are defined and where the reason approximation is
needed is made concrete. The facilitator's job is to unblock, to ask the discussion prompts at module
boundaries, and to keep the group roughly together at the day boundaries. A day ends
when the day's checkpoints pass; participants who finish early do the Challenges.

Withholding `solutions/`: delete or move the directory before distributing. The tests
run against `workshop/` by default and do not import `solutions/`. If you keep it,
participants can run `pytest --solutions` to see the reference behaviour, which is useful
for debugging a failing Monte Carlo test and harmless for learning if you say so.

## Timing

Reference wall-clock times for the test suites against `solutions/` on a 2024-class
laptop CPU, measured during the build. Participant code is typically within 2x of these.

| Module | Tests | Suite time | Module time |
|---|---|---|---|
| m00 Setup | 4 | 1 s | 0.5 h |
| m01 JAX warm-up | 18 | 5 s | 1.5 h |
| m02 Diagnostic | 11 | 2 s | 1.0 h |
| m02b Bayesian inference from first principles | 14 | 7 s | 3.0 h |
| m03 Exponential-family core | 19 | 5 s | 3.0 h |
| m04 KL and Bregman geometry | 13 | 3 s | 2.5 h |
| m05 Conjugacy | 11 | 2 s | 3.0 h |
| m06 ELBO and CAVI | 14 | 10 s | 3.0 h |
| m07 Gradient estimators | 10 | 31 s | 3.0 h |
| m08 Black-box VI engine | 11 | 8 s | 2.5 h |
| m09 MH and Hamiltonian dynamics | 10 | 3 s | 2.5 h |
| m10 Adaptation | 10 | 5 s | 3.0 h |
| m11 Diagnostics and comparison | 12 | 11 s | 3.0 h |
| m12 Traces and effect handlers | 17 | 4 s | 3.0 h |
| m13 Inference through the PPL | 8 | 7 s | 2.5 h |
| m14 Amortised VI, a VAE | 10 | 10 s | 3.0 h |
| m15a Normalising flows | 17 | 21 s | 7.0 h |
| m15b Score-based diffusion | 13 | 22 s | 7.0 h |
| m15c Structural causal models | 16 | 8 s | 7.0 h |

Full suite against solutions: about 2 minutes. `make check-starter`: about 1 minute.

Module times include roughly one hour of reading each. Days 1 to 4 therefore run to
about 8.5 hours. For a strict seven-hour day: ask participants to read the Background
sections of the day's modules the evening before (about 1.5 hours), and treat each
module's last step as optional for anyone who is behind. Which steps are safe to skip:
m04 Step 4, m05 Step 5, m07 Step 5, m08 Step 5, m10 Step 5, m11 Step 4, m13 Step 5,
m14 Step 5. Nothing later depends on them.

## Where people get stuck

Listed by module, most common first. The module pages' Troubleshooting sections give
the fixes; this list tells you what to look for when someone has been silent for twenty
minutes.

**m01.** The all-`-inf` case of `logsumexp`. Consuming PRNG keys in a different order
than the test expects (the AR(1) test). `ConcretizationTypeError` from using `n_steps`
non-statically.

**m02.** Step 3 on a non-symmetric matrix (returning $A^{-1}$ rather than $A^{-\top}$).
Participants who fail two or more steps here should be told plainly to spend Day 1
morning on the review list and join at m04; the exponential-family module assumes the
material.

**m02b.** Exponentiating log posteriors before subtracting the maximum. Forgetting the
Jacobian of $\lambda = e^{z}$ and getting a posterior biased toward small precision;
the page's Jacobian section is the canonical one and ships a three-line checker
(`workshop/jacobian_check.py`) that every later page points back to, so send anyone with
a Jacobian bug on any day back to that section and that tool.
A grid that is too narrow, so posterior mass falls off the edge and the moments are
wrong. Conceptually: treating the posterior as a point (asking "what is the answer"
rather than "what is the distribution"), and reading the likelihood as a probability
over parameters. This module is where the word "expectation under the posterior"
has to land; ask each participant to state one question about the qPCR data as such an
expectation before they move on.

**m03.** The Gaussian natural parameterisation: forgetting the factor $-\tfrac12$ in
$\eta_2$, or symmetrising $\eta_2$ inside `log_partition` and breaking the mean-parameter
gradient. The Categorical minimal form (dropping the last one-hot entry).

**m04.** Undamped Newton leaving the natural-parameter domain. Using Python control flow
in the line search and losing differentiability. Step 4 (natural gradients) is marked
optional on the page and its derivation now lives inside the step rather than in the
Background; people who are behind should implement Steps 1 and 3, read Step 2, and
skip Step 4. The KL-direction widget settles most arguments about asymmetry faster than
the algebra does.

**m05.** `betaln` precision. The $\kappa_n$ placement in the Normal–Gamma $\beta_n$. The
Jacobian shift in mapping Beta$(a, b)$ to $(\chi, \nu) = (a, a + b)$ is the conceptual
sticking point; ask them to derive it on the board.

**m06.** A decreasing ELBO. Nearly always a stale factor used in the ELBO after an
update, or the missing $d\,s_k^2$ term in the expected squared distance. The $K$-selection
step is sensitive to seeding; the page specifies the deterministic seeding.

**m07.** Score-function estimator biased because `stop_gradient` is missing on the
samples. Participants often believe the control variate must reduce variance; the test
that it does is the learning moment when it does not.

**m08.** Adam written for descent while the ELBO is maximised. Missing $N/B$ scaling in
SVI, which converges to the prior. Bijector log-det evaluated at the wrong point in a
`Chain`.

**m09.** Asymmetric half-kicks in leapfrog (reversibility fails, volume preservation
passes). Not holding integration time fixed in the step-size scaling test.

**m10.** Dual averaging fed `nan` acceptance probabilities and collapsing the step size.
Off-by-one in the warm-up window ends. Dtype mismatches in `lax.scan` carries holding
Python floats.

**m11.** ESS exactly $N$ (truncating at lag 0) or absurdly large (Geyer loop not
stopping). Centred model shows no divergences because the threshold was applied to the
acceptance probability. The funnel plot with divergences is now on the page (and the
funnel-in-the-prior widget is on the Module 9 page); make sure everyone has looked at
both before they debug. The notebook regenerates the plot from their own sampler for
the Day 5 report.

**m12.** A handler left on the stack after an exception poisons every later test; insist
on `try/finally`. Key reuse across sites. Storing traced values in module globals.

**m13.** The unconstrained density off by exactly the log-Jacobian. Under-converged SVI
on the enzyme model (the ELBO plateau); the page uses it as the convergence lesson.

**m14.** Bernoulli with probabilities where logits are expected. Posterior collapse.
Decoder evaluated at a prior draw instead of the guide's $z$.

**m15a.** Unbounded coupling log-scale. Masks not alternating. Mode-seeking collapse at
the top of the funnel with too few samples per step.

**m15b.** Dividing by $\sigma_t$ at $t \to 0$. Wrong sign in the reverse drift. Reading
Hutchinson noise as a bug.

**m15c.** Confusing $\E[C \mid A]$ with $\E[C \mid do(A)]$. Conditioning a deterministic
node. Soft-observation scale too small (divergences) or too large (bias).

## Discussion prompts

Ask these at the module boundary, to the room or to a pair. Five minutes each.

- **After m02b.** Every question we asked of the qPCR posterior was an expectation.
  Name a question that is not, and argue whether it can be rewritten as one.
- **After m03.** The Gaussian representation with $\mathrm{vec}(xx^\top)$ is
  over-complete. What breaks downstream if a family is over-complete, and where in the
  rest of the workshop would it bite?
- **After m04.** Mean-field VI minimises $\KL(q\,\|\,p)$. Using the Bregman picture, which
  divergence would you have to minimise to get the moment-matching behaviour of
  expectation propagation, and why is it harder?
- **After m05.** The posterior update is addition of sufficient statistics. What
  property of the data does that make invisible to the posterior, and when does that
  matter in a streaming system?
- **After m06.** CAVI is coordinate ascent on a non-convex objective. Why does the
  single-cell dataset make this benign, and what would make it fail?
- **After m07.** The pathwise estimator uses $\nabla f$; the score-function estimator
  does not. Name a model where you cannot use the pathwise estimator, and what you would
  do instead.
- **After m08.** SVI rescales a minibatch likelihood by $N/B$. What does that assume
  about the data, and what breaks in a hierarchical model with local latents?
- **After m09.** Volume preservation makes the Metropolis correction simple. What would
  the correction look like for a non-symplectic integrator, and why does nobody do that?
- **After m10.** Dual averaging targets an acceptance rate of 0.8. Why that number, and
  how does the optimum change with dimension?
- **After m11.** HMC and mean-field VI agree on posterior means and disagree on
  variances on the churn model. For which decision would the disagreement matter, and
  for which would it not?
- **After m12.** The handler stack is a Python side effect inside traced code. Why does
  this work under `jit`, and what is the one thing it cannot do?
- **After m13.** The unconstraining transform adds a Jacobian term. If you forgot it,
  which quantities would still be correct and which would not?
- **After m14.** The VAE encoder is an amortised variational posterior. What is the
  amortisation gap, and how would you measure it with the tools from Module 8?
- **Capstone review.** Each participant presents the "What went wrong" section of their
  report in two minutes. This is the most valuable hour of Day 5.

## Day-boundary checklist

- Day 0 (before arrival): confirm by email that `pytest tests/m00 tests/m01 tests/m02
  tests/m02b` passes for every participant. Day 0 is now about six hours of pre-work;
  say so when sending the invitation. Anyone failing m02 gets the review list and an offer of
  a 30-minute call.
- End of Day 1: everyone has `pytest tests/m03 tests/m04 tests/m05` passing. The m04
  Challenges absorb fast participants.
- End of Day 2: m08 must pass for Day 3's m11 comparison and Day 4's m13. A participant
  who is behind on m07 can skip its Step 5 and use the notebook plot.
- End of Day 3: m10 and m11 must pass for Day 4's m13 and Day 5's Track A.
- End of Day 4: m12 and m13 must pass for Track C; m14 is self-contained.
- Day 5: assign tracks at the start of the day. Pairs are allowed; reports are
  individual.
