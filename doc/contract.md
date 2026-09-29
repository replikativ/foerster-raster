# The foerster ↔ raster block contract

Status: draft 0, agreed in outline between the inference and raster sides
(2026-09-28; the inference side was spindel's, and is now
[foerster](https://github.com/replikativ/foerster), on spindel's worlds).
Changes go through this file.

## Division of labour

**foerster** owns everything about *which* variables exist and how they are
explored: addresses and traces, the random streams, proposals, acceptance,
particles and forked worlds, and the statistical meaning of every estimate.

**Raster** owns fixed-shape numerical code: evaluating a block, its log
density and derivatives, compilation, caches, buffers and device sessions.

**This bridge** adapts one to the other. It must not add a compiler cache, an
argument binder or a device-session convention of its own. It consumes
raster's public compilation/artifact API; until that exists it calls plain
functions.

## A block

A *block* is a fixed-shape numerical fragment of a model: a group of latent
variables together with the density factors they touch. foerster treats a
block as one choice site whose value is the block's latents.

A block is a *description* plus *capabilities*. The description is data:

```clojure
{:block/id        :logreg
 :block/latents   [{:name :beta  :shape [3] :dtype :f64
                    :support :real}                 ; or :positive, [:interval a b], :simplex …
                   {:name :sigma :shape []  :dtype :f64
                    :support :positive :transform :log}]
 :block/layout    {:order [:beta :sigma]            ; θ = concat, row-major
                   :coordinates :unconstrained}     ; θ lives in R^n
 :block/inputs    [{:name :X :shape [:n 3]} {:name :y :shape [:n]}]  ; not differentiated
 :block/target    :complete-conditional             ; see below
 :block/capabilities #{:log-density :value+grad}}
```

Capabilities are functions over primitive arrays. Required:

| capability | signature | meaning |
|---|---|---|
| `:log-density` | `(f ^doubles θ inputs) → double` | log target at θ, unconstrained coordinates, change-of-variable terms included |
| `:value+grad` | `(f ^doubles θ inputs) → [double ^doubles g]` | the same value and ∂/∂θ; inputs get no gradient |

Optional, each with an explicit domain and convention in the description:
`:jvp`, `:vjp`, `:hvp`, `:constrain` (θ → latent values, for the trace),
`:unconstrain`, `:inverse`, `:log-abs-det-jacobian`, `:eval` (forward
simulation), `:batch` (a leading particle dimension).

A capability a block does not declare is absent: foerster never falls back to
finite differences behind a caller's back. An undeclared derivative rejects
the program.

### The target

`:complete-conditional` means the log density includes **every factor the
block's latents affect**: their priors *and* all downstream observations and
latents that depend on them, with every variable outside the block held
fixed as an input. A site's own prior alone is not enough for HMC.

foerster keeps HMC exact even when a block's target is incomplete. The
leapfrog map is volume preserving and reversible for any position-only force
field, and foerster computes the acceptance ratio on the **full trace log
joint**, by replay. An incomplete target therefore costs efficiency, never
correctness. foerster reports it: when a replay changes the log probability of
a site outside the block, the step records a diagnostic.

### Parameter layout

`θ` is a flat `double[]` in the declared order, in unconstrained
coordinates. Every constrained latent names its transform. The block's log
density includes the log-absolute-Jacobian of each transform. The trace
records both θ and the constrained values: sites downstream see the
constrained values.

## Randomness

Every draw a seeded run makes depends only on what is drawn, never on when:

    stream(world, key) = generator seeded by hash(seed(world), key)

- **Run seed**: a session's root seed, drawn in program order from the seeded
  process generator.
- **Stable site identity**: the site's structural address. World seeds derive
  per fork from `(parent seed, address, fork index)`.
- **Draw counter**: the position within that stream.
- **Replay versus fresh proposal**: a kept or constrained site draws
  nothing. A proposal draws in the proposal's world, which has a fresh seed.

Raster randomness inside a block (e.g. `par/splitmix64`) takes its seed from
the stream of the site that runs the block. Bit-identical floating point
across devices is **not** part of the contract; reproducibility is per device
and backend.

## Residuals and lifetime

Draft 0 is synchronous: capabilities are pure functions and retain nothing
between calls. When residuals arrive (a tape for `vjp`, a device buffer), they
are owned by the call that made them and released when the world that holds
them (a spindel world) is released. That world is a savepoint anchor, a trace or a
particle. Asynchronous completion goes through one shared completion service
that resumes the waiting spin on its executor.

## Verification

Fast and deterministic, in every test run:

- The Gaussian block against its analytic posterior, from a pure-Clojure
  reference block and from raster.
- Finite differences against `:value+grad` at random points, with a stated
  tolerance.
- For derivative capabilities, the adjoint identity ⟨Jv, w⟩ = ⟨v, Jᵀw⟩.
- Logistic regression (non-conjugate) against a long reference run.

Statistical, as a separate seeded experiment with explicit thresholds:

- Simulation-based calibration. Rank statistics are tested against
  uniformity at a stated significance level and sample size. Finite samples
  never produce perfectly uniform ranks.

## Milestones

1. Pure-Clojure reference block + HMC-within-Gibbs (`foerster.block`,
   `foerster.hmc`); the Gaussian oracle. **Done.** The logistic-regression
   oracle is next.
2. Raster block from `deftm` log densities (`foerster-raster.block`), with
   gradients with respect to θ only. **Done.**
3. Batched particles (`:batch`) through raster dimensions; worlds fork only
   where structure diverges.
4. Residuals, asynchronous completion, device execution.
5. Implicit solves; estimator-based inference (ADEV, differentiable SMC)
   against enumerable and analytic oracles.
