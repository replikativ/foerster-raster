# foerster-raster

[![CircleCI](https://circleci.com/gh/replikativ/foerster-raster.svg?style=shield)](https://circleci.com/gh/replikativ/foerster-raster)
[![Clojars Project](https://img.shields.io/clojars/v/org.replikativ/foerster-raster.svg)](https://clojars.org/org.replikativ/foerster-raster)

Compiled numerical blocks for [foerster](https://github.com/replikativ/foerster)
inference: log densities and their gradients from
[raster](https://github.com/replikativ/raster) — compiled to JVM bytecode,
and by raster's backends to WASM and GPU kernels — inside probabilistic
programs whose addresses, traces, proposals and acceptance foerster owns.

## Where it sits

```text
spindel          worlds, forks, savepoints, world scopes, settlement
   ↑
foerster         inference: sites, traces, SMC / particle MCMC / MCMC / BBVI,
                 HMC-within-Gibbs, the block contract (foerster.block),
                 portable distributions (foerster.dist)
   ↑
foerster-raster  block capabilities from raster: a deftm log density and its
                 gradient by raster's reverse-mode AD
```

foerster defines what a block is and runs HMC on it without knowing where
its numbers come from; this library supplies them from raster. It is a
separate library because raster is a JVM-only compiler with a large
dependency tree, while foerster is small and runs on the JVM and in
JavaScript — and so that the block contract stays open to other backends
(hand-written Clojure, other AD systems).

## Blocks

A *block* is a group of latent variables sampled at one site, instead of one
site per variable. HMC moves all of them jointly along the gradient, which
mixes where single-site Metropolis–Hastings does not (many correlated
coordinates), and the density can be compiled numerical code.

A block is a **description** (data: the latents' names, shapes and support,
and what its density covers) plus **capabilities** (functions over a flat
`double[]` θ): `:log-density` and `:value+grad`, and optionally `:sample`.
In a model it is an ordinary choice site:

```clojure
(sample (block/block-dist b inputs) :id :mu :init [0.0 0.0])
```

The density should be the *complete conditional* — the latents' priors and
every observation that depends on them. foerster keeps HMC exact even when
it is not (it accepts on the full trace's log joint), so a missing factor
costs mixing, not correctness. The full contract is
[doc/contract.md](doc/contract.md).

## A raster block

A raster log density is a `deftm` over typed arguments; `raster-block` says
how θ and the site's inputs become those arguments, and which argument slots
hold θ. Gradients are taken with respect to those slots only (raster's
`:wrt`), so the data stays constant.

```clojure
(require '[org.replikativ.foerster-raster.block :as rb]
         '[org.replikativ.foerster.block :as block]
         '[org.replikativ.foerster.core :as infer]
         '[org.replikativ.foerster.kernel :as k]
         '[org.replikativ.foerster.effects :refer [sample]]
         '[org.replikativ.spindel.spin.cps :refer [spin]]
         '[raster.core :refer [deftm]]
         '[raster.numeric :as n]
         '[raster.arrays :as ra]
         '[raster.sci.distributions :as dist])

;; μ ∈ R², μ_j ~ N(0, s0²), y_ij ~ N(μ_j, s²), observations interleaved
(deftm gauss-lp [m0 :- Double, m1 :- Double, ys :- (Array double),
                 cnt :- Long, s0 :- Double, s :- Double] :- Double
  (loop [i 0 acc (n/+ (dist/logpdf (dist/->Normal 0.0 s0) m0)
                      (dist/logpdf (dist/->Normal 0.0 s0) m1))]
    (if (< i cnt)
      (recur (inc i)
             (n/+ acc (n/+ (dist/logpdf (dist/->Normal m0 s) (ra/aget ys (* 2 i)))
                           (dist/logpdf (dist/->Normal m1 s) (ra/aget ys (inc (* 2 i)))))))
      acc)))

(def gauss
  (rb/raster-block {:block/id :gauss
                    :block/latents [{:name :mu :shape [2] :support :real}]
                    :block/target :complete-conditional}
                   #'gauss-lp
                   {:args (fn [^doubles th {:keys [ys cnt s0 s]}]
                            [(aget th 0) (aget th 1) ys (long cnt) (double s0) (double s)])
                    :theta [0 1]}))

;; the site's inputs: the data and the fixed scales
(def inputs {:ys (double-array ys) :cnt (quot (count ys) 2) :s0 3.0 :s 1.0})

;; HMC-within-Gibbs, two chains
(infer/kernel-infer (spin (sample (block/block-dist gauss inputs) :id :mu :init [0.0 0.0]))
                    (k/hmc-kernel 1500 {:step-size 0.1 :steps 8 :samples :all :burn 100})
                    2)
```

The tests check this block against a pure-Clojure reference block, finite
differences, and the analytic posterior under HMC.

### Shapes raster differentiates

What the block tests and raster's AD regression tests cover, as of raster
0.2.1141:

- loops (`loop`/`recur`) and `par/reduce` over observations, with
  constructed densities (`(dist/logpdf (dist/->Normal mu s) y)`) in the body;
- a density seeding the accumulator, inline or `let`-bound;
- indexed and strided reads of data arrays (`ra/aget`, `aget`), which stay
  constant under `:wrt`.

For a shape it cannot differentiate, raster declines with an error (e.g.
"No AD template for …") rather than return a gradient; the tests here also
compare every block's gradient with finite differences. Report such a shape
to raster with a minimal `deftm`.

## Running

```clojure
org.replikativ/foerster-raster {:mvn/version "LATEST"}  ; see the Clojars badge
```

raster's compiler needs JVM options, collected in the `:jvm` alias:

```bash
clojure -M:jvm:test        # the tests
clojure -M:jvm:repl        # a REPL
```

JDK 21 or later (`--add-modules=jdk.incubator.vector`,
`--enable-native-access=ALL-UNNAMED`).

## Roadmap

1. Simulation-based calibration of the blocks (the logistic-regression
   block is checked against grid quadrature; `test/…/logistic_test.clj`).
2. Batching: many rows reduced in parallel, and particles as a leading
   dimension (`:batch`), on CPU and GPU.
3. A conformance test that `foerster.dist` and raster's distributions compute
   the same densities, and then lowering a model's `foerster.dist` sites into
   a compiled block without rewriting the model.
4. Residuals, asynchronous completion and device execution; estimator-based
   inference (ADEV, differentiable SMC).

## License

Copyright © 2026 Christian Weilbach. Apache License 2.0.
