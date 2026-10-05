(ns org.replikativ.foerster-raster.block
  "Raster-compiled log densities as foerster blocks (doc/contract.md).

  A raster block is a `deftm` log density over typed arguments. The block
  says how θ and the site's inputs become those arguments, and which argument
  slots hold θ:

    (raster-block {:block/id :gauss
                   :block/latents [{:name :mu :shape [2] :support :real}]
                   :block/target :complete-conditional}
                  #'gauss-lp
                  {:args  (fn [^doubles theta {:keys [ys n s0 s]}]
                            [(aget theta 0) (aget theta 1) ys n s0 s])
                   :theta [0 1]})

  `:theta` lists, in θ order, the argument slots θ occupies: a scalar slot
  contributes one coordinate, an array slot as many as its gradient has.
  The block's `:log-density` calls the compiled function; its `:value+grad`
  calls raster's reverse mode (`raster.ad.reverse/value+grad`) with respect
  to the θ slots only, built once per block: the other arguments (the data)
  stay constant. Raster owns compilation and caching; this namespace only
  binds arguments."
  (:require [org.replikativ.foerster.block :as block]
            [raster.ad.reverse :as rev]))

(defn- theta-gradient
  "θ's gradient from raster's per-argument gradients: the declared slots in
  order, arrays spliced."
  ^doubles [grads slots]
  (let [part (fn [slot]
               (let [g (nth grads slot)]
                 (when (nil? g)
                   (throw (ex-info "Raster returned no gradient for a θ slot"
                                   {:type ::no-gradient :slot slot})))
                 g))
        size (reduce (fn [n slot]
                       (let [g (part slot)]
                         (+ n (if (number? g) 1 (alength ^doubles g)))))
                     0 slots)
        out (double-array size)]
    (reduce (fn [offset slot]
              (let [g (part slot)]
                (if (number? g)
                  (do (aset out (int offset) (double g)) (inc offset))
                  (let [n (alength ^doubles g)]
                    (System/arraycopy g 0 out (int offset) n)
                    (+ offset n)))))
            0 slots)
    out))

(defn raster-block
  "A foerster block whose log density is the raster function `lp-var`. See
  the namespace for `binding`: {:args (fn [theta inputs]) :theta [slot …]}."
  [description lp-var {:keys [args theta]}]
  (let [value+grad (delay (rev/value+grad lp-var :wrt theta :compile? true))]
    (block/block
     description
     {:log-density (fn [th inputs] (double (apply @lp-var (args th inputs))))
      :value+grad (fn [th inputs]
                    (let [[v & grads] (apply @value+grad (args th inputs))]
                      [(double v) (theta-gradient (vec grads) theta)]))})))

(defn- latent-spec
  "{:sym :shape :support} of a latent form: `b`, `[b support]`, `[b [k]]` or
  `[b [k] support]` — a scalar, or a vector of k coordinates (k any
  expression, evaluated when the block is defined)."
  [form]
  (if (vector? form)
    (let [[sym a b] form]
      (if (vector? a)
        {:sym sym :shape a :support (or b :real)}
        {:sym sym :shape [] :support (or a :real)}))
    {:sym form :shape [] :support :real}))

(defmacro defdensity
  "Define a raster block from its log density, written as the body of a
  `deftm`: `latents` are the block's latents in θ order — a scalar `b`, a
  constrained scalar `[b support]` (support `:positive` or [:interval a b]),
  a vector `[beta [k]]` or a constrained vector `[beta [k] support]` (the
  body sees it as an `(Array double)` of length k) — and `data` the other
  arguments with their raster types. Defines `name` as the block and
  `name-lp` as the compiled density; the gradient with respect to the
  latents comes from raster's reverse mode. The site's inputs are a map
  keyed by the data arguments' names:

    (defdensity logreg [b0 [b [3]]]
      [xs :- (Array double), ys :- (Array double), cnt :- Long]
      (loop … acc))

    (sample (block/block-dist logreg {:xs xs :ys ys :cnt n}) :id :beta :init [0.0 0.0 0.0 0.0])

  Inside a loop, read a vector latent at the loop's index (`(ra/aget beta
  i)`) or at an index computed from it over the data (`(ra/aget alpha
  (ra/aget group i))`, a varying intercept); bind an element read at a
  fixed index before the loop (`(let [b1 (ra/aget beta 0)] (loop …))`).

  The gradient is compiled and safe to call from parallel chains.

  θ is the latents flattened in order. With a constrained latent the body
  sees it in natural coordinates (σ, not log σ): the block declares
  `:block/coordinates :constrained` and foerster transforms."
  [name latents data & body]
  (let [specs (mapv latent-spec latents)
        sizes (mapv (fn [{:keys [shape]}] (if (empty? shape) 1 (first shape))) specs)
        lp (symbol (str name "-lp"))
        data-args (vec (partition 3 data))
        th (gensym "theta")
        inputs (gensym "inputs")
        offsets (gensym "offsets")
        coerce (fn [[sym _ type]]
                 (let [k (keyword sym)]
                   (case type
                     Double `(double (get ~inputs ~k))
                     Long `(long (get ~inputs ~k))
                     `(get ~inputs ~k))))
        constrained? (some #(not= :real (:support %)) specs)
        size-syms (mapv (fn [_] (gensym "k")) specs)]
    `(do
       (raster.core/deftm ~lp ~(vec (concat (mapcat (fn [{:keys [sym shape]}]
                                                      [sym :- (if (empty? shape) 'Double '(Array double))])
                                                    specs)
                                            data)) :- ~'Double
         ~@body)
       (def ~name
         (let [~@(mapcat (fn [k size] [k size]) size-syms sizes)
               ~offsets (vec (reductions + 0 ~size-syms))]
           (raster-block (cond-> {:block/id ~(keyword name)
                                  :block/latents ~(mapv (fn [{:keys [sym shape support]} k]
                                                          {:name (keyword sym)
                                                           :shape (if (empty? shape) [] [k])
                                                           :support support})
                                                        specs size-syms)
                                  :block/target :complete-conditional}
                           ~(boolean constrained?) (assoc :block/coordinates :constrained))
                         (var ~lp)
                         {:args (fn [~(with-meta th {:tag 'doubles}) ~inputs]
                                  (into ~(vec (map-indexed
                                               (fn [i {:keys [shape]}]
                                                 (if (empty? shape)
                                                   `(aget ~th (nth ~offsets ~i))
                                                   `(java.util.Arrays/copyOfRange ~th (int (nth ~offsets ~i))
                                                                                  (int (nth ~offsets ~(inc i))))))
                                               specs))
                                        [~@(map coerce data-args)]))
                          :theta ~(vec (range (count specs)))}))))))
