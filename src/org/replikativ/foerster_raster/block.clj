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
  (double-array
   (mapcat (fn [slot]
             (let [g (nth grads slot)]
               (cond
                 (nil? g) (throw (ex-info "Raster returned no gradient for a θ slot"
                                          {:type ::no-gradient :slot slot}))
                 (number? g) [(double g)]
                 :else (seq ^doubles g))))
           slots)))

(defn raster-block
  "A foerster block whose log density is the raster function `lp-var`. See
  the namespace for `binding`: {:args (fn [theta inputs]) :theta [slot …]}."
  [description lp-var {:keys [args theta]}]
  (let [value+grad (delay (rev/value+grad lp-var :wrt theta))]
    (block/block
     description
     {:log-density (fn [th inputs] (double (apply @lp-var (args th inputs))))
      :value+grad (fn [th inputs]
                    (let [[v & grads] (apply @value+grad (args th inputs))]
                      [(double v) (theta-gradient (vec grads) theta)]))})))

(defmacro defdensity
  "Define a raster block from its log density, written as the body of a
  `deftm`: `latents` are the block's scalar latents in θ order (each a
  symbol, or [symbol support] with support `:positive` or [:interval a b]),
  `data` the other arguments with their raster types. Defines `name` as the
  block and `name-lp` as the compiled density; the gradient with respect to
  the latents comes from raster's reverse mode. The site's inputs are a map
  keyed by the data arguments' names:

    (defdensity logreg [b0 b1 b2]
      [xs :- (Array double), ys :- (Array double), cnt :- Long]
      (loop … acc))

    (sample (block/block-dist logreg {:xs xs :ys ys :cnt n}) :id :beta :init [0.0 0.0 0.0])

  With a constrained latent the body sees it in natural coordinates (σ, not
  log σ): the block declares `:block/coordinates :constrained` and foerster
  transforms."
  [name latents data & body]
  (let [specs (mapv #(if (vector? %) % [% :real]) latents)
        syms (mapv first specs)
        lp (symbol (str name "-lp"))
        data-args (vec (partition 3 data))
        th (gensym "theta")
        inputs (gensym "inputs")
        coerce (fn [[sym _ type]]
                 (let [k (keyword sym)]
                   (case type
                     Double `(double (get ~inputs ~k))
                     Long `(long (get ~inputs ~k))
                     `(get ~inputs ~k))))
        constrained? (some #(not= :real (second %)) specs)]
    `(do
       (raster.core/deftm ~lp ~(vec (concat (mapcat (fn [s] [s :- 'Double]) syms) data)) :- ~'Double
         ~@body)
       (def ~name
         (raster-block (cond-> {:block/id ~(keyword name)
                                :block/latents ~(mapv (fn [[s su]] {:name (keyword s) :shape [] :support su}) specs)
                                :block/target :complete-conditional}
                         ~(boolean constrained?) (assoc :block/coordinates :constrained))
                       (var ~lp)
                       {:args (fn [~(with-meta th {:tag 'doubles}) ~inputs]
                                (into ~(mapv (fn [i] `(aget ~th ~i)) (range (count syms)))
                                      [~@(map coerce data-args)]))
                        :theta ~(vec (range (count syms)))})))))
