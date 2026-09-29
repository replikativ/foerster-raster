(ns org.replikativ.spindel-raster.block
  "Raster-compiled log densities as spindel blocks (doc/contract.md).

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
  "A spindel block whose log density is the raster function `lp-var`. See
  the namespace for `binding`: {:args (fn [theta inputs]) :theta [slot …]}."
  [description lp-var {:keys [args theta]}]
  (let [value+grad (delay (rev/value+grad lp-var :wrt theta))]
    (block/block
     description
     {:log-density (fn [th inputs] (double (apply @lp-var (args th inputs))))
      :value+grad (fn [th inputs]
                    (let [[v & grads] (apply @value+grad (args th inputs))]
                      [(double v) (theta-gradient (vec grads) theta)]))})))
