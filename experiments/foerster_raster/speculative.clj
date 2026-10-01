(ns foerster-raster.speculative
  "Speculative SMC: draws from a large language model p with tokens proposed
  by a small one q (same tokenizer). Each particle extends its text by a chunk
  of k tokens sampled from q, which the large model only scores — it reads
  the tokens, it does not choose them — and the chunk is weighted by
  Σ log p(xₜ | x<ₜ) − log q(xₜ | x<ₜ) (`steer/model`, one barrier per chunk).
  The target is p itself: q's law cancels, so SMC is exact for any q, and
  the evidence of a fixed-length continuation is 1 (E[Ẑ] = 1).

  Both models keep their KV caches as immutable snapshots
  (`pretrained.continuation/export-cpu`) in the particle's state, so a
  resampled copy shares its parent's prefix and restores its own arrays.

    clojure -M:jvm:lm -m foerster-raster.speculative [runs]

  compares, on one prompt, the mean per-token log p of continuations drawn
  from p directly, from q alone, and by speculative SMC, and reports how far
  the weights degenerate (the mean log Ẑ)."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.steer :as steer]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [pretrained.continuation :as cont]
            [pretrained.lm :as lm]))

(defn- log-normalizer
  "log Σ exp(logits)."
  ^double [^floats lg]
  (let [n (alength lg)
        top (loop [i 0 top Double/NEGATIVE_INFINITY]
              (if (< i n) (recur (inc i) (max top (double (aget lg i)))) top))]
    (loop [i 0 s 0.0]
      (if (< i n)
        (recur (inc i) (+ s (Math/exp (- (double (aget lg i)) top))))
        (+ top (Math/log s))))))

(defn- draw
  "A token from softmax(logits) by inverse CDF at `u`, and its log probability."
  [^floats lg ^double u]
  (let [z (log-normalizer lg)
        n (alength lg)]
    (loop [i 0 acc 0.0]
      (let [lp (- (double (aget lg i)) z)
            acc (+ acc (Math/exp lp))]
        (if (or (>= acc u) (= i (dec n)))
          [i lp]
          (recur (inc i) acc))))))

(defn- step!
  "Advance continuation `c` by one token: `choose` gets the logits and returns
  [token log-prob]. Returns [c' token log-prob]."
  [c choose]
  (let [out (volatile! nil)
        [c' _] (cont/step-cpu c (fn [lg _vocab] (let [[t lp] (choose lg)] (vreset! out [t lp]) t)))]
    (into [c'] @out)))

(defn- start
  "A particle's state for `prompt-ids`: both models' snapshots, nothing
  generated."
  [models prompt-ids max-position]
  (into {:tokens [] :log-p 0.0 :log-w 0.0}
        (for [[k model] models]
          [k (cont/export-cpu (cont/start-cpu model prompt-ids {:max-position max-position}))])))

(defn- chunk
  "`state` extended by `k` tokens drawn from `:q` and scored by `:p`, or drawn
  from `:p` alone when `proposal` is :p."
  [models state k max-position proposal]
  (let [restore (fn [key] (cont/restore-cpu (models key) (state key) {:max-position max-position}))
        direct? (= :p proposal)]
    ;; drawing from p directly, q is never consulted
    (loop [i 0 q (when-not direct? (restore :q)) p (restore :p)
           tokens (:tokens state) log-p (:log-p state) log-w (:log-w state)]
      (if (= i k)
        (cond-> (assoc state :p (cont/export-cpu p) :tokens tokens :log-p log-p :log-w log-w)
          q (assoc :q (cont/export-cpu q)))
        (let [u (random/uniform01)]
          (if direct?
            (let [[p' t lp] (step! p #(draw % u))]
              (recur (inc i) nil p' (conj tokens t) (+ log-p lp) log-w))
            (let [[q' t lq] (step! q #(draw % u))
                  [p' _ lp] (step! p (fn [lg] [t (- (double (aget ^floats lg (int t))) (log-normalizer lg))]))]
              (recur (inc i) q' p' (conj tokens t) (+ log-p lp) (+ log-w (- lp lq))))))))))

(defn speculative-model
  "The `steer` program drawing `n-tokens` after `prompt-ids` in chunks of
  `k`: from q weighted to p (`proposal` :q), or from p directly (:p)."
  [models prompt-ids {:keys [n-tokens k proposal] :or {proposal :q}}]
  (let [max-position (+ (count prompt-ids) n-tokens 1)]
    (steer/model {:init (start models prompt-ids max-position)
                  :step #(chunk models % k max-position proposal)
                  :done? #(>= (count (:tokens %)) n-tokens)
                  :value :log-w
                  :reward :log-w
                  ;; the trace keeps the text, not the caches
                  :record #(select-keys % [:tokens :log-p :log-w])
                  :max-steps (inc (quot n-tokens k))})))

(defn run-smc
  "Speculative SMC with `n` particles; resolves the measure."
  [models prompt-ids n opts]
  (let [ctx (context/create-execution-context)]
    (try
      (binding [ec/*execution-context* ctx]
        @(spin (await (infer/smc-infer (speculative-model models prompt-ids opts) n
                                       {:resampling :stratified :resample-threshold 0.5}))))
      (finally (context/stop-context! ctx)))))

(defn- per-token [state] (/ (:log-p state) (count (:tokens state))))

(defn -main [& [runs]]
  (let [runs (Long/parseLong (or runs "10"))
        models {:q (lm/load-lm :smollm2-135m-instruct) :p (lm/load-lm :smollm2-360m-instruct)}
        {:keys [tok encode decode]} (:tokenizer (:p models))
        prompt (vec (encode tok "Once upon a time, in a small village by the sea,"))
        opts {:n-tokens 24 :k 2}
        n 32
        mean (fn [xs] (/ (reduce + xs) (count xs)))
        timed (fn [f] (let [t0 (System/nanoTime) v (f)] [v (/ (- (System/nanoTime) t0) 1e9)]))
        ;; from p directly and from q alone: SMC runs whose weights are 0 or ignored
        ;; each run reduced at once: a measure's particles hold KV caches
        summary (fn [ms]
                  (let [ps (m/get-particles ms)
                        ws (m/normalize-log-weights (mapv second ps))
                        values (mapv (comp per-token m/get-value first) ps)]
                    {:values values
                     :estimate (reduce + (map * ws values))
                     :log-z (m/log-marginal ms)
                     :text (decode tok (:tokens (m/get-value (ffirst ps))))}))
        [direct t-direct] (timed #(vec (for [_ (range runs)]
                                         (:values (summary (run-smc models prompt n (assoc opts :proposal :p)))))))
        [smcs t-smc] (timed #(vec (repeatedly runs (fn [] (summary (run-smc models prompt n opts))))))
        q-only (mapv :values smcs)
        smc-est (mapv :estimate smcs)]
    (println "per-token log p of 24-token continuations," runs "runs ×" n "particles")
    (println (format "  from p directly        %.3f   (%.0f s)" (mean (map mean direct)) t-direct))
    (println (format "  speculative SMC        %.3f   (%.0f s)" (mean smc-est) t-smc))
    (println (format "  q's draws, unweighted  %.3f" (mean (map mean q-only))))
    (println (format "  run-to-run sd: direct %.3f, SMC %.3f"
                     (let [xs (map mean direct) mu (mean xs)] (Math/sqrt (mean (map #(Math/pow (- % mu) 2) xs))))
                     (let [mu (mean smc-est)] (Math/sqrt (mean (map #(Math/pow (- % mu) 2) smc-est))))))
    ;; E[Ẑ] = 1, but Ẑ is skewed: its log sits below 0 by about half its
    ;; variance, so the mean log Ẑ measures how far the weights degenerate
    (println (format "  mean log Ẑ %.2f (0 when q = p)" (mean (map :log-z smcs))))
    (println "  a draw:" (pr-str (:text (first smcs))))
    (shutdown-agents)
    (System/exit 0)))
