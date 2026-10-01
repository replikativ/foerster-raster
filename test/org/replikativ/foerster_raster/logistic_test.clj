(ns org.replikativ.foerster-raster.logistic-test
  "A non-conjugate block: Bayesian logistic regression, β ∈ R³ (an intercept
  and two features), β_j ~ N(0, 2²), y_i ~ Bernoulli(σ(β₀ + β₁x₁ + β₂x₂)).
  The raster-compiled log density and gradient against a pure-Clojure one and
  finite differences, and HMC's posterior means against grid quadrature (the
  exact posterior up to the grid)."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster-raster.block :as rb]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.effects.await :as aw]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [raster.arrays :as ra]
            [raster.core :refer [deftm]]
            [raster.math :as rm]
            [raster.numeric :as n]
            [raster.sci.distributions :as dist]))

;; The features interleaved in one array (x₁ x₂ per row); y as 0.0 / 1.0.
;; log p(y | η) = y·η − log(1 + e^η).
(deftm logreg-lp [b0 :- Double, b1 :- Double, b2 :- Double, xs :- (Array double),
                  ys :- (Array double), cnt :- Long, s0 :- Double] :- Double
  (loop [i 0 acc (n/+ (dist/logpdf (dist/->Normal 0.0 s0) b0)
                      (n/+ (dist/logpdf (dist/->Normal 0.0 s0) b1)
                           (dist/logpdf (dist/->Normal 0.0 s0) b2)))]
    (if (< i cnt)
      (let [eta (n/+ b0 (n/+ (n/* b1 (ra/aget xs (* 2 i))) (n/* b2 (ra/aget xs (inc (* 2 i))))))]
        (recur (inc i)
               (n/+ acc (n/- (n/* (ra/aget ys i) eta) (rm/log (n/+ 1.0 (rm/exp eta)))))))
      acc)))

(def ^:private description
  {:block/id :logreg
   :block/latents [{:name :beta :shape [3] :support :real}]
   :block/target :complete-conditional})

(def ^:private raster-logreg
  (rb/raster-block description #'logreg-lp
                   {:args (fn [^doubles th {:keys [xs ys cnt s0]}]
                            [(aget th 0) (aget th 1) (aget th 2) xs ys (long cnt) (double s0)])
                    :theta [0 1 2]}))

(def ^:private inputs
  ;; 40 rows from β = (−0.5, 1.2, −0.8)
  (let [rng (java.util.Random. 11)
        cnt 40
        xs (vec (repeatedly (* 2 cnt) #(.nextGaussian rng)))
        ys (vec (for [i (range cnt)
                      :let [eta (+ -0.5 (* 1.2 (xs (* 2 i))) (* -0.8 (xs (inc (* 2 i)))))]]
                  (if (< (.nextDouble rng) (/ 1.0 (+ 1.0 (Math/exp (- eta))))) 1.0 0.0)))]
    {:xs (double-array xs) :ys (double-array ys) :cnt cnt :s0 2.0}))

(defn- reference-lp
  "The log density in plain Clojure."
  ^double [^doubles th {:keys [^doubles xs ^doubles ys cnt s0]}]
  (let [prior (reduce + (map #(dist/logpdf (dist/->Normal 0.0 s0) %) th))]
    (+ prior
       (reduce + (for [i (range cnt)
                       :let [eta (+ (aget th 0) (* (aget th 1) (aget xs (* 2 i))) (* (aget th 2) (aget xs (inc (* 2 i)))))]]
                   (- (* (aget ys i) eta) (Math/log (+ 1.0 (Math/exp eta)))))))))

(deftest raster-logistic-block-matches-the-reference-and-finite-differences
  (let [lp (block/capability raster-logreg :log-density)
        vg (block/capability raster-logreg :value+grad)
        h 1e-6]
    (doseq [x [[0.0 0.0 0.0] [-0.5 1.2 -0.8] [1.5 -2.0 0.7]]]
      (let [th (double-array x)
            [v g] (vg th inputs)]
        (is (< (Math/abs (- v (reference-lp th inputs))) 1e-9))
        (is (< (Math/abs (- v (lp th inputs))) 1e-9))
        (doseq [j [0 1 2]]
          (let [fd (/ (- (reference-lp (double-array (update x j + h)) inputs)
                         (reference-lp (double-array (update x j - h)) inputs))
                      (* 2 h))]
            (is (< (Math/abs (- fd (aget ^doubles g j))) 1e-4) (str x " " j))))))))

(def ^:private quadrature-means
  ;; the posterior means on a 61³ grid over [−4, 4]³, where the posterior's
  ;; mass lies (its sd is about 0.4)
  (delay
    (let [k 61 lo -4.0 step (/ 8.0 (dec k))
          pts (mapv #(+ lo (* step %)) (range k))
          cells (for [a pts b pts c pts] (double-array [a b c]))
          lps (mapv #(reference-lp % inputs) cells)
          top (apply max lps)
          ws (mapv #(Math/exp (- % top)) lps)
          z (reduce + ws)]
      (mapv (fn [j] (/ (reduce + (map (fn [^doubles th w] (* w (aget th j))) cells ws)) z)) [0 1 2]))))

(deftest hmc-on-the-logistic-block-recovers-the-posterior
  (random/set-seed! 13)
  (let [root (ctx/create-execution-context)
        draws (try
                (binding [ec/*execution-context* root]
                  (let [meas (deref (future @(spin (aw/await
                                                    (infer/kernel-infer
                                                     (spin (sample (block/block-dist raster-logreg inputs) :id :beta :init [0.0 0.0 0.0]))
                                                     (k/hmc-kernel 2000 {:step-size 0.15 :steps 10 :samples :all :burn 200})
                                                     2 {}))))
                                    300000 ::timeout)]
                    (mapv (comp m/get-value first) (m/get-particles meas))))
                (finally (ctx/stop-context! root)))]
    (doseq [j [0 1 2]]
      (let [mu (/ (reduce + (map #(nth % j) draws)) (count draws))
            exact (nth @quadrature-means j)]
        (is (< (Math/abs (- mu exact)) 0.06) (str "β" j " " mu " vs " exact))))))
