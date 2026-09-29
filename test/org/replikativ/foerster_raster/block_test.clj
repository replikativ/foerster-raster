(ns org.replikativ.foerster-raster.block-test
  "A raster-compiled Gaussian block against a pure-Clojure reference block,
  finite differences, and the analytic posterior under foerster's HMC."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster-raster.block :as rb]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.effects.await :as aw]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.arrays :as ra]
            [raster.sci.distributions :as dist]
            [org.replikativ.foerster.random :as random]))

;; μ ∈ R², μ_j ~ N(0, s0²), y_ij ~ N(μ_j, s²); the observations interleaved
;; in one array, the prior seeding the loop's accumulator.
(deftm gauss-lp [m0 :- Double, m1 :- Double, ys :- (Array double),
                 cnt :- Long, s0 :- Double, s :- Double] :- Double
  (loop [i 0 acc (n/+ (dist/logpdf (dist/->Normal 0.0 s0) m0)
                      (dist/logpdf (dist/->Normal 0.0 s0) m1))]
    (if (< i cnt)
      (recur (inc i)
             (n/+ acc (n/+ (dist/logpdf (dist/->Normal m0 s) (ra/aget ys (* 2 i)))
                           (dist/logpdf (dist/->Normal m1 s) (ra/aget ys (inc (* 2 i)))))))
      acc)))

(def ^:private description
  {:block/id :gauss
   :block/latents [{:name :mu :shape [2] :support :real}]
   :block/target :complete-conditional})

(def ^:private raster-gauss
  (rb/raster-block description #'gauss-lp
                   {:args (fn [^doubles th {:keys [ys cnt s0 s]}]
                            [(aget th 0) (aget th 1) ys (long cnt) (double s0) (double s)])
                    :theta [0 1]}))

(def ^:private reference-gauss
  (let [lp+grad (fn [^doubles th {:keys [ys cnt s0 s]}]
                  (let [m (vec th)
                        pairs (partition 2 (vec ys))
                        lp (- (+ (/ (reduce + (map #(* % %) m)) (* 2 s0 s0))
                                 (reduce + (for [p pairs j [0 1]]
                                             (let [r (- (nth p j) (nth m j))] (/ (* r r) (* 2 s s)))))))
                        g (double-array (for [j [0 1]]
                                          (+ (- (/ (nth m j) (* s0 s0)))
                                             (/ (reduce + (map #(- (nth % j) (nth m j)) pairs)) (* s s)))))]
                    [lp g]))]
    (block/block description {:log-density (fn [th in] (first (lp+grad th in)))
                              :value+grad lp+grad})))

(def ^:private inputs
  (let [rng (java.util.Random. 5)
        n 20]
    {:ys (double-array (mapcat (fn [_] [(+ 1.0 (.nextGaussian rng)) (+ -2.0 (.nextGaussian rng))])
                               (range n)))
     :cnt n :s0 3.0 :s 1.0}))

(defn- close? [a b] (< (Math/abs (- a b)) (* 1e-9 (max 1.0 (Math/abs a)))))

(deftest raster-block-matches-the-reference
  (let [rng (java.util.Random. 1)]
    (dotimes [_ 20]
      (let [th (double-array [(* 3 (.nextGaussian rng)) (* 3 (.nextGaussian rng))])
            [v g] ((block/capability raster-gauss :value+grad) th inputs)
            [v' g'] ((block/capability reference-gauss :value+grad) th inputs)]
        ;; the raster density is normalized, the reference is not: the two
        ;; differ by a constant
        (is (close? (- v v') (- ((block/capability raster-gauss :log-density) (double-array [0.0 0.0]) inputs)
                                ((block/capability reference-gauss :log-density) (double-array [0.0 0.0]) inputs))))
        (is (close? v ((block/capability raster-gauss :log-density) th inputs)))
        (is (every? true? (map close? (vec g') (vec g))))))))

(deftest raster-gradient-matches-finite-differences
  (let [lp (block/capability raster-gauss :log-density)
        vg (block/capability raster-gauss :value+grad)
        h 1e-6]
    (doseq [x [[0.0 0.0] [1.3 -2.1] [-4.0 2.5]]]
      (let [[_ g] (vg (double-array x) inputs)]
        (doseq [j [0 1]]
          (let [fd (/ (- (lp (double-array (update x j + h)) inputs)
                         (lp (double-array (update x j - h)) inputs))
                      (* 2 h))]
            (is (< (Math/abs (- fd (aget ^doubles g j))) 1e-4))))))))

(defn- run-chain [b seed]
  (random/set-seed! seed)
  (let [root (ctx/create-execution-context)]
    (try
      (binding [ec/*execution-context* root]
        (let [meas (deref (future @(spin (aw/await
                                          (infer/kernel-infer
                                           (spin (sample (block/block-dist b inputs) :id :mu :init [0.0 0.0]))
                                           (k/hmc-kernel 1500 {:step-size 0.1 :steps 8 :samples :all :burn 100})
                                           2 {}))))
                          300000 ::timeout)]
          (mapv (comp m/get-value first) (m/get-particles meas))))
      (finally (ctx/stop-context! root)))))

(deftest hmc-on-a-raster-block-recovers-the-posterior
  (let [{:keys [ys cnt s0 s]} inputs
        prec (+ (/ 1.0 (* s0 s0)) (/ cnt (* s s)))
        pairs (partition 2 (vec ys))
        truth (for [j [0 1]] [(/ (/ (reduce + (map #(nth % j) pairs)) (* s s)) prec) (/ 1.0 (Math/sqrt prec))])
        draws (run-chain raster-gauss 3)]
    (doseq [[j [mean sd]] (map-indexed vector truth)]
      (let [xs (map #(nth % j) draws)
            mu (/ (reduce + xs) (count xs))
            sd' (Math/sqrt (/ (reduce + (map #(let [r (- % mu)] (* r r)) xs)) (count xs)))]
        (is (< (Math/abs (- mu mean)) 0.03) (str "mean " mu " vs " mean))
        (is (< (Math/abs (- sd' sd)) 0.03) (str "sd " sd' " vs " sd))))))
