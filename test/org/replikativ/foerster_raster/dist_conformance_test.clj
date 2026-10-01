(ns org.replikativ.foerster-raster.dist-conformance-test
  "foerster.dist and raster's distributions compute the same densities, so a
  model's foerster.dist sites and a raster block over the same factors agree:
  each pair at random parameters and points, in and outside the support.
  Both parameterize as written here (Gamma by shape and scale, the negative
  binomial by failures before the r-th success)."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.dist :as fd]
            [raster.sci.distributions :as rd]))

(defn- close? [a b]
  (or (and (Double/isInfinite a) (= a b))
      (< (Math/abs (- a b)) (* 1e-9 (max 1.0 (Math/abs a))))))

(def ^:private rng (java.util.Random. 17))
(defn- u [lo hi] (+ lo (* (- hi lo) (.nextDouble ^java.util.Random rng))))

(def ^:private continuous
  ;; [name (fn [] [foerster-dist raster-dist points])]
  [["normal" #(let [m (u -3 3) s (u 0.2 3)]
                [(fd/normal m s) (rd/->Normal m s) (repeatedly 5 (fn [] (u -8 8)))])]
   ["uniform" #(let [a (u -3 0) b (u 0.5 3)]
                 [(fd/uniform a b) (rd/->Uniform a b) [(u a b) (u a b) (- a 1.0) (+ b 1.0)]])]
   ["exponential" #(let [l (u 0.2 4)]
                     [(fd/exponential l) (rd/->Exponential l) [(u 0 5) (u 0 1) -1.0]])]
   ["gamma" #(let [a (u 0.5 6) b (u 0.2 3)]
               [(fd/gamma a b) (rd/->Gamma a b) [(u 0.01 10) (u 0.01 2) -0.5]])]
   ["beta" #(let [a (u 0.5 5) b (u 0.5 5)]
              [(fd/beta a b) (rd/->Beta a b) [(u 0.01 0.99) (u 0.01 0.99)]])]
   ["student-t" #(let [nu (u 1 10)]
                   [(fd/student-t nu) (rd/->StudentT nu) (repeatedly 4 (fn [] (u -6 6)))])]
   ["chi-squared" #(let [k (u 1 8)]
                     [(fd/chi-squared k) (rd/->Chisq k) [(u 0.01 12) (u 0.01 2)]])]])

(def ^:private discrete
  [["poisson" #(let [l (u 0.3 8)]
                 [(fd/poisson l) (rd/->Poisson l) [0 1 2 5 11]])]
   ["bernoulli" #(let [p (u 0.05 0.95)]
                   [(fd/bernoulli p) (rd/->Bernoulli p) [0 1]])]
   ["negative-binomial" #(let [r (u 0.5 6) p (u 0.1 0.9)]
                           [(fd/negative-binomial r p) (rd/->NegativeBinomial r p) [0 1 3 8]])]])

(deftest densities-agree
  (doseq [[label make] continuous
          _ (range 10)]
    (let [[f r xs] (make)]
      (testing label
        (doseq [x xs]
          (is (close? (fd/logpdf f x) (rd/logpdf r (double x)))
              (str label " at " x ": " (fd/logpdf f x) " vs " (rd/logpdf r (double x))))))))
  (doseq [[label make] discrete
          _ (range 10)]
    (let [[f r ks] (make)]
      (testing label
        (doseq [k ks]
          (is (close? (fd/logpdf f k) (rd/logpmf r (long k)))
              (str label " at " k ": " (fd/logpdf f k) " vs " (rd/logpmf r (long k)))))))))
