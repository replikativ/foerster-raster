;; Radon varying intercepts as a raster defdensity (gradient by raster's
;; reverse mode), NUTS through foerster — the like-for-like counterpart to
;; PyMC's autodiff. Run from this checkout:
;;   clojure -J-Xmx3g -M -i bench/radon.clj
(ns bench.radon
  (:require [clojure.string :as str]
            [org.replikativ.foerster-raster.block :refer [defdensity]]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [raster.arrays :as ra]
            [raster.math :as rm]
            [raster.numeric :as n]))

(defdensity radon [mu-a [sigma-a :positive] beta [sigma-y :positive] [alpha [85]]]
  [county :- (Array long), floor :- (Array double), y :- (Array double), cnt :- Long, counties :- Long]
  (let [half-log-2pi 0.9189385332046727
        prior (n/+ (n/* -0.5 (n// (n/* mu-a mu-a) 100.0))
                   (n/+ (n/- 0.0 sigma-a)
                        (n/+ (n/* -0.5 (n// (n/* beta beta) 100.0)) (n/- 0.0 sigma-y))))
        groups (loop [c 0 acc 0.0]
                 (if (< c counties)
                   (let [z (n// (n/- (ra/aget alpha c) mu-a) sigma-a)]
                     (recur (inc c) (n/- acc (n/+ (n/* 0.5 (n/* z z)) (n/+ (rm/log sigma-a) half-log-2pi)))))
                   acc))]
    (loop [i 0 acc (n/+ prior groups)]
      (if (< i cnt)
        (let [r (n// (n/- (ra/aget y i) (n/+ (ra/aget alpha (ra/aget county i)) (n/* beta (ra/aget floor i)))) sigma-y)]
          (recur (inc i) (n/- acc (n/+ (n/* 0.5 (n/* r r)) (n/+ (rm/log sigma-y) half-log-2pi)))))
        acc))))

(def data
  (let [[_ & lines] (str/split-lines (slurp "../foerster-compare/experiments/comparison/data/radon.csv"))
        rows (mapv #(let [[c f y] (str/split % #",")] [(Long/parseLong c) (Double/parseDouble f) (Double/parseDouble y)]) lines)]
    {:county (long-array (map first rows)) :floor (double-array (map second rows))
     :y (double-array (map #(nth % 2) rows)) :cnt (count rows) :counties 85}))

;; gradient check against central differences
(let [th (double-array (map #(* 0.1 (Math/sin %)) (range 89)))
      [v g] ((block/capability radon :value+grad) th data)
      ld (block/capability radon :log-density)
      fd (fn [j] (let [h 1e-6 u (aclone th) dn (aclone th)]
                   (aset u j (+ (aget th j) h)) (aset dn j (- (aget th j) h))
                   (/ (- (ld u data) (ld dn data)) (* 2 h))))]
  (println :grad-check-max-err (apply max (map #(Math/abs (- (aget ^doubles g %) (fd %))) [0 1 2 3 4 50 88]))))

(def world (sp/create-execution-context))
(defn run* [seed] (random/set-seed! seed)
  (let [t0 (System/nanoTime)
        r (sp/with-context world @(infer/infer (spin (sample (block/block-dist radon data) :id :theta :init (vec (repeat 89 0.0))))
                                               {:method :nuts :iterations 2000 :burn 1000 :chains 4}))]
    [r (/ (- (System/nanoTime) t0) 1e9)]))
(let [[_ t1] (run* 3) [r t2] (run* 3)
      nat (fn [i] #(nth (block/constrain radon %) i))]
  (println :first_s t1 :warm_s t2)
  (doseq [[k i] [[:beta 2] [:sigma-a 1] [:sigma-y 3]]]
    (println k (select-keys (d/summary r (nat i)) [:mean :sd :ess-bulk :rhat]))))
(shutdown-agents) (System/exit 0)
