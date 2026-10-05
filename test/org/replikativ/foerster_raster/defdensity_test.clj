(ns org.replikativ.foerster-raster.defdensity-test
  "defdensity: the logistic-regression block of logistic-test without the
  argument wiring — the same density and gradient."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster-raster.block :refer [defdensity]]
            [org.replikativ.foerster.block :as block]
            [raster.arrays :as ra]
            [raster.math :as rm]
            [raster.numeric :as n]
            [raster.sci.distributions :as dist]))

(defdensity logreg [b0 b1 b2]
  [xs :- (Array double), ys :- (Array double), cnt :- Long, s0 :- Double]
  (loop [i 0 acc (n/+ (dist/logpdf (dist/->Normal 0.0 s0) b0)
                      (n/+ (dist/logpdf (dist/->Normal 0.0 s0) b1)
                           (dist/logpdf (dist/->Normal 0.0 s0) b2)))]
    (if (< i cnt)
      (let [eta (n/+ b0 (n/+ (n/* b1 (ra/aget xs (* 2 i))) (n/* b2 (ra/aget xs (inc (* 2 i))))))]
        (recur (inc i) (n/+ acc (n/- (n/* (ra/aget ys i) eta) (rm/log (n/+ 1.0 (rm/exp eta)))))))
      acc)))

(def ^:private inputs
  (let [rng (java.util.Random. 11) cnt 40
        xs (vec (repeatedly (* 2 cnt) #(.nextGaussian rng)))
        ys (vec (for [i (range cnt)] (if (< (.nextDouble rng) 0.5) 1.0 0.0)))]
    {:xs (double-array xs) :ys (double-array ys) :cnt cnt :s0 2.0}))

(defn- reference-lp ^double [^doubles th {:keys [^doubles xs ^doubles ys cnt s0]}]
  (+ (reduce + (map #(dist/logpdf (dist/->Normal 0.0 s0) %) th))
     (reduce + (for [i (range cnt)
                     :let [eta (+ (aget th 0) (* (aget th 1) (aget xs (* 2 i))) (* (aget th 2) (aget xs (inc (* 2 i)))))]]
                 (- (* (aget ys i) eta) (Math/log (+ 1.0 (Math/exp eta))))))))

(deftest a-density-becomes-a-block
  (is (= 3 (block/dimension logreg)))
  (doseq [x [[0.0 0.0 0.0] [0.4 -1.1 0.7]]]
    (let [th (double-array x)
          [v g] ((block/capability logreg :value+grad) th inputs)]
      (is (< (Math/abs (- v (reference-lp th inputs))) 1e-9))
      (doseq [j [0 1 2]]
        (let [h 1e-6
              fd (/ (- (reference-lp (double-array (update x j + h)) inputs)
                       (reference-lp (double-array (update x j - h)) inputs))
                    (* 2 h))]
          (is (< (Math/abs (- fd (aget ^doubles g j))) 1e-4)))))))

;; mu ~ N(0, 10), sigma ~ N⁺(0, 2) (half-normal, written as a normal on σ > 0
;; up to a constant), y ~ N(mu, sigma): sigma is :positive, so foerster
;; samples log σ and the body sees σ
(defdensity mean-scale [mu [sigma :positive]]
  [ys :- (Array double), cnt :- Long]
  (loop [i 0 acc (n/+ (dist/logpdf (dist/->Normal 0.0 10.0) mu) (dist/logpdf (dist/->Normal 0.0 2.0) sigma))]
    (if (< i cnt)
      (recur (inc i) (n/+ acc (dist/logpdf (dist/->Normal mu sigma) (ra/aget ys i))))
      acc)))

(deftest a-constrained-latent-in-natural-coordinates
  (let [ys (double-array [2.1 3.4 1.7 2.9 2.5 3.8 2.2 3.1])
        inputs {:ys ys :cnt 8}
        ;; θ = [mu, log σ]: the block's density is the body at σ = e^θ₁ plus θ₁
        lp (fn [mu s] (+ (dist/logpdf (dist/->Normal 0.0 10.0) mu) (dist/logpdf (dist/->Normal 0.0 2.0) s)
                         (reduce + (map #(dist/logpdf (dist/->Normal mu s) %) ys))))
        th (double-array [2.4 (Math/log 0.9)])
        [v g] ((block/capability mean-scale :value+grad) th inputs)
        h 1e-6
        f (fn [a b] (+ (lp a (Math/exp b)) b))]
    (is (< (Math/abs (- v (f 2.4 (Math/log 0.9)))) 1e-9))
    (is (< (Math/abs (- (aget ^doubles g 0) (/ (- (f (+ 2.4 h) (Math/log 0.9)) (f (- 2.4 h) (Math/log 0.9))) (* 2 h)))) 1e-4))
    (is (< (Math/abs (- (aget ^doubles g 1) (/ (- (f 2.4 (+ (Math/log 0.9) h)) (f 2.4 (- (Math/log 0.9) h))) (* 2 h)))) 1e-4))
    (is (= [2.4 0.9] (mapv #(/ (Math/round (* 1e9 %)) 1e9) (block/constrain mean-scale th))))))

;; the same regression with its slopes as one vector latent
(defdensity logreg-vec [b0 [b [2]]]
  [xs :- (Array double), ys :- (Array double), cnt :- Long, s0 :- Double]
  ;; elements read at a fixed index are bound before the loop (see defdensity)
  (let [b1 (ra/aget b 0) b2 (ra/aget b 1)]
    (loop [i 0 acc (n/+ (dist/logpdf (dist/->Normal 0.0 s0) b0)
                        (n/+ (dist/logpdf (dist/->Normal 0.0 s0) b1)
                             (dist/logpdf (dist/->Normal 0.0 s0) b2)))]
      (if (< i cnt)
        (let [eta (n/+ b0 (n/+ (n/* b1 (ra/aget xs (* 2 i))) (n/* b2 (ra/aget xs (inc (* 2 i))))))]
          (recur (inc i) (n/+ acc (n/- (n/* (ra/aget ys i) eta) (rm/log (n/+ 1.0 (rm/exp eta)))))))
        acc))))

(deftest a-vector-latent-is-a-slice-of-theta
  (is (= 3 (block/dimension logreg-vec)))
  (is (= [{:name :b0 :shape [] :support :real} {:name :b :shape [2] :support :real}]
         (:block/latents (:description logreg-vec))))
  (doseq [x [[0.0 0.0 0.0] [0.4 -1.1 0.7]]]
    (let [th (double-array x)
          [v g] ((block/capability logreg-vec :value+grad) th inputs)
          [v' g'] ((block/capability logreg :value+grad) th inputs)]
      (is (< (Math/abs (- v v')) 1e-12))
      (is (every? #(< (Math/abs %) 1e-9) (map - (seq ^doubles g) (seq ^doubles g')))))))

;; k positive scales, each with a half-normal prior and one observation
(defdensity scales [[s [3] :positive]]
  [ys :- (Array double)]
  (loop [i 0 acc 0.0]
    (if (< i 3)
      (recur (inc i) (n/+ acc (n/+ (dist/logpdf (dist/->Normal 0.0 2.0) (ra/aget s i))
                                   (dist/logpdf (dist/->Normal 0.0 (ra/aget s i)) (ra/aget ys i)))))
      acc)))

(deftest a-constrained-vector-latent
  (let [ys (double-array [0.5 -1.2 2.0])
        th [0.1 -0.3 0.4]
        ;; θ = log s: the body at s = e^θ plus Σ θ
        lp (fn [th] (+ (reduce + (map (fn [t y] (let [s (Math/exp t)]
                                                  (+ (dist/logpdf (dist/->Normal 0.0 2.0) s)
                                                     (dist/logpdf (dist/->Normal 0.0 s) y) t)))
                                      th ys))))
        [v g] ((block/capability scales :value+grad) (double-array th) {:ys ys})]
    (is (< (Math/abs (- v (lp th))) 1e-9))
    (doseq [j [0 1 2]]
      (let [h 1e-6 fd (/ (- (lp (update th j + h)) (lp (update th j - h))) (* 2 h))]
        (is (< (Math/abs (- fd (aget ^doubles g j))) 1e-4))))))

;; a vector latent read at a data-dependent index (varying intercepts); NUTS
;; chains call the gradient from several threads at once
(defdensity intercepts [[a [3]] beta]
  [group :- (Array long), x :- (Array double), y :- (Array double), cnt :- Long]
  (loop [i 0 acc 0.0]
    (if (< i cnt)
      (let [r (n/- (ra/aget y i) (n/+ (ra/aget a (ra/aget group i)) (n/* beta (ra/aget x i))))]
        (recur (inc i) (n/- acc (n/* 0.5 (n/* r r)))))
      acc)))

(def ^:private grouped
  (let [rng (java.util.Random. 5) cnt 30]
    {:group (long-array (repeatedly cnt #(.nextInt rng 3)))
     :x (double-array (repeatedly cnt #(.nextGaussian rng)))
     :y (double-array (repeatedly cnt #(.nextGaussian rng)))
     :cnt cnt}))

(defn- intercepts-reference ^double [^doubles th {:keys [^longs group ^doubles x ^doubles y cnt]}]
  (reduce + (for [i (range cnt)
                  :let [r (- (aget y i) (aget th (aget group i)) (* (aget th 3) (aget x i)))]]
              (* -0.5 r r))))

(deftest a-gathered-vector-latent-under-concurrent-calls
  (let [vg (block/capability intercepts :value+grad)
        thetas (vec (for [k (range 4)] (double-array [(* 0.1 k) -0.2 0.3 (- 0.5 (* 0.2 k))])))
        expected (mapv #(vec (second (vg % grouped))) thetas)]
    (doseq [th thetas
            j (range 4)]
      (let [h 1e-6
            at (fn [d] (let [t (aclone ^doubles th)] (aset t j (+ (aget t j) d)) (intercepts-reference t grouped)))]
        (is (< (Math/abs (- (/ (- (at h) (at (- h))) (* 2 h))
                            (aget ^doubles (second (vg th grouped)) j)))
               1e-5))))
    (is (= (mapv hash-set expected)
           (mapv deref (doall (for [k (range 4)]
                                (future (into #{} (repeatedly 2000 #(vec (second (vg (thetas k) grouped)))))))))))))
