(ns org.replikativ.foerster-raster.gradient-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster-raster.gradient :as gradient]
            [org.replikativ.foerster-raster.block :as rb]
            [org.replikativ.foerster.block :as block]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.arrays :as a]
            [raster.par :as p]))

(deftm regression [coefficients :- (Array double) xs :- (Array double) ys :- (Array double) cnt :- Long] :- Double
  (let [intercept (a/aget coefficients 0) slope (a/aget coefficients 1)
        prior (n/* -0.5 (n/+ (n/* intercept intercept) (n/* slope slope)))]
    (n/+ prior
         (p/reduce acc 0.0 i cnt
                   (let [r (n/- (a/aget ys i) (n/+ intercept (n/* slope (a/aget xs i))))]
                     (n/- acc (n/* 0.5 (n/* r r))))))))

(def source
  {:arguments '[coefficients :- (Array double) xs :- (Array double) ys :- (Array double) cnt :- Long]
   :dimension 2
   :scalar-source
   {:args '[intercept :- Double slope :- Double xs :- (Array double) ys :- (Array double) cnt :- Long]
    :body '(let [prior (raster.numeric/* -0.5 (raster.numeric/+ (raster.numeric/* intercept intercept) (raster.numeric/* slope slope)))]
             (raster.numeric/+ prior
                               (raster.par/reduce acc 0.0 i cnt
                                                  (let [r (raster.numeric/- (raster.arrays/aget ys i)
                                                                            (raster.numeric/+ intercept (raster.numeric/* slope (raster.arrays/aget xs i))))]
                                                    (raster.numeric/- acc (raster.numeric/* 0.5 (raster.numeric/* r r)))))))}})

(def compiled
  (delay {:scalar (gradient/compile-gradient #'regression source)
          :array (gradient/compile-gradient #'regression (dissoc source :scalar-source) :simd? false)}))
(def binding-spec
  {:args (fn [theta {:keys [x y]}] [theta (double-array x) (double-array y) (long (count x))])
   :theta [0]})
(def description {:block/id :regression :block/target :complete-conditional
                  :block/latents [{:name :intercept :shape [] :support :real}
                                  {:name :slope :shape [] :support :real}]})
(def cases [{:theta [0.2 0.7] :x [0.0 0.4 1.2] :y [0.1 0.5 1.0]}
            {:theta [-0.5 1.2] :x [0.1 0.3 1.5 2.0 4.0] :y [0.2 0.4 1.1 2.2 3.0]}])
(defn reference [{[b m] :theta :keys [x y]}]
  (reduce (fn [[v db dm] [x y]] (let [r (- y (+ b (* m x)))]
                                  [(- v (* 0.5 r r)) (+ db r) (+ dm (* r x))]))
          [(* -0.5 (+ (* b b) (* m m))) (- b) (- m)] (map vector x y)))
(defn close? [a b] (< (Math/abs (double (- a b))) 1e-10))

(deftest invoked-aot-scalar-and-array-gradients
  (doseq [[_ f] @compiled c cases]
    (let [actual (vec (apply f ((:args binding-spec) (double-array (:theta c)) c)))]
      (is (= 3 (count actual)))
      (is (every? true? (map close? actual (reference c)))))))

(deftest block-capabilities-and-coordinate-ownership
  (let [runtime (rb/raster-block description #'regression binding-spec)]
    (doseq [[_ f] @compiled c cases]
      (let [b (gradient/aot-block description #'regression binding-spec f)
            [v g] ((block/capability b :value+grad) (double-array (:theta c)) c)
            [rv rg] ((block/capability runtime :value+grad) (double-array (:theta c)) c)]
        (is (close? v rv))
        (is (every? true? (map close? g rg)))
        (is (close? v ((block/capability b :log-density) (double-array (:theta c)) c)))
        (aset-double g 0 99999.0)
        (is (every? true? (map close? (second ((block/capability b :value+grad) (double-array (:theta c)) c)) (rest (reference c)))))))
    (let [c (assoc (first cases) :theta [0.8 0.4]) q [(Math/log 0.8) 0.4]
          [value db dm] (reference c)
          desc (-> description (assoc :block/coordinates :constrained)
                   (assoc-in [:block/latents 0 :support] :positive))]
      (doseq [[_ f] @compiled]
        (let [b (gradient/aot-block desc #'regression binding-spec f)
              [v g] ((block/capability b :value+grad) (double-array q) c)]
          (is (close? v (+ value (first q))))
          (is (every? true? (map close? g [(+ (* db 0.8) 1.0) dm]))))))))

(deftest reject-malformed-signatures-and-unsupported-scalar-arity
  (is (= :scalar (gradient/parameter-layout source)))
  (is (= :array (gradient/parameter-layout (dissoc source :scalar-source))))
  (is (= :array (gradient/parameter-layout source :array)))
  (doseq [bad [(assoc source :dimension 0)
               (assoc source :arguments '[x :- Double])
               (assoc-in source [:scalar-source :args] '[intercept :- Double])
               (assoc-in source [:scalar-source :args] '[intercept :- Double slope :- Double ys :- (Array double) xs :- (Array double) cnt :- Long])]]
    (is (thrown? clojure.lang.ExceptionInfo (gradient/parameter-layout bad))))
  (is (thrown? clojure.lang.ExceptionInfo (gradient/parameter-layout source :unknown)))
  (is (thrown? clojure.lang.ExceptionInfo (gradient/parameter-layout (dissoc source :scalar-source) :scalar)))
  (let [data (vec (mapcat #(vector (symbol (str "x" %)) :- 'Double) (range 17)))
        wide {:dimension 2 :arguments (into '[theta :- (Array double)] data)
              :scalar-source {:args (into '[b :- Double m :- Double] data) :body 0.0}}]
    (is (= :array (gradient/parameter-layout wide)))
    (is (thrown? clojure.lang.ExceptionInfo (gradient/parameter-layout wide :scalar)))))

(deftest reject-compiled-gradient-of-wrong-length
  (let [b (gradient/aot-block description #'regression binding-spec (fn [& _] (double-array [0.0 1.0])))]
    (is (thrown? clojure.lang.ExceptionInfo
                 ((block/capability b :value+grad) (double-array [0.2 0.7]) (first cases))))))
