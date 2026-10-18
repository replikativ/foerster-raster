(ns org.replikativ.foerster-raster.gradient
  "JVM-AOT gradient capabilities for vector-parameter Raster densities.
  This adapter emits typed wrappers and delegates differentiation/compilation
  to Raster. It owns no model semantics, compiler pass, binder or cache."
  (:require [org.replikativ.foerster.block :as block]
            [raster.core]
            [raster.ad.reverse]
            [raster.arrays]
            [raster.par]
            [raster.compiler.pipeline :as pipeline]))

(def qualified-scalar-arity 18)

(defn parameter-layout
  "Choose scalar AD only for a supplied scalar source within the qualified
  18-argument boundary. The vector-parameter public interface is unchanged."
  ([source] (parameter-layout source :auto))
  ([{:keys [arguments dimension scalar-source]} requested]
   (when-not (and (integer? dimension) (pos? dimension)
                  (vector? arguments) (zero? (mod (count arguments) 3))
                  (>= (count arguments) 3)
                  (= [:- '(Array double)] (vec (rest (take 3 arguments)))))
     (throw (ex-info "Expected typed arguments beginning with a double-array parameter vector and positive dimension" {})))
   (when (and scalar-source
              (not (and (vector? (:args scalar-source))
                        (= (* 3 (+ dimension (dec (quot (count arguments) 3))))
                           (count (:args scalar-source)))
                        (= (drop (* 3 dimension) (:args scalar-source)) (drop 3 arguments))
                        (every? #(= [:- 'Double] (vec (rest %)))
                                (take dimension (partition 3 (:args scalar-source))))
                        (some? (:body scalar-source)))))
     (throw (ex-info "Scalar source must declare one Double per parameter followed by the same data arguments" {})))
   (let [arity (some-> scalar-source :args count (quot 3))
         eligible? (and arity (<= arity qualified-scalar-arity))]
     (case requested
       :auto (if eligible? :scalar :array)
       :array :array
       :scalar (if eligible? :scalar
                   (throw (ex-info "Scalar AD signature exceeds qualified arity or lacks source"
                                   {:scalar-arity arity :maximum-arity qualified-scalar-arity :jvm-maximum-arity 20})))
       (throw (ex-info "Unknown gradient parameter layout" {:layout requested}))))))

(defn compile-gradient
  "Compile a density whose first argument is a flat double[] parameter vector.
  `source` supplies typed :arguments and :dimension. Optional :scalar-source
  {:args ... :body ...} supplies the equivalent density with scalar parameters
  followed by the same data arguments. :parameter-layout selects :auto, :scalar
  or :array; remaining options pass directly to Raster compile-aot.
  Returns a callable producing double[value, gradient coordinates ...]."
  [density source & {:as options}]
  (when-not (var? density)
    (throw (ex-info "Compiled gradient requires a Raster density Var" {})))
  (let [layout (parameter-layout source (:parameter-layout options :auto))
        args (:arguments source)
        params (mapv first (partition 3 args))
        scalar? (= layout :scalar)
        scalar-name (gensym "scalar_density_")
        density-name (if scalar? scalar-name
                         (symbol (str (ns-name (:ns (meta density)))) (str (:name (meta density)))))
        d (:dimension source)
        wrt (if scalar? (vec (range d)) [0])
        call-params (if scalar?
                      (concat (map #(list 'raster.arrays/aget (first params) %) (range d)) (rest params)) params)
        name (gensym "compiled_density_gradient_")
        result (cons (list 'raster.ad.reverse/value+grad (list 'var density-name) :wrt wrt) call-params)
        body (if scalar?
               (list 'let ['result result]
                     (list 'raster.par/map ['j (inc d)] (list 'double (list 'nth 'result 'j))))
               (list 'let ['result result 'gradient (list 'nth 'result 1)]
                     (list 'raster.par/map ['j (inc d)]
                           (list 'double
                                 (list 'if (list 'zero? 'j) (list 'nth 'result 0)
                                       (list 'raster.arrays/aget 'gradient (list 'dec 'j)))))))
        v (binding [*ns* (:ns (meta density))]
            (when scalar?
              (eval (list 'raster.core/deftm scalar-name (get-in source [:scalar-source :args])
                          :- 'Double (get-in source [:scalar-source :body]))))
            (eval (list 'raster.core/deftm name args :- '(Array double) body))
            (ns-resolve *ns* name))]
    (apply pipeline/compile-aot v (mapcat identity (dissoc options :parameter-layout)))))

(defn aot-block
  "Adapt a compiled [value, gradient...] array to Foerster block capabilities.
  `args` is the caller's existing (theta, inputs)->arguments function. The
  density and gradient operate in the coordinates declared by description;
  Foerster owns constrained transforms and their Jacobians."
  [description density {:keys [args]} compiled]
  (let [dimension (reduce + 0 (map #(reduce * 1 (:shape %)) (:block/latents description)))]
    (block/block
     description
     {:log-density (fn [theta inputs] (double (apply @density (args theta inputs))))
      :value+grad (fn [theta inputs]
                    (let [^doubles result (apply compiled (args theta inputs))]
                      (when-not (= (inc dimension) (alength result))
                        (throw (ex-info "Compiled gradient dimension does not match block latents"
                                        {:dimension dimension :result-length (alength result)})))
                      [(aget result 0)
                       (java.util.Arrays/copyOfRange result 1 (alength result))]))})))
