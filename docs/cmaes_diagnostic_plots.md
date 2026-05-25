# CMA-ES Diagnostic Plots: Comprehensive Explanation

## Panel 1: Objective Statistics

### **Plot 1.1: Convergence - Best, Mean, Median Fitness**

**What it shows:** Three fitness curves over iterations:
- **Best fitness** (blue solid line): The best solution found so far
- **Mean fitness f(m)** (green dashed): Fitness evaluated at the distribution mean
- **Median fitness** (red dotted): Middle value of all population fitnesses

**Interpretation:**
- **Best fitness declining** = algorithm is making progress
- **Mean fitness f(m)** should track best fitness but lag behind - it shows where the "center" of search is
- **Large gap between best and mean** = population is exploring widely; good early on
- **Converging together** = search is focusing around the optimum
- **Mean fitness worse than best** = the distribution center hasn't caught up to the best solution yet (normal)
- **Stagnation (flat lines)** = algorithm may be stuck; check if σ became too small

**Healthy pattern:** All three should decrease steadily, with mean lagging slightly behind best.

---

### **Plot 1.2: Fitness Standard Deviation**

**What it shows:** The spread (standard deviation) of fitness values across the population each iteration.

**Interpretation:**
- **High std early** = good exploration, population sampling diverse regions
- **Decreasing std** = population concentrating around good solutions (convergence)
- **Very low std too early** = premature convergence - lost diversity before finding optimum
- **Oscillations** = healthy sign that algorithm is adapting search range
- **Sudden spike** = algorithm escaped local region (can be good)
- **Stuck at high value** = not converging, might need more iterations

**Healthy pattern:** Steady decline from high to low, possibly with small oscillations.

---

## Panel 2: Adaptation Dynamics

### **Plot 2.1: Step-Size Evolution (σ)**

**What it shows:** The global step-size parameter σ over iterations.

**Interpretation:**
- **σ decreasing overall** = algorithm is zooming in on the optimum (normal as it converges)
- **Small oscillations** = healthy adaptation responding to local geometry
- **σ exploding** = divergence, likely due to numerical issues or inappropriate bounds
- **σ → 0 too fast** = premature convergence, lost exploration capability
- **σ stays constant** = adaptation may have stalled
- **Smooth decay** = ideal convergence pattern

**Formula context:** σ is updated as:
```
σ ← σ · exp((c_σ/d_σ) · (||p_σ||/E||N(0,I)|| - 1))
```
If ||p_σ|| > expected norm, σ increases (making progress). If ||p_σ|| < expected, σ decreases (oscillating).

**Healthy pattern:** Gradual decrease with small oscillations, never reaching exactly zero until very late.

---

### **Plot 2.2: Covariance Condition Number (κ)**

**What it shows:** The ratio κ = λ_max / λ_min where λ are the eigenvalues of the covariance matrix C.

**Interpretation:**
- **κ = 1** = perfectly spherical search (isotropic)
- **κ increasing** = algorithm adapting to elongated fitness landscape
- **κ very large (>10¹⁴)** = numerical instability, ill-conditioned matrix
- **κ stable at moderate value** = adapted to problem structure
- **κ → 1 late in run** = converging to local region where landscape looks isotropic

**Problem-specific meaning:**
- **Sphere function:** κ should stay near 1 (landscape is isotropic)
- **Ellipsoid function:** κ should grow to match problem's condition number
- **CEC functions:** depends on problem structure

**Termination criterion:** CMA-ES stops if κ > tol_condition (default 10¹⁴) to avoid numerical breakdown.

**Healthy pattern:** Gradual increase if problem is ill-conditioned, otherwise stays moderate. Should never explode.

---

## Panel 3: Covariance Properties

### **Plot 3.1: Covariance Determinant (Search Volume)**

**What it shows:** det(C), the determinant of the covariance matrix (on log scale).

**Mathematical note:** The actual search volume is proportional to σⁿ·√det(C), but det(C) alone shows how the shape's "size" evolves independent of σ.

**Interpretation:**
- **det(C) decreasing** = search volume shrinking, focusing on smaller region
- **Smooth decline** = healthy convergence
- **det(C) near 0** = covariance nearly singular, about to converge
- **det(C) increasing** = expanding search (rare, usually only early)
- **Sudden drops** = rapid convergence phase

**Relationship to σ:** Total search volume = σⁿ · √det(C), so even if det(C) changes, σ is the main volume controller.

**Healthy pattern:** Steady decrease, especially in later iterations.

---

### **Plot 3.2: Eigenvalue Spectrum (first 5)**

**What it shows:** The 5 largest eigenvalues of C over time (on log scale).

**Interpretation:**
- **All eigenvalues similar** = isotropic search (sphere-like)
- **Eigenvalues separating** = anisotropic search (ellipsoid adapting to landscape)
- **λ_max growing, λ_min shrinking** = stretching ellipsoid
- **All decreasing together** = overall volume shrinking while maintaining shape
- **One eigenvalue exploding** = likely numerical issue or pathological landscape

**Problem-specific:**
- **Sphere:** All λ should stay roughly equal
- **Ellipsoid:** λ should match problem's eigenvalues
- **Separable problems:** λ adapt to per-coordinate difficulty

**Connection to condition number:** κ = λ_max / λ_min, so this plot shows the components of κ.

**Healthy pattern:** Smooth evolution, no sudden spikes, largest 5 should be well-behaved.

---

## Panel 4: Evolution Paths

### **Plot 4.1: Evolution Path Norms**

**What it shows:** Norms of two evolution paths:
- **||p_c||** (blue solid): Covariance evolution path
- **||p_σ||** (red dashed): Step-size evolution path

**Formulas:**
```
p_c ← (1-c_c)·p_c + √(c_c(2-c_c)μ_eff) · (new_mean - old_mean)/σ
p_σ ← (1-c_σ)·p_σ + √(c_σ(2-c_σ)μ_eff) · C^(-1/2) · (new_mean - old_mean)/σ
```

**Interpretation:**

**||p_c|| (covariance path):**
- **Long path** = consistent direction, algorithm making steady progress
- **Short path** = oscillating direction, searching around current point
- **Moderate, steady values** = healthy adaptation
- **Sudden spikes** = large shift in search direction

**||p_σ|| (step-size path):**
- **||p_σ|| > expected** → σ increases (progress in consistent direction)
- **||p_σ|| < expected** → σ decreases (oscillating, too large steps)
- **Oscillations normal** = CSA responding to local geometry

**Together:**
- **Both high** = strong directional progress
- **Both low** = converged or exploring locally
- **p_c high, p_σ low** = moving but with inappropriate step-size

**Healthy pattern:** Moderate values, gradual decrease as convergence approaches, small oscillations acceptable.

---

### **Plot 4.2: Mean Vector Norm (Distance from Origin)**

**What it shows:** ||m||, the Euclidean distance of the distribution mean from the origin.

**Interpretation:**

**For Sphere function (optimum at 0):**
- **||m|| decreasing** = moving toward optimum (EXACTLY what we want)
- **||m|| → 0** = converging to true optimum
- **Log scale** shows exponential convergence (ideal)
- **Linear rate on log plot** = geometric/exponential convergence

**For shifted functions:**
- ||m|| measures distance from origin, not optimum
- Less directly interpretable, but decreasing is still usually good

**For unknown optima:**
- Use f(m) (mean fitness) instead as convergence indicator

**Connection to convergence:**
On simple functions, ||m|| is often more stable than best fitness because it tracks the distribution center, not just best sample.

**Healthy pattern:** 
- **Sphere/known optimum:** Steady exponential decrease (straight line on log plot)
- **Unknown optimum:** Use f(m) instead

---