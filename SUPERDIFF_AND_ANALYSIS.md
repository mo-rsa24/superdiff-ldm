# SuperDiff AND Operation: Correctness Analysis

## Summary
The current implementation in `dynamics.py::stochastic_super_diff_multi` **does NOT correctly implement** the AND operation according to **Proposition 6**. It uses a heuristic approximation rather than solving the required linear system.

---

## Theoretical Requirements (Proposition 6)

### The Constraint
For AND operation composing M models, we need to find weights κ = [κ₁, κ₂, ..., κₘ] such that:

1. **Equal log-density evolution**: `d log q¹ = d log q² = ... = d log qᴹ`
2. **Normalization**: `Σⱼ κⱼ = 1`

This gives us **M equations for M unknowns** → solvable linear system.

### The SDE (from Proposition 6)
```
dx_τ = Σⱼ κⱼ u_τʲ(x_τ) dτ + g₁₋τ dW̄_τ
```

Where `u_τʲ` is the velocity field for model j.

### Log-Density Evolution (from Theorem 1)
For each model i:
```
d log q^i = ⟨dx_τ, ∇log q^i⟩ + ⟨∇, f₁₋τ⟩ + ⟨f₁₋τ - (g²₁₋τ/2)∇log q^i, ∇log q^i⟩ dτ
```

Setting these equal for all pairs (i,j) and solving for κ gives us the linear system.

---

## Issues with Current Implementation

### Issue 1: Not Solving the Linear System

**Current code (lines 198-229):**
```python
elif operation == "AND":
    # Compute pairwise differences and prepare for kappa optimization
    # For M models, we solve: kappa_i proportional to contribution to joint likelihood

    # Simplified approach: use weighted average based on agreement with independent update
    dx_ind = 2 * dsigma * vel_uncond + noise

    agreements = []
    for m in range(M):
        vel_guided = vel_uncond + guidance_scale * (velocities[m] - vel_uncond)
        agreement = -(dx_ind * (vel_guided - vel_uncond)).sum((1, 2, 3))
        vel_term = (torch.abs(dsigma) * velocities[m] ** 2).sum((1, 2, 3))
        agreements.append(agreement + vel_term + lift / num_inference_steps)

    agreement_stack = torch.stack(agreements, dim=-1)
    kappas[i + 1] = torch.softmax(agreement_stack, dim=-1)  # ❌ WRONG
```

**Problems:**
- Uses an ad-hoc "agreement" metric
- Applies softmax (which is the OR operation according to Algorithm 1!)
- Does NOT enforce the constraint: `d log q^i = d log q^j`
- Comment admits "For simplicity, we use a gradient-based approach" → this is not theoretically justified

### Issue 2: Incorrect Constraint

The constraint `d log q^i = d log q^j` for all i,j is **not satisfied** by the current implementation.

According to Theorem 1, the log-density change is:
```
d log q^i = -|dsigma|/sigma * ||v_i||² - ⟨dx, v_i/sigma⟩
```

For equal evolution, we need:
```
-⟨dx, v_i/sigma⟩ + ⟨dx, v_j/sigma⟩ = |dsigma|/sigma * (||v_i||² - ||v_j||²)
```

Expanding `dx = 2*dsigma*vf + noise` where `vf = vel_uncond + guidance_scale*Σₖ(κₖ*(vₖ - vel_uncond))`:

This becomes a **linear system in κ** that must be solved.

---

## Correct Implementation

### The Linear System Setup

For M models, we set up the system:
- **M-1 equations**: Compare each model to a reference (e.g., model 0)
- **1 equation**: `Σₖ κₖ = 1`

For each pair (0, j) where j ∈ {1, 2, ..., M-1}:
```
Σₖ κₖ * Aⱼₖ = bⱼ
```

Where:
```python
A[j,k] = 2*dsigma*guidance_scale/sigma * ⟨v_k - v_uncond, v_j - v_0⟩
b[j] = |dsigma|/sigma * (||v_0||² - ||v_j||²) - ⟨dx_base, (v_j - v_0)/sigma⟩
```

Plus the last equation:
```
Σₖ κₖ = 1
```

### Corrected Code

See `dynamics_corrected.py::stochastic_super_diff_multi_corrected()` for the full implementation.

Key difference:
```python
elif operation == "AND":
    # ✅ CORRECT: Solve linear system for Proposition 6
    kappas[i + 1] = solve_kappa_and(
        velocities=velocities,
        vel_uncond=vel_uncond,
        dsigma=dsigma,
        sigma=sigma,
        dx_base=dx_base,
        guidance_scale=guidance_scale,
        lift=lift,
        num_inference_steps=num_inference_steps,
        batch_size=batch_size,
        M=M
    )
```

The `solve_kappa_and()` function:
1. Builds the M×M linear system
2. Solves using `torch.linalg.solve()`
3. Clamps and normalizes for numerical stability

---

## Comparison with 2-Prompt Case

### 2-Prompt (Correct)

In `stochastic_super_diff_and()` (lines 96-104):
```python
# Analytically solve for single kappa that balances two models
term1 = (torch.abs(dsigma) * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3))
term2 = (dx_ind * (vel_obj - vel_bg)).sum((1, 2, 3))
term3 = sigma * lift / num_inference_steps

numerator = term1 - term2 + term3
denominator = 2 * dsigma * guidance_scale * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))

kappa[i + 1] = numerator / (denominator + 1e-8)
```

This is **analytically correct** for M=2 because:
- With 2 models, we have 1 degree of freedom (κ₁, and κ₂ = 1 - κ₁)
- The constraint `d log q¹ = d log q²` reduces to a single scalar equation
- Can be solved in closed form

### M-Prompt (Current - Incorrect)

The current M-prompt code tries to use a heuristic + softmax, which:
- ❌ Does NOT generalize the 2-prompt analytical solution
- ❌ Does NOT solve the linear system from Proposition 6
- ❌ Uses softmax (which is actually the OR operation!)

### M-Prompt (Corrected)

The corrected version:
- ✅ Properly sets up the M×M linear system
- ✅ Solves for κ that satisfies `d log q^i = d log q^j` for all i,j
- ✅ Enforces `Σₖ κₖ = 1`
- ✅ Generalizes the 2-prompt case (for M=2, reduces to same solution)

---

## Verification Steps

To verify correctness, you should check:

1. **Constraint satisfaction**: After solving for κ, compute `d log q^i` for all i and verify they're approximately equal
2. **Normalization**: Verify `Σₖ κₖ ≈ 1` at each step
3. **Numerical stability**: Check for NaN or extreme κ values
4. **Degeneracy**: When M=2, verify it matches the 2-prompt analytical solution

---

## Recommendation

**Replace the current AND implementation with the corrected version:**

```python
# In dynamics.py, replace lines 198-229 with:
elif operation == "AND":
    kappas[i + 1] = solve_kappa_and(
        velocities=velocities,
        vel_uncond=vel_uncond,
        dsigma=dsigma,
        sigma=sigma,
        dx_base=2 * dsigma * vel_uncond + noise,
        guidance_scale=guidance_scale,
        lift=lift,
        num_inference_steps=num_inference_steps,
        batch_size=batch_size,
        M=M
    )
```

And add the `solve_kappa_and()` function from `dynamics_corrected.py`.

---

## References

- **Algorithm 1**: SuperDiff pseudocode - specifies "solve Linear Equations" for AND
- **Proposition 6**: [Density control] - defines the constraint system
- **Theorem 1**: [Itô density estimator] - gives log-density evolution equation
- **Appendix C.1**: Formulas for the linear system (mentioned in Proposition 6)

---

## Conclusion

The current implementation is **theoretically incorrect** for AND operation with M>2 prompts. It will not properly compose the models according to the SuperDiff framework. Use the corrected version in `dynamics_corrected.py` for proper implementation.
