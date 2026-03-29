# CBF Neural Control Examples

This directory contains example scripts demonstrating the modular CBF framework.

## Examples

### 1. Single Integrator with 2 Obstacles
**File:** `train_single_integrator.py`

**Dynamics:** ẋ = u (direct velocity control)

**CBF Type:** Standard CBF (relative degree 1)

**Run:**
```bash
cd examples
python train_single_integrator.py
```

**Key Features:**
- Simple integrator dynamics
- 2 circular obstacles
- Standard CBF constraint: Lg h · u ≥ -Lf h - α(h)

---

### 2. Double Integrator with 3 Obstacles (HOCBF)
**File:** `train_double_integrator.py`

**Dynamics:** ẍ = a (acceleration control)

**CBF Type:** Higher-Order CBF (relative degree 2)

**Run:**
```bash
cd examples
python train_double_integrator.py
```

**Key Features:**
- Double integrator dynamics with position + velocity state
- 3 circular obstacles
- HOCBF constraint with auxiliary barrier h₁ = ḣ₀ + α₁(h₀)
- **HARD safety constraints** (vs soft penalties in naive_soft_for_double_int.py)

---

## Modular Design Benefits

### Easy to Modify:

**Change dynamics:**
```python
config = TrainingConfig(
    dynamics_type='single_integrator',  # or 'double_integrator'
    ...
)
```

**Add/remove obstacles:**
```python
config = TrainingConfig(
    obstacles=[
        {'center': [1.0, 1.0], 'radius': 0.5, 'epsilon': 0.1},
        {'center': [2.0, 2.0], 'radius': 0.5, 'epsilon': 0.1},
        # Add more...
    ],
    ...
)
```

**Tune CBF parameters:**
```python
# Standard CBF (single integrator)
cbf_alpha=10.0

# HOCBF (double integrator)
cbf_alpha=(5.0, 5.0)  # (alpha1, alpha2)
```

---

## Understanding HOCBF

For double integrator with position barrier h₀(p) = ||p - c||² - r²:

**Step 1:** First derivative
```
ḣ₀ = ∇h₀ · v = 2(p - c) · v
```
(No control yet!)

**Step 2:** Auxiliary barrier
```
h₁ = ḣ₀ + α₁ h₀
```

**Step 3:** Second derivative (control appears!)
```
ḧ₀ = 2||v||² + 2(p - c) · a
```

**Step 4:** HOCBF constraint
```
ḧ₀ + α₂(h₁) ≥ 0
→ 2(p - c) · a ≥ -2||v||² - α₂(ḣ₀ + α₁ h₀)
```

This ensures safety for systems where control has **indirect effect** on position.

---

## Output

Each script produces:
1. Trained model saved to `../models/`
2. Trajectory visualization saved as PNG
3. Console output with training progress

---

## Next Steps

To create your own example:
1. Copy one of the example scripts
2. Modify `TrainingConfig` for your problem
3. Run training
4. The modular framework handles the rest!
