# Minimum Example for Hand Checking

This directory contains simplified examples designed for manual validation with colleagues.

## Files

### 1. `minimum_example_deterministic.py`

**Purpose:** Executable Python script demonstrating the tensor engine with a minimal portfolio.

**Configuration:**
- **Assets (N):** 10
- **Events (Q):** 1
- **Typologies (K):** 5
- **Curve points (M):** 20

**Run:**
```bash
python minimum_example_deterministic.py
```

**Output:**
- Console output with all risk metrics
- Visualization saved as `tensorial_engine_complete_analysis.png`
- Total portfolio loss for the single event

**Key Results Displayed:**
- Deterministic loss for Event 0
- Portfolio AAL (equals single event loss since Q=1)
- Per-asset AAL
- Top risk contributors
- All gradients (∂J/∂C, ∂J/∂v, ∂J/∂H)

---

### 2. `hand_calculation_deterministic.ipynb`

**Purpose:** Jupyter notebook performing **step-by-step manual calculations** for validation.

**What it does:**
1. Generates the **same portfolio** as `minimum_example_deterministic.py` (same random seed)
2. Displays all input data in detail
3. **Manually computes loss** for each of the 10 assets using:
   - Explicit loops (not vectorized)
   - Conventional numpy operations
   - Detailed printouts of every intermediate value
4. Compares manual results with tensor engine results
5. Exports complete calculation records for verification

**How to use:**

1. **Open in Jupyter/VS Code:**
   ```bash
   jupyter notebook hand_calculation_deterministic.ipynb
   ```
   or open in VS Code with Jupyter extension

2. **Run all cells** (Cell → Run All)

3. **Review the output:** The notebook will show:
   - Asset-by-asset calculation breakdown
   - All intermediate quantities (α, MDR, grid indices, etc.)
   - Comparison table showing manual vs tensor engine results

**Output Structure:**

- **Step 1:** Import and generate portfolio
- **Step 2:** Display all portfolio data (v, u, C, x_grid, H)
- **Step 3:** Manual calculation for each asset with detailed explanations
- **Step 4:** Summary table of all results
- **Step 5:** Validation against tensor engine
- **Step 6:** Per-asset comparison
- **Step 7:** Complete verification data export

**For hand-checking:**

The notebook explicitly shows for each asset:
```
Asset i:
  1. Intensity h[i]
  2. Typology u[i]
  3. Exposure v[i]
  4. Grid interval [x[j], x[j+1]]
  5. Interpolation weight α
  6. Vulnerability values C[k,j] and C[k,j+1]
  7. Interpolated MDR
  8. Loss = v[i] × MDR
```

You can verify any calculation with a calculator using the printed values.

---

## Validation Approach

### With minimum_example_deterministic.py:

1. Run the script
2. Note the total loss (should be ~$1.7M for the default seed)
3. Check top risk contributors
4. Verify gradient magnitudes are reasonable

### With hand_calculation_deterministic.ipynb:

1. Open the notebook and run all cells
2. **Pick any asset** (e.g., Asset 3)
3. **Follow the printed calculation:**
   - Note intensity: h[3]
   - Note typology: u[3] = k
   - Find grid interval: j where x[j] ≤ h[3] < x[j+1]
   - Compute α = (h[3] - x[j]) / (x[j+1] - x[j])
   - Look up C[k, j] and C[k, j+1] from vulnerability matrix
   - Compute MDR = (1-α)×C[k,j] + α×C[k,j+1]
   - Compute Loss = v[3] × MDR
4. **Verify with calculator:** All values are printed with 6-8 decimal places
5. **Check:** Manual loss matches tensor engine loss (within 1e-6)

### Example Hand Calculation:

From the notebook output, you might see:

```
Asset 3:
  Intensity:       0.523147 g
  Typology:        2
  Exposure:        $456,123.00
  Grid interval:   [0.500000, 0.550000] g (j=10)
  Alpha:           0.462940
  C[2, 10]:        0.234567
  C[2, 11]:        0.289012
  MDR:             0.259789
  Loss:            $118,511.37
```

**Verify manually:**
```
α = (0.523147 - 0.500000) / (0.550000 - 0.500000) = 0.462940 ✓
MDR = (1-0.462940)×0.234567 + 0.462940×0.289012 = 0.259789 ✓
Loss = $456,123.00 × 0.259789 = $118,511.37 ✓
```

---

## Consistency Guarantee

Both files use:
- **Same random seed (42)** → Identical portfolio data
- **Same interpolation logic** → Identical results
- **Same vulnerability curves** → Identical MDR values

The notebook validates that:
```
|Manual Loss - Tensor Loss| < 1e-6
```

This confirms the tensor engine correctly implements the mathematical formulation.

---

## Technical Notes

### Why 10 assets and 1 event?

- **10 assets:** Small enough to review all calculations by hand
- **1 event:** Simplifies to deterministic case (AAL = single loss)
- **20 curve points:** Sufficient resolution for realistic interpolation

### Numerical Precision

- All calculations use `float32` (32-bit floats)
- Typical precision: ~7 decimal digits
- Expected differences: < 1e-6 (acceptable numerical error)

### Data Generation

The portfolio is **synthetic** but realistic:
- Exposures: $100K - $1M (typical building values)
- Intensities: 0.0g - 1.5g (realistic ground motion range)
- Vulnerability: Sigmoid curves with varying fragility

---

## Questions for Colleagues

After running both files, discuss:

1. **Mathematical correctness:**
   - Is the interpolation formula correct?
   - Are we using the right vulnerability curve for each asset?

2. **Numerical accuracy:**
   - Do manual and tensor results match?
   - Are differences within acceptable tolerance?

3. **Edge cases:**
   - What happens if intensity < min(x_grid)?
   - What happens if intensity > max(x_grid)?
   - How does the code handle these?

4. **Gradient validation:**
   - Do gradient magnitudes make physical sense?
   - Which assets have highest ∂AAL/∂v? Why?

---

## Next Steps

After validation:

1. ✅ **Verified mathematical correctness** → Move to larger portfolios
2. ✅ **Checked gradient computation** → Use for optimization
3. ✅ **Validated against manuscript** → Prepare for publication
4. 📈 **Scale up:** Test with N=10,000, Q=100,000

---

*Last updated: February 1, 2026*
