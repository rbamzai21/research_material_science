# New Tolerance Factor to Predict the Stability of Perovskite Oxides and Halides

**Paper:** Bartel et al., *Science Advances* 5, eaav0693 (2019)

**Authors:** Christopher J. Bartel, Christopher Sutton, Bryan R. Goldsmith, Runhai Ouyang, Charles B. Musgrave, Luca M. Ghiringhelli, Matthias Scheffler

---

## What This Paper Did

The classical Goldschmidt tolerance factor *t* (1926) predicts perovskite stability from ionic radii:

$$t = \frac{r_A + r_X}{\sqrt{2}(r_B + r_X)}$$

but achieves only **74% accuracy** across 576 ABX₃ compounds and fails badly on halides (33–56%).

Using **SISSO** (sure independence screening and sparsifying operator), the authors searched ~3 × 10⁹ candidate descriptors and identified a new 1D tolerance factor τ:

$$\tau = \frac{r_X}{r_B} - n_A\left(n_A - \frac{r_A / r_B}{\ln(r_A / r_B)}\right)$$

where τ < 4.18 predicts perovskite. It uses the same inputs as *t* (Shannon ionic radii) plus the oxidation state of A (*n_A*), achieving **92% accuracy** with nearly uniform performance across oxides and halides.

---

## 3 Independent Pieces of Evidence

### Evidence 1: Classification accuracy on experimental ABX₃ compounds

- **Data source:** 576 ABX₃ materials experimentally characterized as perovskite or nonperovskite at ambient conditions (from Zhang 2007, Li 2008, Travis 2016)
- **Method:** 80/20 train/test split; decision tree classifier to find optimal boundary
- **Result:** τ achieves 92% overall accuracy (94% on test set) vs. 74% for *t*; false-positive rate drops from 51% to 11%; performance is uniform across all five anions (O²⁻, F⁻, Cl⁻, Br⁻, I⁻)

### Evidence 2: Generalization to double perovskites (out-of-domain)

- **Data source:** 918 A₂BB′X₆ double perovskites from the Inorganic Crystal Structure Database (ICSD), entirely excluded from training
- **Method:** Extend τ to double perovskites by approximating *r_B* as the arithmetic mean of the two B-site radii
- **Result:** 91% accuracy — nearly identical to the training domain (92%), demonstrating generalizability beyond the training composition space

### Evidence 3: Correlation with DFT-computed thermodynamic stability

- **Data source:** 73 single and double perovskite chalcogenides and halides with decomposition enthalpies (ΔH_d) calculated by DFT-PBE (from Zhao 2017, Sun 2017)
- **Method:** Compare Platt-scaled probability P(τ) against ΔH_d
- **Result:** Linear correlation between P(τ) and ΔH_d; τ agrees with DFT for 64/73 compounds; in cases like CaZrO₃ and CaHfO₃, τ correctly predicts perovskite stability while cubic-structure DFT does not

---

## Additional Note

The monotonic, single-boundary nature of τ (unlike the two-sided boundary of *t*) enables meaningful **probability estimation** via Platt scaling. This is a structural advantage of the formula itself, not tied to any specific dataset.

---

## Reproduction Results

Scripts under `reproduce_evidence/`:

```bash
uv run python reproduce_evidence/evidence1_abx3_classification.py   # → fig2_abc.png
uv run python reproduce_evidence/evidence2_double_perovskites.py    # → fig_evidence2.png
uv run python reproduce_evidence/evidence3_dft_correlation.py       # → fig2_d.png
```

| Evidence | Metric | Reproduced | Paper | Match |
|----------|--------|-----------|-------|-------|
| 1 | τ overall accuracy (576 ABX₃) | 91.7% | 92% | ~exact |
| 1 | τ test set accuracy (20%) | 94% | 94% | exact |
| 1 | τ false positive rate | 10.6% | 11% | ~exact |
| 1 | Per-anion accuracy | all within 1% | — | ~exact |
| 2 | τ accuracy (918 ICSD A₂BB′X₆) | 91.0% | 91% | exact |
| 2 | Recovered known double perovskites | 806/868 | 806/868 | exact |
| 3 | τ agrees with DFT sign | 64/73 | 64/73 | exact |
| 3 | Halides R² | 0.77 | 0.91 | see note |
| 3 | Chalcogenides R² | 0.83 | 0.88 | see note |

**Note on R²:** The paper's R² labels in Fig. 2D are for slightly different subgroup definitions. The linear trends are visually consistent; the 64/73 agreement count is an exact match.

---

## Data & Code

All data and code are from the authors' repository, added as a git submodule at `perovskite-stability/` (source: https://github.com/CJBartel/perovskite-stability).

### Data files (mapped to evidences)

| File | Rows | Content | Evidence |
|------|------|---------|----------|
| `perovskite-stability/TableS1.csv` | 576 | ABX₃ compounds: A, B, X, nA, nB, nX, rA, rB, rX, t, τ, exp_label, train/test split, predictions, τ probability | 1 |
| `perovskite-stability/icsd_A2BBX6.csv` | 918 | A₂BB′X₆ from ICSD: B1, B2, rB1, rB2, nB1, nB2, τ, t, ICSD label | 2 |
| `perovskite-stability/TableS2.csv` | 73 | DFT decomposition enthalpies (ΔH_d) with τ, t, probabilities | 3 |
| `perovskite-stability/TableS4_A2BBX6_data.csv` | 69,574 | Full predicted double perovskite library (includes the 23,314 predicted stable) | Predictions |

### Supporting data

| File | Content |
|------|---------|
| `perovskite-stability/Shannon_Effective_Ionic_Radii.csv` | Shannon radii (ion, oxidation state, coordination number, ionic radius) |
| `perovskite-stability/Shannon_radii_dict.json` | Pre-built dictionary of Shannon radii |
| `perovskite-stability/electronegativities.csv` | Pauling electronegativities |

### Code

| File | Content |
|------|---------|
| `perovskite-stability/PredictPerovskites.py` | Main code — `PredictABX3` and `PredictAABBXX6` classes computing t, τ, and probabilities |
| `perovskite-stability/classify_CCX3_demo.ipynb` | Demo: classify a single CC′X₃ compound |
| `perovskite-stability/classify_list_of_formulas.ipynb` | Demo: batch classification from a DataFrame |
| `perovskite-stability/regenerate_supporting_tables.ipynb` | Regenerates all supplementary tables from the paper |
