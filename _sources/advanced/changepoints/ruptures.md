(target-ruptures)=
# Ruptures

Reference: [Ruptures](https://centre-borelli.github.io/ruptures-docs) {cite:p}`truong2020ruptures`

General-purpose changepoint detection using the `ruptures` library.

---

## Methods

Five search methods are available:

| Method | Description |
|--------|-------------|
| **Pelt** | Penalty-based, fast. Good when the number of changepoints is unknown. |
| **Binseg** | Binary segmentation. Fast recursive splitting. |
| **BottomUp** | Bottom-up merging of segments. |
| **Window** | Sliding window approach. |
| **Dynp** | Dynamic programming. Optimal but slow on long signals. |

Each method supports a cost model (`l2`, `l1`, `rbf`, etc.) and parameters
like `min_size`, `jump`, and either `pen` (penalty) or `n_bkps` (fixed number
of breakpoints).

```{note}
Ruptures detection has not been tested as extensively as the kinematic and
audio methods. Results may vary — check visually and adjust parameters as
needed.
```

---

## Usage

1. Select a feature in Data Controls.
2. Open the **Ruptures** panel.
3. Choose a method and click **Configure...**.
4. Click **Detect**.

---

