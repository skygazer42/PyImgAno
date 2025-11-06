# Algorithm Selection Guide

This guide helps you choose the right anomaly detection algorithm for your use case.

## Quick Selection Table

| Your Priority | Recommended Algorithms | Why |
|---------------|------------------------|-----|
| **Best Overall Performance** | ECOD, COPOD, IForest | Top performers in ADBench benchmark |
| **No Parameter Tuning** | ECOD, COPOD | Completely parameter-free |
| **Speed (Large Datasets)** | COPOD, KNN, ECOD | O(n×d) complexity, highly optimized |
| **Interpretability** | ECOD, PCA, KNN | Clear mathematical foundation |
| **Small Datasets** | KNN, LOF, OCSVM | Work well with limited data |
| **High-Dimensional Data** | ECOD, COPOD, PCA | Designed for many features |
| **Deep Learning** | Deep SVDD, VAE, AE | Learn complex patterns |
| **Production Ready** | ECOD, COPOD, IForest | Fast, stable, well-tested |

## Algorithm Categories

### 🔥 Top-Tier Algorithms (Start Here)

#### ECOD (Empirical Cumulative Distribution-based)
**Best for: First-time users, production systems**

```python
detector = models.create_model(
    "vision_ecod",
    feature_extractor=extractor,
    contamination=0.1,
    n_jobs=-1
)
```

**Pros:**
- ✅ Parameter-free (no tuning needed)
- ✅ Top benchmark performance
- ✅ Highly interpretable
- ✅ Fast O(n×d) complexity
- ✅ Works well on high-dimensional data

**Cons:**
- ❌ Requires PyOD >= 0.9.7

**When to use:** Default choice for most applications

---

#### COPOD (Copula-Based Outlier Detection)
**Best for: Speed-critical applications**

```python
detector = models.create_model(
    "vision_copod",
    feature_extractor=extractor,
    contamination=0.1
)
```

**Pros:**
- ✅ Parameter-free
- ✅ Very fast
- ✅ Excellent benchmark results
- ✅ O(n×d) complexity

**Cons:**
- ❌ Requires PyOD >= 0.9.0

**When to use:** When speed is critical, large datasets

---

#### Isolation Forest
**Best for: General-purpose anomaly detection**

```python
detector = models.create_model(
    "vision_iforest",
    feature_extractor=extractor,
    n_estimators=100,
    contamination=0.1
)
```

**Pros:**
- ✅ Robust and reliable
- ✅ Top benchmark performer
- ✅ Handles high dimensions well
- ✅ Widely used and trusted

**Cons:**
- ❌ Random forest requires tuning n_estimators
- ❌ Memory intensive for large datasets

**When to use:** Proven reliability needed

---

### 📊 Classic Algorithms

#### KNN (K-Nearest Neighbors)
**Best for: Simple, interpretable results**

```python
detector = models.create_model(
    "vision_knn",
    feature_extractor=extractor,
    n_neighbors=10,
    method='largest'  # or 'mean', 'median'
)
```

**Pros:**
- ✅ Very interpretable
- ✅ Simple and well-understood
- ✅ No training needed (lazy learning)
- ✅ Multiple scoring methods

**Cons:**
- ❌ Slow on large datasets
- ❌ Memory intensive (stores all training data)
- ❌ Requires choosing k

**When to use:** Small datasets, interpretability important

---

#### PCA (Principal Component Analysis)
**Best for: Linear patterns, dimensionality reduction**

```python
detector = models.create_model(
    "vision_pca",
    feature_extractor=extractor,
    n_components=0.95,  # Keep 95% variance
    whiten=True
)
```

**Pros:**
- ✅ Classic, well-understood
- ✅ Fast
- ✅ Interpretable (principal components)
- ✅ Good for linear patterns

**Cons:**
- ❌ Assumes linear relationships
- ❌ May miss nonlinear anomalies

**When to use:** Data has linear structure

---

#### LOF (Local Outlier Factor)
**Best for: Density-based detection**

```python
detector = models.create_model(
    "vision_lof",
    feature_extractor=extractor,
    n_neighbors=20
)
```

**Pros:**
- ✅ Detects local anomalies
- ✅ Works well with clusters
- ✅ Classic algorithm

**Cons:**
- ❌ Requires tuning n_neighbors
- ❌ Slow on large datasets

**When to use:** Data has varying densities

---

### 🧠 Deep Learning Algorithms

#### Deep SVDD
**Best for: Complex nonlinear patterns**

```python
detector = models.create_model(
    "vision_deep_svdd",
    feature_extractor=extractor,
    epochs=50,
    hidden_neurons=[128, 64]
)
```

**Pros:**
- ✅ Learns complex patterns
- ✅ End-to-end trainable
- ✅ Powerful representation

**Cons:**
- ❌ Requires more data
- ❌ Slower training
- ❌ More parameters to tune

**When to use:** Large datasets, complex patterns

---

#### VAE (Variational Autoencoder)
**Best for: Probabilistic modeling**

```python
detector = models.create_model(
    "vision_vae",
    feature_extractor=extractor,
    encoder_neurons=[128, 64],
    latent_dim=32,
    epochs=100
)
```

**Pros:**
- ✅ Probabilistic framework
- ✅ Generates samples
- ✅ Smooth latent space

**Cons:**
- ❌ Requires substantial data
- ❌ Complex to tune
- ❌ Slower training

**When to use:** Need probabilistic scores, generative model

---

## Decision Tree

```
Do you need parameter-free solution?
├─ YES → Use ECOD or COPOD
└─ NO
   └─ Is speed critical?
      ├─ YES → Use COPOD or KNN
      └─ NO
         └─ Is interpretability important?
            ├─ YES → Use ECOD, PCA, or KNN
            └─ NO
               └─ Large dataset with complex patterns?
                  ├─ YES → Use Deep SVDD or VAE
                  └─ NO → Use Isolation Forest
```

## Benchmark Results

Based on [ADBench](https://github.com/Minqi824/ADBench) (30 algorithms on 57 datasets):

### Top Performers (Average Rank)
1. **ECOD** - Rank 3.5/30
2. **COPOD** - Rank 4.2/30
3. **IForest** - Rank 5.8/30
4. **LODA** - Rank 6.1/30

### Speed Comparison (Relative)
- **Fastest:** COPOD (1.0×)
- **Very Fast:** ECOD (1.2×), KNN (1.5×)
- **Fast:** IForest (2.0×), PCA (2.1×)
- **Moderate:** LOF (3.5×), OCSVM (4.0×)
- **Slow:** Deep SVDD (10×+), VAE (15×+)

## Use Case Examples

### Industrial Quality Control
**Scenario:** Detect defects in manufactured products

**Recommended:** ECOD or COPOD
- Fast enough for real-time inspection
- No parameter tuning (stable in production)
- High accuracy on visual defects

```python
detector = models.create_model("vision_ecod", ...)
```

---

### Medical Imaging
**Scenario:** Identify abnormal X-rays/scans

**Recommended:** Deep SVDD or VAE
- Can learn subtle patterns
- Probabilistic scores useful for doctors
- Worth the computational cost

```python
detector = models.create_model("vision_deep_svdd", ...)
```

---

### Security/Surveillance
**Scenario:** Detect unusual behavior in video frames

**Recommended:** COPOD or IForest
- Real-time processing needed
- High accuracy required
- Robust to varying conditions

```python
detector = models.create_model("vision_copod", ...)
```

---

### Research/Experimentation
**Scenario:** Exploring different approaches

**Recommended:** Try multiple
- ECOD (baseline)
- Deep SVDD (deep learning)
- Your domain-specific method

```python
for algo in ["vision_ecod", "vision_copod", "vision_deep_svdd"]:
    detector = models.create_model(algo, ...)
    # Compare results
```

---

## Parameter Tuning Guide

### Contamination
**What it is:** Expected proportion of outliers (0 < contamination < 0.5)

**How to set:**
- **Known outlier ratio:** Use that value
- **Unknown:** Start with 0.1 (10%)
- **Very rare defects:** Try 0.01-0.05
- **Balanced dataset:** Try 0.1-0.2

### Number of Neighbors (KNN, LOF)
**What it is:** How many neighbors to consider

**How to set:**
- **Small datasets (<100):** Try 5-10
- **Medium datasets (100-1000):** Try 10-20
- **Large datasets (>1000):** Try 20-50
- **Rule of thumb:** √n where n = dataset size

### Number of Components (PCA)
**What it is:** How many principal components to keep

**How to set:**
- **Variance-based:** 0.90-0.99 (keep 90-99% variance)
- **Fixed number:** 10-50 components
- **Rule of thumb:** min(n_samples/10, n_features/2)

## Common Pitfalls

### ❌ Don't
- Use deep learning on small datasets (<1000 samples)
- Ignore parameter tuning for KNN/LOF
- Use slow algorithms for real-time applications
- Forget to validate on test data

### ✅ Do
- Start with ECOD or COPOD
- Use cross-validation for parameter selection
- Benchmark multiple algorithms
- Monitor performance in production
- Use appropriate feature extractors

## Further Resources

- **PyOD Documentation:** https://pyod.readthedocs.io/
- **ADBench Paper:** https://arxiv.org/abs/2206.09426
- **ECOD Paper:** Li et al., TKDE 2022
- **COPOD Paper:** Li et al., ICDM 2020

---

**Need help?** Check examples in `examples/` directory or open an issue on GitHub.
