## Architecture

**Goal:** predict a target vector (e.g. HF simulation output) from several noisy/cheap "source" point clouds (different mesh resolutions, solvers, heuristics), with calibrated uncertainty split into *aleatoric* (irreducible data noise) and *epistemic* (model doesn't know).

### 1. Per-source encoder — PointNet-style
Each source point cloud $X_i \in \mathbb{R}^{N\times 3}$ is mapped pointwise through shared MLPs, then **max-pooled** over points:
$$z_i = \text{MLP}\big(\max_{n} \phi(x_n)\big) \in \mathbb{R}^d$$
Max-pool makes the encoding invariant to point order and count — a permutation-invariant symmetric function, same trick as PointNet/DeepSets. Same weights across all $n$ sources so different-fidelity clouds of the *same* object land near each other in latent space.

### 2. Attention fusion over sources (handles missing sources)
Treat the $n$ source latents as a *set*, not a sequence — a single learned query $q$ attends over them:
$$\alpha = \text{softmax}\!\left(\frac{q K^\top}{\sqrt{d_h}} + \text{mask}\right), \qquad \text{fused} = \alpha V$$
where $K,V$ are projections of the (source-id-tagged) latents $z_i$, multi-head. The mask sets logits to $-\infty$ for absent sources so ragged coverage (not every case has all sources) just falls out of softmax naturally — no imputation needed.

### 3. Disagreement feature (cheap uncertainty signal)
Empirical variance across the *present* source latents:
$$\text{disagree} = \frac{1}{n}\sum_j \text{Var}_i[z_{i,j}]$$
Intuition: if cheap heuristics agree, trust is high; if they scatter, flag it. This is fed into the head as an extra scalar input — "free" epistemic-ish signal from disagreement among sources, distinct from the ensemble-level epistemic term below.

### 4. Heteroscedastic head — aleatoric uncertainty
Instead of predicting only a point estimate, the head predicts a **Gaussian** per output dim:
$$p(y\mid x) = \mathcal{N}(\mu(x), \sigma^2(x)), \qquad \sigma^2(x) = \exp(\log\text{var}(x))$$
Trained via negative log-likelihood:
$$\mathcal{L} = \tfrac12\left[\log\sigma^2 + \frac{(y-\mu)^2}{\sigma^2}\right]$$
This is the standard "heteroscedastic regression" trick (Kendall & Gal, 2017): predicting variance lets the loss automatically **downweight noisy samples** (large $\sigma^2$ shrinks the residual term's gradient) instead of forcing uniform confidence everywhere. $\log\text{var}$ is predicted (not $\text{var}$ directly) for numerical stability/positivity, clipped to $[-8,8]$.

### 5. Deep ensemble — epistemic uncertainty
Train $K$ independently-initialized/shuffled copies of the whole pipeline. At inference, combine via the **law of total variance**:
$$\text{Var}[y] = \underbrace{\mathbb{E}_k[\sigma_k^2]}_{\text{aleatoric: avg predicted noise}} + \underbrace{\text{Var}_k[\mu_k]}_{\text{epistemic: disagreement between models}}$$
- Aleatoric = average of each member's own predicted variance (irreducible noise the data itself has).
- Epistemic = spread of the $K$ point-estimate means (model uncertainty — shrinks with more/better data, unlike aleatoric).

This is the Lakshminarayanan et al. (2017) deep-ensembles decomposition, applied on top of per-member heteroscedastic outputs rather than plain point estimates — so you get both flavors of uncertainty simultaneously, decomposed rather than conflated into one number.
