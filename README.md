# Transversal ARC Solver

A standalone ARC-AGI solver that uses projective geometry (Plucker lines and transversals) to solve grid transformation tasks with **zero learning** — no neural network, no training, no parameters fitted to ARC data.

Built on the [transversal-memory](https://github.com/khalildh/transversal-memory) library.

## Results

Tested on all same-size ARC tasks (input grid = output grid dimensions) from both the training and evaluation sets:

|  | Training set | Evaluation set |
|--|:--:|:--:|
| Same-size tasks | 262 | 270 |
| **Rank 1 (solved)** | **166** | **150** |
| Timeout (>120s) | 70 | 112 |
| Not rank 1 | 26 | 8 |
| **Of non-timeout** | **166/192 (86%)** | **150/158 (95%)** |

**316 total ARC tasks solved at rank 1 with zero learning.** The correct output grid scores above all other candidates — up to 134 million exhaustively checked, or 10 million sampled for larger grids.

No task-specific logic. The same pipeline handles color maps, spatial shifts, fills, rotations, and pattern completions.

---

## The problem

An ARC task gives you 2–10 training pairs, each showing an input grid and its corresponding output grid. You also get a test input. Your job: produce the test output.

```
Training pairs (2-4):          Test pair:
┌───┐    ┌───┐                 ┌───┐    ┌───┐
│1 2│ →  │3 1│                 │2 1│ →  │???│
│0 1│    │0 3│                 │1 0│    │???│
└───┘    └───┘                 └───┘    └───┘
   pair 1    ...                input    output=?
```

The grids are small (typically 3x3 to 30x30) with up to 10 colors. The transformation rules are diverse — color swaps, spatial shifts, pattern fills, reflections, completions — and you have to figure out the rule from just a few examples.

The brute-force challenge: a 3x3 grid with 8 colors has 134 million possible outputs. A 10x10 grid with 5 colors has over 10^69. You need a way to score every possible output and find the one that best matches the transformation pattern demonstrated in the training examples.

---

## How it works: the full pipeline

```
═══════════════════════════════════════════════════════════════
                    OVERVIEW
═══════════════════════════════════════════════════════════════

Training pairs (input, output)     Test input
         │                              │
    ┌────▼────┐                    ┌────▼────┐
    │ Build Plücker lines from     │ Build lines for each    │
    │ 8 embeddings × adj pairs     │ candidate output color  │
    └────┬────┘                    └────┬────┘
         │                              │
    ┌────▼────┐                         │
    │ Find 200 transversals per    │    │
    │ training pair via P3Memory   │    │
    └────┬────┘                         │
         │                              │
    ┌────▼──────────────────────────────▼────┐
    │ Precompute score tables:               │
    │ table[adj][color_a][color_b] =         │
    │   Σ log|line · J₆ · transversal|      │
    └────────────────┬───────────────────────┘
                     │
                ┌────▼────┐
                │ Score all candidates        │
                │ via table lookup + addition  │
                │ Rank 1 = answer              │
                └─────────┘
```

---

### Step 1: Represent cell relationships as Plucker lines

Every ARC grid has structure in how adjacent cells relate to each other. The solver captures this by building an embedding vector for each cell, then combining adjacent cells into a geometric object — a line in projective 3-space.

```
For each adjacent cell pair (r,c)↔(r',c') in a training pair:

   Cell (0,0): in=1, out=3          Cell (0,1): in=2, out=1
        │                                 │
        ▼                                 ▼
  ┌─────────────┐                   ┌─────────────┐
  │  Embedding   │                   │  Embedding   │
  │  e.g.        │                   │  e.g.        │
  │  color_only: │                   │  color_only: │
  │  [in_OH,     │                   │  [in_OH,     │
  │   out_OH]    │                   │   out_OH]    │
  │  = 20-dim    │                   │  = 20-dim    │
  └──────┬──────┘                   └──────┬──────┘
         │                                  │
         ▼                                  ▼
      ea (20d)                           eb (20d)
         │                                  │
         └──────────┬───────────────────────┘
                    ▼
            ┌───────────────┐
            │  make_line()  │   combined = [ea; eb]  (40d)
            │               │   p1 = W1 @ combined   (4d)
            │  W1, W2 are   │   p2 = W2 @ combined   (4d)
            │  random fixed │
            │  projections  │   L = p1 ∧ p2  (exterior product)
            │               │     = 6-dim Plücker vector
            └───────┬───────┘
                    │
                    ▼
             L ∈ R⁶  (a line in P³)
```

A 3x3 grid has 12 adjacency pairs (6 horizontal + 6 vertical), so each training pair produces 12 lines per embedding type. With 2–4 training pairs, that's 24–48 lines per embedding.

The solver uses **8 different embedding functions**, each capturing different aspects:

```
┌──────────────┐
│  hist_color   │──→ color one-hots + histogram difference (30d)
├──────────────┤
│  color_only   │──→ just input/output color one-hots (20d)
├──────────────┤
│  pos_color    │──→ position + colors (22d)
├──────────────┤
│  all          │──→ position + colors + full histograms (42d)
├──────────────┤
│  row_feat     │──→ row color distribution + uniformity (44d)
├──────────────┤
│  col_feat     │──→ column color distribution + uniformity (42d)
├──────────────┤
│  color_count  │──→ color frequency + mode indicators (24d)
├──────────────┤
│  diagonal     │──→ diagonal position features (26d)
└──────────────┘
```

Each embedding type uses its own deterministic random projection matrices W1, W2 (seeded by SHA-256 hash of the embedding name). Different embeddings project the same cell data into completely different geometric spaces, providing complementary views of the transformation.

---

### Step 2: Find transversals from training examples

In the training pairs, we know both input and output, so the Plucker lines encode the *correct* transformation. The solver extracts the geometric essence by finding **transversals** — lines in P³ that simultaneously meet multiple training lines.

A classic result from Schubert calculus: **given 4 lines in general position in P³, there are exactly 0 or 2 lines that meet all four.**

```
A transversal is a line that MEETS 4 other lines in P³:

      L₁ ──────╲
      L₂ ────────╲──── T (transversal)
      L₃ ──────────╱     meets all 4!
      L₄ ────────╱
```

The algorithm to find transversals:

```
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: Build the Constraint System                   │
│                                                         │
│  "T meets Lᵢ" means: T · ★Lᵢ = 0                      │
│  (★ = Hodge dual via J6 matrix)                         │
│                                                         │
│  For 4 lines, this gives 4 linear constraints on T:     │
│                                                         │
│        ┌ ─── ★L₁ ─── ┐   ┌   ┐                        │
│        │ ─── ★L₂ ─── │   │   │                        │
│   A =  │ ─── ★L₃ ─── │ · │ T │ = 0     A is 4×6      │
│        │ ─── ★L₄ ─── │   │   │                        │
│        └              ┘   └   ┘                        │
│                                                         │
│  4 equations, 6 unknowns → 2D null space               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Find the Null Space via SVD                   │
│                                                         │
│  SVD(A) = U · Σ · Vᵀ                                   │
│                                                         │
│  Last 2 rows of Vᵀ = null space basis: v₁, v₂          │
│                                                         │
│  ANY T = t·v₁ + v₂ satisfies all 4 incidence           │
│  constraints (for any scalar t).                        │
│                                                         │
│  But we also need T to be a VALID LINE...               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: Enforce Plücker Relation (solve_p3)           │
│                                                         │
│  Valid line ⟺ p₀₁p₂₃ − p₀₂p₁₃ + p₀₃p₁₂ = 0          │
│                                                         │
│  Substitute T = t·v₁ + v₂ into the relation:           │
│                                                         │
│    α·t² + β·t + γ = 0                                  │
│                                                         │
│  where:                                                 │
│    α = plucker_relation(v₁)                             │
│    γ = plucker_relation(v₂)                             │
│    β = symmetric bilinear cross term                    │
│                                                         │
│  ┌───────────────────────────────────────────┐          │
│  │         -β ± √(β² - 4αγ)                 │          │
│  │  t  =  ─────────────────────              │          │
│  │                2α                         │          │
│  │                                           │          │
│  │  2 real solutions → 2 transversals        │          │
│  │  (by Schubert calculus on Gr(2,4))        │          │
│  └───────────────────────────────────────────┘          │
│                                                         │
│  T₁ = t₁·v₁ + v₂    ← transversal line 1             │
│  T₂ = t₂·v₁ + v₂    ← transversal line 2             │
└─────────────────────────────────────────────────────────┘
```

The solver samples 200 different random 4-tuples from each training pair's lines, finding up to 400 transversals per pair. Across 4 training pairs and 8 embedding types, this accumulates ~6400 transversals — geometric constraints that the correct output must satisfy.

---

### Step 3: Precompute score tables

Naively scoring each candidate output against 6400 transversals would be too slow for 134 million candidates. The key optimization: **the score decomposes by adjacency pair**.

For each adjacency position and each possible color pair, the contribution to the total score depends only on those two colors — not the rest of the grid. So we precompute it:

```
For each adjacency pair (r,c)↔(r',c'):
  For each possible output color pair (a, b):

    1. Build test embedding:  ea = emb(r,c, test_in[r,c], a)
                              eb = emb(r',c', test_in[r',c'], b)
    2. Make Plücker line:     L = make_line(ea, eb)
    3. Score vs transversals: score = Σ log|L · (★T)|
                                       over all transversals T

    ★ = Hodge dual (J6 matrix), |L·(★T)| → 0 iff lines meet

┌──────────────────────────────────┐
│  score_table[adj][a][b] = float  │  nc × nc per adjacency
│                                  │  12 adjacencies × nc²
│  Precomputed once, reused for    │  = 12 × 64 = 768 entries
│  ALL 134M candidates!            │  (for 8 colors)
└──────────────────────────────────┘
```

The inner product uses the Hodge dual matrix J6:

```
      ┌                     ┐
      │ 0  0  0  0  0  1   │
      │ 0  0  0  0 -1  0   │
J6 =  │ 0  0  0  1  0  0   │    Swaps and signs
      │ 0  0  1  0  0  0   │    the Plücker components
      │ 0 -1  0  0  0  0   │
      │ 1  0  0  0  0  0   │
      └                     ┘
```

The precomputation stores `JTm = J6 @ transversals.T` (a 6 × n_trans matrix), then for each candidate line L: `score = Σ log(|L @ JTm| + 1e-10)`.

---

### Step 4: Score all candidates

With precomputed tables, scoring a candidate output grid is just table lookups and addition:

```
For a 3×3 grid with 8 colors: 8⁹ = 134,217,728 candidates

Each candidate = 9 color indices:
┌─────────┐
│ a b c   │
│ d e f   │  → color index at each position
│ g h i   │
└─────────┘

Score = Σ  score_table[adj][ candidate[r,c], candidate[r',c'] ]
       adj

Just 12 TABLE LOOKUPS per candidate! No matrix operations!

┌──────────────────────────────────────────────┐
│  for each of 134M candidates:                │
│    score = 0                                 │
│    for each adjacency (r,c)↔(r',c'):        │
│      score += table[adj][ color_a, color_b ] │
│                                              │
│  → scores 134M candidates in <1 second       │
│    using OpenMP parallelism                  │
└──────────────────────────────────────────────┘
```

For grids too large to enumerate exhaustively (>200M candidates), the solver draws 10 million random candidates and checks whether any score better than the correct answer. If none do, the correct answer is declared rank 1.

---

### Step 5: Histogram-aware scoring (small grids)

For tasks with few colors and small grids, the solver uses a more precise method. The `hist_color` embedding includes the histogram difference between input and output grids. Since different candidate outputs have different histograms, the solver precomputes **separate score tables for each possible histogram**:

```
3 colors, 3×3 grid → C(11,2) = 55 possible histograms
5 colors, 3×3 grid → C(13,4) = 715 possible histograms

┌────────────────────────────────────────────┐
│  hist_tables[(3,4,2)] = tables for         │
│    "3 of color 0, 4 of color 1, 2 of 2"   │
│                                            │
│  For each candidate:                       │
│    1. Compute its histogram                │
│    2. Look up the RIGHT table              │
│    3. Score with histogram-correct tables   │
└────────────────────────────────────────────┘
```

This is why tasks like 25ff71a9 go from rank 12 to rank 1 — the histogram context tells the scorer exactly which color distribution each candidate has.

---

### Step 6: Dual scoring strategy

```
┌─────────────────────────────────────────────┐
│  IF histogram tables feasible (≤2000):      │
│    → Use histogram-only scoring             │
│       (strongest signal, least noise)       │
│                                             │
│  ELSE (many colors or large grids):         │
│    → Raw sum across all 8 embeddings        │
│       (each adds complementary signal)      │
└─────────────────────────────────────────────┘
```

---

## The math

### Plucker coordinates

A line in projective 3-space P³ is represented by 6 coordinates. Given two points a, b on the line (in R⁴ homogeneous coordinates):

```
p_ij = a_i * b_j - a_j * b_i

for (i,j) in {(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)}
```

Valid Plucker coordinates must satisfy:

```
p_01 * p_23 - p_02 * p_13 + p_03 * p_12 = 0
```

This quadric in P⁵ is the Grassmannian G(2,4) — the space of all lines in P³.

### Incidence

Two lines p, q meet if and only if:

```
p^T · J6 · q = 0     (Plucker inner product)
```

### Transversals

Given 4 lines, the constraint matrix A = [J6·L1; J6·L2; J6·L3; J6·L4] is 4×6. SVD gives a 2D null space {v1, v2}. Any transversal T = t·v1 + v2 must also satisfy the Plucker relation, giving:

```
α·t² + β·t + γ = 0

t = (-β ± √(β² - 4αγ)) / 2α
```

Exactly 0 or 2 real solutions — this is the Schubert calculus on Gr(2,4).

### Scoring

```
score(candidate) = Σ   Σ  log( |L(candidate[i], candidate[j]) · J6 · T| + ε )
                  adj  T
```

Lower score = better. The correct output's lines nearly meet the transversals (inner product ≈ 0), giving log(≈0) = very negative, which beats all other candidates.

---

## Build and run

Single file, no dependencies beyond system LAPACK and standard C:

```bash
# macOS
cc -O3 -march=native -framework Accelerate -o arc_solver arc_solver.c -lm

# Linux
cc -O3 -march=native -o arc_solver arc_solver.c -lm -llapack

# Run on a single task
./arc_solver data/ARC-AGI/data/training/25ff71a9.json

# Run on all same-size tasks in a directory
./arc_solver --all data/ARC-AGI/data/training
```

ARC task JSON files: [ARC-AGI](https://github.com/fchollet/ARC-AGI).

### Example output

```
ARC Plucker Transversal Solver (C)
Embeddings: 8, Trans/pair: 200
Tasks: 1

  0d3d703e (4 train, 3x3):
  8 colors, 11440 histograms -- using placeholder for hist embeddings
    SOLVED  rank 1/134217728
    6400 transversals, setup=0.0s, score=0.8s
```

134,217,728 candidates checked in 0.8 seconds. Correct answer confirmed rank 1.

---

## Why it works

- **Multi-transversal scoring**: A single transversal is one scalar constraint — too weak. 200+ transversals from different random 4-tuples provide complementary geometric constraints. Their intersection narrows down to the correct answer.

- **Precomputed tables**: The score decomposes by adjacency pair, so the expensive linear algebra is done once during setup. Scoring is just array indexing — O(adjacency_pairs) per candidate.

- **Eight complementary embeddings**: Color-only captures the color mapping. Position captures spatial structure. Row/column captures line-level patterns. Histogram captures global distribution changes. Each provides a different geometric view.

- **Zero learning**: W1 and W2 are deterministic random matrices (seeded by embedding name). Transversals are found by exact SVD + quadratic formula. No gradient descent, no loss function, no optimization. The entire pipeline is pure linear algebra on the training examples.

## Limitations

- Only handles same-size tasks (input and output grids must have the same dimensions)
- Large grids (>20x20) can be slow for sampling (>120s timeout)
- Tasks requiring counting, object segmentation, or multi-step reasoning are not captured by pairwise adjacency features
- Results have some sensitivity to the random projection seed (different seeds solve slightly different task subsets)
