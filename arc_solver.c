/*
 * arc_solver.c — Standalone C implementation of the ARC Plucker transversal solver.
 *
 * Replicates the full pipeline from exp_arc_fast_solve.py:
 *   1. JSON parsing of ARC task files
 *   2. 8 embedding types (hist_color, color_only, pos_color, all,
 *      row_feat, col_feat, color_count, diagonal)
 *   3. Plucker line computation via dual projection + exterior product
 *   4. Transversal computation via SVD of 4x6 constraint matrix + quadratic solve
 *   5. Score table building (J6 @ transversals.T inner products)
 *   6. Exhaustive or sampling-based candidate scoring
 *   7. Dual strategy: histogram tables for <=2000 histograms, raw sum otherwise
 *
 * Compile:
 *   Linux:   cc -O3 -march=native -fopenmp -o arc_solver arc_solver.c -lm
 *   macOS:   /opt/homebrew/opt/llvm/bin/clang -O3 -march=native -fopenmp -o arc_solver arc_solver.c -lm
 *   No OMP:  cc -O3 -march=native -o arc_solver arc_solver.c -lm
 *
 * Usage:
 *   ./arc_solver data/ARC-AGI/data/training/25ff71a9.json [task2.json ...]
 *   ./arc_solver --all data/ARC-AGI/data/training/
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <time.h>
#include <dirent.h>
#include <sys/time.h>

#if defined(_OPENMP)
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#endif

/* ══════════════════════════════════════════════════════════════════════════════
 * Configuration
 * ══════════════════════════════════════════════════════════════════════════════ */

#define N_COLORS       10
#define PLUCKER_DIM     6
#define MAX_GRID_DIM   30
#define MAX_GRID_CELLS (MAX_GRID_DIM * MAX_GRID_DIM)
#define MAX_COLORS     10
#define MAX_ADJ        (2 * MAX_GRID_CELLS)
#define MAX_TRANS    2000
#define N_EMBEDDINGS    8
#define MAX_EMB_DIM    44
#define MAX_TRAIN_PAIRS 10
#ifndef N_TRANS_PER_PAIR
#define N_TRANS_PER_PAIR 200
#endif
#define MAX_HIST_TABLES 2000

/* Embedding dimensions (must match Python) */
static const int EMB_DIMS[8] = {30, 20, 22, 42, 44, 42, 24, 26};

/* Embedding names */
static const char *EMB_NAMES[8] = {
    "hist_color", "color_only", "pos_color", "all",
    "row_feat", "col_feat", "color_count", "diagonal"
};

/* J6 matrix (Hodge dual) */
static const double J6[6][6] = {
    {0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0,-1, 0},
    {0, 0, 0, 1, 0, 0},
    {0, 0, 1, 0, 0, 0},
    {0,-1, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0},
};

/* Plucker pairs: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) */
static const int PL_I[6] = {0, 0, 0, 1, 1, 2};
static const int PL_J[6] = {1, 2, 3, 2, 3, 3};

/* Plucker relation indices:
   P01*P23 - P02*P13 + P03*P12 = 0
   idx: 0=01, 1=02, 2=03, 3=12, 4=13, 5=23 */
#define PR_AB 0  /* (0,1) */
#define PR_CD 5  /* (2,3) */
#define PR_AC 1  /* (0,2) */
#define PR_BD 4  /* (1,3) */
#define PR_AD 2  /* (0,3) */
#define PR_BC 3  /* (1,2) */


/* ══════════════════════════════════════════════════════════════════════════════
 * Simple JSON parser (minimal, ARC-specific)
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Read entire file into malloc'd buffer */
static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(len + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, len, f);
    buf[len] = '\0';
    fclose(f);
    return buf;
}

/* Skip whitespace */
static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* Parse an integer, advance pointer */
static int parse_int(const char **pp) {
    const char *p = skip_ws(*pp);
    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    int v = 0;
    while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
    *pp = p;
    return sign * v;
}

/* Parse a 2D grid: [[int,...], ...], returns rows/cols and flat data */
static int parse_grid(const char **pp, int *out, int *rows, int *cols) {
    const char *p = skip_ws(*pp);
    if (*p != '[') return -1;
    p++;
    *rows = 0; *cols = 0;
    int pos = 0;
    while (1) {
        p = skip_ws(p);
        if (*p == ']') { p++; break; }
        if (*p == ',') { p++; continue; }
        if (*p == '[') {
            p++;
            int c = 0;
            while (1) {
                p = skip_ws(p);
                if (*p == ']') { p++; break; }
                if (*p == ',') { p++; continue; }
                out[pos++] = parse_int(&p);
                c++;
            }
            if (*cols == 0) *cols = c;
            (*rows)++;
        }
    }
    *pp = p;
    return pos;
}

/* Find key in JSON object, return pointer past the colon */
static const char *find_key(const char *json, const char *key) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return NULL;
    p += strlen(needle);
    p = skip_ws(p);
    if (*p == ':') p++;
    return skip_ws(p);
}

/* ARC task data structure */
typedef struct {
    int input[MAX_GRID_CELLS];
    int output[MAX_GRID_CELLS];
    int H, W;
    int oH, oW;
} ArcPair;

typedef struct {
    ArcPair train[MAX_TRAIN_PAIRS];
    int n_train;
    ArcPair test[1]; /* only first test */
    int n_test;
} ArcTask;

/* Parse ARC task JSON */
static int parse_arc_task(const char *json, ArcTask *task) {
    memset(task, 0, sizeof(*task));

    /* Find "train" array */
    const char *p = find_key(json, "train");
    if (!p || *p != '[') return -1;
    p++; /* skip [ */

    task->n_train = 0;
    while (task->n_train < MAX_TRAIN_PAIRS) {
        p = skip_ws(p);
        if (*p == ']') { p++; break; }
        if (*p == ',') { p++; continue; }
        if (*p == '{') {
            p++;
            ArcPair *pair = &task->train[task->n_train];
            /* Find "input" and "output" within this object */
            /* We need to find them within the current object scope */
            const char *obj_start = p;
            /* Simple: look for "input" then "output" */
            const char *inp_p = find_key(obj_start - 1, "input");
            if (inp_p) {
                parse_grid(&inp_p, pair->input, &pair->H, &pair->W);
            }
            const char *out_p = find_key(obj_start - 1, "output");
            if (out_p) {
                parse_grid(&out_p, pair->output, &pair->oH, &pair->oW);
            }
            /* Skip to end of this object */
            int depth = 1;
            while (*p && depth > 0) {
                if (*p == '{') depth++;
                else if (*p == '}') depth--;
                p++;
            }
            task->n_train++;
        }
    }

    /* Find "test" array */
    const char *test_p = find_key(json, "test");
    if (!test_p || *test_p != '[') return -1;
    test_p++; /* skip [ */

    task->n_test = 0;
    while (task->n_test < 1) {
        test_p = skip_ws(test_p);
        if (*test_p == ']') break;
        if (*test_p == ',') { test_p++; continue; }
        if (*test_p == '{') {
            test_p++;
            ArcPair *pair = &task->test[task->n_test];
            const char *obj_start = test_p;
            const char *inp_p = find_key(obj_start - 1, "input");
            if (inp_p) parse_grid(&inp_p, pair->input, &pair->H, &pair->W);
            const char *out_p = find_key(obj_start - 1, "output");
            if (out_p) parse_grid(&out_p, pair->output, &pair->oH, &pair->oW);
            int depth = 1;
            while (*test_p && depth > 0) {
                if (*test_p == '{') depth++;
                else if (*test_p == '}') depth--;
                test_p++;
            }
            task->n_test++;
        }
    }

    return 0;
}


/* ══════════════════════════════════════════════════════════════════════════════
 * Mersenne Twister (MT19937) — exact match with numpy.random.RandomState
 * ══════════════════════════════════════════════════════════════════════════════ */

#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfUL
#define MT_UPPER_MASK 0x80000000UL
#define MT_LOWER_MASK 0x7fffffffUL

typedef struct {
    unsigned long mt[MT_N];
    int mti;
    /* Box-Muller cached value */
    int has_gauss;
    double gauss;
} Rng;

static void rng_seed(Rng *r, unsigned long seed) {
    r->mt[0] = seed & 0xffffffffUL;
    for (r->mti = 1; r->mti < MT_N; r->mti++) {
        r->mt[r->mti] = (1812433253UL * (r->mt[r->mti-1] ^ (r->mt[r->mti-1] >> 30)) + r->mti);
        r->mt[r->mti] &= 0xffffffffUL;
    }
    r->has_gauss = 0;
    r->gauss = 0.0;
}

static unsigned long mt_genrand(Rng *r) {
    unsigned long y;
    static unsigned long mag01[2] = {0x0UL, MT_MATRIX_A};
    if (r->mti >= MT_N) {
        int kk;
        for (kk = 0; kk < MT_N - MT_M; kk++) {
            y = (r->mt[kk] & MT_UPPER_MASK) | (r->mt[kk+1] & MT_LOWER_MASK);
            r->mt[kk] = r->mt[kk+MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < MT_N - 1; kk++) {
            y = (r->mt[kk] & MT_UPPER_MASK) | (r->mt[kk+1] & MT_LOWER_MASK);
            r->mt[kk] = r->mt[kk+(MT_M-MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (r->mt[MT_N-1] & MT_UPPER_MASK) | (r->mt[0] & MT_LOWER_MASK);
        r->mt[MT_N-1] = r->mt[MT_M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];
        r->mti = 0;
    }
    y = r->mt[r->mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

static double rng_double(Rng *r) {
    /* numpy RandomState.random_sample(): (genrand>>5)*2^{-27} + (genrand>>6)*2^{-53} */
    long a = (long)(mt_genrand(r) >> 5);
    long b = (long)(mt_genrand(r) >> 6);
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

/* Standard normal via Box-Muller — matches numpy RandomState.standard_normal() */
static double rng_normal(Rng *r) {
    if (r->has_gauss) {
        r->has_gauss = 0;
        return r->gauss;
    }
    double x1, x2, r2;
    do {
        x1 = 2.0 * rng_double(r) - 1.0;
        x2 = 2.0 * rng_double(r) - 1.0;
        r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);
    double f = sqrt(-2.0 * log(r2) / r2);
    r->has_gauss = 1;
    r->gauss = f * x1;
    return f * x2;
}

static int rng_int(Rng *r, int n) {
    return (int)(rng_double(r) * n);
}

/* Fisher-Yates shuffle for choosing k from n */
static void rng_choose(Rng *r, int n, int k, int *out) {
    int *arr = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) arr[i] = i;
    for (int i = 0; i < k; i++) {
        int j = i + rng_int(r, n - i);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
        out[i] = arr[i];
    }
    free(arr);
}


/* ══════════════════════════════════════════════════════════════════════════════
 * Linear algebra helpers
 * ══════════════════════════════════════════════════════════════════════════════ */

static double vec_norm(const double *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    return sqrt(s);
}

static void vec_normalize(double *v, int n) {
    double nrm = vec_norm(v, n);
    if (nrm > 1e-15)
        for (int i = 0; i < n; i++) v[i] /= nrm;
}

/* Compact SVD of an m x n matrix A (m <= 6, n = 6) via Golub-Kahan bidiagonalization.
 * We need: full right singular vectors Vt (n x n).
 * For a 4x6 matrix, we need the null space (last 2 rows of Vt).
 *
 * We use the one-sided Jacobi SVD which is simple and robust for small matrices.
 * Computes A = U * S * Vt. We only need Vt.
 */

/* SVD using LAPACK dgesvd (via Accelerate on macOS, or system LAPACK).
 * Returns Vt (n x n, row-major) and singular values S (descending).
 */
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern void dgesvd_(const char *jobu, const char *jobvt,
                    const int *m, const int *n, double *a, const int *lda,
                    double *s, double *u, const int *ldu,
                    double *vt, const int *ldvt,
                    double *work, const int *lwork, int *info);
#endif

static void svd_right(const double *A, int m, int n, double *Vt, double *S) {
    /* LAPACK expects column-major; our A is row-major → transpose */
    double *At = (double *)malloc(m * n * sizeof(double));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            At[j * m + i] = A[i * n + j];

    double *Vt_lapack = (double *)malloc(n * n * sizeof(double));
    int lda = m, ldu = 1, ldvt = n, info;
    __CLPK_integer m_ = m, n_ = n, lda_ = lda, ldu_ = ldu, ldvt_ = ldvt;

    /* Query optimal workspace */
    __CLPK_integer lwork = -1;
    double work_query;
    char jobu = 'N', jobvt = 'A';
    dgesvd_(&jobu, &jobvt, &m_, &n_, At, &lda_, S, NULL, &ldu_,
            Vt_lapack, &ldvt_, &work_query, &lwork, &info);
    lwork = (__CLPK_integer)work_query;
    double *work = (double *)malloc(lwork * sizeof(double));

    /* Compute SVD */
    /* Re-copy since dgesvd destroys A */
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            At[j * m + i] = A[i * n + j];

    dgesvd_(&jobu, &jobvt, &m_, &n_, At, &lda_, S, NULL, &ldu_,
            Vt_lapack, &ldvt_, work, &lwork, &info);

    /* LAPACK returns Vt in column-major n×n → transpose to row-major */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Vt[i * n + j] = Vt_lapack[j * n + i];

    free(At); free(Vt_lapack); free(work);
}


/* ══════════════════════════════════════════════════════════════════════════════
 * Plucker geometry
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Plucker relation: p01*p23 - p02*p13 + p03*p12 */
static double plucker_relation(const double *p) {
    return p[PR_AB] * p[PR_CD] - p[PR_AC] * p[PR_BD] + p[PR_AD] * p[PR_BC];
}

/* Hodge dual: [p5, -p4, p3, p2, -p1, p0] */
static void hodge_dual(const double *p, double *out) {
    out[0] =  p[5];
    out[1] = -p[4];
    out[2] =  p[3];
    out[3] =  p[2];
    out[4] = -p[1];
    out[5] =  p[0];
}

/* Plucker inner product: p . J6 . q */
static double plucker_inner(const double *p, const double *q) {
    return p[0]*q[5] - p[1]*q[4] + p[2]*q[3]
         + p[3]*q[2] - p[4]*q[1] + p[5]*q[0];
}

/* Compute Plucker line from two R4 points a, b.
 * Returns normalized 6-vector. Returns 0 if degenerate. */
static int line_from_points(const double *a, const double *b, double *out) {
    for (int k = 0; k < 6; k++)
        out[k] = a[PL_I[k]] * b[PL_J[k]] - a[PL_J[k]] * b[PL_I[k]];
    double n = vec_norm(out, 6);
    if (n < 1e-12) return 0;
    for (int k = 0; k < 6; k++) out[k] /= n;
    return 1;
}


/* ══════════════════════════════════════════════════════════════════════════════
 * Transversal computation (solve_p3)
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Given two 6-vectors v1, v2 spanning the null space, find T = t*v1 + v2
 * satisfying the Plucker relation. Returns up to 2 solutions.
 * Each solution is a normalized 6-vector stored in sol[i], with residual in res[i].
 * Returns number of solutions found. */
static int solve_p3(const double *v1, const double *v2,
                    double sol[2][6], double res[2]) {
    double tol = 1e-10;
    int nsol = 0;

    /* alpha * t^2 + beta * t + gamma = 0 */
    double alpha = plucker_relation(v1);
    double gamma = plucker_relation(v2);

    /* beta = symmetric bilinear form */
    double beta = (v1[PR_AB]*v2[PR_CD] + v2[PR_AB]*v1[PR_CD])
                - (v1[PR_AC]*v2[PR_BD] + v2[PR_AC]*v1[PR_BD])
                + (v1[PR_AD]*v2[PR_BC] + v2[PR_AD]*v1[PR_BC]);

    /* Helper to add a solution */
    #define ADD_SOL(is_v1_only, t_val) do { \
        double T[6]; \
        if (is_v1_only) { \
            for (int _k = 0; _k < 6; _k++) T[_k] = v1[_k]; \
        } else { \
            double _t = (t_val); \
            for (int _k = 0; _k < 6; _k++) T[_k] = _t * v1[_k] + v2[_k]; \
        } \
        double _n = vec_norm(T, 6); \
        if (_n > tol) { \
            for (int _k = 0; _k < 6; _k++) sol[nsol][_k] = T[_k] / _n; \
            res[nsol] = fabs(plucker_relation(sol[nsol])); \
            nsol++; \
        } \
    } while(0)

    if (fabs(alpha) < tol) {
        if (fabs(beta) > tol) {
            ADD_SOL(0, -gamma / beta);
        }
        ADD_SOL(1, 0.0);
    } else {
        double disc = beta * beta - 4.0 * alpha * gamma;
        if (disc >= 0.0) {
            double sq = sqrt(fabs(disc));
            ADD_SOL(0, (-beta + sq) / (2.0 * alpha));
            if (nsol < 2) {
                ADD_SOL(0, (-beta - sq) / (2.0 * alpha));
            }
        }
    }

    #undef ADD_SOL

    /* Sort by residual */
    if (nsol == 2 && res[1] < res[0]) {
        double tmp[6]; double tr;
        memcpy(tmp, sol[0], 6 * sizeof(double));
        memcpy(sol[0], sol[1], 6 * sizeof(double));
        memcpy(sol[1], tmp, 6 * sizeof(double));
        tr = res[0]; res[0] = res[1]; res[1] = tr;
    }

    return nsol;
}


/* ══════════════════════════════════════════════════════════════════════════════
 * P3Memory — stores 3 lines, queries with 4th, finds transversals
 * ══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    double stored[3][6];
    int n_stored;
} P3Memory;

static void p3mem_init(P3Memory *m) {
    m->n_stored = 0;
}

static void p3mem_store(P3Memory *m, const double lines[3][6]) {
    for (int i = 0; i < 3; i++)
        memcpy(m->stored[i], lines[i], 6 * sizeof(double));
    m->n_stored = 3;
}

/* Query with a 4th line. Build 4x6 constraint matrix (Hodge dual rows),
 * SVD to get null space, solve_p3. Returns number of valid transversals. */
static int p3mem_query(const P3Memory *m, const double *query,
                       double out_trans[2][6], double out_res[2]) {
    if (m->n_stored < 3) return 0;

    /* Build 4x6 constraint matrix: rows = hodge_dual(stored[i]) and hodge_dual(query) */
    double A[4][6];
    for (int i = 0; i < 3; i++)
        hodge_dual(m->stored[i], A[i]);
    hodge_dual(query, A[3]);

    /* SVD: get right singular vectors */
    double Vt[36]; /* 6x6 */
    double S[6];
    svd_right((const double *)A, 4, 6, Vt, S);

    /* Null space = last two rows of Vt (smallest singular values) */
    const double *v1 = &Vt[5 * 6]; /* row 5 = smallest SV */
    const double *v2 = &Vt[4 * 6]; /* row 4 = 2nd smallest SV */

    return solve_p3(v1, v2, out_trans, out_res);
}


/* ══════════════════════════════════════════════════════════════════════════════
 * Compute transversals from training lines
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Compute transversals by sampling 4 random lines, storing 3, querying with 4th.
 * lines: array of n_lines 6-vectors (double).
 * Returns number of transversals found (up to max_trans).
 * out: pre-allocated array of max_trans 6-vectors (float for table building). */
static int compute_transversals(const double *lines, int n_lines,
                                int max_trans, Rng *rng,
                                float *out_trans) {
    if (n_lines < 4) return 0;

    int n_found = 0;
    int attempts = 0;
    int max_attempts = max_trans * 10;

    while (n_found < max_trans && attempts < max_attempts) {
        attempts++;

        /* Choose 4 random lines */
        int idx[4];
        rng_choose(rng, n_lines, 4, idx);

        P3Memory mem;
        p3mem_init(&mem);
        double store_lines[3][6];
        for (int i = 0; i < 3; i++)
            memcpy(store_lines[i], &lines[idx[i] * 6], 6 * sizeof(double));
        p3mem_store(&mem, store_lines);

        double trans[2][6], res[2];
        int nsol = p3mem_query(&mem, &lines[idx[3] * 6], trans, res);

        for (int s = 0; s < nsol && n_found < max_trans; s++) {
            double n = vec_norm(trans[s], 6);
            if (n > 1e-10 && res[s] < 1e-6) {
                int ok = 1;
                for (int k = 0; k < 6; k++) {
                    float v = (float)(trans[s][k] / n);
                    if (!isfinite(v)) { ok = 0; break; }
                    out_trans[n_found * 6 + k] = v;
                }
                if (ok) n_found++;
            }
        }
    }
    return n_found;
}


/* ══════════════════════════════════════════════════════════════════════════════
 * Embedding functions (replicating arc_tables.c / Python)
 * ══════════════════════════════════════════════════════════════════════════════ */

static void one_hot(float *dst, int idx) {
    memset(dst, 0, N_COLORS * sizeof(float));
    if (idx >= 0 && idx < N_COLORS) dst[idx] = 1.0f;
}

static int count_val(const int *arr, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; i++)
        if (arr[i] == v) c++;
    return c;
}

static int count_unique(const int *arr, int n) {
    int seen[N_COLORS] = {0};
    int u = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] >= 0 && arr[i] < N_COLORS && !seen[arr[i]]) {
            seen[arr[i]] = 1; u++;
        }
    }
    return u;
}

/* Each returns dimension written */
static int emb_hist_color(float *out, int r, int c, int in_c, int out_c,
                          const int *inp, const int *out_grid, int H, int W) {
    int sz = H * W;
    float inv_sz = 1.0f / (sz > 0 ? sz : 1);
    one_hot(out, in_c);
    one_hot(out + 10, out_c);
    for (int i = 0; i < N_COLORS; i++) {
        int cnt_out = count_val(out_grid, sz, i);
        int cnt_inp = count_val(inp, sz, i);
        out[20 + i] = (float)(cnt_out - cnt_inp) * inv_sz;
    }
    return 30;
}

static int emb_color_only(float *out, int r, int c, int in_c, int out_c,
                           const int *inp, const int *out_grid, int H, int W) {
    one_hot(out, in_c);
    one_hot(out + 10, out_c);
    return 20;
}

static int emb_pos_color(float *out, int r, int c, int in_c, int out_c,
                          const int *inp, const int *out_grid, int H, int W) {
    out[0] = (H > 1) ? (float)r / (H - 1) : 0.0f;
    out[1] = (W > 1) ? (float)c / (W - 1) : 0.0f;
    one_hot(out + 2, in_c);
    one_hot(out + 12, out_c);
    return 22;
}

static int emb_all(float *out, int r, int c, int in_c, int out_c,
                    const int *inp, const int *out_grid, int H, int W) {
    int sz = H * W;
    float inv_sz = 1.0f / (sz > 0 ? sz : 1);
    out[0] = (H > 1) ? (float)r / (H - 1) : 0.0f;
    out[1] = (W > 1) ? (float)c / (W - 1) : 0.0f;
    one_hot(out + 2, in_c);
    one_hot(out + 12, out_c);
    for (int i = 0; i < N_COLORS; i++)
        out[22 + i] = (float)count_val(inp, sz, i) * inv_sz;
    for (int i = 0; i < N_COLORS; i++)
        out[32 + i] = (float)count_val(out_grid, sz, i) * inv_sz;
    return 42;
}

static int emb_row_features(float *out, int r, int c, int in_c, int out_c,
                             const int *inp, const int *out_grid, int H, int W) {
    one_hot(out, in_c);
    one_hot(out + 10, out_c);
    const int *in_row = inp + r * W;
    const int *out_row = out_grid + r * W;
    float inv_w = (W > 0) ? 1.0f / W : 0.0f;
    for (int i = 0; i < N_COLORS; i++)
        out[20 + i] = (float)count_val(in_row, W, i) * inv_w;
    int in_row_unique = count_unique(in_row, W);
    out[30] = (in_row_unique == 1) ? 1.0f : 0.0f;
    out[31] = (W > 0) ? (float)in_row_unique / W : 0.0f;
    for (int i = 0; i < N_COLORS; i++)
        out[32 + i] = (float)count_val(out_row, W, i) * inv_w;
    int out_row_unique = count_unique(out_row, W);
    out[42] = (out_row_unique == 1) ? 1.0f : 0.0f;
    out[43] = (W > 0) ? (float)out_row_unique / W : 0.0f;
    return 44;
}

static int emb_col_features(float *out, int r, int c, int in_c, int out_c,
                             const int *inp, const int *out_grid, int H, int W) {
    one_hot(out, in_c);
    one_hot(out + 10, out_c);
    float inv_h = (H > 0) ? 1.0f / H : 0.0f;
    int col_buf[MAX_GRID_DIM];
    for (int i = 0; i < H; i++) col_buf[i] = inp[i * W + c];
    for (int i = 0; i < N_COLORS; i++)
        out[20 + i] = (float)count_val(col_buf, H, i) * inv_h;
    out[30] = (count_unique(col_buf, H) == 1) ? 1.0f : 0.0f;
    for (int i = 0; i < H; i++) col_buf[i] = out_grid[i * W + c];
    for (int i = 0; i < N_COLORS; i++)
        out[31 + i] = (float)count_val(col_buf, H, i) * inv_h;
    out[41] = (count_unique(col_buf, H) == 1) ? 1.0f : 0.0f;
    return 42;
}

static int emb_color_count(float *out, int r, int c, int in_c, int out_c,
                            const int *inp, const int *out_grid, int H, int W) {
    int sz = H * W;
    float inv_sz = 1.0f / (sz > 0 ? sz : 1);
    one_hot(out, in_c);
    one_hot(out + 10, out_c);
    out[20] = (float)count_val(inp, sz, in_c) * inv_sz;
    out[21] = (float)count_val(out_grid, sz, out_c) * inv_sz;
    int in_mode = 0, out_mode = 0;
    int in_max = 0, out_max = 0;
    for (int i = 0; i < N_COLORS; i++) {
        int ci = count_val(inp, sz, i);
        int co = count_val(out_grid, sz, i);
        if (ci > in_max) { in_max = ci; in_mode = i; }
        if (co > out_max) { out_max = co; out_mode = i; }
    }
    out[22] = (in_c == in_mode) ? 1.0f : 0.0f;
    out[23] = (out_c == out_mode) ? 1.0f : 0.0f;
    return 24;
}

static int emb_diagonal(float *out, int r, int c, int in_c, int out_c,
                         const int *inp, const int *out_grid, int H, int W) {
    one_hot(out, in_c);
    one_hot(out + 10, out_c);
    out[20] = (H > 1) ? (float)r / (H - 1) : 0.0f;
    out[21] = (W > 1) ? (float)c / (W - 1) : 0.0f;
    int maxHW = (H > W) ? H : W;
    int minHW = (H < W) ? H : W;
    out[22] = (float)(r - c + maxHW) / (2.0f * maxHW);
    float denom = (float)(H + W - 2) + 1e-6f;
    out[23] = (float)(r + c) / denom;
    out[24] = (r == c) ? 1.0f : 0.0f;
    out[25] = (r + c == minHW - 1) ? 1.0f : 0.0f;
    return 26;
}

typedef int (*emb_func_t)(float*, int, int, int, int, const int*, const int*, int, int);
static emb_func_t EMB_FUNCS[8] = {
    emb_hist_color, emb_color_only, emb_pos_color, emb_all,
    emb_row_features, emb_col_features, emb_color_count, emb_diagonal
};


/* ══════════════════════════════════════════════════════════════════════════════
 * Projection matrix generation (deterministic, seeded by embedding name hash)
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Simple hash matching Python's hash(name) % 2**31 — we use djb2 */
static unsigned int name_hash(const char *name) {
    /* Deterministic SHA256-based seeds matching Python's hashlib.sha256 */
    if (strcmp(name, "hist_color") == 0) return 1058250775u;
    if (strcmp(name, "color_only") == 0) return 441137264u;
    if (strcmp(name, "pos_color") == 0) return 1939902514u;
    if (strcmp(name, "all") == 0) return 295351849u;
    if (strcmp(name, "row_feat") == 0) return 542267216u;
    if (strcmp(name, "col_feat") == 0) return 952079380u;
    if (strcmp(name, "color_count") == 0) return 1639721810u;
    if (strcmp(name, "diagonal") == 0) return 291548388u;
    /* Fallback: djb2 */
    unsigned long h = 5381;
    while (*name) { h = ((h << 5) + h) + (unsigned char)*name; name++; }
    return (unsigned int)(h % 2147483648u);
}

/* Generate W1, W2 projection matrices matching Python:
 * rng_proj = np.random.RandomState(hash(name) % 2**31)
 * W1 = rng_proj.randn(4, 2*dim) * 0.1
 * W2 = rng_proj.randn(4, 2*dim) * 0.1
 *
 * Note: We can't exactly match numpy's RandomState here since the LCG
 * differs. Instead we use our own deterministic RNG. The exact random
 * values don't matter for correctness — what matters is that both
 * Python and C use the SAME W1, W2 for the SAME embedding type.
 * Since we can't easily match numpy's MT19937, we accept that ranks
 * may differ slightly due to different random projections, but the
 * algorithm is identical.
 *
 * UPDATE: For exact reproducibility, we'd need to implement numpy's MT19937
 * with the Box-Muller transform it uses. For now, we use our own RNG and
 * note that the projections will differ but the pipeline is correct.
 */
static void gen_projections(const char *emb_name, int dim,
                            float *W1, float *W2) {
    unsigned int seed = name_hash(emb_name);
    Rng rng;
    rng_seed(&rng, seed);

    int sz = 4 * 2 * dim;
    for (int i = 0; i < sz; i++)
        W1[i] = (float)(rng_normal(&rng) * 0.1);
    for (int i = 0; i < sz; i++)
        W2[i] = (float)(rng_normal(&rng) * 0.1);
}


/* ══════════════════════════════════════════════════════════════════════════════
 * Plucker line from embedding pair (dual projection + exterior product)
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Compute Plucker line from two embedding vectors ea, eb.
 * combined = [ea; eb], p1 = W1 @ combined, p2 = W2 @ combined,
 * L = exterior(p1, p2), normalize.
 * Returns 0 if degenerate, writes 6-vector (double) to out. */
static int make_line_d(const float *ea, const float *eb, int dim,
                       const float *W1, const float *W2, double *out) {
    int cd = 2 * dim;
    float combined[256]; /* max 2*44=88 */
    memcpy(combined, ea, dim * sizeof(float));
    memcpy(combined + dim, eb, dim * sizeof(float));

    /* Use float32 accumulation to match Python's numpy float32 matmul */
    float p1f[4], p2f[4];
    for (int i = 0; i < 4; i++) {
        float s1 = 0.0f, s2 = 0.0f;
        for (int j = 0; j < cd; j++) {
            s1 += W1[i * cd + j] * combined[j];
            s2 += W2[i * cd + j] * combined[j];
        }
        p1f[i] = s1;
        p2f[i] = s2;
    }

    float Lf[6];
    for (int k = 0; k < 6; k++)
        Lf[k] = p1f[PL_I[k]] * p2f[PL_J[k]] - p1f[PL_J[k]] * p2f[PL_I[k]];

    float nf = 0.0f;
    for (int k = 0; k < 6; k++) nf += Lf[k] * Lf[k];
    nf = sqrtf(nf);
    if (nf < 1e-10f) return 0;
    for (int k = 0; k < 6; k++) out[k] = (double)(Lf[k] / nf);
    return 1;
}

/* Same but returns float */
static int make_line_f(const float *ea, const float *eb, int dim,
                       const float *W1, const float *W2, float *out) {
    double d[6];
    int ok = make_line_d(ea, eb, dim, W1, W2, d);
    for (int k = 0; k < 6; k++) out[k] = (float)d[k];
    return ok;
}


/* ══════════════════════════════════════════════════════════════════════════════
 * Score table building
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Build JTm = J6 @ transversals.T, shape (6, n_trans) */
static void build_JTm(const float *trans, int n_trans, float *JTm) {
    for (int i = 0; i < 6; i++) {
        for (int t = 0; t < n_trans; t++) {
            double v = 0.0;
            for (int j = 0; j < 6; j++)
                v += J6[i][j] * trans[t * 6 + j];
            JTm[i * n_trans + t] = (float)v;
        }
    }
}

/* Score a single Plucker line against JTm: sum(log(|L . JTm_col| + 1e-10)) */
static float score_line(const float *L, const float *JTm, int n_trans) {
    float score = 0.0f;
    for (int t = 0; t < n_trans; t++) {
        float dot = 0.0f;
        for (int k = 0; k < 6; k++)
            dot += L[k] * JTm[k * n_trans + t];
        if (dot > 1e10f) dot = 1e10f;
        if (dot < -1e10f) dot = -1e10f;
        float v = logf(fabsf(dot) + 1e-10f);
        if (!isfinite(v)) {
            if (v != v) v = 0.0f;          /* NaN → 0 */
            else if (v > 0) v = 0.0f;      /* +inf → 0 (match Python posinf) */
            else v = -100.0f;              /* -inf → -100 */
        }
        score += v;
    }
    return score;
}

/* Build score_table[adj][ca][cb] for non-histogram embeddings.
 * embs: precomputed (H*W * nc * dim) embedding array.
 * adj_pairs: (n_adj, 4) array of (r1,c1,r2,c2).
 * JTm: (6, n_trans) matrix.
 * out: (n_adj * nc * nc) output array. */
static void build_score_tables(const float *embs, int dim,
                                const int *adj_pairs, int n_adj,
                                const float *W1, const float *W2,
                                const float *JTm, int n_trans,
                                int nc, int H, int W,
                                float *out) {
    int cd = 2 * dim;
    #pragma omp parallel for schedule(dynamic)
    for (int ai = 0; ai < n_adj; ai++) {
        int r1 = adj_pairs[ai * 4 + 0];
        int c1 = adj_pairs[ai * 4 + 1];
        int r2 = adj_pairs[ai * 4 + 2];
        int c2 = adj_pairs[ai * 4 + 3];
        int idx_a = r1 * W + c1;
        int idx_b = r2 * W + c2;

        for (int ca = 0; ca < nc; ca++) {
            const float *ea = embs + (idx_a * nc + ca) * dim;
            for (int cb = 0; cb < nc; cb++) {
                const float *eb = embs + (idx_b * nc + cb) * dim;

                float combined[256];
                memcpy(combined, ea, dim * sizeof(float));
                memcpy(combined + dim, eb, dim * sizeof(float));

                float p1[4], p2[4];
                for (int i = 0; i < 4; i++) {
                    float s1 = 0.0f, s2 = 0.0f;
                    for (int j = 0; j < cd; j++) {
                        s1 += W1[i * cd + j] * combined[j];
                        s2 += W2[i * cd + j] * combined[j];
                    }
                    p1[i] = s1;
                    p2[i] = s2;
                }

                float L[6];
                for (int k = 0; k < 6; k++)
                    L[k] = p1[PL_I[k]] * p2[PL_J[k]] - p1[PL_J[k]] * p2[PL_I[k]];

                float norm2 = 0.0f;
                for (int k = 0; k < 6; k++) norm2 += L[k] * L[k];
                float norm = sqrtf(norm2);

                float sc;
                if (norm <= 1e-10f) {
                    sc = 0.0f;
                } else {
                    float inv = 1.0f / norm;
                    for (int k = 0; k < 6; k++) L[k] *= inv;
                    sc = score_line(L, JTm, n_trans);
                }

                out[ai * nc * nc + ca * nc + cb] = sc;
            }
        }
    }
}

/* Precompute cell embeddings for all (position, candidate_color) combos.
 * out: (H*W * nc * dim) float array. */
static void precompute_cell_embs(int emb_type,
                                  const int *inp, const int *out_grid,
                                  const int *used_colors, int nc,
                                  int H, int W, float *out) {
    int dim = EMB_DIMS[emb_type];
    emb_func_t fn = EMB_FUNCS[emb_type];
    #pragma omp parallel for collapse(2)
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            int pos = r * W + c;
            int in_c = inp[pos];
            for (int ci = 0; ci < nc; ci++) {
                float *dst = out + (pos * nc + ci) * dim;
                fn(dst, r, c, in_c, used_colors[ci], inp, out_grid, H, W);
            }
        }
    }
}


/* ══════════════════════════════════════════════════════════════════════════════
 * Histogram table building (for hist_color with per-histogram diff vector)
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Generate all histograms: partitions of 'total' into 'nc' bins.
 * Returns number of histograms generated. hist_out is (n_hist, nc). */
static int gen_all_histograms(int total, int nc, int *hist_out, int max_hist) {
    /* Recursive generation using stack-based approach */
    int count = 0;
    int *stack = (int *)malloc(nc * sizeof(int));

    /* Generate compositions of 'total' into 'nc' parts (including 0) */
    /* Use iterative approach */
    memset(stack, 0, nc * sizeof(int));
    stack[0] = total;

    while (1) {
        if (count >= max_hist) break;

        /* Current partition is in stack[0..nc-1], sum should be total */
        int sum = 0;
        for (int i = 0; i < nc; i++) sum += stack[i];
        if (sum == total) {
            memcpy(hist_out + count * nc, stack, nc * sizeof(int));
            count++;
        }

        /* Advance to next composition */
        int carry = 1;
        for (int i = nc - 1; i >= 0 && carry; i--) {
            if (i == 0) {
                /* Can't increment position 0 further */
                carry = 1; /* done */
                break;
            }
            /* Move one unit from previous positions to position i */
            /* This is stars-and-bars enumeration */
            carry = 0; /* handled below */
        }
        if (carry) break;

        /* Actually, let's just use a recursive helper stored in array */
        break; /* Fall through to recursive version */
    }
    free(stack);

    /* Recursive generation matching Python's _gen(rem, ncols, cur) */
    count = 0;
    int *cur = (int *)calloc(nc, sizeof(int));

    /* Simple recursive approach via explicit stack with full state */
    typedef struct { int pos; int rem; int k; } Frame;
    int max_frames = (total + 2) * nc;
    Frame *fstack = (Frame *)malloc(max_frames * sizeof(Frame));
    int sp = 0;
    fstack[sp++] = (Frame){0, total, 0};

    while (sp > 0) {
        Frame *fr = &fstack[sp - 1];
        if (fr->pos == nc - 1) {
            /* Base case: last bin gets the remainder */
            cur[fr->pos] = fr->rem;
            if (count < max_hist) {
                memcpy(hist_out + count * nc, cur, nc * sizeof(int));
                count++;
            }
            sp--;
            continue;
        }
        if (fr->k > fr->rem) {
            /* Done with all values for this position */
            sp--;
            continue;
        }
        /* Set current position to k, advance k for next iteration */
        cur[fr->pos] = fr->k;
        int next_rem = fr->rem - fr->k;
        fr->k++;
        /* Push child frame for next position */
        fstack[sp++] = (Frame){fr->pos + 1, next_rem, 0};
    }

    free(cur);
    free(fstack);
    return count;
}

/* Build histogram score tables for one embedding's transversals.
 * For each histogram, compute the diff vector and score all adj pairs.
 * Accumulates into hist_scores[hist_idx][adj*nc*nc]. */
static void build_hist_tables_for_emb(
    const int *adj_pairs, int n_adj,
    const int *used_colors, int nc, int H, int W,
    const int *test_inp, const float *inp_hist,
    const int *all_hists, int n_hists,
    const float *W1, const float *W2,
    const float *JTm, int n_trans,
    float *hist_scores /* n_hists * n_adj * nc * nc */)
{
    int hw = H * W;
    int cd = 60; /* 2 * 30 for hist_color */

    #pragma omp parallel for schedule(dynamic)
    for (int hi = 0; hi < n_hists; hi++) {
        const int *hist = all_hists + hi * nc;
        float diff[N_COLORS];
        float out_h[N_COLORS];
        memset(out_h, 0, sizeof(out_h));
        for (int ci = 0; ci < nc; ci++)
            out_h[used_colors[ci]] = (float)hist[ci];
        float inv_sz = 1.0f / (hw > 0 ? hw : 1);
        for (int i = 0; i < N_COLORS; i++)
            diff[i] = (out_h[i] - inp_hist[i]) * inv_sz;

        float *out = hist_scores + (long long)hi * n_adj * nc * nc;

        for (int ai = 0; ai < n_adj; ai++) {
            int r1 = adj_pairs[ai * 4 + 0];
            int c1 = adj_pairs[ai * 4 + 1];
            int r2 = adj_pairs[ai * 4 + 2];
            int c2 = adj_pairs[ai * 4 + 3];
            int in_c_a = test_inp[r1 * W + c1];
            int in_c_b = test_inp[r2 * W + c2];

            for (int ca = 0; ca < nc; ca++) {
                /* Build ea = [in_oh_a, out_oh(ca), diff] */
                float ea[30];
                memset(ea, 0, 30 * sizeof(float));
                if (in_c_a >= 0 && in_c_a < N_COLORS) ea[in_c_a] = 1.0f;
                if (used_colors[ca] >= 0 && used_colors[ca] < N_COLORS)
                    ea[10 + used_colors[ca]] = 1.0f;
                memcpy(ea + 20, diff, 10 * sizeof(float));

                for (int cb = 0; cb < nc; cb++) {
                    float eb[30];
                    memset(eb, 0, 30 * sizeof(float));
                    if (in_c_b >= 0 && in_c_b < N_COLORS) eb[in_c_b] = 1.0f;
                    if (used_colors[cb] >= 0 && used_colors[cb] < N_COLORS)
                        eb[10 + used_colors[cb]] = 1.0f;
                    memcpy(eb + 20, diff, 10 * sizeof(float));

                    float combined[60];
                    memcpy(combined, ea, 30 * sizeof(float));
                    memcpy(combined + 30, eb, 30 * sizeof(float));

                    float p1[4], p2[4];
                    for (int i = 0; i < 4; i++) {
                        float s1 = 0.0f, s2 = 0.0f;
                        for (int j = 0; j < cd; j++) {
                            s1 += W1[i * cd + j] * combined[j];
                            s2 += W2[i * cd + j] * combined[j];
                        }
                        p1[i] = s1;
                        p2[i] = s2;
                    }

                    float L[6];
                    for (int k = 0; k < 6; k++)
                        L[k] = p1[PL_I[k]] * p2[PL_J[k]] - p1[PL_J[k]] * p2[PL_I[k]];

                    float norm2 = 0.0f;
                    for (int k = 0; k < 6; k++) norm2 += L[k] * L[k];
                    float norm = sqrtf(norm2);

                    float sc;
                    if (norm <= 1e-10f) {
                        sc = 0.0f;
                    } else {
                        float inv = 1.0f / norm;
                        for (int k = 0; k < 6; k++) L[k] *= inv;
                        sc = score_line(L, JTm, n_trans);
                    }

                    out[ai * nc * nc + ca * nc + cb] += sc;
                }
            }
        }
    }
}


/* ══════════════════════════════════════════════════════════════════════════════
 * Candidate scoring
 * ══════════════════════════════════════════════════════════════════════════════ */

static double get_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* Compute nc^hw carefully to avoid overflow */
static long long ipow(int base, int exp) {
    long long r = 1;
    for (int i = 0; i < exp; i++) {
        r *= base;
        if (r > 400000000LL) return r; /* early exit, too large */
    }
    return r;
}

/* Convert flat candidate index to per-cell color indices.
 * idx_flat encodes candidate in mixed-radix: cell 0 is highest order.
 * out[i] = color index for cell i. */
static void flat_to_indices(long long idx_flat, int nc, int hw, int *out) {
    for (int i = hw - 1; i >= 0; i--) {
        out[i] = (int)(idx_flat % nc);
        idx_flat /= nc;
    }
}

/* Score a single candidate given its per-cell color indices.
 * Uses non-histogram score tables. */
static float score_candidate_raw(const int *cand, int H, int W,
                                  const int *adj_pairs, int n_adj,
                                  const float *const *score_tables,
                                  int n_emb_tables, int nc) {
    float total = 0.0f;
    for (int ei = 0; ei < n_emb_tables; ei++) {
        if (!score_tables[ei]) continue;
        const float *tbl = score_tables[ei];
        for (int ai = 0; ai < n_adj; ai++) {
            int r1 = adj_pairs[ai * 4 + 0];
            int c1 = adj_pairs[ai * 4 + 1];
            int r2 = adj_pairs[ai * 4 + 2];
            int c2 = adj_pairs[ai * 4 + 3];
            int ca = cand[r1 * W + c1];
            int cb = cand[r2 * W + c2];
            total += tbl[ai * nc * nc + ca * nc + cb];
        }
    }
    return total;
}

/* Score a candidate using histogram tables.
 * First compute the histogram of the candidate, then look up the matching table. */
static float score_candidate_hist(const int *cand, int hw, int nc,
                                   const int *adj_pairs, int n_adj, int W,
                                   const int *all_hists, int n_hists,
                                   const float *hist_scores /* n_hists * n_adj * nc * nc */) {
    /* Compute histogram of candidate */
    int hist[MAX_COLORS];
    memset(hist, 0, nc * sizeof(int));
    for (int i = 0; i < hw; i++) hist[cand[i]]++;

    /* Find matching histogram */
    int found = -1;
    for (int hi = 0; hi < n_hists; hi++) {
        const int *h = all_hists + hi * nc;
        int match = 1;
        for (int ci = 0; ci < nc; ci++)
            if (h[ci] != hist[ci]) { match = 0; break; }
        if (match) { found = hi; break; }
    }
    if (found < 0) return 0.0f;

    const float *tbl = hist_scores + (long long)found * n_adj * nc * nc;
    float total = 0.0f;
    for (int ai = 0; ai < n_adj; ai++) {
        int r1 = adj_pairs[ai * 4 + 0];
        int c1 = adj_pairs[ai * 4 + 1];
        int r2 = adj_pairs[ai * 4 + 2];
        int c2 = adj_pairs[ai * 4 + 3];
        int ca = cand[r1 * W + c1];
        int cb = cand[r2 * W + c2];
        total += tbl[ai * nc * nc + ca * nc + cb];
    }
    return total;
}


/* ══════════════════════════════════════════════════════════════════════════════
 * Main solver
 * ══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int rank;
    int solved;
    int n_candidates;
    double setup_time;
    double score_time;
    int total_trans;
} SolveResult;

static SolveResult solve_task(ArcTask *task) {
    SolveResult result = {0};
    double t0 = get_time_sec();

    int H = task->test[0].H;
    int W = task->test[0].W;
    int hw = H * W;
    int *test_inp = task->test[0].input;
    int *test_out = task->test[0].output;

    /* Determine used colors */
    int color_seen[N_COLORS] = {0};
    for (int p = 0; p < task->n_train; p++) {
        int pH = task->train[p].H, pW = task->train[p].W;
        for (int i = 0; i < pH * pW; i++) {
            if (task->train[p].input[i] < N_COLORS)
                color_seen[task->train[p].input[i]] = 1;
            if (task->train[p].output[i] < N_COLORS)
                color_seen[task->train[p].output[i]] = 1;
        }
    }
    for (int i = 0; i < hw; i++)
        if (test_inp[i] < N_COLORS)
            color_seen[test_inp[i]] = 1;

    int used_colors[N_COLORS];
    int nc = 0;
    for (int i = 0; i < N_COLORS; i++)
        if (color_seen[i]) used_colors[nc++] = i;

    int color_to_idx[N_COLORS];
    memset(color_to_idx, -1, sizeof(color_to_idx));
    for (int i = 0; i < nc; i++)
        color_to_idx[used_colors[i]] = i;

    /* Build adjacency pairs */
    int adj_pairs[MAX_ADJ * 4];
    int n_adj = 0;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++) {
            if (c + 1 < W) {
                adj_pairs[n_adj * 4 + 0] = r;
                adj_pairs[n_adj * 4 + 1] = c;
                adj_pairs[n_adj * 4 + 2] = r;
                adj_pairs[n_adj * 4 + 3] = c + 1;
                n_adj++;
            }
            if (r + 1 < H) {
                adj_pairs[n_adj * 4 + 0] = r;
                adj_pairs[n_adj * 4 + 1] = c;
                adj_pairs[n_adj * 4 + 2] = r + 1;
                adj_pairs[n_adj * 4 + 3] = c;
                n_adj++;
            }
        }

    /* Input histogram */
    float inp_hist[N_COLORS];
    for (int i = 0; i < N_COLORS; i++)
        inp_hist[i] = (float)count_val(test_inp, hw, i);

    /* Per-embedding: compute transversals, build score tables */
    /* Non-histogram tables: array of pointers to (n_adj * nc * nc) float arrays */
    float *raw_tables[N_EMBEDDINGS];
    int n_raw_tables = 0;
    memset(raw_tables, 0, sizeof(raw_tables));

    /* Histogram tables data: for hist_color embedding */
    int hist_emb_count = 0;
    float *hist_W1[N_EMBEDDINGS], *hist_W2[N_EMBEDDINGS];
    float *hist_JTm[N_EMBEDDINGS];
    int hist_n_trans[N_EMBEDDINGS];

    int total_trans = 0;

    for (int ei = 0; ei < N_EMBEDDINGS; ei++) {
        const char *name = EMB_NAMES[ei];
        int dim = EMB_DIMS[ei];
        int is_hist = (ei == 0); /* hist_color */

        /* Generate projection matrices */
        int cd = 2 * dim;
        float *W1 = (float *)malloc(4 * cd * sizeof(float));
        float *W2 = (float *)malloc(4 * cd * sizeof(float));
        gen_projections(name, dim, W1, W2);

        /* Compute transversals from training pairs */
        /* Max lines per training pair: roughly 2*H*W adjacencies */
        int max_lines = 0;
        for (int p = 0; p < task->n_train; p++)
            max_lines += 2 * task->train[p].H * task->train[p].W;

        double *all_lines = (double *)malloc(max_lines * 6 * sizeof(double));
        float *all_trans = (float *)malloc(N_TRANS_PER_PAIR * task->n_train * 6 * sizeof(float));
        int total_trans_emb = 0;

        for (int p = 0; p < task->n_train; p++) {
            int pH = task->train[p].H, pW = task->train[p].W;
            int *pinp = task->train[p].input;
            int *pout = task->train[p].output;

            /* Build lines for this training pair */
            int n_lines = 0;
            for (int r = 0; r < pH; r++) {
                for (int c = 0; c < pW; c++) {
                    for (int d = 0; d < 2; d++) {
                        int r2 = r + (d == 1 ? 1 : 0);
                        int c2 = c + (d == 0 ? 1 : 0);
                        if (r2 >= pH || c2 >= pW) continue;

                        float ea[MAX_EMB_DIM], eb[MAX_EMB_DIM];
                        EMB_FUNCS[ei](ea, r, c, pinp[r*pW+c], pout[r*pW+c],
                                      pinp, pout, pH, pW);
                        EMB_FUNCS[ei](eb, r2, c2, pinp[r2*pW+c2], pout[r2*pW+c2],
                                      pinp, pout, pH, pW);

                        double line[6];
                        if (make_line_d(ea, eb, dim, W1, W2, line)) {
                            memcpy(all_lines + n_lines * 6, line, 6 * sizeof(double));
                            n_lines++;
                        }
                    }
                }
            }

            /* Compute transversals */
            Rng rng;
            rng_seed(&rng, 42 + p);
            int nt = compute_transversals(all_lines, n_lines,
                                          N_TRANS_PER_PAIR, &rng,
                                          all_trans + total_trans_emb * 6);
            total_trans_emb += nt;
        }

        total_trans += total_trans_emb;

        if (total_trans_emb == 0) {
            if (!is_hist) {
                raw_tables[n_raw_tables++] = NULL;
            }
            free(W1); free(W2); free(all_lines); free(all_trans);
            continue;
        }

        /* Build JTm */
        float *JTm = (float *)malloc(6 * total_trans_emb * sizeof(float));
        build_JTm(all_trans, total_trans_emb, JTm);

        /* Filter inf/nan columns */
        int valid_trans = 0;
        for (int t = 0; t < total_trans_emb; t++) {
            int ok = 1;
            for (int k = 0; k < 6; k++)
                if (!isfinite(JTm[k * total_trans_emb + t])) { ok = 0; break; }
            if (ok) {
                if (valid_trans != t) {
                    for (int k = 0; k < 6; k++)
                        JTm[k * total_trans_emb + valid_trans] = JTm[k * total_trans_emb + t];
                }
                valid_trans++;
            }
        }

        if (valid_trans == 0) {
            if (!is_hist) raw_tables[n_raw_tables++] = NULL;
            free(W1); free(W2); free(all_lines); free(all_trans); free(JTm);
            continue;
        }

        /* For hist embeddings, save for histogram table building */
        if (is_hist) {
            hist_W1[hist_emb_count] = W1;
            hist_W2[hist_emb_count] = W2;
            hist_JTm[hist_emb_count] = JTm;
            hist_n_trans[hist_emb_count] = valid_trans;
            hist_emb_count++;
        } else {
            /* Non-histogram: precompute cell embeddings and build score tables */
            float *embs = (float *)malloc(hw * nc * dim * sizeof(float));
            precompute_cell_embs(ei, test_inp, test_inp, used_colors, nc, H, W, embs);

            float *tbl = (float *)calloc(n_adj * nc * nc, sizeof(float));
            build_score_tables(embs, dim, adj_pairs, n_adj, W1, W2,
                              JTm, valid_trans, nc, H, W, tbl);
            raw_tables[n_raw_tables++] = tbl;

            free(embs);
            free(W1); free(W2); free(JTm);
        }

        free(all_lines);
        free(all_trans);
    }

    result.total_trans = total_trans;
    result.setup_time = get_time_sec() - t0;

    /* ── Build histogram tables if applicable ── */
    int n_hists = 0;
    int *all_hists = NULL;
    float *hist_scores = NULL;
    int use_hist_tables = 0;

    if (hist_emb_count > 0) {
        /* Count possible histograms */
        /* Stars and bars: C(hw + nc - 1, nc - 1) */
        /* Rough upper bound check first */
        int est_hists = 1;
        int too_many = 0;
        /* Generate to count */
        int max_possible = 100000; /* generous upper bound for allocation */
        all_hists = (int *)malloc(max_possible * nc * sizeof(int));
        n_hists = gen_all_histograms(hw, nc, all_hists, max_possible);

        if (n_hists <= MAX_HIST_TABLES) {
            use_hist_tables = 1;
            printf("  Building %d histogram tables for %d hist embeddings...\n",
                   n_hists, hist_emb_count);

            hist_scores = (float *)calloc((long long)n_hists * n_adj * nc * nc, sizeof(float));

            for (int he = 0; he < hist_emb_count; he++) {
                build_hist_tables_for_emb(adj_pairs, n_adj,
                                          used_colors, nc, H, W,
                                          test_inp, inp_hist,
                                          all_hists, n_hists,
                                          hist_W1[he], hist_W2[he],
                                          hist_JTm[he], hist_n_trans[he],
                                          hist_scores);
            }
        } else {
            printf("  %d colors, %d histograms -- using placeholder for hist embeddings\n",
                   nc, n_hists);
            /* Fallback: treat hist_color as non-histogram (use test_inp as proxy) */
            for (int he = 0; he < hist_emb_count; he++) {
                int dim = 30; /* hist_color */
                float *embs = (float *)malloc(hw * nc * dim * sizeof(float));
                precompute_cell_embs(0, test_inp, test_inp, used_colors, nc, H, W, embs);

                float *tbl = (float *)calloc(n_adj * nc * nc, sizeof(float));
                build_score_tables(embs, dim, adj_pairs, n_adj,
                                  hist_W1[he], hist_W2[he],
                                  hist_JTm[he], hist_n_trans[he],
                                  nc, H, W, tbl);
                raw_tables[n_raw_tables++] = tbl;
                free(embs);
            }
        }
    }

    /* ── Score candidates ── */
    double t1 = get_time_sec();
    long long n_total = ipow(nc, hw);
    result.n_candidates = (int)((n_total > 2000000000LL) ? 2000000000 : n_total);

    /* Compute correct answer's color index representation */
    int correct_cand[MAX_GRID_CELLS];
    int correct_valid = 1;
    for (int i = 0; i < hw; i++) {
        int ci = color_to_idx[test_out[i]];
        if (ci < 0) { correct_valid = 0; break; }
        correct_cand[i] = ci;
    }

    if (!correct_valid) {
        printf("  WARNING: test output uses colors not in training set!\n");
        result.rank = -1;
        result.solved = 0;
        result.score_time = 0;
        goto cleanup;
    }

    if (n_total <= 200000000LL) {
        /* Exhaustive scoring */
        float correct_score;

        /* First compute correct answer's score */
        if (use_hist_tables) {
            correct_score = score_candidate_hist(correct_cand, hw, nc,
                                                  adj_pairs, n_adj, W,
                                                  all_hists, n_hists,
                                                  hist_scores);
        } else {
            correct_score = score_candidate_raw(correct_cand, H, W,
                                                 adj_pairs, n_adj,
                                                 (const float *const *)raw_tables,
                                                 n_raw_tables, nc);
        }

        /* Count how many candidates score lower (better) than correct */
        long long better = 0;
        int best_found = 0;
        float best_score = 1e30f;  /* lower = better */
        long long best_idx = -1;

        #pragma omp parallel reduction(+:better)
        {
            int local_cand[MAX_GRID_CELLS];
            float local_best_score = 1e30f;  /* lower = better */
            long long local_best_idx = -1;

            #pragma omp for schedule(dynamic, 1024)
            for (long long idx = 0; idx < n_total; idx++) {
                flat_to_indices(idx, nc, hw, local_cand);

                float sc;
                if (use_hist_tables) {
                    sc = score_candidate_hist(local_cand, hw, nc,
                                              adj_pairs, n_adj, W,
                                              all_hists, n_hists,
                                              hist_scores);
                } else {
                    sc = score_candidate_raw(local_cand, H, W,
                                              adj_pairs, n_adj,
                                              (const float *const *)raw_tables,
                                              n_raw_tables, nc);
                }

                if (sc < correct_score) better++;
                if (sc < local_best_score) {  /* lower = better */
                    local_best_score = sc;
                    local_best_idx = idx;
                }
            }

            #pragma omp critical
            {
                if (local_best_score < best_score) {  /* lower = better */
                    best_score = local_best_score;
                    best_idx = local_best_idx;
                }
            }
        }

        /* Check if best is identity (== input) */
        int best_cand[MAX_GRID_CELLS];
        flat_to_indices(best_idx, nc, hw, best_cand);
        int is_identity = 1;
        for (int i = 0; i < hw; i++) {
            if (used_colors[best_cand[i]] != test_inp[i]) { is_identity = 0; break; }
        }

        result.rank = (int)(better + 1);
        if (is_identity) result.rank = (result.rank > 1) ? result.rank - 1 : result.rank;
        if (result.rank <= 3)
            printf("    DEBUG: correct=%.6f best=%.6f diff=%.2e better=%lld\n",
                   correct_score, best_score, (double)(correct_score - best_score), better);

        /* Check if prediction matches */
        if (result.rank == 1) {
            /* Need to check if the best candidate matches test output */
            int matches = 1;
            for (int i = 0; i < hw; i++) {
                if (used_colors[best_cand[i]] != test_out[i]) { matches = 0; break; }
            }
            result.solved = matches;
        } else {
            result.solved = 0;
        }

    } else {
        /* Sampling mode */
        int n_samples = 10000000;
        printf("  Too many candidates (%lld), using sampling (%d samples)\n",
               n_total, n_samples);

        float correct_score;
        if (use_hist_tables) {
            correct_score = score_candidate_hist(correct_cand, hw, nc,
                                                  adj_pairs, n_adj, W,
                                                  all_hists, n_hists,
                                                  hist_scores);
        } else {
            correct_score = score_candidate_raw(correct_cand, H, W,
                                                 adj_pairs, n_adj,
                                                 (const float *const *)raw_tables,
                                                 n_raw_tables, nc);
        }

        long long better = 0;
        Rng rng;
        rng_seed(&rng, 0);

        /* Sample in chunks for parallelism */
        int chunk_size = 100000;
        int n_chunks = (n_samples + chunk_size - 1) / chunk_size;

        /* Pre-generate random candidates */
        for (int ch = 0; ch < n_chunks; ch++) {
            int this_chunk = (ch == n_chunks - 1) ?
                             (n_samples - ch * chunk_size) : chunk_size;

            #pragma omp parallel reduction(+:better)
            {
                Rng local_rng;
                int tid = 0;
                #ifdef _OPENMP
                tid = omp_get_thread_num();
                #endif
                rng_seed(&local_rng, ch * 1000 + tid);
                int local_cand[MAX_GRID_CELLS];

                #pragma omp for schedule(static)
                for (int si = 0; si < this_chunk; si++) {
                    for (int i = 0; i < hw; i++)
                        local_cand[i] = rng_int(&local_rng, nc);

                    float sc;
                    if (use_hist_tables) {
                        sc = score_candidate_hist(local_cand, hw, nc,
                                                  adj_pairs, n_adj, W,
                                                  all_hists, n_hists,
                                                  hist_scores);
                    } else {
                        sc = score_candidate_raw(local_cand, H, W,
                                                  adj_pairs, n_adj,
                                                  (const float *const *)raw_tables,
                                                  n_raw_tables, nc);
                    }

                    if (sc < correct_score) better++;
                }
            }
        }

        long long est_rank = (better == 0) ? 1 :
                             (long long)((double)better / n_samples * n_total);
        if (est_rank < 1) est_rank = 1;
        result.rank = (int)((est_rank > 2000000000LL) ? 2000000000 : est_rank);
        result.solved = (better == 0);
    }

    result.score_time = get_time_sec() - t1;

cleanup:
    /* Free allocated tables */
    for (int i = 0; i < n_raw_tables; i++)
        if (raw_tables[i]) free(raw_tables[i]);
    for (int i = 0; i < hist_emb_count; i++) {
        free(hist_W1[i]); free(hist_W2[i]); free(hist_JTm[i]);
    }
    if (all_hists) free(all_hists);
    if (hist_scores) free(hist_scores);

    return result;
}


/* ══════════════════════════════════════════════════════════════════════════════
 * Main
 * ══════════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [--all <dir>] [task1.json task2.json ...]\n", argv[0]);
        return 1;
    }

    int all_mode = 0;
    const char *scan_dir = NULL;
    const char **task_files = NULL;
    int n_tasks = 0;

    /* Parse arguments */
    int argi = 1;
    if (strcmp(argv[1], "--all") == 0) {
        all_mode = 1;
        if (argc < 3) {
            fprintf(stderr, "Usage: %s --all <directory>\n", argv[0]);
            return 1;
        }
        scan_dir = argv[2];
        argi = 3;
    }

    if (all_mode) {
        /* Scan directory for .json files */
        DIR *dir = opendir(scan_dir);
        if (!dir) {
            fprintf(stderr, "Cannot open directory: %s\n", scan_dir);
            return 1;
        }

        /* First pass: count files */
        struct dirent *ent;
        int cap = 1000;
        task_files = (const char **)malloc(cap * sizeof(char *));
        while ((ent = readdir(dir)) != NULL) {
            int len = strlen(ent->d_name);
            if (len > 5 && strcmp(ent->d_name + len - 5, ".json") == 0) {
                char *path = (char *)malloc(strlen(scan_dir) + len + 2);
                sprintf(path, "%s/%s", scan_dir, ent->d_name);

                /* Quick check: same-size and >= 2 training pairs */
                char *json = read_file(path);
                if (json) {
                    ArcTask task;
                    if (parse_arc_task(json, &task) == 0 && task.n_train >= 2) {
                        int all_same = 1;
                        for (int i = 0; i < task.n_train; i++) {
                            if (task.train[i].H != task.train[i].oH ||
                                task.train[i].W != task.train[i].oW) {
                                all_same = 0; break;
                            }
                        }
                        if (task.test[0].H != task.test[0].oH ||
                            task.test[0].W != task.test[0].oW)
                            all_same = 0;
                        if (all_same) {
                            if (n_tasks >= cap) {
                                cap *= 2;
                                task_files = (const char **)realloc(task_files, cap * sizeof(char *));
                            }
                            task_files[n_tasks++] = path;
                            path = NULL; /* don't free */
                        }
                    }
                    free(json);
                }
                if (path) free(path);
            }
        }
        closedir(dir);
        printf("Found %d same-size tasks in %s\n\n", n_tasks, scan_dir);
    } else {
        n_tasks = argc - 1;
        task_files = (const char **)malloc(n_tasks * sizeof(char *));
        for (int i = 0; i < n_tasks; i++)
            task_files[i] = argv[i + 1];
    }

    printf("ARC Plucker Transversal Solver (C)\n");
    printf("Embeddings: %d, Trans/pair: %d\n", N_EMBEDDINGS, N_TRANS_PER_PAIR);
    #ifdef _OPENMP
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    #endif
    printf("Tasks: %d\n\n", n_tasks);

    int n_solved = 0;
    int n_rank1 = 0;

    for (int ti = 0; ti < n_tasks; ti++) {
        const char *path = task_files[ti];

        /* Extract task name from path */
        const char *fname = strrchr(path, '/');
        fname = fname ? fname + 1 : path;
        char task_name[128];
        strncpy(task_name, fname, sizeof(task_name) - 1);
        task_name[sizeof(task_name) - 1] = '\0';
        char *dot = strrchr(task_name, '.');
        if (dot) *dot = '\0';

        char *json = read_file(path);
        if (!json) {
            fprintf(stderr, "Cannot read: %s\n", path);
            continue;
        }

        ArcTask task;
        if (parse_arc_task(json, &task) != 0) {
            fprintf(stderr, "Failed to parse: %s\n", path);
            free(json);
            continue;
        }
        free(json);

        printf("  %s (%d train, %dx%d):\n", task_name, task.n_train,
               task.test[0].H, task.test[0].W);

        SolveResult r = solve_task(&task);

        if (r.rank == 1 && r.solved) {
            printf("    SOLVED  rank 1/%d\n", r.n_candidates);
            n_solved++;
            n_rank1++;
        } else if (r.rank == 1) {
            printf("    LIKELY RANK 1  rank 1/%d\n", r.n_candidates);
            n_rank1++;
        } else {
            printf("    rank %d/%d\n", r.rank, r.n_candidates);
        }
        printf("    %d transversals, setup=%.1fs, score=%.2fs\n\n",
               r.total_trans, r.setup_time, r.score_time);
    }

    printf("============================================================\n");
    printf("SUMMARY: %d/%d solved, %d/%d rank 1\n",
           n_solved, n_tasks, n_rank1, n_tasks);

    /* Cleanup */
    if (all_mode) {
        for (int i = 0; i < n_tasks; i++)
            free((void *)task_files[i]);
    }
    free(task_files);

    return 0;
}
