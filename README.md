<p align="center">
  <h1 align="center">tukey_test</h1>
  <p align="center">
    <strong>Statistical post-hoc testing for Rust</strong>
  </p>
  <p align="center">
    Pure Rust &bull; Zero required dependencies &bull; No unsafe code
  </p>
  <p align="center">
    <a href="https://crates.io/crates/tukey_test"><img src="https://img.shields.io/crates/v/tukey_test.svg" alt="crates.io"></a>
    <a href="https://docs.rs/tukey_test"><img src="https://img.shields.io/docsrs/tukey_test" alt="docs.rs"></a>
    <a href="https://github.com/TechieTeee/Rust_Tukey_Test/blob/main/LICENSE"><img src="https://img.shields.io/crates/l/tukey_test.svg" alt="MIT License"></a>
  </p>
</p>

---

## The Problem

You're comparing three drug dosages, five marketing campaigns, or four manufacturing processes. A simple t-test won't work - running multiple pairwise t-tests inflates your false positive rate. With 5 groups, that's 10 comparisons, and your chance of a spurious "significant" result climbs from 5% to nearly 40%.

**Post-hoc tests solve this.** They control the family-wise error rate, giving you honest answers about which groups truly differ - not statistical noise.

> **ANOVA tells you *something* differs. Post-hoc tests tell you *what*.**

---

## Real-World Applications

**Medicine & Clinical Trials** - Compare patient outcomes across treatment arms. A hospital testing three pain medications needs to know not just that outcomes differ, but *which drug works best* and by how much, while controlling for multiple comparisons that could lead to approving an ineffective treatment.

**Agriculture & Food Science** - Compare crop yields across fertilizer types, or taste scores across formulations. The Tukey test was originally developed by John Tukey at Princeton for exactly these kinds of agricultural experiments - its design is purpose-built for "which of these treatments actually made a difference?"

**Manufacturing & Quality Control** - Compare defect rates or tolerances across production lines, shifts, or suppliers. When a factory runs five machines making the same part, Dunnett's test can flag which machines deviate from the reference line without false alarms.

**Software & A/B Testing** - Compare conversion rates, latency, or engagement metrics across multiple variants. With 4+ variants, pairwise t-tests give misleading results; Tukey HSD or Games-Howell gives you defensible answers.

**Education & Social Science** - Compare test scores across teaching methods, survey responses across demographics, or behavioral outcomes across intervention groups. These fields routinely analyze 3-10 groups and need post-hoc tests that reviewers and journals accept.

**Environmental Science** - Compare pollution levels across sites, species counts across habitats, or water quality across treatment methods. Ragged data (unequal sample sizes) is the norm here, which is why Tukey-Kramer and Games-Howell matter.

---

## Why Rust?

Rust has been ranked the **most admired programming language** in the Stack Overflow Developer Survey for nine consecutive years (2016–2024). Developer adoption is accelerating - Rust is now used in the Linux kernel, the Windows kernel, the Android platform, and major infrastructure at AWS, Microsoft, Google, and Meta. It is no longer a niche systems language; it is becoming the default choice for any new software where correctness, performance, and long-term maintainability matter.

The statistical computing ecosystem hasn't kept up. `tukey_test` is part of closing that gap.

---

R, Python, and Mathematica are great for interactive analysis in a notebook. But when you're **building something that runs statistical tests in production** - a clinical trial pipeline, a manufacturing quality system, a real-time A/B testing platform - those tools create serious problems:

- You have to shell out to another language mid-pipeline, adding subprocess overhead, serialization, and a fragile runtime dependency
- Python and R's numerical output can vary by OS, version, and BLAS library - meaning results on your laptop may not match production
- Deploying a Python + scipy + numpy stack (or an R environment) on edge hardware, in Docker, or in WebAssembly is genuinely painful

**This is the gap `tukey_test` fills.** Post-hoc statistical tests have essentially zero native Rust coverage. If you needed Tukey HSD in a Rust service today, your options were: reimplement it yourself, shell out to R, or change languages. None of those are acceptable in a serious production codebase.

### Four reasons this matters

**1. Correctness you can trust**
Rust's type system and borrow checker eliminate whole categories of bugs - buffer overruns, data races, silent memory corruption - that have caused real errors in scientific software. Reproducibility is a well-documented crisis in research. A statistically correct implementation that *cannot silently corrupt memory* is a different class of software than a Python script that happens to produce the right answer most of the time.

**2. Deterministic results everywhere**
Rust produces bit-for-bit identical output across platforms and operating systems. For auditable systems - FDA regulatory submissions, financial reporting, clinical trial analysis - that determinism is not a nice-to-have, it's a requirement. R and Python cannot make that guarantee.

**3. Zero-overhead integration**
A Rust service can call `tukey_hsd()` directly in the same process, with no FFI, no subprocess, no round-trip serialization. For high-frequency data processing or embedded applications - real-time sensor analysis, manufacturing QC on the line - that's the difference between feasible and not.

**4. Simple, self-contained deployment**
A Rust binary with zero required dependencies ships as a single executable. No runtime to install, no package manager to invoke, no version conflicts. Compare that to deploying a full R or Python scientific stack on edge hardware or in a serverless function.

### Who this is for

The intended user is not a statistician choosing between tools for a one-off analysis - use R or Python for that. This crate is for the **Rust engineer** who already knows what statistical test they need and is now building the system that runs it reliably, repeatedly, and at scale. That person has had no good option until now.

---

## Available Tests

| Test | When to Use | Function |
|:-----|:------------|:---------|
| **One-way ANOVA** | Check if any group means differ overall | `one_way_anova()` |
| **Tukey HSD** | All pairwise comparisons (equal variances) | `tukey_hsd()` |
| **Games-Howell** | All pairwise comparisons (unequal variances) | `games_howell()` |
| **Dunnett's test** | Compare treatments against a single control | `dunnett()` |
| **Levene's test** | Test equality of variances (Brown-Forsythe variant) | `levene_test()` |
| **q critical value** | Studentized range distribution lookup | `q_critical()` |
| **Dunnett critical value** | Dunnett distribution lookup | `dunnett_critical()` |
| **Studentized range CDF** | Exact p-values for post-hoc tests | `ptukey_cdf()` |

All tests support **unequal group sizes**. Tukey HSD applies the Tukey-Kramer adjustment automatically.

All pairwise comparison results include an exact **p-value** computed via `ptukey_cdf`. ANOVA results include **η²** (eta-squared) and **ω²** (omega-squared) effect sizes. `q_critical` supports any α and any number of groups k (not limited to the table range).

---

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tukey_test = "0.2"
```

### ANOVA + Tukey HSD

The most common workflow - test for an overall effect, then find which pairs differ:

```rust
use tukey_test::{one_way_anova, tukey_hsd};

let data = vec![
    vec![6.0, 8.0, 4.0, 5.0, 3.0, 4.0],   // Group A
    vec![8.0, 12.0, 9.0, 11.0, 6.0, 8.0],  // Group B
    vec![13.0, 9.0, 11.0, 8.0, 12.0, 14.0], // Group C
];

// Step 1: Is there an overall difference?
let anova = one_way_anova(&data).unwrap();
println!("{anova}");
// F = 13.18, p = 0.0005

// Step 2: Which pairs differ?
let result = tukey_hsd(&data, 0.05).unwrap();
for pair in result.significant_pairs() {
    println!("Groups {} and {} differ (q = {:.4})",
        pair.group_i, pair.group_j, pair.q_statistic);
}
```

### Games-Howell - when variances aren't equal

```rust
use tukey_test::games_howell;

let data = vec![
    vec![4.0, 5.0, 3.0, 4.0, 6.0],
    vec![20.0, 30.0, 25.0, 35.0, 28.0],  // high variance
    vec![5.0, 7.0, 6.0, 4.0, 5.0],
];
let result = games_howell(&data, 0.05).unwrap();
println!("{result}");
```

### Dunnett's test - compare against a control

```rust
use tukey_test::dunnett;

let data = vec![
    vec![10.0, 12.0, 11.0, 9.0, 10.0],   // control (group 0)
    vec![15.0, 17.0, 14.0, 16.0, 15.0],   // treatment A
    vec![11.0, 13.0, 10.0, 12.0, 11.0],   // treatment B
];
let result = dunnett(&data, 0, 0.05).unwrap();
for t in result.significant_treatments() {
    println!("Treatment {} differs from control (t = {:.4})",
        t.treatment, t.t_statistic);
}
```

### Levene's test - check variance equality before choosing Tukey vs Games-Howell

```rust
use tukey_test::levene_test;

let data = vec![
    vec![5.0, 6.0, 5.5, 5.2, 5.8],   // group A (low variance)
    vec![1.0, 9.0, 3.0, 8.0, 5.0],   // group B (high variance)
];
let result = levene_test(&data, 0.05).unwrap();
if result.significant {
    println!("Unequal variances detected - use games_howell()");
} else {
    println!("Variances are homogeneous - tukey_hsd() is appropriate");
}
```

### Exact p-values and effect sizes

Every pairwise comparison result includes an exact **p-value** computed via the studentized range CDF:

```rust
use tukey_test::tukey_hsd;

let data = vec![
    vec![6.0, 8.0, 4.0, 5.0, 3.0, 4.0],
    vec![8.0, 12.0, 9.0, 11.0, 6.0, 8.0],
    vec![13.0, 9.0, 11.0, 8.0, 12.0, 14.0],
];
let result = tukey_hsd(&data, 0.05).unwrap();
for pair in &result.comparisons {
    println!("Groups ({}, {}): p = {:.4}", pair.group_i, pair.group_j, pair.p_value);
}
```

ANOVA results include **η²** (eta-squared) and **ω²** (omega-squared) for practical effect size:

```rust
use tukey_test::one_way_anova;

let data = vec![
    vec![6.0, 8.0, 4.0, 5.0, 3.0, 4.0],
    vec![8.0, 12.0, 9.0, 11.0, 6.0, 8.0],
    vec![13.0, 9.0, 11.0, 8.0, 12.0, 14.0],
];
let result = one_way_anova(&data).unwrap();
println!("η² = {:.3}, ω² = {:.3}", result.eta_squared, result.omega_squared);
// η² = 0.637, ω² = 0.589
```

---

## Data Input

### Flexible types in the library

All test functions accept any type implementing `AsRef<[f64]>` for groups - no need to convert into `Vec<Vec<f64>>`:

```rust
use tukey_test::one_way_anova;

// Slices
let data: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]];
one_way_anova(data).unwrap();

// Fixed-size arrays
let data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
one_way_anova(&data).unwrap();
```

### Load from CSV

Read data from CSV files in **wide format** (each column is a group). Headers are auto-detected and skipped. Columns can have unequal lengths. Values of `NaN`, `Infinity`, and `-Infinity` are rejected with a `ParseError`.

```
control,treatment_a,treatment_b
10,15,11
12,17,13
11,14,10
9,16,12
10,15,11
```

```rust
use tukey_test::{parse_csv_file, tukey_hsd};

let groups = parse_csv_file("experiment.csv").unwrap();
let result = tukey_hsd(&groups, 0.05).unwrap();
println!("{result}");
```

You can also parse from any `std::io::Read` source:

```rust
use tukey_test::parse_csv;

let csv_data = "a,b\n1,4\n2,5\n3,6\n";
let groups = parse_csv(csv_data.as_bytes()).unwrap();
```

---

## Optional Features

### Serde

Serialize any result to JSON, CSV, or any serde-supported format:

```toml
[dependencies]
tukey_test = { version = "0.2", features = ["serde"] }
```

---

## CLI

The crate also ships as a command-line tool.

```sh
git clone https://github.com/TechieTeee/Rust_Tukey_Test.git
cd Rust_Tukey_Test
cargo build --release
```

### Data input options

```sh
# Inline - separate groups with --
tukey_test hsd 0.05 6 8 4 5 3 4 -- 8 12 9 11 6 8 -- 13 9 11 8 12 14

# From a CSV file
tukey_test hsd 0.05 --file experiment.csv

# From stdin (pipe-friendly)
cat data.csv | tukey_test hsd 0.05 --file -
```

### Commands

```sh
# One-way ANOVA
tukey_test anova 6 8 4 5 3 4 -- 8 12 9 11 6 8 -- 13 9 11 8 12 14
tukey_test anova --file data.csv

# Tukey HSD (alpha = 0.05)
tukey_test hsd 0.05 6 8 4 5 3 4 -- 8 12 9 11 6 8 -- 13 9 11 8 12 14
tukey_test hsd 0.05 --file data.csv

# Games-Howell (alpha = 0.05)
tukey_test games-howell 0.05 4 5 3 4 6 -- 20 30 25 35 28 -- 5 7 6 4 5
tukey_test games-howell 0.05 --file data.csv

# Dunnett (alpha = 0.05, control = group 0)
tukey_test dunnett 0.05 0 10 12 11 9 10 -- 15 17 14 16 15 -- 11 13 10 12 11
tukey_test dunnett 0.05 0 --file data.csv

# Critical value lookup
tukey_test q 3 15 0.05
```

### Example Output

```
One-Way ANOVA

Source                 SS     df           MS          F    p-value
--------------------------------------------------------------------
Between          117.4444      2      58.7222    13.1796   0.000497
Within            66.8333     15       4.4556
Total            184.2778     17
```

```
Tukey HSD Test Results (alpha = 0.05, df = 15, MSE = 4.4556, q_critical = 3.6700)

Comparison      Mean Diff     q-stat  Significant   CI
------------------------------------------------------------------------
( 0,  1)           4.0000     4.6418          Yes   [-7.1626, -0.8374]
( 0,  2)           6.1667     7.1561          Yes   [-9.3292, -3.0041]
( 1,  2)           2.1667     2.5143           No   [-5.3292, 0.9959]
```

```
Dunnett's Test Results (alpha = 0.05, control = group 0, df = 12, d_critical = 2.3800)

Treatment           Mean Diff     t-stat  Significant   CI
--------------------------------------------------------------------------
Group  1 vs  0        5.0000     6.9338          Yes   [3.2838, 6.7162]
Group  2 vs  0        1.0000     1.3868           No   [-0.7162, 2.7162]
```

---

## How to Choose the Right Test

```
  Got multiple groups to compare?
  |
  |-- Start with one_way_anova()
  |   p < 0.05? --> significant overall difference
  |   eta_squared / omega_squared --> practical effect size
  |
  |-- Want all pairwise comparisons?
  |   |
  |   |-- Check variances first: levene_test()
  |   |   |-- Not significant (homogeneous)? --> tukey_hsd()
  |   |   +-- Significant (unequal)?         --> games_howell()
  |   |
  |   +-- All pairwise results include exact p-values
  |
  +-- Comparing treatments to a control?
      +-- dunnett()
```

---

## Roadmap

This crate aims to be the go-to resource for post-hoc and related statistical tests in Rust. Future plans include:

- **Scheffe's test** - the most conservative post-hoc test, for complex contrasts
- **Bonferroni / Holm corrections** - general-purpose p-value adjustment for multiple comparisons
- **Welch's ANOVA** - one-way ANOVA without equal variance assumption
- **Cohen's d** - pairwise effect size for individual comparisons
- **Dunnett's test for any k/df** - remove table-based limits via numerical multivariate-t integration (currently requires df ≥ 5 and k ≤ 9)
- **Games-Howell non-integer df** - fix Welch-Satterthwaite df truncation for more accurate p-values at small sample sizes
- **`ptukey_cdf` extended validation** - comprehensive accuracy benchmarks against R across the full (k, df, q) grid, especially for k > 20 and df < 5
- **`no_std` / `f32` support** - for embedded and WASM scientific computing targets

These items are targeted for **0.3.0**.

Contributions, feature requests, and bug reports are welcome!

---

## Prerequisites

Rust **1.56+** (edition 2021). Install from [rust-lang.org](https://www.rust-lang.org).

## License

[MIT](LICENSE)
