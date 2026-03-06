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

## Why tukey_test?

Rust's ecosystem for data science is growing fast — but access to foundational statistical tests has lagged behind R and Python. Researchers, engineers, and analysts working in Rust shouldn't have to shell out to another language just to answer: *"which groups are actually different?"*

`tukey_test` brings that capability natively to Rust. It's lightweight, correct, and designed to drop into existing data pipelines — whether you're running clinical trials, optimizing manufacturing processes, analyzing A/B tests, or building research tools.

> **ANOVA tells you *something* differs. Post-hoc tests tell you *what*.**

---

## Available Tests

| Test | When to Use | Function |
|:-----|:------------|:---------|
| **One-way ANOVA** | Check if any group means differ overall | `one_way_anova()` |
| **Tukey HSD** | All pairwise comparisons (equal variances) | `tukey_hsd()` |
| **Games-Howell** | All pairwise comparisons (unequal variances) | `games_howell()` |
| **Dunnett's test** | Compare treatments against a single control | `dunnett()` |
| **q critical value** | Studentized range distribution lookup | `q_critical()` |
| **Dunnett critical value** | Dunnett distribution lookup | `dunnett_critical()` |

All tests support **unequal group sizes**. Tukey HSD applies the Tukey-Kramer adjustment automatically.

---

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tukey_test = "0.2"
```

### ANOVA + Tukey HSD

The most common workflow — test for an overall effect, then find which pairs differ:

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

### Games-Howell — when variances aren't equal

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

### Dunnett's test — compare against a control

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

---

## Data Input

### Flexible types in the library

All test functions accept any type implementing `AsRef<[f64]>` for groups — no need to convert into `Vec<Vec<f64>>`:

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

Read data from CSV files in **wide format** (each column is a group). Headers are auto-detected and skipped. Columns can have unequal lengths.

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
# Inline — separate groups with --
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
  |
  |-- Want all pairwise comparisons?
  |   |-- Equal variances?   --> tukey_hsd()
  |   +-- Unequal variances? --> games_howell()
  |
  +-- Comparing treatments to a control?
      +-- dunnett()
```

---

## Roadmap

This crate aims to be the go-to resource for post-hoc and related statistical tests in Rust. Future plans include:

- **Scheffe's test** — the most conservative post-hoc test, for complex contrasts
- **Bonferroni / Holm corrections** — general-purpose p-value adjustment for multiple comparisons
- **Levene's test** — test for equality of variances (helps choose between Tukey and Games-Howell)
- **Welch's ANOVA** — one-way ANOVA without equal variance assumption
- **Effect sizes** — Cohen's d, eta-squared, omega-squared for practical significance
- **Numerical q-distribution** — continuous p-value computation beyond table lookup

Contributions, feature requests, and bug reports are welcome!

---

## Prerequisites

Rust **1.56+** (edition 2021). Install from [rust-lang.org](https://www.rust-lang.org).

## License

[MIT](LICENSE)
