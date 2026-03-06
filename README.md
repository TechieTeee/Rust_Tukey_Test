# Rust Tukey Test

This script performs a Tukey test based on user inputs. It calculates the critical q value for a Tukey test, given the number of groups and total observations. The critical q value can be used to determine if there are statistically significant differences between the means of multiple groups.

## Background

The Tukey test, also known as the Tukey's honest significant difference (HSD) test, is a statistical test used to compare the means of multiple groups. It is often applied after performing an analysis of variance (ANOVA) to determine if there are significant differences between the means of the groups.

The test calculates a critical value (q value) based on the number of groups and the total number of observations. By comparing the differences between the means of the groups to this critical value, the Tukey test helps identify which groups have significantly different means from each other.

The Tukey test is useful in various fields, such as experimental research, social sciences, and business analytics. It allows researchers and analysts to gain insights into the significant differences between multiple groups, enabling them to make informed decisions or draw meaningful conclusions.

## Features

- **Tukey HSD test** — full pairwise comparison of group means with confidence intervals
- **Tukey-Kramer adjustment** — automatically handles unequal group sizes
- **Studentized range (q) critical values** — lookup table for k = 2-10 groups, df = 1-120, at alpha = 0.05 and 0.01
- **Library + CLI** — use as a Rust dependency or from the command line

## Using as a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
tukey-test = "0.2"
```

```rust
use tukey_test::tukey_hsd;

let data = vec![
    vec![23.0, 25.0, 21.0, 24.0],  // Group A
    vec![30.0, 28.0, 33.0, 31.0],  // Group B
    vec![22.0, 24.0, 20.0, 23.0],  // Group C
];

let result = tukey_hsd(&data, 0.05).unwrap();
println!("{result}");

for pair in result.significant_pairs() {
    println!("Groups {} and {} differ significantly (q = {:.4})",
        pair.group_i, pair.group_j, pair.q_statistic);
}
```

You can also look up critical q values directly:

```rust
let q = tukey_test::q_critical(3, 15, 0.05).unwrap();
println!("q = {q:.4}"); // q = 3.6700
```

## Prerequisites

- Rust programming language (1.56+ for edition 2021). Install from [https://www.rust-lang.org](https://www.rust-lang.org)

## CLI Usage

1. Clone and build:

```sh
git clone https://github.com/TechieTeee/Rust_Tukey_Test.git
cd Rust_Tukey_Test
cargo build --release
```

2. **Run the full Tukey HSD test** with your data (groups separated by `--`):

```sh
cargo run -- hsd 0.05 6 8 4 5 3 4 -- 8 12 9 11 6 8 -- 13 9 11 8 12 14
```

Example output:

```
Tukey HSD Test Results (alpha = 0.05, df = 15, MSE = 4.4556, q_critical = 3.6700)

Comparison      Mean Diff     q-stat  Significant   CI
------------------------------------------------------------------------
( 0,  1)           4.0000     4.6418          Yes   [-7.1626, -0.8374]
( 0,  2)           6.1667     7.1561          Yes   [-9.3292, -3.0041]
( 1,  2)           2.1667     2.5143           No   [-5.3292, 0.9959]
```

3. **Look up a critical q value** by specifying the number of groups, degrees of freedom, and alpha:

```sh
cargo run -- q 3 15 0.05
# q_critical(k=3, df=15, alpha=0.05) = 3.6700
```

