//! Statistical tests for pairwise group comparisons.
//!
//! This crate provides:
//! - [`one_way_anova`] — one-way analysis of variance (F-test)
//! - [`tukey_hsd`] — Tukey HSD test (assumes equal variances)
//! - [`games_howell`] — Games-Howell test (does **not** assume equal variances)
//! - [`dunnett`] — Dunnett's test (compare treatments against a control)
//! - [`q_critical`] — studentized range distribution critical value lookup
//! - [`dunnett_critical`] — Dunnett's critical value lookup
//! - [`parse_csv`] — load grouped data from CSV (wide format)
//!
//! All test functions accept any type implementing `AsRef<[f64]>` for groups,
//! so you can pass `&[Vec<f64>]`, `&[&[f64]]`, or `&[[f64; N]]`.
//!
//! # Example
//!
//! ```
//! use tukey_test::{one_way_anova, tukey_hsd};
//!
//! let data = vec![
//!     vec![23.0, 25.0, 21.0, 24.0],  // Group A
//!     vec![30.0, 28.0, 33.0, 31.0],  // Group B
//!     vec![22.0, 24.0, 20.0, 23.0],  // Group C
//! ];
//!
//! // Step 1: check if there is an overall difference
//! let anova = one_way_anova(&data).unwrap();
//! println!("{anova}");
//!
//! // Step 2: find which pairs differ
//! let result = tukey_hsd(&data, 0.05).unwrap();
//! println!("{result}");
//!
//! for pair in result.significant_pairs() {
//!     println!("Groups {} and {} differ significantly (q = {:.4})",
//!         pair.group_i, pair.group_j, pair.q_statistic);
//! }
//! ```

use std::fmt;
use std::io;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during Tukey test operations.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TukeyError {
    /// Fewer than 2 groups were provided.
    TooFewGroups,
    /// More groups than the lookup table supports (max 10).
    TooManyGroups(usize),
    /// A group contained no observations.
    EmptyGroup(usize),
    /// Not enough degrees of freedom (need at least 1).
    InsufficientDf,
    /// Alpha level not supported — use 0.05 or 0.01.
    UnsupportedAlpha(f64),
    /// Within-group variance is zero (all observations identical).
    ZeroVariance,
    /// A group needs at least 2 observations (for per-group variance).
    GroupTooSmall(usize),
    /// Control group index is out of range.
    ControlGroupOutOfRange(usize),
    /// Too many treatment groups for the Dunnett table (max 9).
    TooManyTreatments(usize),
    /// I/O error when reading data.
    IoError(String),
    /// Failed to parse a value in CSV data.
    ParseError { line: usize, column: usize, value: String },
    /// CSV data has no columns.
    EmptyCsv,
}

impl fmt::Display for TukeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TukeyError::TooFewGroups => write!(f, "at least 2 groups are required"),
            TukeyError::TooManyGroups(k) => {
                write!(f, "too many groups ({k}), maximum supported is 10")
            }
            TukeyError::EmptyGroup(i) => write!(f, "group {i} is empty"),
            TukeyError::InsufficientDf => {
                write!(f, "insufficient degrees of freedom (need at least 1)")
            }
            TukeyError::UnsupportedAlpha(a) => {
                write!(f, "unsupported alpha level ({a}), use 0.05 or 0.01")
            }
            TukeyError::ZeroVariance => {
                write!(f, "within-group variance is zero (all observations identical)")
            }
            TukeyError::GroupTooSmall(i) => {
                write!(f, "group {i} needs at least 2 observations")
            }
            TukeyError::ControlGroupOutOfRange(i) => {
                write!(f, "control group index {i} is out of range")
            }
            TukeyError::TooManyTreatments(p) => {
                write!(f, "too many treatment groups ({p}), maximum supported is 9")
            }
            TukeyError::IoError(msg) => write!(f, "I/O error: {msg}"),
            TukeyError::ParseError { line, column, value } => {
                write!(f, "failed to parse \"{value}\" as number (line {line}, column {column})")
            }
            TukeyError::EmptyCsv => {
                write!(f, "CSV data has no columns")
            }
        }
    }
}

impl std::error::Error for TukeyError {}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Results of a one-way ANOVA.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnovaResult {
    /// Sum of squares between groups.
    pub ss_between: f64,
    /// Sum of squares within groups.
    pub ss_within: f64,
    /// Total sum of squares.
    pub ss_total: f64,
    /// Degrees of freedom between groups (k − 1).
    pub df_between: usize,
    /// Degrees of freedom within groups (N − k).
    pub df_within: usize,
    /// Total degrees of freedom (N − 1).
    pub df_total: usize,
    /// Mean square between groups.
    pub ms_between: f64,
    /// Mean square within groups.
    pub ms_within: f64,
    /// F statistic.
    pub f_statistic: f64,
    /// p-value (probability of observing this F or larger under H0).
    pub p_value: f64,
    /// Grand mean of all observations.
    pub grand_mean: f64,
    /// Mean of each group.
    pub group_means: Vec<f64>,
    /// Number of observations in each group.
    pub group_sizes: Vec<usize>,
}

impl fmt::Display for AnovaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "One-Way ANOVA")?;
        writeln!(f)?;
        writeln!(
            f,
            "{:<14} {:>10} {:>6} {:>12} {:>10} {:>10}",
            "Source", "SS", "df", "MS", "F", "p-value"
        )?;
        writeln!(f, "{}", "-".repeat(68))?;
        writeln!(
            f,
            "{:<14} {:>10.4} {:>6} {:>12.4} {:>10.4} {:>10.6}",
            "Between", self.ss_between, self.df_between, self.ms_between, self.f_statistic, self.p_value
        )?;
        writeln!(
            f,
            "{:<14} {:>10.4} {:>6} {:>12.4}",
            "Within", self.ss_within, self.df_within, self.ms_within
        )?;
        writeln!(
            f,
            "{:<14} {:>10.4} {:>6}",
            "Total", self.ss_total, self.df_total
        )?;
        Ok(())
    }
}

/// A single pairwise comparison between two groups.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PairwiseComparison {
    /// Index of the first group.
    pub group_i: usize,
    /// Index of the second group.
    pub group_j: usize,
    /// Mean of group i.
    pub mean_i: f64,
    /// Mean of group j.
    pub mean_j: f64,
    /// Absolute difference between the group means.
    pub mean_diff: f64,
    /// Observed q statistic for this pair.
    pub q_statistic: f64,
    /// Whether the difference is statistically significant at the chosen alpha.
    pub significant: bool,
    /// Lower bound of the confidence interval for (mean_i − mean_j).
    pub ci_lower: f64,
    /// Upper bound of the confidence interval for (mean_i − mean_j).
    pub ci_upper: f64,
}

/// Full results of a Tukey HSD test.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TukeyResult {
    /// Number of groups.
    pub groups: usize,
    /// Significance level used.
    pub alpha: f64,
    /// Critical q value from the studentized range distribution.
    pub q_critical: f64,
    /// Degrees of freedom (within groups).
    pub df: usize,
    /// Mean square error (pooled within-group variance).
    pub mse: f64,
    /// Mean of each group.
    pub group_means: Vec<f64>,
    /// Number of observations in each group.
    pub group_sizes: Vec<usize>,
    /// All pairwise comparisons.
    pub comparisons: Vec<PairwiseComparison>,
}

impl TukeyResult {
    /// Returns only the comparisons that are statistically significant.
    #[must_use]
    pub fn significant_pairs(&self) -> Vec<&PairwiseComparison> {
        self.comparisons.iter().filter(|c| c.significant).collect()
    }
}

impl fmt::Display for TukeyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Tukey HSD Test Results (alpha = {}, df = {}, MSE = {:.4}, q_critical = {:.4})",
            self.alpha, self.df, self.mse, self.q_critical
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "{:<14} {:>10} {:>10} {:>12}   {}",
            "Comparison", "Mean Diff", "q-stat", "Significant", "CI"
        )?;
        writeln!(f, "{}", "-".repeat(72))?;
        for c in &self.comparisons {
            writeln!(
                f,
                "({:>2}, {:>2})       {:>10.4} {:>10.4} {:>12}   [{:.4}, {:.4}]",
                c.group_i,
                c.group_j,
                c.mean_diff,
                c.q_statistic,
                if c.significant { "Yes" } else { "No" },
                c.ci_lower,
                c.ci_upper
            )?;
        }
        Ok(())
    }
}

/// Full results of a Games-Howell test.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GamesHowellResult {
    /// Number of groups.
    pub groups: usize,
    /// Significance level used.
    pub alpha: f64,
    /// Mean of each group.
    pub group_means: Vec<f64>,
    /// Number of observations in each group.
    pub group_sizes: Vec<usize>,
    /// Sample variance of each group.
    pub group_variances: Vec<f64>,
    /// All pairwise comparisons.
    pub comparisons: Vec<PairwiseComparison>,
}

impl GamesHowellResult {
    /// Returns only the comparisons that are statistically significant.
    #[must_use]
    pub fn significant_pairs(&self) -> Vec<&PairwiseComparison> {
        self.comparisons.iter().filter(|c| c.significant).collect()
    }
}

impl fmt::Display for GamesHowellResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Games-Howell Test Results (alpha = {})", self.alpha)?;
        writeln!(f)?;
        writeln!(
            f,
            "{:<14} {:>10} {:>10} {:>5} {:>12}   {}",
            "Comparison", "Mean Diff", "q-stat", "df", "Significant", "CI"
        )?;
        writeln!(f, "{}", "-".repeat(78))?;
        for c in &self.comparisons {
            // Recover the Welch-Satterthwaite df from the q_critical we'd need
            // Instead, just display the q_statistic (df is per-pair, stored implicitly)
            writeln!(
                f,
                "({:>2}, {:>2})       {:>10.4} {:>10.4} {:>5} {:>12}   [{:.4}, {:.4}]",
                c.group_i,
                c.group_j,
                c.mean_diff,
                c.q_statistic,
                "",
                if c.significant { "Yes" } else { "No" },
                c.ci_lower,
                c.ci_upper
            )?;
        }
        Ok(())
    }
}

/// A single comparison of a treatment group against the control.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DunnettComparison {
    /// Index of the treatment group.
    pub treatment: usize,
    /// Mean of the treatment group.
    pub treatment_mean: f64,
    /// Mean of the control group.
    pub control_mean: f64,
    /// Absolute difference between means.
    pub mean_diff: f64,
    /// Observed t statistic.
    pub t_statistic: f64,
    /// Whether the difference is statistically significant.
    pub significant: bool,
    /// Lower bound of the confidence interval for (treatment − control).
    pub ci_lower: f64,
    /// Upper bound of the confidence interval for (treatment − control).
    pub ci_upper: f64,
}

/// Full results of a Dunnett's test.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DunnettResult {
    /// Index of the control group.
    pub control: usize,
    /// Number of treatment groups (excluding control).
    pub treatments: usize,
    /// Significance level used.
    pub alpha: f64,
    /// Critical value from the Dunnett distribution.
    pub d_critical: f64,
    /// Degrees of freedom (within groups).
    pub df: usize,
    /// Mean square error (pooled within-group variance).
    pub mse: f64,
    /// Mean of each group (including control).
    pub group_means: Vec<f64>,
    /// Number of observations in each group.
    pub group_sizes: Vec<usize>,
    /// Comparisons of each treatment against the control.
    pub comparisons: Vec<DunnettComparison>,
}

impl DunnettResult {
    /// Returns only the comparisons that are statistically significant.
    #[must_use]
    pub fn significant_treatments(&self) -> Vec<&DunnettComparison> {
        self.comparisons.iter().filter(|c| c.significant).collect()
    }
}

impl fmt::Display for DunnettResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Dunnett's Test Results (alpha = {}, control = group {}, df = {}, d_critical = {:.4})",
            self.alpha, self.control, self.df, self.d_critical
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "{:<18} {:>10} {:>10} {:>12}   {}",
            "Treatment", "Mean Diff", "t-stat", "Significant", "CI"
        )?;
        writeln!(f, "{}", "-".repeat(74))?;
        for c in &self.comparisons {
            writeln!(
                f,
                "Group {:>2} vs {:>2}    {:>10.4} {:>10.4} {:>12}   [{:.4}, {:.4}]",
                c.treatment,
                self.control,
                c.mean_diff,
                c.t_statistic,
                if c.significant { "Yes" } else { "No" },
                c.ci_lower,
                c.ci_upper
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Studentized range (q) critical-value table
// ---------------------------------------------------------------------------

/// Degrees-of-freedom breakpoints used in the lookup table.
const DF_VALUES: [usize; 25] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 30, 40, 60, 120,
];

/// q critical values at alpha = 0.05, indexed by [k-2][df_index].
#[rustfmt::skip]
const Q_TABLE_05: [[f64; 25]; 9] = [
    // k=2
    [17.97, 6.08, 4.50, 3.93, 3.64, 3.46, 3.34, 3.26, 3.20, 3.15, 3.11, 3.08, 3.06, 3.03, 3.01, 3.00, 2.98, 2.97, 2.96, 2.95, 2.92, 2.89, 2.86, 2.83, 2.80],
    // k=3
    [26.98, 8.33, 5.91, 5.04, 4.60, 4.34, 4.16, 4.04, 3.95, 3.88, 3.82, 3.77, 3.73, 3.70, 3.67, 3.65, 3.63, 3.61, 3.59, 3.58, 3.53, 3.49, 3.44, 3.40, 3.36],
    // k=4
    [32.82, 9.80, 6.82, 5.76, 5.22, 4.90, 4.68, 4.53, 4.41, 4.33, 4.26, 4.20, 4.15, 4.11, 4.08, 4.05, 4.02, 4.00, 3.98, 3.96, 3.90, 3.85, 3.79, 3.74, 3.68],
    // k=5
    [37.08, 10.88, 7.50, 6.29, 5.67, 5.30, 5.06, 4.89, 4.76, 4.65, 4.57, 4.51, 4.45, 4.41, 4.37, 4.33, 4.30, 4.28, 4.25, 4.23, 4.17, 4.10, 4.04, 3.98, 3.92],
    // k=6
    [40.41, 11.74, 8.04, 6.71, 6.03, 5.63, 5.36, 5.17, 5.02, 4.91, 4.82, 4.75, 4.69, 4.64, 4.59, 4.56, 4.52, 4.49, 4.47, 4.45, 4.37, 4.30, 4.23, 4.16, 4.10],
    // k=7
    [43.12, 12.44, 8.48, 7.05, 6.33, 5.90, 5.61, 5.40, 5.24, 5.12, 5.03, 4.95, 4.88, 4.83, 4.78, 4.74, 4.70, 4.67, 4.65, 4.62, 4.54, 4.46, 4.39, 4.31, 4.24],
    // k=8
    [45.40, 13.03, 8.85, 7.35, 6.58, 6.12, 5.82, 5.60, 5.43, 5.30, 5.20, 5.12, 5.05, 4.99, 4.94, 4.90, 4.86, 4.82, 4.79, 4.77, 4.68, 4.60, 4.52, 4.44, 4.36],
    // k=9
    [47.36, 13.54, 9.18, 7.60, 6.80, 6.32, 6.00, 5.77, 5.59, 5.46, 5.35, 5.27, 5.19, 5.13, 5.08, 5.03, 4.99, 4.96, 4.92, 4.90, 4.81, 4.72, 4.63, 4.55, 4.47],
    // k=10
    [49.07, 13.99, 9.46, 7.83, 6.99, 6.49, 6.16, 5.92, 5.74, 5.60, 5.49, 5.39, 5.32, 5.25, 5.20, 5.15, 5.11, 5.07, 5.04, 5.01, 4.92, 4.82, 4.73, 4.65, 4.56],
];

/// q critical values at alpha = 0.01, indexed by [k-2][df_index].
#[rustfmt::skip]
const Q_TABLE_01: [[f64; 25] ; 9] = [
    // k=2
    [90.03, 14.04, 8.26, 6.51, 5.70, 5.24, 4.95, 4.75, 4.60, 4.48, 4.39, 4.32, 4.26, 4.21, 4.17, 4.13, 4.10, 4.07, 4.05, 4.02, 3.96, 3.89, 3.82, 3.76, 3.70],
    // k=3
    [135.0, 19.02, 10.62, 8.12, 6.98, 6.33, 5.92, 5.64, 5.43, 5.27, 5.15, 5.05, 4.96, 4.89, 4.84, 4.79, 4.74, 4.70, 4.67, 4.64, 4.55, 4.45, 4.37, 4.28, 4.20],
    // k=4
    [164.3, 22.29, 12.17, 9.17, 7.80, 7.03, 6.54, 6.20, 5.96, 5.77, 5.62, 5.50, 5.40, 5.32, 5.25, 5.19, 5.14, 5.09, 5.05, 5.02, 4.91, 4.80, 4.70, 4.59, 4.50],
    // k=5
    [185.6, 24.72, 13.33, 9.96, 8.42, 7.56, 7.01, 6.62, 6.35, 6.14, 5.97, 5.84, 5.73, 5.63, 5.56, 5.49, 5.43, 5.38, 5.33, 5.29, 5.17, 5.05, 4.93, 4.82, 4.71],
    // k=6
    [202.2, 26.63, 14.24, 10.58, 8.91, 7.97, 7.37, 6.96, 6.66, 6.43, 6.25, 6.10, 5.98, 5.88, 5.80, 5.72, 5.66, 5.60, 5.55, 5.51, 5.37, 5.24, 5.11, 4.99, 4.87],
    // k=7
    [215.8, 28.20, 15.00, 11.10, 9.32, 8.32, 7.68, 7.24, 6.91, 6.67, 6.48, 6.32, 6.19, 6.08, 5.99, 5.92, 5.85, 5.79, 5.73, 5.69, 5.54, 5.40, 5.26, 5.13, 5.01],
    // k=8
    [227.2, 29.53, 15.64, 11.55, 9.67, 8.61, 7.94, 7.47, 7.13, 6.87, 6.67, 6.51, 6.37, 6.26, 6.16, 6.08, 6.01, 5.94, 5.89, 5.84, 5.69, 5.54, 5.39, 5.25, 5.12],
    // k=9
    [237.0, 30.68, 16.20, 11.93, 9.97, 8.87, 8.17, 7.68, 7.33, 7.05, 6.84, 6.67, 6.53, 6.41, 6.31, 6.22, 6.15, 6.08, 6.02, 5.97, 5.81, 5.65, 5.50, 5.36, 5.21],
    // k=10
    [245.6, 31.69, 16.69, 12.27, 10.24, 9.10, 8.37, 7.86, 7.49, 7.21, 6.99, 6.81, 6.67, 6.54, 6.44, 6.35, 6.27, 6.20, 6.14, 6.09, 5.92, 5.76, 5.60, 5.45, 5.30],
];

// ---------------------------------------------------------------------------
// Dunnett's critical-value table (two-sided)
// ---------------------------------------------------------------------------

/// Degrees-of-freedom breakpoints for the Dunnett table.
const DUNNETT_DF_VALUES: [usize; 21] = [
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 30, 40, 60, 120,
];

/// Dunnett two-sided critical values at alpha = 0.05.
/// Indexed by [p-1][df_index] where p = number of treatment groups (1..=9).
#[rustfmt::skip]
const DUNNETT_TABLE_05: [[f64; 21]; 9] = [
    // p=1
    [2.57, 2.45, 2.36, 2.31, 2.26, 2.23, 2.20, 2.18, 2.16, 2.14, 2.13, 2.12, 2.11, 2.10, 2.09, 2.09, 2.06, 2.04, 2.02, 2.00, 1.98],
    // p=2
    [2.86, 2.70, 2.60, 2.53, 2.48, 2.44, 2.41, 2.38, 2.36, 2.34, 2.33, 2.32, 2.31, 2.30, 2.29, 2.28, 2.25, 2.23, 2.21, 2.19, 2.16],
    // p=3
    [3.03, 2.85, 2.74, 2.66, 2.61, 2.57, 2.53, 2.50, 2.48, 2.46, 2.44, 2.43, 2.42, 2.41, 2.40, 2.39, 2.36, 2.34, 2.31, 2.28, 2.26],
    // p=4
    [3.15, 2.95, 2.84, 2.76, 2.69, 2.65, 2.61, 2.58, 2.56, 2.53, 2.51, 2.50, 2.49, 2.48, 2.47, 2.46, 2.43, 2.40, 2.37, 2.35, 2.32],
    // p=5
    [3.24, 3.03, 2.91, 2.83, 2.76, 2.71, 2.67, 2.64, 2.61, 2.59, 2.57, 2.56, 2.54, 2.53, 2.52, 2.51, 2.48, 2.45, 2.42, 2.39, 2.37],
    // p=6
    [3.31, 3.10, 2.97, 2.88, 2.81, 2.77, 2.73, 2.69, 2.66, 2.64, 2.62, 2.60, 2.59, 2.58, 2.57, 2.56, 2.53, 2.50, 2.47, 2.44, 2.41],
    // p=7
    [3.37, 3.15, 3.02, 2.93, 2.86, 2.81, 2.77, 2.73, 2.70, 2.68, 2.66, 2.64, 2.62, 2.61, 2.60, 2.59, 2.56, 2.53, 2.50, 2.47, 2.44],
    // p=8
    [3.42, 3.20, 3.06, 2.97, 2.90, 2.85, 2.80, 2.77, 2.74, 2.71, 2.69, 2.67, 2.66, 2.64, 2.63, 2.62, 2.59, 2.56, 2.53, 2.49, 2.47],
    // p=9
    [3.47, 3.24, 3.10, 3.01, 2.93, 2.88, 2.84, 2.80, 2.77, 2.74, 2.72, 2.70, 2.69, 2.67, 2.66, 2.65, 2.62, 2.58, 2.55, 2.52, 2.49],
];

/// Dunnett two-sided critical values at alpha = 0.01.
/// Indexed by [p-1][df_index] where p = number of treatment groups (1..=9).
#[rustfmt::skip]
const DUNNETT_TABLE_01: [[f64; 21]; 9] = [
    // p=1
    [4.03, 3.71, 3.50, 3.36, 3.25, 3.17, 3.11, 3.05, 3.01, 2.98, 2.95, 2.92, 2.90, 2.88, 2.86, 2.85, 2.80, 2.75, 2.70, 2.66, 2.62],
    // p=2
    [4.36, 3.99, 3.75, 3.58, 3.46, 3.37, 3.30, 3.24, 3.19, 3.16, 3.12, 3.09, 3.07, 3.05, 3.03, 3.01, 2.96, 2.90, 2.85, 2.80, 2.75],
    // p=3
    [4.56, 4.16, 3.90, 3.73, 3.60, 3.49, 3.42, 3.36, 3.31, 3.27, 3.23, 3.20, 3.17, 3.15, 3.13, 3.11, 3.05, 3.00, 2.94, 2.89, 2.84],
    // p=4
    [4.69, 4.28, 4.01, 3.83, 3.69, 3.59, 3.51, 3.44, 3.39, 3.35, 3.31, 3.28, 3.25, 3.23, 3.21, 3.19, 3.13, 3.07, 3.01, 2.95, 2.90],
    // p=5
    [4.80, 4.37, 4.09, 3.90, 3.76, 3.65, 3.57, 3.50, 3.45, 3.40, 3.36, 3.33, 3.30, 3.28, 3.26, 3.24, 3.18, 3.12, 3.06, 3.00, 2.95],
    // p=6
    [4.88, 4.44, 4.16, 3.96, 3.82, 3.71, 3.63, 3.56, 3.50, 3.45, 3.41, 3.38, 3.35, 3.33, 3.30, 3.29, 3.22, 3.16, 3.10, 3.04, 2.98],
    // p=7
    [4.95, 4.50, 4.21, 4.02, 3.87, 3.76, 3.67, 3.60, 3.54, 3.49, 3.45, 3.42, 3.39, 3.37, 3.34, 3.32, 3.26, 3.19, 3.13, 3.07, 3.01],
    // p=8
    [5.01, 4.56, 4.26, 4.06, 3.91, 3.80, 3.71, 3.64, 3.58, 3.53, 3.49, 3.45, 3.42, 3.40, 3.37, 3.35, 3.29, 3.22, 3.16, 3.09, 3.04],
    // p=9
    [5.06, 4.60, 4.31, 4.10, 3.95, 3.83, 3.74, 3.67, 3.61, 3.56, 3.52, 3.48, 3.45, 3.43, 3.40, 3.38, 3.31, 3.25, 3.18, 3.12, 3.06],
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Look up the critical q value from the studentized range distribution.
///
/// # Arguments
/// * `k` — number of groups (2..=10)
/// * `df` — degrees of freedom within groups (≥ 1)
/// * `alpha` — significance level (0.05 or 0.01)
///
/// For `df` values between table entries, inverse-df interpolation is used.
/// For `df` > 120 the df = 120 value is returned (conservative).
///
/// # Example
/// ```
/// let q = tukey_test::q_critical(3, 12, 0.05).unwrap();
/// assert!((q - 3.77).abs() < 0.01);
/// ```
pub fn q_critical(k: usize, df: usize, alpha: f64) -> Result<f64, TukeyError> {
    if k < 2 {
        return Err(TukeyError::TooFewGroups);
    }
    if k > 10 {
        return Err(TukeyError::TooManyGroups(k));
    }
    if df < 1 {
        return Err(TukeyError::InsufficientDf);
    }

    let table = if (alpha - 0.05).abs() < 1e-10 {
        &Q_TABLE_05
    } else if (alpha - 0.01).abs() < 1e-10 {
        &Q_TABLE_01
    } else {
        return Err(TukeyError::UnsupportedAlpha(alpha));
    };

    let row = &table[k - 2];

    // df at or beyond the last table entry
    if df >= 120 {
        return Ok(row[24]);
    }

    // Exact match
    for (i, &d) in DF_VALUES.iter().enumerate() {
        if d == df {
            return Ok(row[i]);
        }
    }

    // Find bounding entries and interpolate on 1/df
    let mut lower_idx = 0;
    for (i, &d) in DF_VALUES.iter().enumerate() {
        if d < df {
            lower_idx = i;
        } else {
            break;
        }
    }
    let upper_idx = lower_idx + 1;

    let inv_df = 1.0 / df as f64;
    let inv_lo = 1.0 / DF_VALUES[lower_idx] as f64;
    let inv_hi = 1.0 / DF_VALUES[upper_idx] as f64;
    let t = (inv_df - inv_hi) / (inv_lo - inv_hi);

    Ok(row[lower_idx] * t + row[upper_idx] * (1.0 - t))
}

/// Look up the critical value from the Dunnett distribution (two-sided).
///
/// # Arguments
/// * `p` — number of treatment groups, excluding the control (1..=9)
/// * `df` — degrees of freedom within groups (≥ 5)
/// * `alpha` — significance level (0.05 or 0.01)
///
/// # Example
/// ```
/// let d = tukey_test::dunnett_critical(2, 20, 0.05).unwrap();
/// assert!((d - 2.28).abs() < 0.01);
/// ```
pub fn dunnett_critical(p: usize, df: usize, alpha: f64) -> Result<f64, TukeyError> {
    if p < 1 || p > 9 {
        return Err(TukeyError::TooManyTreatments(p));
    }
    if df < 5 {
        return Err(TukeyError::InsufficientDf);
    }

    let table = if (alpha - 0.05).abs() < 1e-10 {
        &DUNNETT_TABLE_05
    } else if (alpha - 0.01).abs() < 1e-10 {
        &DUNNETT_TABLE_01
    } else {
        return Err(TukeyError::UnsupportedAlpha(alpha));
    };

    let row = &table[p - 1];

    if df >= 120 {
        return Ok(row[20]);
    }

    // Exact match
    for (i, &d) in DUNNETT_DF_VALUES.iter().enumerate() {
        if d == df {
            return Ok(row[i]);
        }
    }

    // Interpolate on 1/df
    let mut lower_idx = 0;
    for (i, &d) in DUNNETT_DF_VALUES.iter().enumerate() {
        if d < df {
            lower_idx = i;
        } else {
            break;
        }
    }
    let upper_idx = lower_idx + 1;

    let inv_df = 1.0 / df as f64;
    let inv_lo = 1.0 / DUNNETT_DF_VALUES[lower_idx] as f64;
    let inv_hi = 1.0 / DUNNETT_DF_VALUES[upper_idx] as f64;
    let t = (inv_df - inv_hi) / (inv_lo - inv_hi);

    Ok(row[lower_idx] * t + row[upper_idx] * (1.0 - t))
}

/// Perform a Tukey HSD test on grouped data.
///
/// Accepts groups with unequal sizes (Tukey-Kramer adjustment is applied
/// automatically, which reduces to standard Tukey HSD when all groups are
/// equal in size).
///
/// # Arguments
/// * `data` — slice of groups; each group can be any type implementing `AsRef<[f64]>`
/// * `alpha` — significance level (0.05 or 0.01)
///
/// # Example
/// ```
/// use tukey_test::{tukey_hsd, TukeyResult};
///
/// let data = vec![
///     vec![6.0, 8.0, 4.0, 5.0, 3.0, 4.0],
///     vec![8.0, 12.0, 9.0, 11.0, 6.0, 8.0],
///     vec![13.0, 9.0, 11.0, 8.0, 12.0, 14.0],
/// ];
/// let result = tukey_hsd(&data, 0.05).unwrap();
/// assert_eq!(result.significant_pairs().len(), 2);
/// ```
pub fn tukey_hsd<T: AsRef<[f64]>>(data: &[T], alpha: f64) -> Result<TukeyResult, TukeyError> {
    let k = data.len();
    if k < 2 {
        return Err(TukeyError::TooFewGroups);
    }
    if k > 10 {
        return Err(TukeyError::TooManyGroups(k));
    }

    // Validate groups, compute means and sizes
    let mut group_means = Vec::with_capacity(k);
    let mut group_sizes = Vec::with_capacity(k);
    let mut n_total: usize = 0;

    for (i, group) in data.iter().enumerate() {
        let g = group.as_ref();
        if g.is_empty() {
            return Err(TukeyError::EmptyGroup(i));
        }
        let n = g.len();
        group_sizes.push(n);
        n_total += n;
        group_means.push(g.iter().sum::<f64>() / n as f64);
    }

    let df = n_total - k;
    if df < 1 {
        return Err(TukeyError::InsufficientDf);
    }

    // Mean square error (pooled within-group variance)
    let mut ss_within = 0.0;
    for (i, group) in data.iter().enumerate() {
        let mean = group_means[i];
        for &x in group.as_ref() {
            ss_within += (x - mean).powi(2);
        }
    }
    let mse = ss_within / df as f64;

    if mse == 0.0 {
        return Err(TukeyError::ZeroVariance);
    }

    let q_crit = q_critical(k, df, alpha)?;

    // Pairwise comparisons using Tukey-Kramer formula
    let mut comparisons = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            let raw_diff = group_means[i] - group_means[j];
            let mean_diff = raw_diff.abs();

            // Standard error: sqrt(MSE/2 * (1/n_i + 1/n_j))
            let se = (mse / 2.0 * (1.0 / group_sizes[i] as f64 + 1.0 / group_sizes[j] as f64))
                .sqrt();

            let q_stat = mean_diff / se;
            let significant = q_stat > q_crit;

            let ci_half = q_crit * se;
            comparisons.push(PairwiseComparison {
                group_i: i,
                group_j: j,
                mean_i: group_means[i],
                mean_j: group_means[j],
                mean_diff,
                q_statistic: q_stat,
                significant,
                ci_lower: raw_diff - ci_half,
                ci_upper: raw_diff + ci_half,
            });
        }
    }

    Ok(TukeyResult {
        groups: k,
        alpha,
        q_critical: q_crit,
        df,
        mse,
        group_means,
        group_sizes,
        comparisons,
    })
}

// ---------------------------------------------------------------------------
// Numerical helpers (F-distribution via incomplete beta function)
// ---------------------------------------------------------------------------

/// Log-gamma function using the Lanczos approximation.
fn ln_gamma(x: f64) -> f64 {
    const COEFFS: [f64; 6] = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015_f64;
    for (j, &c) in COEFFS.iter().enumerate() {
        ser += c / (x + 1.0 + j as f64);
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// Continued fraction evaluation for the regularized incomplete beta function.
fn betacf(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3.0e-12;
    const FPMIN: f64 = 1.0e-30;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0_f64;
    let mut d = (1.0 - qab * x / qap).recip();
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f = m as f64;

        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + 2.0 * m_f) * (a + 2.0 * m_f));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN { d = FPMIN; }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN { c = FPMIN; }
        d = d.recip();
        h *= d * c;

        // Odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + 2.0 * m_f) * (qap + 2.0 * m_f));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN { d = FPMIN; }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN { c = FPMIN; }
        d = d.recip();
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPS {
            return h;
        }
    }
    h
}

/// Regularized incomplete beta function I_x(a, b).
fn betai(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(x, a, b) / a
    } else {
        1.0 - front * betacf(1.0 - x, b, a) / b
    }
}

/// Survival function (1 − CDF) of the F-distribution: P(F ≥ x | d1, d2).
fn f_sf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    betai(d2 / (d2 + d1 * x), d2 / 2.0, d1 / 2.0)
}

// ---------------------------------------------------------------------------
// One-way ANOVA
// ---------------------------------------------------------------------------

/// Perform a one-way analysis of variance (ANOVA).
///
/// Tests the null hypothesis that all group means are equal.
///
/// # Example
/// ```
/// use tukey_test::one_way_anova;
///
/// let data = vec![
///     vec![6.0, 8.0, 4.0, 5.0, 3.0, 4.0],
///     vec![8.0, 12.0, 9.0, 11.0, 6.0, 8.0],
///     vec![13.0, 9.0, 11.0, 8.0, 12.0, 14.0],
/// ];
/// let result = one_way_anova(&data).unwrap();
/// assert!(result.p_value < 0.05);
/// ```
pub fn one_way_anova<T: AsRef<[f64]>>(data: &[T]) -> Result<AnovaResult, TukeyError> {
    let k = data.len();
    if k < 2 {
        return Err(TukeyError::TooFewGroups);
    }

    let mut group_means = Vec::with_capacity(k);
    let mut group_sizes = Vec::with_capacity(k);
    let mut n_total: usize = 0;
    let mut grand_sum = 0.0_f64;

    for (i, group) in data.iter().enumerate() {
        let g = group.as_ref();
        if g.is_empty() {
            return Err(TukeyError::EmptyGroup(i));
        }
        let n = g.len();
        let s: f64 = g.iter().sum();
        group_sizes.push(n);
        group_means.push(s / n as f64);
        n_total += n;
        grand_sum += s;
    }

    let df_between = k - 1;
    let df_within = n_total - k;
    if df_within < 1 {
        return Err(TukeyError::InsufficientDf);
    }
    let df_total = n_total - 1;
    let grand_mean = grand_sum / n_total as f64;

    let mut ss_between = 0.0;
    for (i, &mean) in group_means.iter().enumerate() {
        ss_between += group_sizes[i] as f64 * (mean - grand_mean).powi(2);
    }

    let mut ss_within = 0.0;
    for (i, group) in data.iter().enumerate() {
        let mean = group_means[i];
        for &x in group.as_ref() {
            ss_within += (x - mean).powi(2);
        }
    }

    let ss_total = ss_between + ss_within;
    let ms_between = ss_between / df_between as f64;
    let ms_within = ss_within / df_within as f64;

    let f_statistic = if ms_within > 0.0 {
        ms_between / ms_within
    } else {
        f64::INFINITY
    };

    let p_value = f_sf(f_statistic, df_between as f64, df_within as f64);

    Ok(AnovaResult {
        ss_between,
        ss_within,
        ss_total,
        df_between,
        df_within,
        df_total,
        ms_between,
        ms_within,
        f_statistic,
        p_value,
        grand_mean,
        group_means,
        group_sizes,
    })
}

// ---------------------------------------------------------------------------
// Games-Howell test
// ---------------------------------------------------------------------------

/// Perform a Games-Howell post-hoc test.
///
/// Unlike Tukey HSD, Games-Howell does **not** assume equal variances or
/// equal sample sizes. It uses per-pair standard errors and
/// Welch-Satterthwaite degrees of freedom.
///
/// # Example
/// ```
/// use tukey_test::games_howell;
///
/// // Groups with very different variances
/// let data = vec![
///     vec![4.0, 5.0, 3.0, 4.0, 6.0],
///     vec![20.0, 30.0, 25.0, 35.0, 28.0],
///     vec![5.0, 7.0, 6.0, 4.0, 5.0],
/// ];
/// let result = games_howell(&data, 0.05).unwrap();
/// assert!(!result.significant_pairs().is_empty());
/// ```
pub fn games_howell<T: AsRef<[f64]>>(data: &[T], alpha: f64) -> Result<GamesHowellResult, TukeyError> {
    let k = data.len();
    if k < 2 {
        return Err(TukeyError::TooFewGroups);
    }
    if k > 10 {
        return Err(TukeyError::TooManyGroups(k));
    }

    let mut group_means = Vec::with_capacity(k);
    let mut group_sizes = Vec::with_capacity(k);
    let mut group_variances = Vec::with_capacity(k);

    for (i, group) in data.iter().enumerate() {
        let g = group.as_ref();
        let n = g.len();
        if n < 2 {
            return Err(TukeyError::GroupTooSmall(i));
        }
        let mean = g.iter().sum::<f64>() / n as f64;
        let var = g.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        group_means.push(mean);
        group_sizes.push(n);
        group_variances.push(var);
    }

    let mut comparisons = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            let ni = group_sizes[i] as f64;
            let nj = group_sizes[j] as f64;
            let vi = group_variances[i];
            let vj = group_variances[j];

            let raw_diff = group_means[i] - group_means[j];
            let mean_diff = raw_diff.abs();

            // Games-Howell standard error
            let se = ((vi / ni + vj / nj) / 2.0).sqrt();

            // Welch-Satterthwaite degrees of freedom
            let a_term = vi / ni;
            let b_term = vj / nj;
            let numerator = (a_term + b_term).powi(2);
            let denominator =
                a_term.powi(2) / (ni - 1.0) + b_term.powi(2) / (nj - 1.0);
            let df_welch = (numerator / denominator).floor() as usize;
            let df_welch = df_welch.max(1);

            let q_crit = q_critical(k, df_welch, alpha)?;
            let q_stat = mean_diff / se;
            let significant = q_stat > q_crit;

            let ci_half = q_crit * se;
            comparisons.push(PairwiseComparison {
                group_i: i,
                group_j: j,
                mean_i: group_means[i],
                mean_j: group_means[j],
                mean_diff,
                q_statistic: q_stat,
                significant,
                ci_lower: raw_diff - ci_half,
                ci_upper: raw_diff + ci_half,
            });
        }
    }

    Ok(GamesHowellResult {
        groups: k,
        alpha,
        group_means,
        group_sizes,
        group_variances,
        comparisons,
    })
}

// ---------------------------------------------------------------------------
// Dunnett's test
// ---------------------------------------------------------------------------

/// Perform Dunnett's test (two-sided) comparing each treatment group against a
/// control group.
///
/// # Arguments
/// * `data` — slice of groups; each group can be any type implementing `AsRef<[f64]>`
/// * `control` — index of the control group (usually 0)
/// * `alpha` — significance level (0.05 or 0.01)
///
/// # Example
/// ```
/// use tukey_test::dunnett;
///
/// let data = vec![
///     vec![10.0, 12.0, 11.0, 9.0, 10.0],  // control
///     vec![15.0, 17.0, 14.0, 16.0, 15.0],  // treatment A
///     vec![11.0, 13.0, 10.0, 12.0, 11.0],  // treatment B
/// ];
/// let result = dunnett(&data, 0, 0.05).unwrap();
/// assert!(!result.significant_treatments().is_empty());
/// ```
pub fn dunnett<T: AsRef<[f64]>>(data: &[T], control: usize, alpha: f64) -> Result<DunnettResult, TukeyError> {
    let k = data.len();
    if k < 2 {
        return Err(TukeyError::TooFewGroups);
    }
    if control >= k {
        return Err(TukeyError::ControlGroupOutOfRange(control));
    }
    let p = k - 1; // number of treatment groups
    if p > 9 {
        return Err(TukeyError::TooManyTreatments(p));
    }

    let mut group_means = Vec::with_capacity(k);
    let mut group_sizes = Vec::with_capacity(k);
    let mut n_total: usize = 0;

    for (i, group) in data.iter().enumerate() {
        let g = group.as_ref();
        if g.is_empty() {
            return Err(TukeyError::EmptyGroup(i));
        }
        let n = g.len();
        group_sizes.push(n);
        n_total += n;
        group_means.push(g.iter().sum::<f64>() / n as f64);
    }

    let df = n_total - k;
    if df < 5 {
        return Err(TukeyError::InsufficientDf);
    }

    // Pooled MSE
    let mut ss_within = 0.0;
    for (i, group) in data.iter().enumerate() {
        let mean = group_means[i];
        for &x in group.as_ref() {
            ss_within += (x - mean).powi(2);
        }
    }
    let mse = ss_within / df as f64;

    if mse == 0.0 {
        return Err(TukeyError::ZeroVariance);
    }

    let d_crit = dunnett_critical(p, df, alpha)?;
    let control_mean = group_means[control];
    let n_control = group_sizes[control] as f64;

    let mut comparisons = Vec::new();
    for i in 0..k {
        if i == control {
            continue;
        }
        let raw_diff = group_means[i] - control_mean;
        let mean_diff = raw_diff.abs();
        let se = (mse * (1.0 / group_sizes[i] as f64 + 1.0 / n_control)).sqrt();
        let t_stat = mean_diff / se;
        let significant = t_stat > d_crit;

        let ci_half = d_crit * se;
        comparisons.push(DunnettComparison {
            treatment: i,
            treatment_mean: group_means[i],
            control_mean,
            mean_diff,
            t_statistic: t_stat,
            significant,
            ci_lower: raw_diff - ci_half,
            ci_upper: raw_diff + ci_half,
        });
    }

    Ok(DunnettResult {
        control,
        treatments: p,
        alpha,
        d_critical: d_crit,
        df,
        mse,
        group_means,
        group_sizes,
        comparisons,
    })
}

// ---------------------------------------------------------------------------
// CSV parsing
// ---------------------------------------------------------------------------

/// Parse CSV data in wide format (each column is a group).
///
/// Reads from any `std::io::Read` source — files, stdin, byte slices, etc.
/// The first row is treated as a header and skipped if any cell is not a valid
/// number. Columns may have unequal lengths (ragged data). Empty cells and
/// trailing commas are ignored.
///
/// # Example
///
/// ```
/// use tukey_test::parse_csv;
///
/// let csv = "control,treatment_a,treatment_b\n10,15,11\n12,17,13\n11,14,10\n";
/// let groups = parse_csv(csv.as_bytes()).unwrap();
/// assert_eq!(groups.len(), 3);
/// assert_eq!(groups[0], vec![10.0, 12.0, 11.0]);
/// ```
pub fn parse_csv<R: io::Read>(reader: R) -> Result<Vec<Vec<f64>>, TukeyError> {
    let content = {
        let mut buf = String::new();
        let mut r = reader;
        r.read_to_string(&mut buf)
            .map_err(|e| TukeyError::IoError(e.to_string()))?;
        buf
    };

    let mut lines = content.lines().peekable();
    let first_line = match lines.peek() {
        Some(line) => *line,
        None => return Err(TukeyError::EmptyCsv),
    };

    // Detect number of columns from first row
    let first_cells: Vec<&str> = first_line.split(',').collect();
    let num_cols = first_cells.len();
    if num_cols == 0 {
        return Err(TukeyError::EmptyCsv);
    }

    // Check if first row is a header (any cell fails to parse as f64)
    let is_header = first_cells.iter().any(|c| c.trim().parse::<f64>().is_err());
    if is_header {
        lines.next(); // skip header
    }

    let mut groups: Vec<Vec<f64>> = vec![Vec::new(); num_cols];

    for (line_idx, line) in lines.enumerate() {
        let line_num = if is_header { line_idx + 2 } else { line_idx + 1 };
        for (col, cell) in line.split(',').enumerate() {
            if col >= num_cols {
                break; // ignore extra columns
            }
            let trimmed = cell.trim();
            if trimmed.is_empty() {
                continue; // skip empty cells (ragged data)
            }
            let val: f64 = trimmed.parse().map_err(|_| TukeyError::ParseError {
                line: line_num,
                column: col + 1,
                value: trimmed.to_string(),
            })?;
            groups[col].push(val);
        }
    }

    // Remove any completely empty columns
    groups.retain(|g| !g.is_empty());

    if groups.is_empty() {
        return Err(TukeyError::EmptyCsv);
    }

    Ok(groups)
}

/// Convenience wrapper: parse CSV from a file path.
///
/// # Example
///
/// ```no_run
/// use tukey_test::parse_csv_file;
///
/// let groups = parse_csv_file("data.csv").unwrap();
/// ```
pub fn parse_csv_file(path: &str) -> Result<Vec<Vec<f64>>, TukeyError> {
    let file = std::fs::File::open(path)
        .map_err(|e| TukeyError::IoError(format!("{path}: {e}")))?;
    parse_csv(io::BufReader::new(file))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_critical_exact_lookup() {
        // k=3, df=12, alpha=0.05 should be 3.77
        let q = q_critical(3, 12, 0.05).unwrap();
        assert!((q - 3.77).abs() < 0.01, "got {q}");
    }

    #[test]
    fn q_critical_interpolation() {
        // k=2, df=25 is between df=24 (2.92) and df=30 (2.89)
        let q = q_critical(2, 25, 0.05).unwrap();
        assert!(q > 2.89 && q < 2.92, "got {q}");
    }

    #[test]
    fn q_critical_large_df() {
        // df >= 120 should return the df=120 entry
        let q = q_critical(4, 500, 0.05).unwrap();
        assert!((q - 3.68).abs() < 0.01, "got {q}");
    }

    #[test]
    fn q_critical_errors() {
        assert_eq!(q_critical(1, 10, 0.05), Err(TukeyError::TooFewGroups));
        assert_eq!(q_critical(11, 10, 0.05), Err(TukeyError::TooManyGroups(11)));
        assert_eq!(q_critical(3, 0, 0.05), Err(TukeyError::InsufficientDf));
        assert_eq!(q_critical(3, 10, 0.10), Err(TukeyError::UnsupportedAlpha(0.10)));
    }

    #[test]
    fn tukey_hsd_basic() {
        // Classic textbook example: 3 groups, equal size
        let data = vec![
            vec![6.0, 8.0, 4.0, 5.0, 3.0, 4.0],
            vec![8.0, 12.0, 9.0, 11.0, 6.0, 8.0],
            vec![13.0, 9.0, 11.0, 8.0, 12.0, 14.0],
        ];
        let result = tukey_hsd(&data, 0.05).unwrap();

        assert_eq!(result.groups, 3);
        assert_eq!(result.df, 15);
        assert_eq!(result.comparisons.len(), 3); // C(3,2) = 3 pairs

        // Group means: 5.0, 9.0, 11.167
        assert!((result.group_means[0] - 5.0).abs() < 0.01);
        assert!((result.group_means[1] - 9.0).abs() < 0.01);
        assert!((result.group_means[2] - 11.1667).abs() < 0.01);

        // Groups 0-1 and 0-2 should differ; 1-2 may or may not
        let sig = result.significant_pairs();
        assert!(
            sig.iter().any(|c| c.group_i == 0 && c.group_j == 2),
            "groups 0 and 2 should differ"
        );
    }

    #[test]
    fn tukey_hsd_unequal_sizes() {
        // Should work with unequal group sizes (Tukey-Kramer)
        let data = vec![
            vec![2.0, 4.0, 3.0],
            vec![10.0, 12.0, 11.0, 13.0, 9.0],
            vec![5.0, 6.0, 4.0, 7.0],
        ];
        let result = tukey_hsd(&data, 0.05).unwrap();
        assert_eq!(result.group_sizes, vec![3, 5, 4]);
        assert_eq!(result.df, 9); // 12 - 3
    }

    #[test]
    fn tukey_hsd_errors() {
        assert!(tukey_hsd(&[vec![1.0]], 0.05).is_err()); // too few groups
        assert!(tukey_hsd(&[vec![1.0], vec![]], 0.05).is_err()); // empty group
        assert!(tukey_hsd(&[vec![1.0], vec![1.0]], 0.05).is_err()); // insufficient df
        assert!(tukey_hsd(&[vec![5.0, 5.0], vec![5.0, 5.0]], 0.05).is_err()); // zero variance
    }

    #[test]
    fn display_output() {
        let data = vec![
            vec![23.0, 25.0, 21.0, 24.0],
            vec![30.0, 28.0, 33.0, 31.0],
            vec![22.0, 24.0, 20.0, 23.0],
        ];
        let result = tukey_hsd(&data, 0.05).unwrap();
        let output = format!("{result}");
        assert!(output.contains("Tukey HSD"));
        assert!(output.contains("q_critical"));
    }

    // --- ANOVA tests ---

    #[test]
    fn anova_basic() {
        let data = vec![
            vec![6.0, 8.0, 4.0, 5.0, 3.0, 4.0],
            vec![8.0, 12.0, 9.0, 11.0, 6.0, 8.0],
            vec![13.0, 9.0, 11.0, 8.0, 12.0, 14.0],
        ];
        let r = one_way_anova(&data).unwrap();
        assert_eq!(r.df_between, 2);
        assert_eq!(r.df_within, 15);
        assert_eq!(r.df_total, 17);
        // F should be significant
        assert!(r.p_value < 0.01, "p = {}", r.p_value);
        // Check SS_total = SS_between + SS_within
        assert!((r.ss_total - r.ss_between - r.ss_within).abs() < 1e-10);
    }

    #[test]
    fn anova_no_difference() {
        // Groups with same mean — F should be small, p large
        let data = vec![
            vec![10.0, 11.0, 9.0, 10.0],
            vec![10.0, 9.0, 11.0, 10.0],
        ];
        let r = one_way_anova(&data).unwrap();
        assert!(r.p_value > 0.5, "p = {}", r.p_value);
    }

    #[test]
    fn anova_display() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let r = one_way_anova(&data).unwrap();
        let output = format!("{r}");
        assert!(output.contains("One-Way ANOVA"));
        assert!(output.contains("Between"));
        assert!(output.contains("Within"));
    }

    #[test]
    fn anova_errors() {
        assert!(one_way_anova(&[vec![1.0]]).is_err());
        assert!(one_way_anova(&[vec![1.0], vec![]]).is_err());
    }

    // --- F-distribution helper tests ---

    #[test]
    fn f_sf_known_values() {
        // F(1, 1) at x=161.45 should give p ≈ 0.05 (approximately)
        // More stable check: F(2, 15) at x=3.68 gives p ≈ 0.05
        let p = f_sf(3.68, 2.0, 15.0);
        assert!((p - 0.05).abs() < 0.01, "p = {p}");

        // Very large F should give p ≈ 0
        let p = f_sf(100.0, 2.0, 15.0);
        assert!(p < 0.001, "p = {p}");

        // F = 0 should give p = 1
        let p = f_sf(0.0, 2.0, 15.0);
        assert!((p - 1.0).abs() < 1e-10, "p = {p}");
    }

    // --- Games-Howell tests ---

    #[test]
    fn games_howell_basic() {
        // Groups with different variances
        let data = vec![
            vec![4.0, 5.0, 3.0, 4.0, 6.0],
            vec![20.0, 30.0, 25.0, 35.0, 28.0],
            vec![5.0, 7.0, 6.0, 4.0, 5.0],
        ];
        let r = games_howell(&data, 0.05).unwrap();
        assert_eq!(r.groups, 3);
        assert_eq!(r.comparisons.len(), 3);

        // Group 1 (mean ~27.6) should differ from groups 0 and 2
        let sig = r.significant_pairs();
        assert!(
            sig.iter().any(|c| c.group_i == 0 && c.group_j == 1),
            "groups 0 and 1 should differ"
        );
        assert!(
            sig.iter().any(|c| c.group_i == 1 && c.group_j == 2),
            "groups 1 and 2 should differ"
        );
    }

    #[test]
    fn games_howell_unequal_sizes() {
        let data = vec![
            vec![2.0, 4.0, 3.0],
            vec![10.0, 12.0, 11.0, 13.0, 9.0],
            vec![5.0, 6.0, 4.0, 7.0],
        ];
        let r = games_howell(&data, 0.05).unwrap();
        assert_eq!(r.group_sizes, vec![3, 5, 4]);
    }

    #[test]
    fn games_howell_errors() {
        // Need at least 2 observations per group for variance
        let err = games_howell(&[vec![1.0], vec![2.0, 3.0]], 0.05).unwrap_err();
        assert_eq!(err, TukeyError::GroupTooSmall(0));
    }

    // --- Dunnett tests ---

    #[test]
    fn dunnett_critical_lookup() {
        // p=2, df=20, alpha=0.05 should be 2.28
        let d = dunnett_critical(2, 20, 0.05).unwrap();
        assert!((d - 2.28).abs() < 0.01, "got {d}");
    }

    #[test]
    fn dunnett_critical_interpolation() {
        // df=25 is between df=24 and df=30
        let d = dunnett_critical(1, 25, 0.05).unwrap();
        assert!(d > 2.04 && d < 2.06, "got {d}");
    }

    #[test]
    fn dunnett_critical_errors() {
        assert_eq!(dunnett_critical(0, 10, 0.05), Err(TukeyError::TooManyTreatments(0)));
        assert_eq!(dunnett_critical(10, 10, 0.05), Err(TukeyError::TooManyTreatments(10)));
        assert_eq!(dunnett_critical(1, 4, 0.05), Err(TukeyError::InsufficientDf));
        assert_eq!(dunnett_critical(1, 10, 0.10), Err(TukeyError::UnsupportedAlpha(0.10)));
    }

    #[test]
    fn dunnett_basic() {
        let data = vec![
            vec![10.0, 12.0, 11.0, 9.0, 10.0],  // control
            vec![15.0, 17.0, 14.0, 16.0, 15.0],  // treatment A
            vec![11.0, 13.0, 10.0, 12.0, 11.0],  // treatment B
        ];
        let r = dunnett(&data, 0, 0.05).unwrap();
        assert_eq!(r.control, 0);
        assert_eq!(r.treatments, 2);
        assert_eq!(r.comparisons.len(), 2);

        // Treatment A (mean 15.4) should differ from control (mean 10.4)
        let sig = r.significant_treatments();
        assert!(
            sig.iter().any(|c| c.treatment == 1),
            "treatment A should differ from control"
        );
    }

    #[test]
    fn dunnett_non_zero_control() {
        // Use the last group as control
        let data = vec![
            vec![15.0, 17.0, 14.0, 16.0, 15.0],
            vec![11.0, 13.0, 10.0, 12.0, 11.0],
            vec![10.0, 12.0, 11.0, 9.0, 10.0],  // control
        ];
        let r = dunnett(&data, 2, 0.05).unwrap();
        assert_eq!(r.control, 2);
        assert_eq!(r.comparisons.len(), 2);
    }

    #[test]
    fn dunnett_errors() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        assert_eq!(
            dunnett(&data, 5, 0.05).unwrap_err(),
            TukeyError::ControlGroupOutOfRange(5)
        );
    }

    // --- Generic input tests ---

    #[test]
    fn accepts_slice_refs() {
        let data: &[&[f64]] = &[
            &[6.0, 8.0, 4.0, 5.0, 3.0, 4.0],
            &[8.0, 12.0, 9.0, 11.0, 6.0, 8.0],
            &[13.0, 9.0, 11.0, 8.0, 12.0, 14.0],
        ];
        let r = tukey_hsd(data, 0.05).unwrap();
        assert_eq!(r.groups, 3);

        let a = one_way_anova(data).unwrap();
        assert!(a.p_value < 0.05);

        let g = games_howell(data, 0.05).unwrap();
        assert_eq!(g.comparisons.len(), 3);

        let d = dunnett(data, 0, 0.05).unwrap();
        assert_eq!(d.comparisons.len(), 2);
    }

    #[test]
    fn accepts_fixed_arrays() {
        let data = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ];
        let r = one_way_anova(&data).unwrap();
        assert!(r.p_value < 0.05);
    }

    // --- CSV parsing tests ---

    #[test]
    fn parse_csv_with_header() {
        let csv = "a,b,c\n1,4,7\n2,5,8\n3,6,9\n";
        let groups = parse_csv(csv.as_bytes()).unwrap();
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(groups[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(groups[2], vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn parse_csv_no_header() {
        let csv = "1,4,7\n2,5,8\n3,6,9\n";
        let groups = parse_csv(csv.as_bytes()).unwrap();
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn parse_csv_ragged() {
        let csv = "a,b\n1,4\n2,5\n3,\n";
        let groups = parse_csv(csv.as_bytes()).unwrap();
        assert_eq!(groups[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(groups[1], vec![4.0, 5.0]); // shorter column
    }

    #[test]
    fn parse_csv_into_test() {
        let csv = "control,treatment\n10,15\n12,17\n11,14\n9,16\n10,15\n";
        let groups = parse_csv(csv.as_bytes()).unwrap();
        let result = tukey_hsd(&groups, 0.05).unwrap();
        assert_eq!(result.groups, 2);
    }

    #[test]
    fn parse_csv_errors() {
        assert!(parse_csv("".as_bytes()).is_err()); // empty
        assert!(parse_csv("a,b\n1,foo\n".as_bytes()).is_err()); // bad value
    }
}
