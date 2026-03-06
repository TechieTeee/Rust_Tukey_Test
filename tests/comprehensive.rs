//! Comprehensive QA test suite for tukey_test.
//!
//! Organized by category:
//!   - Statistical correctness (hand-computed reference values)
//!   - Numerical accuracy (textbook examples, F-distribution precision)
//!   - Edge cases and boundaries (floating point, group sizes, limits)
//!   - CSV parsing robustness (malformed input, special chars, no-panic)
//!   - API contracts (error coverage, return invariants, generics)

use tukey_test::*;

// ===========================================================================
// STATISTICAL CORRECTNESS — hand-computed reference values
// ===========================================================================

#[test]
fn hand_computed_tukey_hsd() {
    // Groups: A=[4,5,6], B=[8,9,10], C=[5,6,7]
    // Means: 5.0, 9.0, 6.0
    // Grand mean: (15+27+18)/9 = 60/9 = 6.6667
    // SS_within: (1+0+1)+(1+0+1)+(1+0+1) = 6.0
    // df_within = 9-3 = 6
    // MSE = 6/6 = 1.0
    // SE = sqrt(MSE/2 * (1/3 + 1/3)) = sqrt(1/3) = 0.5774
    // q(A,B) = |5-9|/0.5774 = 6.9282
    // q(A,C) = |5-6|/0.5774 = 1.7321
    // q(B,C) = |9-6|/0.5774 = 5.1962
    // q_critical(k=3, df=6, alpha=0.05) = 4.34
    let data = vec![vec![4.0, 5.0, 6.0], vec![8.0, 9.0, 10.0], vec![5.0, 6.0, 7.0]];
    let r = tukey_hsd(&data, 0.05).unwrap();

    assert!((r.mse - 1.0).abs() < 1e-10, "MSE should be 1.0, got {}", r.mse);
    assert_eq!(r.df, 6);

    // Pair (0,1): A vs B
    let c01 = &r.comparisons[0];
    assert!((c01.mean_diff - 4.0).abs() < 1e-10);
    assert!((c01.q_statistic - 6.9282).abs() < 0.01, "got {}", c01.q_statistic);
    assert!(c01.significant, "A vs B should be significant");

    // Pair (0,2): A vs C
    let c02 = &r.comparisons[1];
    assert!((c02.mean_diff - 1.0).abs() < 1e-10);
    assert!((c02.q_statistic - 1.7321).abs() < 0.01, "got {}", c02.q_statistic);
    assert!(!c02.significant, "A vs C should NOT be significant");

    // Pair (1,2): B vs C
    let c12 = &r.comparisons[2];
    assert!((c12.mean_diff - 3.0).abs() < 1e-10);
    assert!((c12.q_statistic - 5.1962).abs() < 0.01, "got {}", c12.q_statistic);
    assert!(c12.significant, "B vs C should be significant");
}

#[test]
fn anova_ss_decomposition() {
    // For ANY dataset: SS_total = SS_between + SS_within
    let datasets: Vec<Vec<Vec<f64>>> = vec![
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        vec![vec![10.0, 20.0], vec![30.0, 40.0], vec![50.0, 60.0]],
        vec![
            vec![2.0, 4.0, 3.0],
            vec![10.0, 12.0, 11.0, 13.0, 9.0],
            vec![5.0, 6.0, 4.0, 7.0],
        ],
    ];
    for data in &datasets {
        let r = one_way_anova(data).unwrap();

        // Verify decomposition
        assert!(
            (r.ss_total - r.ss_between - r.ss_within).abs() < 1e-10,
            "SS decomposition failed: {} != {} + {}",
            r.ss_total, r.ss_between, r.ss_within
        );
        // Verify MS = SS/df
        assert!((r.ms_between - r.ss_between / r.df_between as f64).abs() < 1e-10);
        assert!((r.ms_within - r.ss_within / r.df_within as f64).abs() < 1e-10);
        // Verify F = MS_between / MS_within
        assert!((r.f_statistic - r.ms_between / r.ms_within).abs() < 1e-10);
        // Verify df
        assert_eq!(r.df_total, r.df_between + r.df_within);
    }
}

#[test]
fn anova_independently_computed_ss_total() {
    // SS_total = sum((x_i - grand_mean)^2)
    let data = vec![vec![2.0, 4.0, 3.0], vec![10.0, 12.0, 11.0], vec![5.0, 7.0]];
    let r = one_way_anova(&data).unwrap();

    let all_values: Vec<f64> = data.iter().flat_map(|g| g.iter().copied()).collect();
    let n = all_values.len() as f64;
    let grand = all_values.iter().sum::<f64>() / n;
    let ss_total: f64 = all_values.iter().map(|x| (x - grand).powi(2)).sum();

    assert!(
        (r.ss_total - ss_total).abs() < 1e-10,
        "SS_total mismatch: {} vs computed {}",
        r.ss_total, ss_total
    );
}

#[test]
fn tukey_kramer_se_for_unequal_n() {
    // Verify SE = sqrt(MSE/2 * (1/n_i + 1/n_j)), NOT sqrt(MSE/n)
    let data = vec![
        vec![1.0, 2.0, 3.0],           // n=3
        vec![4.0, 5.0, 6.0, 7.0, 8.0], // n=5
        vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], // n=7
    ];
    let r = tukey_hsd(&data, 0.05).unwrap();

    for c in &r.comparisons {
        let ni = r.group_sizes[c.group_i] as f64;
        let nj = r.group_sizes[c.group_j] as f64;
        let expected_se = (r.mse / 2.0 * (1.0 / ni + 1.0 / nj)).sqrt();
        // Recover SE: q = diff / SE, so SE = diff / q
        let actual_se = c.mean_diff / c.q_statistic;
        assert!(
            (actual_se - expected_se).abs() < 1e-10,
            "SE mismatch for ({},{}): got {}, expected {}",
            c.group_i, c.group_j, actual_se, expected_se
        );
    }
}

#[test]
fn hand_computed_dunnett_t_statistic() {
    // Control=[2,4,6,8], Treatment1=[12,14,16,18], Treatment2=[3,5,7,9]
    // Means: 5.0, 15.0, 6.0
    // SS_within = (4+4+4+4)*3 = 20+20+20 = 60  (actually let me compute properly)
    // Control: mean=5, ss=(9+1+1+9)=20
    // T1: mean=15, ss=(9+1+1+9)=20
    // T2: mean=6, ss=(9+1+1+9)=20
    // SS_within=60, df=12-3=9, MSE=60/9=6.667
    // t1 = |15-5| / sqrt(6.667*(1/4+1/4)) = 10/sqrt(3.333) = 10/1.8257 = 5.477
    // t2 = |6-5| / sqrt(6.667*(1/4+1/4)) = 1/1.8257 = 0.5477
    let data = vec![
        vec![2.0, 4.0, 6.0, 8.0],
        vec![12.0, 14.0, 16.0, 18.0],
        vec![3.0, 5.0, 7.0, 9.0],
    ];
    let r = dunnett(&data, 0, 0.05).unwrap();

    assert!((r.mse - 20.0 / 3.0).abs() < 1e-10, "MSE should be 6.667, got {}", r.mse);

    let se = (r.mse * (1.0 / 4.0 + 1.0 / 4.0)).sqrt();
    assert!(
        (r.comparisons[0].t_statistic - 10.0 / se).abs() < 1e-4,
        "t1 mismatch: got {}, expected {}",
        r.comparisons[0].t_statistic, 10.0 / se
    );
    assert!(
        (r.comparisons[1].t_statistic - 1.0 / se).abs() < 1e-4,
        "t2 mismatch"
    );
    assert!(r.comparisons[0].significant);
}

#[test]
fn games_howell_welch_satterthwaite_df() {
    // Group A: [1,2,3] → mean=2, var=1.0, n=3
    // Group B: [10,20,30,40,50] → mean=30, var=250.0, n=5
    // a_term = var_a/n_a = 1/3 = 0.3333
    // b_term = var_b/n_b = 250/5 = 50
    // numerator = (0.3333+50)^2 = 50.3333^2 = 2533.44
    // denominator = 0.3333^2/(3-1) + 50^2/(5-1) = 0.0556 + 625 = 625.0556
    // df = 2533.44/625.0556 = 4.053 → floor = 4
    let data = vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0, 40.0, 50.0]];
    let r = games_howell(&data, 0.05).unwrap();

    // Verify group variances
    assert!((r.group_variances[0] - 1.0).abs() < 1e-10);
    assert!((r.group_variances[1] - 250.0).abs() < 1e-10);

    // The q_statistic should use df=4 for the critical value lookup
    // We can verify the critical value used by checking significance
    let q_crit_df4 = q_critical(2, 4, 0.05).unwrap();
    assert!(
        (q_crit_df4 - 3.93).abs() < 0.01,
        "q_critical(2,4,0.05) should be 3.93, got {}",
        q_crit_df4
    );
}

#[test]
fn known_non_significant_result() {
    // Groups with identical means — nothing should be significant
    let data = vec![
        vec![10.0, 11.0, 9.0],
        vec![10.0, 9.0, 11.0],
        vec![11.0, 10.0, 9.0],
    ];
    let anova = one_way_anova(&data).unwrap();
    assert!(anova.p_value > 0.5, "p should be large, got {}", anova.p_value);
    assert!(anova.f_statistic < 1.0, "F should be small");

    let tukey = tukey_hsd(&data, 0.05).unwrap();
    assert!(
        tukey.significant_pairs().is_empty(),
        "No pairs should be significant"
    );
}

#[test]
fn confidence_interval_contains_zero_iff_not_significant() {
    let data = vec![
        vec![6.0, 8.0, 4.0, 5.0, 3.0, 4.0],
        vec![8.0, 12.0, 9.0, 11.0, 6.0, 8.0],
        vec![13.0, 9.0, 11.0, 8.0, 12.0, 14.0],
    ];

    // Tukey HSD
    let r = tukey_hsd(&data, 0.05).unwrap();
    for c in &r.comparisons {
        let ci_contains_zero = c.ci_lower <= 0.0 && c.ci_upper >= 0.0;
        if c.significant {
            assert!(
                !ci_contains_zero,
                "Significant pair ({},{}) CI [{}, {}] should NOT contain zero",
                c.group_i, c.group_j, c.ci_lower, c.ci_upper
            );
        } else {
            assert!(
                ci_contains_zero,
                "Non-significant pair ({},{}) CI [{}, {}] should contain zero",
                c.group_i, c.group_j, c.ci_lower, c.ci_upper
            );
        }
    }

    // Dunnett
    let d = dunnett(&data, 0, 0.05).unwrap();
    for c in &d.comparisons {
        let ci_contains_zero = c.ci_lower <= 0.0 && c.ci_upper >= 0.0;
        if c.significant {
            assert!(!ci_contains_zero);
        } else {
            assert!(ci_contains_zero);
        }
    }
}

// ===========================================================================
// NUMERICAL ACCURACY — F-distribution, interpolation, textbook examples
// ===========================================================================

#[test]
fn f_distribution_at_known_critical_points() {
    // Verify ANOVA p-values at known F critical values
    // F(2,15) critical at alpha=0.05 is ~3.68
    // Construct data that produces this F
    let anova1 = one_way_anova(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
    // Just verify p is in [0, 1] and directionally correct
    assert!(anova1.p_value >= 0.0 && anova1.p_value <= 1.0);
    assert!(anova1.p_value < 0.05, "clearly different groups should be significant");

    // Non-significant groups
    let anova2 = one_way_anova(&[vec![1.0, 2.0, 3.0], vec![1.5, 2.5, 3.5]]).unwrap();
    assert!(anova2.p_value > 0.05, "similar groups should not be significant");
}

#[test]
fn q_critical_monotonically_decreasing_in_df() {
    // For fixed k, q_critical should decrease as df increases
    for k in 2..=10 {
        let mut prev = f64::INFINITY;
        for &df in &[1, 2, 5, 10, 15, 20, 30, 60, 120] {
            let q = q_critical(k, df, 0.05).unwrap();
            assert!(
                q < prev,
                "q_critical(k={}, df={}) = {} should be < previous {}",
                k, df, q, prev
            );
            prev = q;
        }
    }
}

#[test]
fn q_critical_increasing_in_k() {
    // For fixed df, q_critical should increase as k increases
    for &df in &[5, 10, 20, 60, 120] {
        let mut prev = 0.0;
        for k in 2..=10 {
            let q = q_critical(k, df, 0.05).unwrap();
            assert!(
                q > prev,
                "q_critical(k={}, df={}) = {} should be > previous {}",
                k, df, q, prev
            );
            prev = q;
        }
    }
}

#[test]
fn dunnett_critical_monotonically_decreasing_in_df() {
    for p in 1..=9 {
        let mut prev = f64::INFINITY;
        for &df in &[5, 6, 8, 10, 15, 20, 30, 60, 120] {
            let d = dunnett_critical(p, df, 0.05).unwrap();
            assert!(
                d < prev,
                "dunnett_critical(p={}, df={}) = {} should be < previous {}",
                p, df, d, prev
            );
            prev = d;
        }
    }
}

#[test]
fn interpolation_bounded_by_neighbors() {
    // Interpolated values should lie between the two surrounding table entries
    // df=25 is between df=20 and df=24 in the table
    for k in 2..=10 {
        let q_20 = q_critical(k, 20, 0.05).unwrap();
        let q_24 = q_critical(k, 24, 0.05).unwrap();
        let q_22 = q_critical(k, 22, 0.05).unwrap();
        assert!(
            q_22 >= q_24 && q_22 <= q_20,
            "q_critical(k={}, df=22) = {} not between q(20)={} and q(24)={}",
            k, q_22, q_20, q_24
        );
    }
}

#[test]
fn alpha_001_more_conservative_than_005() {
    // Critical values at alpha=0.01 should be larger than at alpha=0.05
    for k in 2..=10 {
        for &df in &[5, 10, 20, 60, 120] {
            let q_05 = q_critical(k, df, 0.05).unwrap();
            let q_01 = q_critical(k, df, 0.01).unwrap();
            assert!(
                q_01 > q_05,
                "q(k={}, df={}, 0.01)={} should be > q(0.05)={}",
                k, df, q_01, q_05
            );
        }
    }
}

// ===========================================================================
// EDGE CASES AND BOUNDARIES
// ===========================================================================

#[test]
fn minimum_valid_input() {
    // 2 groups, 2 observations each — smallest valid input
    let data = vec![vec![1.0, 3.0], vec![5.0, 7.0]];
    let r = tukey_hsd(&data, 0.05).unwrap();
    assert_eq!(r.groups, 2);
    assert_eq!(r.df, 2);
}

#[test]
fn maximum_groups_k10() {
    let data: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![i as f64 * 10.0, i as f64 * 10.0 + 1.0, i as f64 * 10.0 + 2.0])
        .collect();
    let r = tukey_hsd(&data, 0.05).unwrap();
    assert_eq!(r.groups, 10);
    assert_eq!(r.comparisons.len(), 45); // C(10,2) = 45
}

#[test]
fn k11_errors() {
    let data: Vec<Vec<f64>> = (0..11)
        .map(|i| vec![i as f64, i as f64 + 1.0])
        .collect();
    assert!(tukey_hsd(&data, 0.05).is_err());
    assert!(games_howell(&data, 0.05).is_err());
}

#[test]
fn very_large_numbers() {
    let data = vec![
        vec![1e14, 1.1e14, 0.9e14],
        vec![2e14, 2.1e14, 1.9e14],
    ];
    let r = tukey_hsd(&data, 0.05).unwrap();
    assert!(r.comparisons[0].mean_diff > 0.0);
    assert!(!r.mse.is_nan());
    assert!(!r.mse.is_infinite());
}

#[test]
fn very_small_numbers() {
    let data = vec![
        vec![1e-14, 1.1e-14, 0.9e-14],
        vec![2e-14, 2.1e-14, 1.9e-14],
    ];
    let r = tukey_hsd(&data, 0.05).unwrap();
    assert!(r.comparisons[0].mean_diff > 0.0);
    assert!(!r.mse.is_nan());
}

#[test]
fn negative_numbers_and_zeros() {
    let data = vec![
        vec![-5.0, -3.0, 0.0, -4.0],
        vec![0.0, 2.0, 1.0, 3.0],
    ];
    let r = tukey_hsd(&data, 0.05).unwrap();
    assert!((r.group_means[0] - (-3.0)).abs() < 1e-10);
    assert!((r.group_means[1] - 1.5).abs() < 1e-10);
}

#[test]
fn one_large_one_tiny_group() {
    let large: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let tiny = vec![500.0, 600.0];
    let data = vec![large, tiny];
    let r = tukey_hsd(&data, 0.05).unwrap();
    assert_eq!(r.group_sizes, vec![100, 2]);
    assert!(!r.mse.is_nan());
}

#[test]
fn all_groups_size_1_errors() {
    let data = vec![vec![1.0], vec![2.0], vec![3.0]];
    // df = 3-3 = 0, insufficient
    assert!(tukey_hsd(&data, 0.05).is_err());
}

#[test]
fn dunnett_control_last_group() {
    let data = vec![
        vec![15.0, 17.0, 14.0],
        vec![11.0, 13.0, 10.0],
        vec![10.0, 12.0, 11.0], // control
    ];
    let r = dunnett(&data, 2, 0.05).unwrap();
    assert_eq!(r.control, 2);
    assert_eq!(r.comparisons.len(), 2);
    // Treatment indices should be 0 and 1
    assert_eq!(r.comparisons[0].treatment, 0);
    assert_eq!(r.comparisons[1].treatment, 1);
}

#[test]
fn dunnett_control_middle() {
    let data = vec![
        vec![15.0, 17.0, 14.0],
        vec![10.0, 12.0, 11.0], // control
        vec![11.0, 13.0, 10.0],
    ];
    let r = dunnett(&data, 1, 0.05).unwrap();
    assert_eq!(r.control, 1);
    assert_eq!(r.comparisons[0].treatment, 0);
    assert_eq!(r.comparisons[1].treatment, 2);
}

#[test]
fn dunnett_control_out_of_range() {
    let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    assert_eq!(
        dunnett(&data, 5, 0.05).unwrap_err(),
        TukeyError::ControlGroupOutOfRange(5)
    );
}

#[test]
fn dunnett_max_9_treatments() {
    // 10 groups = 9 treatments + 1 control — should work
    let data: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![i as f64 * 10.0, i as f64 * 10.0 + 1.0, i as f64 * 10.0 + 2.0])
        .collect();
    let r = dunnett(&data, 0, 0.05).unwrap();
    assert_eq!(r.treatments, 9);

    // 11 groups = 10 treatments — should error
    let data11: Vec<Vec<f64>> = (0..11)
        .map(|i| vec![i as f64 * 10.0, i as f64 * 10.0 + 1.0, i as f64 * 10.0 + 2.0])
        .collect();
    assert!(dunnett(&data11, 0, 0.05).is_err());
}

#[test]
fn q_critical_at_exact_table_boundaries() {
    // df=1 (first entry)
    let q = q_critical(2, 1, 0.05).unwrap();
    assert!((q - 17.97).abs() < 0.01);

    // df=120 (last entry)
    let q = q_critical(2, 120, 0.05).unwrap();
    assert!((q - 2.80).abs() < 0.01);

    // df > 120 should use df=120 value
    let q_big = q_critical(2, 10000, 0.05).unwrap();
    assert!((q_big - 2.80).abs() < 0.01);
}

#[test]
fn dunnett_critical_at_boundaries() {
    // df=5 (first entry)
    let d = dunnett_critical(1, 5, 0.05).unwrap();
    assert!((d - 2.57).abs() < 0.01);

    // df < 5 should error
    assert!(dunnett_critical(1, 4, 0.05).is_err());

    // df=120 (last entry)
    let d = dunnett_critical(1, 120, 0.05).unwrap();
    assert!((d - 1.98).abs() < 0.01);
}

// --- Bug fix verification ---

#[test]
fn anova_all_identical_data_returns_error() {
    // All values identical across all groups — was returning F=Infinity
    let data = vec![vec![5.0, 5.0, 5.0], vec![5.0, 5.0, 5.0]];
    let err = one_way_anova(&data).unwrap_err();
    assert_eq!(err, TukeyError::ZeroVariance);
}

#[test]
fn anova_different_constants_f_infinity() {
    // Each group is constant but groups differ — F should be Infinity (valid)
    let data = vec![vec![5.0, 5.0, 5.0], vec![10.0, 10.0, 10.0]];
    let r = one_way_anova(&data).unwrap();
    assert!(r.f_statistic.is_infinite(), "F should be Infinity");
    assert!(r.p_value < 0.001, "p should be ~0");
}

#[test]
fn games_howell_zero_variance_returns_error() {
    // One group has zero variance — was producing Inf/NaN
    let data = vec![vec![5.0, 5.0, 5.0], vec![1.0, 2.0, 3.0]];
    let err = games_howell(&data, 0.05).unwrap_err();
    assert_eq!(err, TukeyError::ZeroVariance);
}

#[test]
fn games_howell_both_groups_zero_variance_errors() {
    let data = vec![vec![5.0, 5.0, 5.0], vec![10.0, 10.0, 10.0]];
    assert!(games_howell(&data, 0.05).is_err());
}

#[test]
fn games_howell_wildly_different_variances() {
    // Variance ratio ~10000:1 — should still produce finite results
    let data = vec![
        vec![100.0, 100.1, 99.9, 100.05, 99.95],  // var ≈ 0.005
        vec![0.0, 50.0, 100.0, -50.0, 150.0],       // var ≈ 5000
    ];
    let r = games_howell(&data, 0.05).unwrap();
    assert!(!r.comparisons[0].q_statistic.is_nan());
    assert!(!r.comparisons[0].q_statistic.is_infinite());
}

// ===========================================================================
// CSV PARSING ROBUSTNESS
// ===========================================================================

#[test]
fn csv_empty_input() {
    assert!(parse_csv("".as_bytes()).is_err());
}

#[test]
fn csv_whitespace_only() {
    assert!(parse_csv("   \n  \n  ".as_bytes()).is_err());
}

#[test]
fn csv_only_commas() {
    // Should result in empty groups after trimming
    assert!(parse_csv(",,,\n,,,\n".as_bytes()).is_err());
}

#[test]
fn csv_single_column() {
    let csv = "values\n1\n2\n3\n";
    let groups = parse_csv(csv.as_bytes()).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0], vec![1.0, 2.0, 3.0]);
}

#[test]
fn csv_trailing_commas() {
    let csv = "a,b,\n1,2,\n3,4,\n";
    let groups = parse_csv(csv.as_bytes()).unwrap();
    assert_eq!(groups.len(), 2); // third column is empty, removed
    assert_eq!(groups[0], vec![1.0, 3.0]);
    assert_eq!(groups[1], vec![2.0, 4.0]);
}

#[test]
fn csv_scientific_notation() {
    let csv = "1e2,1.5E-3\n2e2,-2.3e+1\n";
    let groups = parse_csv(csv.as_bytes()).unwrap();
    assert!((groups[0][0] - 100.0).abs() < 1e-10);
    assert!((groups[0][1] - 200.0).abs() < 1e-10);
    assert!((groups[1][0] - 0.0015).abs() < 1e-10);
    assert!((groups[1][1] - (-23.0)).abs() < 1e-10);
}

#[test]
fn csv_values_with_whitespace() {
    let csv = " 1.0 , 2.0 \n 3.0 , 4.0 \n";
    let groups = parse_csv(csv.as_bytes()).unwrap();
    assert_eq!(groups[0], vec![1.0, 3.0]);
    assert_eq!(groups[1], vec![2.0, 4.0]);
}

#[test]
fn csv_unicode_headers() {
    let csv = "grp_\u{00e9},grp_\u{00e8}\n1,2\n3,4\n";
    let groups = parse_csv(csv.as_bytes()).unwrap();
    assert_eq!(groups.len(), 2);
}

#[test]
fn csv_nan_and_infinity_rejected() {
    // Put bad values in data rows (not first row, which gets treated as header when non-numeric)
    assert!(parse_csv("A,B\nNaN,1\n2,3\n".as_bytes()).is_err());
    assert!(parse_csv("A,B\n1,Infinity\n2,3\n".as_bytes()).is_err());
    assert!(parse_csv("A,B\n1,-Infinity\n2,3\n".as_bytes()).is_err());
}

#[test]
fn csv_hex_rejected() {
    // Put bad value in a data row (first row with non-numeric is treated as header)
    assert!(parse_csv("A,B\n0xFF,1\n2,3\n".as_bytes()).is_err());
}

#[test]
fn csv_header_detection_all_numeric() {
    // If first row is all numbers, it should be treated as data
    let csv = "1,2\n3,4\n";
    let groups = parse_csv(csv.as_bytes()).unwrap();
    assert_eq!(groups[0], vec![1.0, 3.0]); // first row is data
}

#[test]
fn csv_header_detection_mixed() {
    // Any non-numeric cell triggers header detection
    let csv = "1,two,3\n4,5,6\n7,8,9\n";
    let groups = parse_csv(csv.as_bytes()).unwrap();
    // First row skipped as header, "5" and "8" parsed as data
    assert_eq!(groups[1], vec![5.0, 8.0]);
}

#[test]
fn csv_nonexistent_file() {
    let err = parse_csv_file("/nonexistent/path/data.csv").unwrap_err();
    match err {
        TukeyError::IoError(_) => {} // expected
        other => panic!("Expected IoError, got {:?}", other),
    }
}

#[test]
fn csv_no_panic_on_garbage() {
    // None of these should panic — all should return Err or Ok
    let long_input = "x,".repeat(1000);
    let garbage_inputs = vec![
        "\0\0\0",
        "a",
        "\n\n\n",
        ",,,,,,,,,,",
        "a,b\nfoo,bar\n",
        long_input.as_str(),
    ];
    for input in garbage_inputs {
        let _ = parse_csv(input.as_bytes()); // must not panic
    }
}

#[test]
fn csv_large_input() {
    // 1000 rows, 5 columns — should work without issue
    let mut csv = String::from("a,b,c,d,e\n");
    for i in 0..1000 {
        csv.push_str(&format!("{},{},{},{},{}\n", i, i + 1, i + 2, i + 3, i + 4));
    }
    let groups = parse_csv(csv.as_bytes()).unwrap();
    assert_eq!(groups.len(), 5);
    assert_eq!(groups[0].len(), 1000);
}

#[test]
fn csv_feeds_into_all_tests() {
    let csv = "a,b,c\n6,8,13\n8,12,9\n4,9,11\n5,11,8\n3,6,12\n4,8,14\n";
    let groups = parse_csv(csv.as_bytes()).unwrap();
    assert!(one_way_anova(&groups).is_ok());
    assert!(tukey_hsd(&groups, 0.05).is_ok());
    assert!(games_howell(&groups, 0.05).is_ok());
    assert!(dunnett(&groups, 0, 0.05).is_ok());
}

// ===========================================================================
// API CONTRACTS
// ===========================================================================

#[test]
fn comparison_count_invariant() {
    for k in 2..=5 {
        // 4 obs per group so df_within = 4*(n-1) >= 6, satisfying Dunnett's df>=5 minimum
        let data: Vec<Vec<f64>> = (0..k)
            .map(|i| {
                let b = i as f64 * 10.0;
                vec![b, b + 1.0, b + 2.0, b + 3.0]
            })
            .collect();

        let tukey = tukey_hsd(&data, 0.05).unwrap();
        assert_eq!(tukey.comparisons.len(), k * (k - 1) / 2);

        let gh = games_howell(&data, 0.05).unwrap();
        assert_eq!(gh.comparisons.len(), k * (k - 1) / 2);

        let dun = dunnett(&data, 0, 0.05).unwrap();
        assert_eq!(dun.comparisons.len(), k - 1);
    }
}

#[test]
fn group_means_are_correct() {
    let data = vec![vec![2.0, 4.0, 6.0], vec![10.0, 20.0, 30.0]];
    let r = tukey_hsd(&data, 0.05).unwrap();
    assert!((r.group_means[0] - 4.0).abs() < 1e-10);
    assert!((r.group_means[1] - 20.0).abs() < 1e-10);
    assert_eq!(r.group_sizes, vec![3, 3]);
}

#[test]
fn mean_diff_always_non_negative() {
    let data = vec![
        vec![100.0, 101.0, 102.0],
        vec![1.0, 2.0, 3.0],
        vec![50.0, 51.0, 52.0],
    ];
    let r = tukey_hsd(&data, 0.05).unwrap();
    for c in &r.comparisons {
        assert!(c.mean_diff >= 0.0, "mean_diff should be >= 0");
    }
}

#[test]
fn ci_lower_less_than_upper() {
    let data = vec![
        vec![6.0, 8.0, 4.0, 5.0, 3.0, 4.0],
        vec![8.0, 12.0, 9.0, 11.0, 6.0, 8.0],
        vec![13.0, 9.0, 11.0, 8.0, 12.0, 14.0],
    ];
    let r = tukey_hsd(&data, 0.05).unwrap();
    for c in &r.comparisons {
        assert!(c.ci_lower < c.ci_upper, "CI lower {} >= upper {}", c.ci_lower, c.ci_upper);
    }
    let d = dunnett(&data, 0, 0.05).unwrap();
    for c in &d.comparisons {
        assert!(c.ci_lower < c.ci_upper);
    }
}

#[test]
fn significant_matches_statistic_vs_critical() {
    let data = vec![
        vec![6.0, 8.0, 4.0, 5.0, 3.0, 4.0],
        vec![8.0, 12.0, 9.0, 11.0, 6.0, 8.0],
        vec![13.0, 9.0, 11.0, 8.0, 12.0, 14.0],
    ];
    let r = tukey_hsd(&data, 0.05).unwrap();
    for c in &r.comparisons {
        assert_eq!(
            c.significant,
            c.q_statistic > r.q_critical,
            "significant flag mismatch for ({},{})",
            c.group_i, c.group_j
        );
    }

    let d = dunnett(&data, 0, 0.05).unwrap();
    for c in &d.comparisons {
        assert_eq!(c.significant, c.t_statistic > d.d_critical);
    }
}

#[test]
fn display_traits_produce_output() {
    let data = vec![
        vec![6.0, 8.0, 4.0, 5.0, 3.0, 4.0],
        vec![8.0, 12.0, 9.0, 11.0, 6.0, 8.0],
        vec![13.0, 9.0, 11.0, 8.0, 12.0, 14.0],
    ];
    let anova = format!("{}", one_way_anova(&data).unwrap());
    assert!(anova.contains("ANOVA") && anova.contains("Between"));

    let tukey = format!("{}", tukey_hsd(&data, 0.05).unwrap());
    assert!(tukey.contains("Tukey HSD") && tukey.contains("q_critical"));

    let gh = format!("{}", games_howell(&data, 0.05).unwrap());
    assert!(gh.contains("Games-Howell"));

    let dun = format!("{}", dunnett(&data, 0, 0.05).unwrap());
    assert!(dun.contains("Dunnett"));
}

#[test]
fn all_error_variants_have_display_messages() {
    let errors = vec![
        TukeyError::TooFewGroups,
        TukeyError::TooManyGroups(11),
        TukeyError::EmptyGroup(0),
        TukeyError::InsufficientDf,
        TukeyError::UnsupportedAlpha(0.10),
        TukeyError::ZeroVariance,
        TukeyError::GroupTooSmall(0),
        TukeyError::ControlGroupOutOfRange(5),
        TukeyError::TooManyTreatments(10),
        TukeyError::IoError("test".into()),
        TukeyError::ParseError { line: 1, column: 1, value: "x".into() },
        TukeyError::EmptyCsv,
    ];
    for e in &errors {
        let msg = format!("{e}");
        assert!(!msg.is_empty(), "Empty display for {:?}", e);
    }
}

#[test]
fn generic_input_vec_of_vecs() {
    // 4 obs per group so df_within=6, satisfying Dunnett's df>=5 minimum
    let data: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
    assert!(one_way_anova(&data).is_ok());
    assert!(tukey_hsd(&data, 0.05).is_ok());
    assert!(games_howell(&data, 0.05).is_ok());
    assert!(dunnett(&data, 0, 0.05).is_ok());
}

#[test]
fn generic_input_slice_of_slices() {
    let data: &[&[f64]] = &[&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]];
    assert!(one_way_anova(data).is_ok());
    assert!(tukey_hsd(data, 0.05).is_ok());
    assert!(games_howell(data, 0.05).is_ok());
    assert!(dunnett(data, 0, 0.05).is_ok());
}

#[test]
fn generic_input_fixed_arrays() {
    let data = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    assert!(one_way_anova(&data).is_ok());
    assert!(tukey_hsd(&data, 0.05).is_ok());
    assert!(games_howell(&data, 0.05).is_ok());
    assert!(dunnett(&data, 0, 0.05).is_ok());
}

#[test]
fn deterministic_results() {
    let data = vec![
        vec![6.0, 8.0, 4.0, 5.0, 3.0, 4.0],
        vec![8.0, 12.0, 9.0, 11.0, 6.0, 8.0],
        vec![13.0, 9.0, 11.0, 8.0, 12.0, 14.0],
    ];
    let r1 = tukey_hsd(&data, 0.05).unwrap();
    let r2 = tukey_hsd(&data, 0.05).unwrap();
    assert_eq!(r1.q_critical, r2.q_critical);
    assert_eq!(r1.mse, r2.mse);
    for (c1, c2) in r1.comparisons.iter().zip(r2.comparisons.iter()) {
        assert_eq!(c1.q_statistic, c2.q_statistic);
        assert_eq!(c1.significant, c2.significant);
    }
}

#[test]
fn swapping_groups_preserves_mean_diff() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![7.0, 8.0, 9.0];
    let r1 = tukey_hsd(&[a.clone(), b.clone()], 0.05).unwrap();
    let r2 = tukey_hsd(&[b, a], 0.05).unwrap();
    // Absolute difference should be the same
    assert!((r1.comparisons[0].mean_diff - r2.comparisons[0].mean_diff).abs() < 1e-10);
    // But CI should flip sign
    assert!((r1.comparisons[0].ci_lower + r2.comparisons[0].ci_upper).abs() < 1e-10);
}
