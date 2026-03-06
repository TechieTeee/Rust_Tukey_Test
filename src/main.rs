use std::env;
use std::process;

use tukey_test::{games_howell, one_way_anova, q_critical, tukey_hsd};

fn main() {
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("anova") => run_anova(&args[2..]),
        Some("hsd") => run_hsd(&args[2..]),
        Some("games-howell") => run_games_howell(&args[2..]),
        Some("q") => run_q_lookup(&args[2..]),
        _ => {
            eprintln!("Usage:");
            eprintln!("  tukey-test anova <group1> -- <group2> -- <group3> [-- ...]");
            eprintln!("  tukey-test hsd <alpha> <group1> -- <group2> -- <group3> [-- ...]");
            eprintln!("  tukey-test games-howell <alpha> <group1> -- <group2> -- <group3> [-- ...]");
            eprintln!("  tukey-test q <k> <df> <alpha>");
            eprintln!();
            eprintln!("Examples:");
            eprintln!("  tukey-test anova 6 8 4 5 3 4 -- 8 12 9 11 6 8 -- 13 9 11 8 12 14");
            eprintln!("  tukey-test hsd 0.05 6 8 4 5 3 4 -- 8 12 9 11 6 8 -- 13 9 11 8 12 14");
            eprintln!("  tukey-test games-howell 0.05 4 5 3 4 6 -- 20 30 25 35 28 -- 5 7 6 4 5");
            eprintln!("  tukey-test q 3 15 0.05");
            process::exit(1);
        }
    }
}

/// Parse "--"-separated groups from args.
fn parse_groups(args: &[String]) -> Vec<Vec<f64>> {
    let mut groups: Vec<Vec<f64>> = Vec::new();
    let mut current_group: Vec<f64> = Vec::new();

    for arg in args {
        if arg == "--" {
            if !current_group.is_empty() {
                groups.push(current_group);
                current_group = Vec::new();
            }
        } else {
            let val: f64 = arg.parse().unwrap_or_else(|_| {
                eprintln!("Invalid number: {arg}");
                process::exit(1);
            });
            current_group.push(val);
        }
    }
    if !current_group.is_empty() {
        groups.push(current_group);
    }
    groups
}

fn run_q_lookup(args: &[String]) {
    if args.len() != 3 {
        eprintln!("Usage: tukey-test q <k> <df> <alpha>");
        process::exit(1);
    }

    let k: usize = args[0].parse().unwrap_or_else(|_| {
        eprintln!("Invalid k (number of groups): {}", args[0]);
        process::exit(1);
    });
    let df: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("Invalid df (degrees of freedom): {}", args[1]);
        process::exit(1);
    });
    let alpha: f64 = args[2].parse().unwrap_or_else(|_| {
        eprintln!("Invalid alpha: {}", args[2]);
        process::exit(1);
    });

    match q_critical(k, df, alpha) {
        Ok(q) => println!("q_critical(k={k}, df={df}, alpha={alpha}) = {q:.4}"),
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    }
}

fn run_anova(args: &[String]) {
    let groups = parse_groups(args);
    if groups.len() < 2 {
        eprintln!("Error: need at least 2 groups (separate with --)");
        process::exit(1);
    }

    match one_way_anova(&groups) {
        Ok(result) => print!("{result}"),
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    }
}

fn run_hsd(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: tukey-test hsd <alpha> <group1> -- <group2> -- <group3> [-- ...]");
        process::exit(1);
    }

    let alpha: f64 = args[0].parse().unwrap_or_else(|_| {
        eprintln!("Invalid alpha: {}", args[0]);
        process::exit(1);
    });

    let groups = parse_groups(&args[1..]);
    if groups.len() < 2 {
        eprintln!("Error: need at least 2 groups (separate with --)");
        process::exit(1);
    }

    match tukey_hsd(&groups, alpha) {
        Ok(result) => print!("{result}"),
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    }
}

fn run_games_howell(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: tukey-test games-howell <alpha> <group1> -- <group2> -- <group3> [-- ...]");
        process::exit(1);
    }

    let alpha: f64 = args[0].parse().unwrap_or_else(|_| {
        eprintln!("Invalid alpha: {}", args[0]);
        process::exit(1);
    });

    let groups = parse_groups(&args[1..]);
    if groups.len() < 2 {
        eprintln!("Error: need at least 2 groups (separate with --)");
        process::exit(1);
    }

    match games_howell(&groups, alpha) {
        Ok(result) => print!("{result}"),
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    }
}
