use std::env;
use std::io;
use std::process;

use tukey_test::{dunnett, games_howell, one_way_anova, parse_csv, parse_csv_file, q_critical, tukey_hsd};

fn main() {
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("anova") => run_anova(&args[2..]),
        Some("hsd") => run_hsd(&args[2..]),
        Some("games-howell") => run_games_howell(&args[2..]),
        Some("dunnett") => run_dunnett(&args[2..]),
        Some("q") => run_q_lookup(&args[2..]),
        _ => {
            eprintln!("Usage:");
            eprintln!("  tukey_test anova [--file <path>] <group1> -- <group2> [-- ...]");
            eprintln!("  tukey_test hsd <alpha> [--file <path>] <group1> -- <group2> [-- ...]");
            eprintln!("  tukey_test games-howell <alpha> [--file <path>] <group1> -- <group2> [-- ...]");
            eprintln!("  tukey_test dunnett <alpha> <control_index> [--file <path>] <group1> -- <group2> [-- ...]");
            eprintln!("  tukey_test q <k> <df> <alpha>");
            eprintln!();
            eprintln!("Data input:");
            eprintln!("  Inline:  separate groups with --");
            eprintln!("  File:    --file data.csv (wide format, each column = one group)");
            eprintln!("  Stdin:   --file - (pipe CSV data)");
            eprintln!();
            eprintln!("Examples:");
            eprintln!("  tukey_test anova 6 8 4 5 3 4 -- 8 12 9 11 6 8 -- 13 9 11 8 12 14");
            eprintln!("  tukey_test hsd 0.05 --file experiment.csv");
            eprintln!("  cat data.csv | tukey_test anova --file -");
            eprintln!("  tukey_test dunnett 0.05 0 --file treatments.csv");
            eprintln!("  tukey_test q 3 15 0.05");
            process::exit(1);
        }
    }
}

/// Try to extract `--file <path>` from args, returning the groups and remaining args.
/// Returns None if --file is not present.
fn extract_file_groups(args: &[String]) -> Option<Vec<Vec<f64>>> {
    let pos = args.iter().position(|a| a == "--file")?;
    if pos + 1 >= args.len() {
        eprintln!("Error: --file requires a path argument (use - for stdin)");
        process::exit(1);
    }
    let path = &args[pos + 1];
    let groups = if path == "-" {
        parse_csv(io::stdin().lock()).unwrap_or_else(|e| {
            eprintln!("Error reading stdin: {e}");
            process::exit(1);
        })
    } else {
        parse_csv_file(path).unwrap_or_else(|e| {
            eprintln!("Error: {e}");
            process::exit(1);
        })
    };
    Some(groups)
}

/// Parse "--"-separated groups from args, or read from --file.
fn parse_groups_or_file(args: &[String]) -> Vec<Vec<f64>> {
    if let Some(groups) = extract_file_groups(args) {
        return groups;
    }

    // Inline: parse "--"-separated groups
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
        eprintln!("Usage: tukey_test q <k> <df> <alpha>");
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
    let groups = parse_groups_or_file(args);
    if groups.len() < 2 {
        eprintln!("Error: need at least 2 groups (separate with -- or use --file)");
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
        eprintln!("Usage: tukey_test hsd <alpha> [--file <path>] <groups...>");
        process::exit(1);
    }

    let alpha: f64 = args[0].parse().unwrap_or_else(|_| {
        eprintln!("Invalid alpha: {}", args[0]);
        process::exit(1);
    });

    let groups = parse_groups_or_file(&args[1..]);
    if groups.len() < 2 {
        eprintln!("Error: need at least 2 groups (separate with -- or use --file)");
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
        eprintln!("Usage: tukey_test games-howell <alpha> [--file <path>] <groups...>");
        process::exit(1);
    }

    let alpha: f64 = args[0].parse().unwrap_or_else(|_| {
        eprintln!("Invalid alpha: {}", args[0]);
        process::exit(1);
    });

    let groups = parse_groups_or_file(&args[1..]);
    if groups.len() < 2 {
        eprintln!("Error: need at least 2 groups (separate with -- or use --file)");
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

fn run_dunnett(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Usage: tukey_test dunnett <alpha> <control_index> [--file <path>] <groups...>");
        process::exit(1);
    }

    let alpha: f64 = args[0].parse().unwrap_or_else(|_| {
        eprintln!("Invalid alpha: {}", args[0]);
        process::exit(1);
    });

    let control: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("Invalid control group index: {}", args[1]);
        process::exit(1);
    });

    let groups = parse_groups_or_file(&args[2..]);
    if groups.len() < 2 {
        eprintln!("Error: need at least 2 groups (separate with -- or use --file)");
        process::exit(1);
    }

    match dunnett(&groups, control, alpha) {
        Ok(result) => print!("{result}"),
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    }
}
