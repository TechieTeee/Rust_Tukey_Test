use std::env;
use std::process;

fn main() {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();

    // Check if the required number of arguments is provided
    if args.len() < 3 {
        println!("Usage: tukey_test <group_count> <total_count>");
        process::exit(1);
    }

    // Parse command line arguments
    let group_count: usize = match args[1].parse() {
        Ok(value) => value,
        Err(_) => {
            println!("Invalid group count: {}", args[1]);
            process::exit(1);
        }
    };
    let total_count: usize = match args[2].parse() {
        Ok(value) => value,
        Err(_) => {
            println!("Invalid total count: {}", args[2]);
            process::exit(1);
        }
    };

    // Perform Tukey test
    let q_value = tukey_test(group_count, total_count);

    // Print the result
    println!("The critical q value for a Tukey test with {} groups and a total of {} observations is: {:.4}", group_count, total_count, q_value);
}

fn tukey_test(group_count: usize, total_count: usize) -> f64 {
    let q_value = ((group_count as f64) - 1.0).sqrt() * critical_value(total_count);
    q_value
}

fn critical_value(total_count: usize) -> f64 {
    // Define critical values for different alpha levels
    let alpha_values = vec![
        (0.05, 3.673),
        (0.01, 4.692),
        (0.001, 5.842),
    ];

    // Find the critical value for the given total count
    let mut critical_value = 0.0;
    for (alpha, value) in alpha_values {
        if total_count >= value as usize {
            critical_value = alpha;
            break;
        }
    }
    critical_value
}
