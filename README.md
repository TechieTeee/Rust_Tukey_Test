# Rust Tukey Test

This script performs a Tukey test based on user inputs. It calculates the critical q value for a Tukey test, given the number of groups and total observations. The critical q value can be used to determine if there are statistically significant differences between the means of multiple groups.

## Background

The Tukey test, also known as the Tukey's honest significant difference (HSD) test, is a statistical test used to compare the means of multiple groups. It is often applied after performing an analysis of variance (ANOVA) to determine if there are significant differences between the means of the groups.

The test calculates a critical value (q value) based on the number of groups and the total number of observations. By comparing the differences between the means of the groups to this critical value, the Tukey test helps identify which groups have significantly different means from each other.

The Tukey test is useful in various fields, such as experimental research, social sciences, and business analytics. It allows researchers and analysts to gain insights into the significant differences between multiple groups, enabling them to make informed decisions or draw meaningful conclusions.

## Prerequisites

- Rust programming language (compiler and tools) installed. You can download Rust from the official website: [https://www.rust-lang.org](https://www.rust-lang.org)

## How to Run the Code

1. Clone or download this repository to your local machine. 
Run this command in your terminal `git clone https://github.com/TechieTeee/Rust_Tukey_Test.git` or use the download button.

2. If haven't already opened your terminal, open a terminal and navigate to the project directory.
`cd Rust_Tukey_Test`

3. Compile the Rust script using the following command:
`$ rustc tukey_test.rs`


4. Run the compiled binary with the desired command line arguments. The required arguments are:
- `<group_count>`: Number of groups in the data.
- `<total_count>`: Total number of observations.

For example, to perform a Tukey test with 4 groups and a total of 20 observations, run the following command:
`$ ./tukey_test 4 20`


Replace `<group_count>` and `<total_count>` with the actual values for your data.

5. The script will calculate the critical q value and display the result.





