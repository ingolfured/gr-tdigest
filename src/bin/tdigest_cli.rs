// src/bin/tdigest_cli.rs

//! Command-line interface for the `tdigest-rs` library.
//!
//! Build a t-digest from values supplied via one of three inputs (values, file,
//! or stdin) and compute either a CDF for each input value or a single quantile.
//!
//! # Input sources (choose exactly one)
//! - `--values "1, 2; 3 4"`: inline values (commas/spaces/semicolons accepted)
//! - `--file data.txt`: read values from a file (same separators accepted)
//! - `--stdin`: read values from standard input (stream or paste)
//!
//! # Output formats
//! - `--output csv` (default): CSV with header `x,p` (CDF) or `quantile,value`
//! - `--output tsv`: tab-separated with same headers
//! - `--output jsonl`: JSON Lines; CDF emits `{"x": ..., "p": ...}` per line
//!
//! Use `--no-header` to omit headers for CSV/TSV.
//!
//! # Examples
//! ```text
//! # CDF from inline values (defaults: --output csv, --scale k2, --singleton-policy use)
//! tdigest --values "1,2,3,4,5" --cmd cdf
//!
//! # Quantile (q=0.99) from a file, TSV output without header
//! tdigest --file data.txt --cmd quantile --p 0.99 --output tsv --no-header
//!
//! # CDF from stdin, JSONL output
//! cat numbers.txt | tdigest --stdin --cmd cdf --output jsonl
//!
//! # Protect edges (keep 4 per tail) for heavy-tailed data
//! tdigest --file data.txt --cmd cdf --scale quad --singleton-policy edges --keep 4
//! ```

use clap::{builder::EnumValueParser, Arg, ArgAction, ArgGroup, Command, ValueEnum};
use std::error::Error;
use std::fs;
use std::io::{self, Read};

use tdigest_rs::tdigest::singleton_policy::SingletonPolicy;
use tdigest_rs::tdigest::ScaleFamily;

/// User-facing scale options for the CLI.
#[derive(Debug, Clone, ValueEnum)]
enum Scale {
    Quad,
    K1,
    K2,
    K3,
}
impl From<Scale> for ScaleFamily {
    fn from(s: Scale) -> Self {
        match s {
            Scale::Quad => ScaleFamily::Quad,
            Scale::K1 => ScaleFamily::K1,
            Scale::K2 => ScaleFamily::K2,
            Scale::K3 => ScaleFamily::K3,
        }
    }
}

/// How edge singletons are handled.
#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
enum Policy {
    Off,
    Use,
    /// Maps to `SingletonPolicy::UseWithProtectedEdges(keep)`.
    Edges,
}
impl Policy {
    fn into_singleton_policy(self, keep: Option<usize>) -> SingletonPolicy {
        match self {
            Policy::Off => SingletonPolicy::Off,
            Policy::Use => SingletonPolicy::Use,
            Policy::Edges => SingletonPolicy::UseWithProtectedEdges(keep.unwrap_or(3)),
        }
    }
}

/// Which computation to run.
#[derive(Debug, Clone, ValueEnum)]
enum CmdKind {
    /// Emit `P(X ≤ x)` for each input x.
    Cdf,
    /// Emit q-quantile; requires `--p q` with `0 ≤ q ≤ 1`.
    Quantile,
}

/// Output encoding.
#[derive(Debug, Clone, ValueEnum)]
enum Output {
    /// CSV with header (unless `--no-header`).
    Csv,
    /// TSV with header (unless `--no-header`).
    Tsv,
    /// JSON Lines (one JSON object per line).
    Jsonl,
}

/// Parse a flat list of f64 from a string with flexible separators.
fn parse_numbers(s: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut out = Vec::new();
    for tok in s
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .filter(|t| !t.is_empty())
    {
        out.push(tok.parse::<f64>()?);
    }
    Ok(out)
}

fn read_values_from_file(path: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    parse_numbers(&text)
}

fn read_values_from_stdin() -> Result<Vec<f64>, Box<dyn Error>> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    parse_numbers(&buf)
}

/// Print CDF rows in the selected format.
fn print_cdf(values: &[f64], ps: &[f64], out: &Output, header: bool) {
    match out {
        Output::Csv => {
            if header {
                println!("x,p");
            }
            for (x, p) in values.iter().zip(ps.iter()) {
                println!("{x},{p}");
            }
        }
        Output::Tsv => {
            if header {
                println!("x\tp");
            }
            for (x, p) in values.iter().zip(ps.iter()) {
                println!("{x}\t{p}");
            }
        }
        Output::Jsonl => {
            for (x, p) in values.iter().zip(ps.iter()) {
                // Small, stable JSON without dependencies.
                println!("{{\"x\":{x},\"p\":{p}}}");
            }
        }
    }
}

/// Print quantile result in the selected format.
fn print_quantile(q: f64, v: f64, out: &Output, header: bool) {
    match out {
        Output::Csv => {
            if header {
                println!("quantile,value");
            }
            println!("{q},{v}");
        }
        Output::Tsv => {
            if header {
                println!("quantile\tvalue");
            }
            println!("{q}\t{v}");
        }
        Output::Jsonl => {
            println!("{{\"quantile\":{q},\"value\":{v}}}");
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let cmd = Command::new("tdigest")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Build a t-digest from values and compute CDFs or quantiles.")
        .long_about(
            "Construct a t-digest (approximate quantile sketch) from input values and run \
             either a CDF over those same values or a single quantile query. \
             Choose one input source: --values, --file, or --stdin.",
        )
        .after_help(
            r#"EXAMPLES
  tdigest --values "1,2,3,4,5" --cmd cdf
  tdigest --file data.txt --cmd quantile --p 0.99 --output tsv --no-header
  cat numbers.txt | tdigest --stdin --cmd cdf --output jsonl
  tdigest --file data.txt --cmd cdf --scale quad --singleton-policy edges --keep 4
"#,
        )
        // Core tuning
        .arg(
            Arg::new("max_size")
                .long("max-size")
                .help("Maximum centroids (compression). Higher → more accuracy and memory.")
                .value_parser(clap::value_parser!(usize))
                .default_value("1000"),
        )
        .arg(
            Arg::new("scale")
                .long("scale")
                .help("Scale family: quad | k1 | k2 | k3 (tail/center precision trade-offs).")
                .value_parser(EnumValueParser::<Scale>::new())
                .default_value("k2"),
        )
        .arg(
            Arg::new("singleton_policy")
                .long("singleton-policy")
                .help("Edge singleton handling: off | use | edges (use --keep with edges).")
                .value_parser(EnumValueParser::<Policy>::new())
                .default_value("use"),
        )
        .arg(
            Arg::new("keep")
                .long("keep")
                .help(
                    "When --singleton-policy=edges, keep this many raw items per tail (default 3).",
                )
                .value_parser(clap::value_parser!(usize)),
        )
        // Input choices (exactly one)
        .arg(
            Arg::new("values")
                .long("values")
                .short('v')
                .help("Inline values (comma/space/semicolon separated). Example: \"1, 2; 3 4\"")
                .value_parser(clap::builder::NonEmptyStringValueParser::new()),
        )
        .arg(
            Arg::new("file")
                .long("file")
                .help("Read values from a file (any separators: comma/space/semicolon).")
                .value_parser(clap::builder::NonEmptyStringValueParser::new()),
        )
        .arg(
            Arg::new("stdin")
                .long("stdin")
                .help("Read values from standard input.")
                .action(ArgAction::SetTrue),
        )
        .group(
            ArgGroup::new("input")
                .args(["values", "file", "stdin"])
                .required(true)
                .multiple(false),
        )
        // Command to run
        .arg(
            Arg::new("cmd")
                .long("cmd")
                .help("Operation: cdf | quantile")
                .value_parser(EnumValueParser::<CmdKind>::new())
                .required(true),
        )
        .arg(
            Arg::new("p")
                .long("p")
                .help("Quantile probability in [0,1] (required for --cmd quantile).")
                .value_parser(clap::value_parser!(f64))
                .action(ArgAction::Set),
        )
        // Output formatting
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .help("Output format: csv (default) | tsv | jsonl")
                .value_parser(EnumValueParser::<Output>::new())
                .default_value("csv"),
        )
        .arg(
            Arg::new("no_header")
                .long("no-header")
                .help("Suppress header row for CSV/TSV output.")
                .action(ArgAction::SetTrue),
        );

    let matches = cmd.get_matches();

    // Extract arguments
    let max_size = *matches.get_one::<usize>("max_size").unwrap();
    let scale = matches.get_one::<Scale>("scale").unwrap().clone();
    let policy = matches
        .get_one::<Policy>("singleton_policy")
        .unwrap()
        .clone();
    let keep = matches.get_one::<usize>("keep").copied();
    let out_fmt = matches.get_one::<Output>("output").unwrap().clone();
    let no_header = matches.get_flag("no_header");
    let run = matches.get_one::<CmdKind>("cmd").unwrap();
    let p = matches.get_one::<f64>("p").copied();

    // Validate cross-arg rules
    if matches.contains_id("keep") && policy != Policy::Edges {
        eprintln!("--keep is only valid when --singleton-policy=edges");
        std::process::exit(2);
    }
    if let CmdKind::Quantile = run {
        let Some(q) = p else {
            eprintln!("--p is required for --cmd quantile");
            std::process::exit(2);
        };
        if !(0.0..=1.0).contains(&q) {
            eprintln!("--p must be in [0,1]");
            std::process::exit(2);
        }
    }

    // Resolve input
    let values: Vec<f64> = if let Some(vstr) = matches.get_one::<String>("values") {
        parse_numbers(vstr)?
    } else if let Some(path) = matches.get_one::<String>("file") {
        read_values_from_file(path)?
    } else if matches.get_flag("stdin") {
        read_values_from_stdin()?
    } else {
        // Should be unreachable due to ArgGroup, but keep a guard.
        eprintln!("Choose exactly one input source: --values, --file, or --stdin");
        std::process::exit(2);
    };

    // Build digest and run
    let digest = tdigest_rs::tdigest::TDigest::builder()
        .max_size(max_size)
        .scale(scale.into())
        .singleton_policy(policy.into_singleton_policy(keep))
        .build()
        .merge_unsorted(values.clone());

    match run {
        CmdKind::Cdf => {
            let ps = digest.cdf(&values);
            let header = !no_header && !matches!(out_fmt, Output::Jsonl);
            print_cdf(&values, &ps, &out_fmt, header);
        }
        CmdKind::Quantile => {
            let q = p.unwrap(); // validated above
            let v = digest.quantile(q);
            let header = !no_header && !matches!(out_fmt, Output::Jsonl);
            print_quantile(q, v, &out_fmt, header);
        }
    }

    Ok(())
}
#[cfg(test)]
mod cli_smoke {
    use assert_cmd::Command;
    use std::process::Command as Proc;
    use std::sync::Once;

    // Run `cargo build --bin tdigest` exactly once before any tests use it.
    static BUILD_BIN_ONCE: Once = Once::new();

    fn ensure_cli_built() {
        BUILD_BIN_ONCE.call_once(|| {
            let status = Proc::new("cargo")
                .args(["build", "--bin", "tdigest"])
                .current_dir(env!("CARGO_MANIFEST_DIR")) // crate root
                .status()
                .expect("failed to spawn `cargo build --bin tdigest`");
            assert!(status.success(), "`cargo build --bin tdigest` failed");
        });
    }

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn smoke_stdin_to_file_quantile_and_cdf() -> Result<(), Box<dyn std::error::Error>> {
        ensure_cli_built(); // <-- pre-hook

        // ---------- QUANTILE ----------
        let data = "1,2,2,3,100\n";
        let assert = Command::cargo_bin("tdigest")?
            .arg("--stdin")
            .arg("--cmd")
            .arg("quantile")
            .arg("--p")
            .arg("0.5")
            .arg("--output")
            .arg("csv")
            .arg("--max-size")
            .arg("1000")
            .write_stdin(data)
            .assert()
            .success();

        let stdout = String::from_utf8(assert.get_output().stdout.clone())?;
        let mut lines = stdout.lines();
        assert_eq!(lines.next().unwrap_or_default(), "quantile,value");
        let mut cols = lines.next().unwrap().split(',');
        let q: f64 = cols.next().unwrap().parse()?;
        let v: f64 = cols.next().unwrap().parse()?;
        assert!(approx(q, 0.5, 1e-12));
        assert!(approx(v, 2.0, 1e-6));
        assert!(lines.next().is_none());

        // ---------- CDF ----------
        let data2 = "1 2 3 4 5\n";
        let assert2 = Command::cargo_bin("tdigest")?
            .arg("--stdin")
            .arg("--cmd")
            .arg("cdf")
            .arg("--output")
            .arg("csv")
            .arg("--max-size")
            .arg("1000")
            .write_stdin(data2)
            .assert()
            .success();

        let out2 = String::from_utf8(assert2.get_output().stdout.clone())?;
        let mut lines2 = out2.lines();
        assert_eq!(lines2.next().unwrap_or_default(), "x,p");

        let mut p1 = None;
        let mut p3 = None;
        let mut p5 = None;
        for line in lines2 {
            let mut c = line.split(',');
            let x: f64 = match c.next().and_then(|s| s.parse().ok()) {
                Some(v) => v,
                None => continue,
            };
            let p: f64 = c.next().unwrap().parse()?;
            if approx(x, 1.0, 1e-12) {
                p1 = Some(p)
            }
            if approx(x, 3.0, 1e-12) {
                p3 = Some(p)
            }
            if approx(x, 5.0, 1e-12) {
                p5 = Some(p)
            }
            assert!(p >= -1e-12 && p <= 1.0 + 1e-12);
        }
        let (p1, p3, p5) = (p1.unwrap(), p3.unwrap(), p5.unwrap());
        assert!(approx(p1, 0.1, 1e-3));
        assert!(approx(p3, 0.5, 1e-3));
        assert!(approx(p5, 0.9, 1e-3));
        assert!(p1 <= p3 && p3 <= p5);

        Ok(())
    }
}
