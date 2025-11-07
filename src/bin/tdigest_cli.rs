// src/bin/tdigest_cli.rs

use std::io::{self, Read};

use clap::{ArgAction, Parser, ValueEnum};

use gr_tdigest::tdigest::frontends::{parse_scale_str, parse_singleton_policy_str};
use gr_tdigest::tdigest::singleton_policy::SingletonPolicy;
use gr_tdigest::tdigest::{ScaleFamily, TDigest, TDigestBuilder};

/// Simple CLI wrapper for gr-tdigest used in integration/api_coherence tests.
#[derive(Debug, Clone, ValueEnum)]
enum Cmd {
    Quantile,
    Cdf,
}

#[derive(Debug, Clone, ValueEnum)]
enum ScaleOpt {
    Quad,
    K1,
    K2,
    K3,
}

impl ToString for ScaleOpt {
    fn to_string(&self) -> String {
        match self {
            ScaleOpt::Quad => "quad",
            ScaleOpt::K1 => "k1",
            ScaleOpt::K2 => "k2",
            ScaleOpt::K3 => "k3",
        }
        .to_string()
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum PrecisionOpt {
    F32,
    F64,
    Auto,
}

/// Accept only: off | use | edges
#[derive(Debug, Clone, Copy, ValueEnum)]
enum PolicyOpt {
    Off,
    Use,
    Edges,
}

#[derive(Debug, Parser)]
#[command(name = "tdigest", version, disable_help_subcommand = true)]
struct Cli {
    /// Read values from stdin (whitespace/comma separated)
    #[arg(long, action = ArgAction::SetTrue)]
    stdin: bool,

    /// Command to run: quantile or cdf
    #[arg(long, value_enum)]
    cmd: Cmd,

    /// Quantile probability.
    /// Required for --cmd quantile.
    #[arg(long)]
    #[arg(long, allow_hyphen_values = true)]
    p: Option<String>,

    /// Output format (tests use csv)
    #[arg(long, default_value = "csv")]
    output: String,

    /// Suppress header row when output=csv
    #[arg(long, action = ArgAction::SetTrue)]
    no_header: bool,

    /// TDigest max size
    #[arg(long = "max-size", default_value_t = 1000)]
    max_size: usize,

    /// Scale family: quad|k1|k2|k3
    #[arg(long, value_enum, default_value_t = ScaleOpt::K2)]
    scale: ScaleOpt,

    /// Singleton policy: off|use|edges
    #[arg(long = "singleton-policy", value_enum)]
    singleton_policy: Option<PolicyOpt>,

    /// Edges to pin per side when using 'edges' policy
    #[arg(long = "pin-per-side")]
    pin_per_side: Option<usize>,

    /// Precision hint f32|f64|auto (currently informational; digests built as f64)
    #[arg(long, value_enum, default_value_t = PrecisionOpt::F64)]
    precision: PrecisionOpt,
}

fn read_floats_from_stdin() -> Result<Vec<f64>, String> {
    let mut buf = String::new();
    io::stdin()
        .read_to_string(&mut buf)
        .map_err(|e| format!("stdin read error: {e}"))?;

    let mut out = Vec::new();
    for tok in buf.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() {
            continue;
        }
        match tok.parse::<f64>() {
            Ok(v) => {
                if !v.is_finite() {
                    return Err(
                        "tdigest: non-finite values are not allowed in training data (NaN or ±inf)"
                            .to_string(),
                    );
                }
                out.push(v);
            }
            Err(_) => {
                // Ignore junk tokens to match forgiving parsing for non-numerics.
            }
        }
    }
    Ok(out)
}

fn policy_from_opts(
    opt: Option<PolicyOpt>,
    edges: Option<usize>,
) -> Result<SingletonPolicy, String> {
    let as_str = match opt.unwrap_or(PolicyOpt::Use) {
        PolicyOpt::Off => "off",
        PolicyOpt::Use => "use",
        PolicyOpt::Edges => "edges",
    };
    parse_singleton_policy_str(Some(as_str), edges).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();

    // Read data
    let values: Vec<f64> = if cli.stdin {
        read_floats_from_stdin()?
    } else {
        return Err("no values provided (pass --stdin)".into());
    };

    // Build digest (f64 storage; precision flag is accepted but informational)
    let scale = parse_scale_str(Some(&cli.scale.to_string())).unwrap_or(ScaleFamily::K2);
    let policy = policy_from_opts(cli.singleton_policy, cli.pin_per_side)?;

    let base: TDigest<f64> = TDigestBuilder::<f64>::new()
        .max_size(cli.max_size)
        .scale(scale)
        .singleton_policy(policy)
        .build();

    let digest: TDigest<f64> = base
        .merge_unsorted(values.clone())
        .map_err(|e| e.to_string())?;

    // Output CSV as tests expect
    let want_csv = cli.output.eq_ignore_ascii_case("csv");
    if want_csv && !cli.no_header {
        match cli.cmd {
            Cmd::Quantile => println!("p,value"),
            Cmd::Cdf => println!("x,p"),
        }
    }

    match cli.cmd {
        Cmd::Quantile => {
            let p_raw = cli
                .p
                .as_ref()
                .ok_or_else(|| "--p is required for --cmd quantile".to_string())?;
            let p_norm = p_raw.trim().to_lowercase();

            // Reject non-finite tokens explicitly
            if p_norm == "nan" || p_norm == "inf" || p_norm == "+inf" || p_norm == "-inf" {
                return Err("p must be a finite number in [0,1]".to_string());
            }

            // Finite numeric p → must be in [0,1]
            let p_val: f64 = p_raw
                .parse::<f64>()
                .map_err(|_| format!("p must be a finite number in [0,1] (got {p_raw:?})"))?;
            if !p_val.is_finite() {
                return Err("p must be a finite number in [0,1]".into());
            }
            if !(0.0..=1.0).contains(&p_val) {
                return Err("--p must be in [0,1]".into());
            }

            let q = digest.quantile(p_val);
            if want_csv {
                println!("{},{}", p_val, q);
            } else {
                println!("{q}");
            }
        }
        Cmd::Cdf => {
            for x in values {
                let p = digest.cdf_or_nan(&[x])[0];
                if want_csv {
                    println!("{},{}", x, p);
                } else {
                    println!("{p}");
                }
            }
        }
    }

    Ok(())
}
