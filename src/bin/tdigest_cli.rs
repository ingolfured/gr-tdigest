// src/bin/tdigest_cli.rs

use std::io::{self, Read};
use std::str::FromStr;

use clap::{ArgAction, Parser, ValueEnum};

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

impl From<ScaleOpt> for ScaleFamily {
    fn from(s: ScaleOpt) -> Self {
        match s {
            ScaleOpt::Quad => ScaleFamily::Quad,
            ScaleOpt::K1 => ScaleFamily::K1,
            ScaleOpt::K2 => ScaleFamily::K2,
            ScaleOpt::K3 => ScaleFamily::K3,
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum PrecisionOpt {
    F32,
    F64,
}

/// Accept "off|use|edges|use_edges" (the test passes `use_edges`).
#[derive(Debug, Clone)]
enum PolicyOpt {
    Off,
    Use,
    Edges,
}

impl FromStr for PolicyOpt {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let t = s.trim().to_lowercase();
        match t.as_str() {
            "off" => Ok(PolicyOpt::Off),
            "use" | "on" | "respect" => Ok(PolicyOpt::Use),
            "edges" | "use_edges" | "use-with-protected-edges" | "usewithprotectededges" => {
                Ok(PolicyOpt::Edges)
            }
            other => Err(format!(
                "invalid value '{}' for --singleton-policy (expected off|use|edges|use_edges)",
                other
            )),
        }
    }
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

    /// Quantile probability p in
    /// \[0,1\] (required for --cmd quantile)
    #[arg(long)]
    p: Option<f64>,

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

    /// Singleton policy: off|use|edges|use_edges
    #[arg(long = "singleton-policy")]
    singleton_policy: Option<PolicyOpt>,

    /// Edges to pin per side when using 'edges' policy
    #[arg(long = "pin-per-side")]
    pin_per_side: Option<usize>,

    /// Precision hint f32|f64 (currently informational; digests built as f64)
    #[arg(long, value_enum, default_value_t = PrecisionOpt::F64)]
    precision: PrecisionOpt,
}

fn read_floats_from_stdin() -> io::Result<Vec<f64>> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    let mut out = Vec::new();
    for tok in buf.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() {
            continue;
        }
        match tok.parse::<f64>() {
            Ok(v) if v.is_finite() => out.push(v),
            _ => { /* ignore non-finite / bad tokens */ }
        }
    }
    Ok(out)
}

fn policy_from_opts(
    opt: Option<PolicyOpt>,
    edges: Option<usize>,
) -> Result<SingletonPolicy, String> {
    match opt.unwrap_or(PolicyOpt::Use) {
        PolicyOpt::Off => Ok(SingletonPolicy::Off),
        PolicyOpt::Use => Ok(SingletonPolicy::Use),
        PolicyOpt::Edges => {
            let k = edges.ok_or_else(|| "use_edges requires --pin-per-side".to_string())?;
            if k < 1 {
                return Err("--pin-per-side must be >= 1".into());
            }
            Ok(SingletonPolicy::UseWithProtectedEdges(k))
        }
    }
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();

    // Read data
    let values: Vec<f64> = if cli.stdin {
        read_floats_from_stdin().map_err(|e| format!("stdin read error: {e}"))?
    } else {
        return Err("no values provided (pass --stdin)".into());
    };

    // Build digest (f64 storage; matches Python tests’ expectation)
    let scale = ScaleFamily::from(cli.scale);
    let policy = policy_from_opts(cli.singleton_policy, cli.pin_per_side)?;

    let base: TDigest<f64> = TDigestBuilder::<f64>::new()
        .max_size(cli.max_size)
        .scale(scale)
        .singleton_policy(policy)
        .build();

    let digest = base.merge_unsorted(values.clone());

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
            let p = cli
                .p
                .ok_or_else(|| "--p is required for --cmd quantile".to_string())?;
            if !(0.0..=1.0).contains(&p) {
                return Err("--p must be in [0,1]".into());
            }
            let q = digest.quantile(p);
            if want_csv {
                println!("{},{}", p, q);
            } else {
                println!("{q}");
            }
        }
        Cmd::Cdf => {
            // For CSV, echo rows "x,p" in the order of inputs.
            for x in values {
                let ps = digest.cdf(&[x]);
                let p = ps[0];
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
