use clap::{Parser, Subcommand, ValueEnum};
use polars_tdigest::tdigest::{ScaleFamily, TDigest};
use std::error::Error;
use std::io::{self, Read};

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

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Max digest size (compression parameter)
    #[arg(short = 'm', long, default_value_t = 100)]
    max_size: usize,

    /// Scale family (quad|k1|k2|k3)
    #[arg(short = 's', long, value_enum, default_value_t = Scale::K2)]
    scale: Scale,

    /// Optional: probe xs (space/comma/newline separated). If omitted, uses input xs.
    #[arg(long)]
    probes: Option<String>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build a digest from stdin (space/newline separated numbers) and print CDF at --probes (or xs if omitted)
    Cdf,
    /// Build a digest from stdin and print a single quantile value
    Quantile {
        /// q in [0,1]
        #[arg(short, long)]
        q: f64,
    },
}

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

fn read_stdin_f64s() -> Result<Vec<f64>, Box<dyn Error>> {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s)?;
    let mut xs = parse_numbers(&s)?;
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(xs)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let scale: ScaleFamily = args.scale.into();

    let xs = read_stdin_f64s()?;
    let digest = TDigest::new_with_size_and_scale(args.max_size, scale).merge_sorted(xs.clone());

    match args.cmd {
        Cmd::Cdf => {
            let probes = if let Some(p) = args.probes {
                parse_numbers(&p)?
            } else {
                xs
            };
            let cdf = digest.estimate_cdf(&probes);
            for (x, p) in probes.iter().zip(cdf.iter()) {
                println!("{x}\t{p}");
            }
        }
        Cmd::Quantile { q } => {
            if !(0.0..=1.0).contains(&q) {
                eprintln!("q must be in [0,1]");
                std::process::exit(2);
            }
            let v = digest.estimate_quantile(q);
            println!("{v}");
        }
    }
    Ok(())
}
