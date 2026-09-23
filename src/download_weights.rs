#!/usr/bin/env rust-script
//! Single-file Rust script for downloading GroundingDINO weights in parallel.
//! Run with: rust-script src/download_weights.rs
//!
//! Install rust-script first: cargo install rust-script
//!
//! ```cargo
//! [dependencies]
//! tokio = { version = "1", features = ["rt-multi-thread", "macros", "fs"] }
//! reqwest = { version = "0.12", features = ["stream"] }
//! futures = "0.3"
//! indicatif = { version = "0.17", features = ["tokio"] }
//! console = "0.15"
//! clap = { version = "4", features = ["derive"] }
//! zip = "2"
//! ```

use clap::{Parser, ValueEnum};
use console::{style, Emoji};
use futures::future::join_all;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::Client;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

// ── Emoji (falls back to text on non-unicode terminals) ─────────────────────

static LOOKING_GLASS: Emoji<'_, '_> = Emoji("🔍 ", "");
static DOWNLOAD: Emoji<'_, '_> = Emoji("📥 ", "");
static CHECKMARK: Emoji<'_, '_> = Emoji("✅ ", "");
static CROSS: Emoji<'_, '_> = Emoji("❌ ", "");
static SPARKLE: Emoji<'_, '_> = Emoji("✨ ", "");
static SKIP: Emoji<'_, '_> = Emoji("⏭️  ", "");
static ROCKET: Emoji<'_, '_> = Emoji("🚀 ", "");

// ── Weight definitions ──────────────────────────────────────────────────────

struct WeightInfo {
    key: &'static str,
    filename: &'static str,
    url: &'static str,
}

const WEIGHTS: &[WeightInfo] = &[
    WeightInfo {
        key: "swint",
        filename: "groundingdino_swint_ogc.pth",
        url: "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    },
    WeightInfo {
        key: "swinb",
        filename: "groundingdino_swinb_cogcoor.pth",
        url: "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
    },
];

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Clone, ValueEnum)]
enum Variant {
    Swint,
    Swinb,
    Both,
}

#[derive(Parser)]
#[command(
    name = "download-weights",
    about = "Download GroundingDINO checkpoint weights (parallel)"
)]
struct Cli {
    /// Output directory
    #[arg(short, long, default_value = "model_weights")]
    output: PathBuf,

    /// Which variant to download
    #[arg(long, value_enum, default_value = "both")]
    variant: Variant,
}

// ── Download logic ──────────────────────────────────────────────────────────

async fn download_one(
    client: &Client,
    weight: &WeightInfo,
    out_dir: &Path,
    multi: &MultiProgress,
) -> Result<(), String> {
    let dest = out_dir.join(weight.filename);

    // Skip if already present
    if dest.exists() {
        if let Ok(meta) = dest.metadata() {
            let size_mb = meta.len() as f64 / 1_000_000.0;
            multi.println(format!(
                "  {}{}  {} ({:.1} MB)",
                SKIP,
                style("already present, skipping:").dim(),
                style(weight.filename).cyan(),
                size_mb,
            )).ok();
        }
        return Ok(());
    }

    // Start the download
    let resp = client
        .get(weight.url)
        .send()
        .await
        .map_err(|e| format!("{} HTTP error for {}: {}", CROSS, weight.key, e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "{} {} HTTP {} for {}\n    Release asset may have moved — check https://github.com/IDEA-Research/GroundingDINO/releases",
            CROSS,
            weight.key,
            resp.status(),
            weight.url,
        ));
    }

    let total_size = resp.content_length().unwrap_or(0);

    // ── pixi-style progress bar ─────────────────────────────────────────
    let pb = multi.add(ProgressBar::new(total_size));
    pb.set_style(
        ProgressStyle::with_template(
            "  {spinner:.green} {msg}  [{bar:30.cyan/dim}]  {bytes}/{total_bytes}  {bytes_per_sec}  {eta}",
        )
        .unwrap()
        .progress_chars("━╸─"),
    );
    pb.set_message(format!("{}", style(weight.filename).bold()));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // Stream to a .part temp file
    let part_path = dest.with_extension("pth.part");
    let mut file = fs::File::create(&part_path)
        .await
        .map_err(|e| format!("Failed to create {}: {}", part_path.display(), e))?;

    let mut stream = resp.bytes_stream();
    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download interrupted for {}: {}", weight.key, e))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Write error: {}", e))?;
        pb.inc(chunk.len() as u64);
    }

    file.flush().await.map_err(|e| format!("Flush error: {}", e))?;
    drop(file);

    // Validate: .pth files are actually zip archives (PyTorch convention)
    let part_path_clone = part_path.clone();
    let is_valid = tokio::task::spawn_blocking(move || {
        zip::ZipArchive::new(std::fs::File::open(&part_path_clone).unwrap()).is_ok()
    })
    .await
    .unwrap_or(false);

    if !is_valid {
        let _ = fs::remove_file(&part_path).await;
        pb.finish_and_clear();
        return Err(format!(
            "{} {}: downloaded file isn't a valid archive (interrupted download or HTML error page). Re-run to retry.",
            CROSS, weight.key,
        ));
    }

    // Atomic rename
    fs::rename(&part_path, &dest)
        .await
        .map_err(|e| format!("Rename failed: {}", e))?;

    let final_size = dest.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1_000_000.0;
    pb.finish_and_clear();
    multi.println(format!(
        "  {}{} verified and saved ({:.1} MB)",
        CHECKMARK,
        style(weight.filename).green().bold(),
        final_size,
    )).ok();

    Ok(())
}

// ── Main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Banner
    eprintln!();
    eprintln!(
        "  {}{}",
        ROCKET,
        style("GroundingDINO Weight Downloader").bold()
    );
    eprintln!();

    // Filter variants
    let selected: Vec<&WeightInfo> = match cli.variant {
        Variant::Both => WEIGHTS.iter().collect(),
        Variant::Swint => WEIGHTS.iter().filter(|w| w.key == "swint").collect(),
        Variant::Swinb => WEIGHTS.iter().filter(|w| w.key == "swinb").collect(),
    };

    eprintln!(
        "  {}downloading {} variant(s) → {}/",
        LOOKING_GLASS,
        style(selected.len()).cyan().bold(),
        style(cli.output.display()).cyan(),
    );
    eprintln!();

    // Create output dir
    fs::create_dir_all(&cli.output).await.unwrap_or_else(|e| {
        eprintln!("  {}Failed to create output dir: {}", CROSS, e);
        std::process::exit(1);
    });

    // Shared HTTP client (connection pooling)
    let client = Client::builder()
        .user_agent("groundingdino-downloader/1.0")
        .build()
        .expect("Failed to build HTTP client");

    // Multi-progress for parallel bars
    let multi = MultiProgress::new();

    // Launch all downloads in parallel
    let tasks: Vec<_> = selected
        .iter()
        .map(|weight| {
            let client = client.clone();
            let out_dir = cli.output.clone();
            let multi = multi.clone();
            async move { download_one(&client, weight, &out_dir, &multi).await }
        })
        .collect();

    let results = join_all(tasks).await;

    // Report results
    eprintln!();
    let mut had_error = false;
    for result in &results {
        if let Err(msg) = result {
            eprintln!("  {}", msg);
            had_error = true;
        }
    }

    if had_error {
        eprintln!();
        std::process::exit(1);
    }

    eprintln!(
        "  {}{}",
        SPARKLE,
        style("All weights downloaded successfully!").green().bold()
    );
    eprintln!();

    // Usage hint
    eprintln!(
        "  Point your script at this directory:"
    );
    eprintln!(
        "    {}",
        style(format!(
            "python -m src.main -i <image> -w {} --objects ...",
            std::fs::canonicalize(&cli.output)
                .unwrap_or(cli.output.clone())
                .display()
        ))
        .dim()
    );
    eprintln!();
}
