//! Runs the built `dluth` binary and checks how it reports misuse: one line on stderr,
//! exit status 2, never a panic.

use std::process::{Command, Output};

fn dluth(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_dluth")).args(args).output().expect("run dluth")
}

/// Asserts a refusal: status 2, nothing on stdout, exactly one stderr line containing `needle`.
fn assert_refused(out: &Output, needle: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr: {stderr}");
    assert!(out.stdout.is_empty(), "stdout: {}", String::from_utf8_lossy(&out.stdout));
    assert_eq!(stderr.trim_end().lines().count(), 1, "stderr: {stderr}");
    assert!(stderr.contains(needle), "expected {needle:?} in: {stderr}");
    assert!(!stderr.contains("panicked"), "stderr: {stderr}");
}

#[test]
fn no_arguments_prints_usage() {
    assert_refused(&dluth(&[]), "usage: dluth generate");
}

#[test]
fn unknown_command_is_named() {
    assert_refused(&dluth(&["serve"]), "serve");
}

#[test]
fn unknown_flag_is_named() {
    assert_refused(&dluth(&["generate", "--model", "x", "--prompt", "hi", "--frobnicate"]), "--frobnicate");
}

#[test]
fn missing_required_flag_is_named() {
    assert_refused(&dluth(&["generate", "--prompt", "hi"]), "--model");
    assert_refused(&dluth(&["generate", "--model", "x"]), "--prompt");
}

#[test]
fn flag_without_a_value_is_named() {
    assert_refused(&dluth(&["generate", "--model"]), "--model");
}

#[test]
fn non_numeric_count_is_named() {
    assert_refused(&dluth(&["generate", "--model", "x", "--prompt", "hi", "--max-new", "many"]), "--max-new");
}

#[test]
fn model_directory_without_config_names_the_file() {
    let dir = std::env::temp_dir().join(format!("dluth-cli-empty-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dluth(&["generate", "--model", dir.to_str().unwrap(), "--prompt", "hi"]);
    std::fs::remove_dir_all(&dir).unwrap();
    assert_refused(&out, "config.json");
}

#[test]
fn a_path_containing_a_newline_still_gives_one_line() {
    let out = dluth(&["generate", "--model", "no-such-dir\nsecond line", "--prompt", "hi"]);
    assert_refused(&out, "config.json");
}

#[test]
fn help_flag_prints_usage_and_succeeds() {
    for flag in ["--help", "-h"] {
        let out = dluth(&[flag]);
        assert_eq!(out.status.code(), Some(0));
        assert!(String::from_utf8_lossy(&out.stdout).starts_with("usage: dluth generate"));
    }
}

#[test]
fn version_flag_prints_the_version() {
    let out = dluth(&["--version"]);
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), format!("dluth {}", env!("CARGO_PKG_VERSION")));
}

fn tiny_model_dir() -> String {
    format!("{}/../../tests/fixtures/tiny-qwen3", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn generates_from_the_tiny_model_end_to_end() {
    let run =
        || dluth(&["generate", "--model", &tiny_model_dir(), "--prompt", "hi", "--max-new", "6", "--max-seq", "64"]);
    let out = run();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr: {stderr}");
    // "hi" in the chat format is 5 chat tokens plus 22 single-byte tokens.
    assert!(stderr.contains("prompt: 27 tokens") && stderr.contains("generated: 6 tokens"), "stderr: {stderr}");
    assert!(out.stdout.ends_with(b"\n"));
    assert_eq!(run().stdout, out.stdout, "greedy generation must be repeatable");
}

#[test]
fn request_larger_than_the_cache_is_refused_before_generating() {
    let out =
        dluth(&["generate", "--model", &tiny_model_dir(), "--prompt", "hi", "--max-new", "40", "--max-seq", "64"]);
    assert_refused(&out, "exceeds the cache of 64");
}

#[test]
fn cache_larger_than_the_model_allows_is_refused() {
    let out = dluth(&["generate", "--model", &tiny_model_dir(), "--prompt", "hi", "--max-seq", "65"]);
    assert_refused(&out, "65");
}
