#!/usr/bin/env bash
# Mutation checks: apply one deliberate bug at a time and require a test to fail.
#
# Each tools/mutants/*.patch starts with comment lines:
#   # test: cargo test ...          the test that must fail with the bug applied (required)
#   # test-weights: cargo test ...  a second test that must also fail; needs real model weights
#   # needs: gpu                    the patch only makes sense on a machine with the CUDA toolkit
#
# Usage: tools/mutants.sh [cpu|all]
#   cpu (default)  skip gpu patches and weights tests; this is what CI runs
#   all            also run gpu patches when the CUDA toolkit is present, and weights tests
#                  when DLUTH_QWEN3_8B points at the model directory
#
# Results per patch:
#   KILLED    the test failed with the bug applied (good)
#   SURVIVED  the test still passed: the test does not catch this bug
#   STALE     the patch no longer applies: the code moved, refresh the patch
#   BROKEN    the mutated code did not compile, so nothing was proven
#   SKIPPED   not runnable in this mode or on this machine
# Exit status is 0 only if at least one patch ran and nothing SURVIVED, is STALE or is BROKEN.
# Before a patch is applied its test must pass on the unmodified code; otherwise a test
# that is already failing would make every planted bug look caught.
# Mutations are applied to a temporary worktree of HEAD, never to your working tree.
set -u

mode="${1:-cpu}"
case "$mode" in cpu | all) ;; *) echo "usage: $0 [cpu|all]" >&2; exit 2 ;; esac

root="$(git rev-parse --show-toplevel)" || exit 2
work="$(mktemp -d)"
tree="$work/tree"
cleanup() {
    git -C "$root" worktree remove --force "$tree" >/dev/null 2>&1
    rm -rf "$work"
}
trap cleanup EXIT
git -C "$root" worktree add --detach --quiet "$tree" HEAD || exit 2
export CARGO_TARGET_DIR="$root/target/mutants"

# The CUDA compiler, found the same way the dluth-cuda build script finds it.
have_nvcc() { command -v nvcc >/dev/null 2>&1 || [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; }

header() { sed -n "s/^# $2: *//p" "$1" | head -1; }

# baseline <test command>: the test must pass on unmodified code. Each command is checked once.
: >"$work/green"
baseline() {
    grep -Fxq -- "$1" "$work/green" && return 0
    case "$1" in "cargo test "*) ;; *) return 1 ;; esac
    # shellcheck disable=SC2086
    (cd "$tree" && $1) >"$work/log" 2>&1 || return 1
    echo "$1" >>"$work/green"
}

# run_one <patch> <test command>: prints KILLED, SURVIVED or BROKEN.
run_one() {
    local cmd="$2" log="$work/log"
    case "$cmd" in "cargo test "*) ;; *) echo "BROKEN (test command must start with 'cargo test ')"; return ;; esac
    # Compile first so a build failure is not mistaken for a caught bug.
    # shellcheck disable=SC2086
    if ! (cd "$tree" && $cmd --no-run) >"$log" 2>&1; then
        echo "BROKEN (does not compile)"
        tail -5 "$log" | sed 's/^/      /' >&2
        return
    fi
    # shellcheck disable=SC2086
    if (cd "$tree" && $cmd) >"$log" 2>&1; then echo "SURVIVED"; else echo "KILLED"; fi
}

bad=0 killed=0 skipped=0
for patch in "$root"/tools/mutants/*.patch; do
    [ -e "$patch" ] || continue
    name="$(basename "$patch" .patch)"
    test_cmd="$(header "$patch" test)"
    weights_cmd="$(header "$patch" test-weights)"
    needs="$(header "$patch" needs)"

    if [ "$needs" = "gpu" ] && { [ "$mode" != all ] || ! have_nvcc; }; then
        printf '%-34s SKIPPED (needs gpu)\n' "$name"; skipped=$((skipped + 1)); continue
    fi
    if [ -z "$test_cmd" ]; then
        printf '%-34s BROKEN (no "# test:" line)\n' "$name"; bad=$((bad + 1)); continue
    fi
    if ! baseline "$test_cmd"; then
        printf '%-34s BROKEN (test fails without the bug: %s)\n' "$name" "$test_cmd"; bad=$((bad + 1)); continue
    fi
    run_weights=0
    if [ -n "$weights_cmd" ] && [ "$mode" = all ] && [ -n "${DLUTH_QWEN3_8B:-}" ]; then
        run_weights=1
        if ! baseline "$weights_cmd"; then
            printf '%-34s BROKEN (test fails without the bug: %s)\n' "$name (weights)" "$weights_cmd"; bad=$((bad + 1)); continue
        fi
    fi
    if ! git -C "$tree" apply "$patch" 2>/dev/null; then
        printf '%-34s STALE\n' "$name"; bad=$((bad + 1)); continue
    fi

    result="$(run_one "$patch" "$test_cmd")"
    printf '%-34s %-9s %s\n' "$name" "$result" "$test_cmd"
    case "$result" in KILLED) killed=$((killed + 1)) ;; *) bad=$((bad + 1)) ;; esac

    if [ "$run_weights" -eq 1 ]; then
        result="$(run_one "$patch" "$weights_cmd")"
        printf '%-34s %-9s %s\n' "$name (weights)" "$result" "$weights_cmd"
        case "$result" in KILLED) killed=$((killed + 1)) ;; *) bad=$((bad + 1)) ;; esac
    fi

    git -C "$tree" checkout --quiet -- . && git -C "$tree" clean -fdq
done

echo "mutants: $killed killed, $bad not killed, $skipped skipped"
if [ "$killed" -eq 0 ] && [ "$bad" -eq 0 ]; then
    echo "mutants: no patch was run; tools/mutants/ is empty or everything was skipped" >&2
    exit 1
fi
[ "$bad" -eq 0 ]
