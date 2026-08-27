#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEDULER_REPOSITORY="git@github.com:D-Diaa/gpu_scheduler.git"
SCHEDULER_COMMIT="a86c356d13c8b5fad0097e0847e11dcda4579f3d"
DESTINATION="${SWORDSTAMP_TOOLS_DIR:-$REPO_ROOT/.tools}/gpu_scheduler"
ASSUME_YES=0
PRINT_PLAN=0

usage() {
    printf '%s\n' \
        'Usage: bash scripts/install_gpu_scheduler.sh [--yes] [--print-plan] [--destination PATH]' \
        '' \
        'Clones and installs the exact gpu_scheduler revision used by SwordStamp.'
}

while (($#)); do
    case "$1" in
        --yes)
            ASSUME_YES=1
            shift
            ;;
        --print-plan)
            PRINT_PLAN=1
            shift
            ;;
        --destination)
            (($# >= 2)) || { printf '%s\n' 'missing value for --destination' >&2; exit 2; }
            DESTINATION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

print_plan() {
    printf '%s\n' \
        "Repository:  $SCHEDULER_REPOSITORY" \
        "Commit:      $SCHEDULER_COMMIT" \
        "Clone path:  $DESTINATION" \
        '' \
        'The upstream installer copies code into ~/.gpu-scheduler and commands into ~/.local/bin.' \
        'If a dispatcher is running, the upstream installer stops it and restarts it after copying files.' \
        'This script does NOT initialize a GPU pool and does NOT start a new dispatcher.'
}

print_plan
((PRINT_PLAN == 0)) || exit 0

command -v git >/dev/null 2>&1 || { printf '%s\n' 'git is required' >&2; exit 1; }
[[ -n "${HOME:-}" ]] || { printf '%s\n' 'HOME must be set for the upstream user installer' >&2; exit 1; }

if ((ASSUME_YES == 0)); then
    if [[ ! -t 0 ]]; then
        printf '%s\n' 'Refusing a non-interactive install without --yes.' >&2
        exit 2
    fi
    read -r -p 'Continue with these user-level side effects? [y/N] ' answer
    [[ "$answer" == y || "$answer" == Y ]] || exit 0
fi

if [[ -e "$DESTINATION" ]]; then
    [[ -d "$DESTINATION/.git" ]] || {
        printf 'destination exists but is not a Git checkout: %s\n' "$DESTINATION" >&2
        exit 1
    }
else
    mkdir -p "$(dirname "$DESTINATION")"
    git clone --no-checkout "$SCHEDULER_REPOSITORY" "$DESTINATION"
    git -C "$DESTINATION" checkout --detach "$SCHEDULER_COMMIT"
fi

actual_commit="$(git -C "$DESTINATION" rev-parse HEAD)"
[[ "$actual_commit" == "$SCHEDULER_COMMIT" ]] || {
    printf 'existing checkout is at %s; expected %s\n' "$actual_commit" "$SCHEDULER_COMMIT" >&2
    exit 1
}
[[ -f "$DESTINATION/install.sh" ]] || {
    printf 'pinned checkout has no install.sh: %s\n' "$DESTINATION" >&2
    exit 1
}

bash "$DESTINATION/install.sh"
printf '%s\n' \
    '' \
    'gpu_scheduler installed at the pinned revision.' \
    'No GPU pool was initialized. Choose the local GPU indices yourself, then run:' \
    '  gpu-scheduler init --gpus 0,1,...' \
    '  gpu-scheduler start'
exit 0
