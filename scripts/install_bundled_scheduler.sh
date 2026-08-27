#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="$REPO_ROOT/tools/gpu_scheduler"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
SCHEDULER_HOME="${HOME:-}/.gpu-scheduler"
LIB_DIR="$SCHEDULER_HOME/lib"
RUNTIME_BIN_DIR="$SCHEDULER_HOME/bin"
USER_BIN_DIR="${HOME:-}/.local/bin"
ASSUME_YES=0
PRINT_PLAN=0

LIBRARIES=(common.py dispatcher.py warmup.py)
COMMANDS=(gpu-scheduler gpu-enqueue gpu-delete gpu-status)

die() {
    printf 'scheduler install error: %s\n' "$*" >&2
    exit 1
}

usage() {
    printf '%s\n' \
        'Usage: bash scripts/install_bundled_scheduler.sh [--print-plan] [--yes]' \
        '' \
        'Install the bundled scheduler without overwriting an existing installation.'
}

while (($#)); do
    case "$1" in
        --print-plan) PRINT_PLAN=1 ;;
        --yes) ASSUME_YES=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

[[ -n "${HOME:-}" ]] || die 'HOME must be set'

emit_launcher() {
    local command_name="$1"
    local quoted_python quoted_script
    printf -v quoted_python '%q' "$VENV_PYTHON"
    printf -v quoted_script '%q' "$RUNTIME_BIN_DIR/$command_name"
    printf '#!/usr/bin/env bash\nexec %s %s "$@"\n' "$quoted_python" "$quoted_script"
}

mode_is() {
    [[ "$(stat -c '%a' "$1" 2>/dev/null || true)" == "$2" ]]
}

file_is() {
    local expected="$1" actual="$2" mode="$3"
    [[ -f "$actual" && ! -L "$actual" ]] || return 1
    cmp -s "$expected" "$actual" && mode_is "$actual" "$mode"
}

launcher_is() {
    local command_name="$1" actual="$USER_BIN_DIR/$1"
    [[ -f "$actual" && ! -L "$actual" ]] || return 1
    cmp -s <(emit_launcher "$command_name") "$actual" && mode_is "$actual" 755
}

has_managed_target() {
    local name
    for name in "${LIBRARIES[@]}"; do
        [[ -e "$LIB_DIR/$name" || -L "$LIB_DIR/$name" ]] && return 0
    done
    for name in "${COMMANDS[@]}"; do
        [[ -e "$RUNTIME_BIN_DIR/$name" || -L "$RUNTIME_BIN_DIR/$name" ]] && return 0
        [[ -e "$USER_BIN_DIR/$name" || -L "$USER_BIN_DIR/$name" ]] && return 0
    done
    return 1
}

is_exact_install() {
    local name
    [[ -d "$SCHEDULER_HOME" && ! -L "$SCHEDULER_HOME" ]] || return 1
    for name in "${LIBRARIES[@]}"; do
        file_is "$SOURCE_ROOT/src/$name" "$LIB_DIR/$name" 644 || return 1
    done
    for name in "${COMMANDS[@]}"; do
        file_is "$SOURCE_ROOT/bin/$name" "$RUNTIME_BIN_DIR/$name" 644 || return 1
        launcher_is "$name" || return 1
    done
}

dispatcher_is_running() {
    local pid_file="$SCHEDULER_HOME/dispatcher.pid" pid=''
    [[ -f "$pid_file" ]] || return 1
    IFS= read -r pid < "$pid_file" || true
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

installation_state() {
    if is_exact_install; then
        printf 'exact'
    elif dispatcher_is_running; then
        printf 'running'
    elif [[ -e "$SCHEDULER_HOME" || -L "$SCHEDULER_HOME" ]] || has_managed_target; then
        printf 'conflict'
    else
        printf 'fresh'
    fi
}

STATE="$(installation_state)"
if ((PRINT_PLAN)); then
    printf '%s\n' \
        "Bundled source: $SOURCE_ROOT" \
        "Frozen Python: $VENV_PYTHON" \
        "State home:    $SCHEDULER_HOME" \
        "Launchers:     $USER_BIN_DIR" \
        "Current state: $STATE" \
        '' \
        'The installer performs no network access and never starts a dispatcher.' \
        'It never modifies a running dispatcher and refuses partial or different installations.'
    exit 0
fi

[[ "$STATE" != running ]] || die 'refusing to modify a running scheduler'
if [[ "$STATE" == exact ]]; then
    printf '%s\n' 'Bundled scheduler is already installed exactly; no files changed.'
    exit 0
fi
[[ "$STATE" == fresh ]] || die 'refusing to overwrite a partial or different scheduler installation'

[[ -x "$VENV_PYTHON" ]] || die 'run scripts/setup.sh first; .venv/bin/python is missing'
python_version="$($VENV_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$python_version" == 3.11 ]] || die "frozen Python 3.11 is required; found $python_version"

for path in \
    "$SOURCE_ROOT/LICENSE" \
    "${LIBRARIES[@]/#/$SOURCE_ROOT/src/}" \
    "${COMMANDS[@]/#/$SOURCE_ROOT/bin/}"
do
    [[ -f "$path" && ! -L "$path" ]] || die "bundled scheduler file is missing: $path"
done

if ((ASSUME_YES == 0)); then
    [[ -t 0 ]] || { printf '%s\n' 'Refusing non-interactive installation without --yes.' >&2; exit 2; }
    read -r -p 'Install the bundled scheduler into your user account? [y/N] ' answer
    [[ "$answer" == y || "$answer" == Y ]] || exit 0
fi

# Recheck immediately before the first write.
STATE="$(installation_state)"
[[ "$STATE" == fresh ]] || die "installation state changed during confirmation: $STATE"

mkdir "$SCHEDULER_HOME"
mkdir "$LIB_DIR" "$RUNTIME_BIN_DIR"
mkdir "$SCHEDULER_HOME/queue" "$SCHEDULER_HOME/running" \
      "$SCHEDULER_HOME/done" "$SCHEDULER_HOME/failed" \
      "$SCHEDULER_HOME/logs" "$SCHEDULER_HOME/exit-codes" \
      "$SCHEDULER_HOME/requests" "$SCHEDULER_HOME/request-results"
mkdir -p "$USER_BIN_DIR"
chmod 700 "$SCHEDULER_HOME" "$SCHEDULER_HOME/queue" \
    "$SCHEDULER_HOME/running" "$SCHEDULER_HOME/done" \
    "$SCHEDULER_HOME/failed" "$SCHEDULER_HOME/logs" \
    "$SCHEDULER_HOME/exit-codes" "$SCHEDULER_HOME/requests" \
    "$SCHEDULER_HOME/request-results"

copy_new() {
    local source="$1" destination="$2" mode="$3"
    [[ ! -e "$destination" && ! -L "$destination" ]] || die "target appeared during installation: $destination"
    cp --no-clobber -- "$source" "$destination" \
        || die "could not create target without overwriting: $destination"
    cmp -s "$source" "$destination" \
        || die "target differs after non-overwriting copy: $destination"
    chmod "$mode" "$destination"
}

write_launcher_new() {
    local command_name="$1" destination="$USER_BIN_DIR/$1"
    [[ ! -e "$destination" && ! -L "$destination" ]] || die "target appeared during installation: $destination"
    (set -o noclobber; umask 077; emit_launcher "$command_name" > "$destination") \
        || die "could not create launcher without overwriting: $destination"
    chmod 755 "$destination"
    launcher_is "$command_name" || die "launcher verification failed: $destination"
}

for name in "${LIBRARIES[@]}"; do
    copy_new "$SOURCE_ROOT/src/$name" "$LIB_DIR/$name" 644
done
for name in "${COMMANDS[@]}"; do
    copy_new "$SOURCE_ROOT/bin/$name" "$RUNTIME_BIN_DIR/$name" 644
    write_launcher_new "$name"
done

is_exact_install || die 'post-install verification failed'
printf '%s\n' \
    'Bundled scheduler installed without starting a dispatcher.' \
    "Launchers use: $VENV_PYTHON" \
    'Add $HOME/.local/bin to PATH, then initialize only the GPUs allocated to you.'
