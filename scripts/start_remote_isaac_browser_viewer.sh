#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIS_ROOT="${ROOT}/outputs/remote_visualization"
WEB_ROOT="${VIS_ROOT}/web-viewer-sample"
NODE_ROOT="${VIS_ROOT}/tools/node-v20.19.4-linux-x64"
PLAYWRIGHT_ROOT="${VIS_ROOT}/tools/playwright-browsers"
ROOTFS="${VIS_ROOT}/rootfs"
RUN_DIR="${VIS_ROOT}/run"
LOG_DIR="${VIS_ROOT}/logs"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

NODE="${NODE_ROOT}/bin/node"
NPM="${NODE_ROOT}/bin/npm"
PLAYWRIGHT_CHROMIUM="${PLAYWRIGHT_ROOT}/chromium-1181/chrome-linux/chrome"
if [[ -n "${CHROMIUM_PATH_OVERRIDE:-}" ]]; then
    CHROMIUM="${CHROMIUM_PATH_OVERRIDE}"
elif [[ -x "${PLAYWRIGHT_CHROMIUM}" ]]; then
    CHROMIUM="${PLAYWRIGHT_CHROMIUM}"
else
    CHROMIUM="${VIS_ROOT}/tools/google-chrome/opt/google/chrome/chrome"
fi
LIB_PATH="${ROOTFS}/usr/lib/x86_64-linux-gnu:${ROOTFS}/lib/x86_64-linux-gnu"

for required in "${NODE}" "${NPM}" "${CHROMIUM}" "${WEB_ROOT}/package.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required viewer component: ${required}" >&2
        exit 1
    fi
done

process_is_alive() {
    local pid_file="$1"
    local expected="$2"
    [[ -r "${pid_file}" ]] || return 1
    local pid
    pid="$(<"${pid_file}")"
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -Fq "${expected}"
}

if process_is_alive "${RUN_DIR}/web-viewer.pid" "npm run dev" \
    && curl -fsS http://127.0.0.1:5173/ >/dev/null; then
    :
else
    if process_is_alive "${RUN_DIR}/web-viewer.pid" "npm run dev"; then
        stale_pid="$(<"${RUN_DIR}/web-viewer.pid")"
        kill -TERM "${stale_pid}"
        sleep 2
    fi
    (
        cd "${WEB_ROOT}"
        export PATH="${NODE_ROOT}/bin:${PATH}"
        export npm_config_cache="${VIS_ROOT}/cache/npm"
        nohup "${NPM}" run dev -- --host 127.0.0.1 --port 5173 --strictPort \
            > "${LOG_DIR}/web-viewer.log" 2>&1 &
        echo "$!" > "${RUN_DIR}/web-viewer.pid"
    )
fi

for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:5173/ >/dev/null; then
        break
    fi
    sleep 1
done
curl -fsS http://127.0.0.1:5173/ >/dev/null

if ! process_is_alive "${RUN_DIR}/viewer-browser.pid" "remote_isaac_web_viewer_browser.mjs"; then
    (
        cd "${WEB_ROOT}"
        export DISPLAY="${DISPLAY:-:1}"
        export LD_LIBRARY_PATH="${LIB_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_ROOT}"
        export CHROMIUM_PATH="${CHROMIUM}"
        export ISAAC_WEB_VIEWER_URL="http://127.0.0.1:5173"
        nohup "${NODE}" "${ROOT}/scripts/remote_isaac_web_viewer_browser.mjs" \
            > "${LOG_DIR}/viewer-browser.log" 2>&1 &
        echo "$!" > "${RUN_DIR}/viewer-browser.pid"
    )
fi

sleep 3

echo "Web Viewer: http://127.0.0.1:5173"
echo "VNC display: ${DISPLAY:-:1}"
echo "Web Viewer PID: $(<"${RUN_DIR}/web-viewer.pid")"
echo "Browser PID: $(<"${RUN_DIR}/viewer-browser.pid")"
tail -n 20 "${LOG_DIR}/viewer-browser.log" || true
