#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VIS_ROOT="${VIS_ROOT:-${PROJECT_ROOT}/outputs/remote_visualization}"
ROOTFS="${VIS_ROOT}/rootfs"
PYTHON_DIR="${VIS_ROOT}/python"
RUN_DIR="${VIS_ROOT}/run"
LOG_DIR="${VIS_ROOT}/logs"
DISPLAY_NUMBER="${DISPLAY_NUMBER:-1}"
SCREEN_SIZE="${SCREEN_SIZE:-1920x1080x24}"
VNC_PORT="${VNC_PORT:-5901}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
CONTACTDIFF_PYTHON="${CONTACTDIFF_PYTHON:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/miniconda3/envs/contactdiff/bin/python}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

XSERVER="${ROOTFS}/usr/bin/Xvfb"
X11VNC="${ROOTFS}/usr/bin/x11vnc"
FLUXBOX="${ROOTFS}/usr/bin/fluxbox"
PROOT="${ROOTFS}/usr/bin/proot"
XKBCOMP="${ROOTFS}/usr/bin/xkbcomp"
NOVNC_WEB="${ROOTFS}/usr/share/novnc"
LOCAL_LIB="${ROOTFS}/usr/lib/x86_64-linux-gnu:${ROOTFS}/lib/x86_64-linux-gnu"

for required in \
    "${XSERVER}" "${X11VNC}" "${PROOT}" "${XKBCOMP}" \
    "${CONTACTDIFF_PYTHON}" "${NOVNC_WEB}/vnc.html"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing ${required}; run scripts/setup_remote_novnc_local.sh first." >&2
        exit 1
    fi
done

stop_from_pidfile() {
    local pidfile="$1"
    if [[ -f "${pidfile}" ]]; then
        local pid
        pid="$(<"${pidfile}")"
        if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}"
        fi
    fi
}

stop_from_pidfile "${RUN_DIR}/websockify.pid"
stop_from_pidfile "${RUN_DIR}/x11vnc.pid"
stop_from_pidfile "${RUN_DIR}/fluxbox.pid"
stop_from_pidfile "${RUN_DIR}/xvfb.pid"

export LD_LIBRARY_PATH="${LOCAL_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${ROOTFS}/usr/bin:${PATH}"
export XKB_CONFIG_ROOT="${ROOTFS}/usr/share/X11/xkb"
export DISPLAY=":${DISPLAY_NUMBER}"

nohup "${PROOT}" -b "${XKBCOMP}:/usr/bin/xkbcomp" \
    "${XSERVER}" "${DISPLAY}" -screen 0 "${SCREEN_SIZE}" -ac -noreset \
    -xkbdir "${XKB_CONFIG_ROOT}" \
    >"${LOG_DIR}/xvfb.log" 2>&1 < /dev/null &
echo "$!" >"${RUN_DIR}/xvfb.pid"

for _ in $(seq 1 30); do
    if [[ -S "/tmp/.X11-unix/X${DISPLAY_NUMBER}" ]]; then
        break
    fi
    sleep 1
done
if [[ ! -S "/tmp/.X11-unix/X${DISPLAY_NUMBER}" ]]; then
    echo "Xvfb did not create ${DISPLAY}; see ${LOG_DIR}/xvfb.log" >&2
    exit 1
fi

nohup "${X11VNC}" -display "${DISPLAY}" -forever -shared -nopw \
    -rfbport "${VNC_PORT}" -localhost \
    >"${LOG_DIR}/x11vnc.log" 2>&1 < /dev/null &
echo "$!" >"${RUN_DIR}/x11vnc.pid"

sleep 1
if ! kill -0 "$(<"${RUN_DIR}/x11vnc.pid")" 2>/dev/null; then
    echo "x11vnc failed; see ${LOG_DIR}/x11vnc.log" >&2
    exit 1
fi

nohup env PYTHONPATH="${PYTHON_DIR}" "${CONTACTDIFF_PYTHON}" -m websockify \
    --web "${NOVNC_WEB}" "0.0.0.0:${NOVNC_PORT}" "127.0.0.1:${VNC_PORT}" \
    >"${LOG_DIR}/websockify.log" 2>&1 < /dev/null &
echo "$!" >"${RUN_DIR}/websockify.pid"

sleep 2
if ! kill -0 "$(<"${RUN_DIR}/websockify.pid")" 2>/dev/null; then
    echo "websockify failed; see ${LOG_DIR}/websockify.log" >&2
    exit 1
fi

echo "DISPLAY=${DISPLAY}"
echo "noVNC is listening on 0.0.0.0:${NOVNC_PORT}"
echo "Use the authenticated Inspire /proxy/${NOVNC_PORT}/ URL in your browser."
