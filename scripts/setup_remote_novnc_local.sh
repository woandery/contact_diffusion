#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VIS_ROOT="${VIS_ROOT:-${PROJECT_ROOT}/outputs/remote_visualization}"
DEB_DIR="${VIS_ROOT}/debs"
ROOTFS="${VIS_ROOT}/rootfs"
PYTHON_DIR="${VIS_ROOT}/python"
CONTACTDIFF_PYTHON="${CONTACTDIFF_PYTHON:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/miniconda3/envs/contactdiff/bin/python}"

mkdir -p "${DEB_DIR}" "${ROOTFS}" "${PYTHON_DIR}" "${VIS_ROOT}/logs"

PACKAGE_SET_VERSION="7"
if [[ ! -f "${VIS_ROOT}/.package_set_${PACKAGE_SET_VERSION}" ]]; then
    (
        cd "${DEB_DIR}"
        apt-get download \
            xvfb xserver-common x11vnc libvncclient1 libvncserver1 \
            novnc fluxbox libsm6 libice6 \
            libgl1 libglx0 libglvnd0 libpixman-1-0 libxfont2 \
            libxtst6 libxinerama1 libxrandr2 libxfixes3 libxdamage1 \
            libavahi-common3 libavahi-client3 libxi6 \
            libfontenc1 libfreetype6 liblzo2-2 libxrender1 \
            xkb-data x11-xkb-utils libxkbfile1 proot libtalloc2 \
            libgtk-3-0t64 libpango-1.0-0 libcairo2 \
            libpangocairo-1.0-0 libpangoft2-1.0-0 libcairo-gobject2 \
            libgdk-pixbuf-2.0-0 libepoxy0 \
            libatk-bridge2.0-0t64 libatspi2.0-0t64 libcups2t64 \
            libdrm2 libxcomposite1 libxkbcommon0 libnss3 libnspr4 \
            libgbm1 libasound2t64 libwayland-client0 libwayland-cursor0 \
            libwayland-egl1 libxcursor1
    )
    for package in "${DEB_DIR}"/*.deb; do
        dpkg-deb -x "${package}" "${ROOTFS}"
    done
    touch "${VIS_ROOT}/.package_set_${PACKAGE_SET_VERSION}"
fi

if [[ ! -f "${PYTHON_DIR}/websockify/__init__.py" ]]; then
    "${CONTACTDIFF_PYTHON}" -m pip install \
        --disable-pip-version-check \
        --no-input \
        --target "${PYTHON_DIR}" \
        websockify
fi

NOVNC_WEB="${ROOTFS}/usr/share/novnc"
if [[ ! -f "${NOVNC_WEB}/vnc.html" ]]; then
    echo "noVNC web root was not found at ${NOVNC_WEB}" >&2
    exit 1
fi

echo "Remote visualization dependencies are ready under ${VIS_ROOT}"
echo "No files outside ${PROJECT_ROOT} were installed or modified."
