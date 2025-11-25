#!/usr/bin/env bash
set -euo pipefail

# ================================
# Config
# ================================
VERSION="${1:-8.1.0}"   # You can override: ./graphviz_from_source.sh 8.0.5
TARBALL="graphviz-${VERSION}.tar.gz"
SRC_DIR="$HOME/src/graphviz-${VERSION}"
PREFIX_DIR="$HOME/graphviz-${VERSION}"

# MIT mirror with graphviz source tarballs
BASE_URL="https://mirrors.mit.edu/macports/distfiles/graphviz"
DOWNLOAD_URL="${BASE_URL}/${TARBALL}"

echo ">>> Building Graphviz ${VERSION}"
echo ">>> Source:   ${DOWNLOAD_URL}"
echo ">>> Prefix:   ${PREFIX_DIR}"
echo

# ================================
# Check for wget / curl
# ================================
if command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget -O"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl -L -o"
else
    echo "Error: neither wget nor curl is installed." >&2
    exit 1
fi

# ================================
# Download tarball
# ================================
if [ -f "${TARBALL}" ]; then
    echo ">>> Reusing existing tarball: ${TARBALL}"
else
    echo ">>> Downloading tarball..."
    ${DOWNLOADER} "${TARBALL}" "${DOWNLOAD_URL}"
fi

# ================================
# Extract source
# ================================
mkdir -p "$(dirname "${SRC_DIR}")"
rm -rf "${SRC_DIR}"
echo ">>> Extracting source to ${SRC_DIR}"
tar -xvzf "${TARBALL}" -C "$(dirname "${SRC_DIR}")"
# tar creates $HOME/src/graphviz-${VERSION}, so we're good

# ================================
# Configure + build + install
# ================================
cd "${SRC_DIR}"
echo ">>> Configuring with prefix=${PREFIX_DIR}"
./configure --prefix="${PREFIX_DIR}"

echo ">>> Running make (this may take a while)..."
make -j"$(nproc)"

echo ">>> Running make install..."
make install

# ================================
# Update PATH + LD_LIBRARY_PATH
# ================================
BASHRC="$HOME/.bashrc"

PATH_EXPORT="export PATH=\$HOME/graphviz-${VERSION}/bin:\$PATH"
LD_EXPORT="export LD_LIBRARY_PATH=\$HOME/graphviz-${VERSION}/lib:\$LD_LIBRARY_PATH"

if ! grep -Fq "${PATH_EXPORT}" "${BASHRC}"; then
    echo "" >> "${BASHRC}"
    echo "# Graphviz ${VERSION}" >> "${BASHRC}"
    echo "${PATH_EXPORT}" >> "${BASHRC}"
fi

if ! grep -Fq "${LD_EXPORT}" "${BASHRC}"; then
    echo "${LD_EXPORT}" >> "${BASHRC}"
fi

echo
echo ">>> Graphviz ${VERSION} installed under:"
echo "    ${PREFIX_DIR}"
echo
echo "Now run:"
echo "    source ~/.bashrc"
echo "and then:"
echo "    dot -V"
echo
echo "If you're on a cluster, run this on a compute node if login builds are discouraged."
