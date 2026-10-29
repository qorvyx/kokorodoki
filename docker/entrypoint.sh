#!/bin/bash
set -e

# The image installs with pip, not uv, so run the venv directly
KOKORODOKI=(/root/.venvs/kdvenv/bin/python3.12 /root/.local/bin/kdoki/src/main.py)

echo "Starting..."
echo "TTY: $(tty)"

if [ "$SKIP_STARTUP_MSG" != "1" ]; then
    "${KOKORODOKI[@]}" -t "Starting..."
fi

exec "${KOKORODOKI[@]}" "$@"
