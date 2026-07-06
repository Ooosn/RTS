#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHODS="${METHODS:-gs3_release microtexgs rng_forward rng_deferred}"

bash "${SCRIPT_DIR}/preflight_alignment.sh"

for method in ${METHODS}; do
  case "${method}" in
    gs3_release) "${SCRIPT_DIR}/run_gs3_release.sh" ;;
    microtexgs) "${SCRIPT_DIR}/run_microtexgs.sh" ;;
    rng_forward) "${SCRIPT_DIR}/run_rng_forward.sh" ;;
    rng_deferred) "${SCRIPT_DIR}/run_rng_deferred.sh" ;;
    *) echo "Unknown method: ${method}" >&2; exit 2 ;;
  esac
done
