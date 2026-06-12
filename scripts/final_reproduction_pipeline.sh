#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

COMMANDS=(
  "python3 -m py_compile src/run_otpfloodguard_experiment.py scripts/check_public_artifacts.py scripts/check_manuscript_results.py"
  "python3 src/run_otpfloodguard_experiment.py"
  "python3 scripts/check_public_artifacts.py"
  "python3 scripts/check_manuscript_results.py"
  "python3 src/build_ieee_pdf.py"
)

echo "OTPFloodGuard final reproduction pipeline"
echo "Repository: ${ROOT}"
echo "This script does not commit, push, tag, or create a release."

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "Dry-run mode. Planned commands:"
  for cmd in "${COMMANDS[@]}"; do
    echo "  ${cmd}"
  done
  exit 0
fi

cd "${ROOT}"
for cmd in "${COMMANDS[@]}"; do
  echo
  echo "+ ${cmd}"
  eval "${cmd}"
done
