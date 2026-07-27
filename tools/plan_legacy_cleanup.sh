#!/usr/bin/env bash
# Read-only legacy inventory.  This script never deletes files because the
# current v5/v6 implementation still imports shared v3/v4 modules.
set -Eeuo pipefail

patterns=(
  'src/tri_fair_v2*.py'
  'src/tri_fair_v3*.py'
  'src/tri_fair_v4*.py'
  'src/config/v2*.py'
  'src/config/v3*.py'
  'src/config/v4*.py'
  'src/fairness/v2*.py'
  'src/fairness/v3*.py'
  'src/fairness/v4*.py'
  'scripts/*v2*.py'
  'scripts/*v3*.py'
  'scripts/*v4*.py'
  'jobs/*v2*'
  'jobs/*v3*'
  'jobs/*v4*'
)

mapfile -t candidates < <(
  git ls-files |
  while IFS= read -r path; do
    for pattern in "${patterns[@]}"; do
      if [[ "$path" == $pattern ]]; then
        printf '%s\n' "$path"
        break
      fi
    done
  done |
  sort -u
)

printf 'Legacy tracked-file inventory (%d):\n' "${#candidates[@]}"
printf '  %s\n' "${candidates[@]}"

cat <<'EOF'

READ-ONLY REPORT: nothing was removed.

Do not delete these files yet.  Tri-Fair v5/v6 currently inherit from and import
shared v3/v4 modules.  First make v6 self-contained in a separate refactor,
validate it, create an archive branch/tag, and only then remove truly unused
entry points.
EOF
