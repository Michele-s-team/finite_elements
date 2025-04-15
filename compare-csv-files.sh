#!/bin/bash

# Usage: ./compare-csv-files.sh pathA pathB
PATH_A=$1
PATH_B=$2

find "$PATH_A" -name '*.csv' -exec bash -c '
  FILE_A="$1"
  FILE_B="$2/${1#"$3"/}"

  if [ -f "$FILE_B" ]; then
    if ! diff -q "$FILE_A" "$FILE_B" > /dev/null; then
      echo "DIFFERS: ${FILE_A#"$3/"}"
    fi
  else
    echo "ONLY in $3: ${FILE_A#"$3/"}"
  fi
' _ {} "$PATH_B" "$PATH_A" \;