#!/bin/bash
# this script compares folders $1 and $2 by checking if each csv file in $1 is equal to the corresponding csv file in $2, including subfolders 
# Usage: ./compare-csv-files.sh pathA pathB

PATH_A=$(realpath "$1")
PATH_B=$(realpath "$2")

# Find all CSV files in both directories
FILES_A=$(find "$PATH_A" -type f -name '*.csv')
FILES_B=$(find "$PATH_B" -type f -name '*.csv')

# Create temporary files for comparison
TMP_A=$(mktemp)
TMP_B=$(mktemp)

# Generate list of relative paths
echo "$FILES_A" | sed "s|^$PATH_A/||" | sort > "$TMP_A"
echo "$FILES_B" | sed "s|^$PATH_B/||" | sort > "$TMP_B"

# Compare the file lists
comm -23 "$TMP_A" "$TMP_B" | while read -r REL_PATH; do
  echo "ONLY in $PATH_A: $REL_PATH"
done

comm -13 "$TMP_A" "$TMP_B" | while read -r REL_PATH; do
  echo "ONLY in $PATH_B: $REL_PATH"
done

# Compare common files
comm -12 "$TMP_A" "$TMP_B" | while read -r REL_PATH; do
  FILE_A="$PATH_A/$REL_PATH"
  FILE_B="$PATH_B/$REL_PATH"

  if ! diff -q "$FILE_A" "$FILE_B" > /dev/null; then
    echo "DIFFERS: $REL_PATH"
  fi
done

# Cleanup
rm "$TMP_A" "$TMP_B"
