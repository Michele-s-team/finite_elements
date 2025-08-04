#!/bin/bash
#ths script copies from abacus all files of typs from a path on abacus to a local path, by copying only files matching patterns given by the user

# Usage:
#   ./copy_from_abacus.sh [remote_dir] [pattern1] [pattern2] ... [local_dir]
#Example
# ./copy_from_abacus.sh elastic_obstacle_2/solution/snapshots/csv/nodal_values 'def_v_n_*' 'line_mesh_el_n_*' 'line_mesh_msh_n_' 'u_msh_n_*' ~/Desktop
set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 [remote_dir] [pattern1] [pattern2] ... [local_dir]"
    exit 1
fi

REMOTE_BASE="/mnt/beegfs/home/mcastel1"
REMOTE_DIR="$1"
OUT_DIR="${@: -1}"           # Last argument is output directory
PATTERNS=("${@:2:$#-2}")     # All arguments except first and last are patterns

IN_DIR="$REMOTE_BASE/$REMOTE_DIR"

clear; clear

echo "Remote directory: $IN_DIR"
echo "Patterns to match: ${PATTERNS[*]}"
echo "Local output directory: $OUT_DIR"

# Remove and recreate local output dir
rm -rf "$OUT_DIR"
mkdir -p $OUT_DIR/$REMOTE_DIR

FIND_CMD="rm -f \"$IN_DIR/file_list.txt\"; "
FIND_CMD+="cd \"$REMOTE_BASE\" && "
FIND_CMD+="( "

for pattern in "${PATTERNS[@]}"; do
  FIND_CMD+="find \"$REMOTE_DIR\" -type f -name \"$pattern\" -printf \"%P\n\"; "
done

FIND_CMD+=") > \"$IN_DIR/file_list.txt\""


echo "Building remote file list..."
ssh mcastel1@abacus "$FIND_CMD"

echo "Copying file list locally..."
rsync --stats --size-only -P -v -e ssh mcastel1@abacus:"$IN_DIR/file_list.txt" "$OUT_DIR"

echo "Number of files to copy:"
wc -l "$OUT_DIR/file_list.txt"

echo "Starting recursive copy..."
rsync -avz --files-from="$OUT_DIR/file_list.txt" --relative -e ssh mcastel1@abacus:"$IN_DIR/" "$OUT_DIR/$REMOTE_DIR"

echo "✅ Done copying files."
