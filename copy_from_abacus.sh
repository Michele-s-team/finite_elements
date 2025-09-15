#!/bin/bash
# This script copies from abacus all files matching one or more patterns,
# filtering them by numeric suffix (min, max, stride).

# Usage:
#   ./copy_from_abacus.sh [remote_dir] [pattern1] [pattern2] ... [local_dir] [min] [max] [stride]
# Example:
#   ./copy_from_abacus.sh elastic_obstacle_2/solution/snapshots/csv/nodal_values 'u_msh_n_*' 'def_v_n*' ~/Desktop 1 40000 100
#   ./copy_from_abacus.sh elastic_obstacle_2/solution/snapshots/csv 'line_mesh_el_n_*' 'line_mesh_msh_n*' ~/Desktop 1 40000 100

set -e

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 [remote_dir] [pattern1] [pattern2] ... [local_dir] [min] [max] [stride]"
    exit 1
fi

# Parse fixed arguments
REMOTE_BASE="/mnt/beegfs/home/mcastel1"
REMOTE_DIR="$1"
OUT_DIR="${@: -4:1}"   # 4th-to-last arg is local output dir
MIN="${@: -3:1}"       # 3rd-to-last
MAX="${@: -2:1}"       # 2nd-to-last
STRIDE="${@: -1:1}"    # last arg
PATTERNS=("${@:2:$#-5}")  # Everything between remote_dir and local_dir

IN_DIR="$REMOTE_BASE/$REMOTE_DIR"

clear; clear

echo "Remote directory: $IN_DIR"
echo "Patterns to match: ${PATTERNS[*]}"
echo "Local output directory: $OUT_DIR"
echo "Filter: min=$MIN, max=$MAX, stride=$STRIDE"

# Clean and recreate output directory
#uncomment this if you want to remove the existing directory, if any
#rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/$REMOTE_DIR"

# Build the remote command to generate file_list.txt
REMOTE_CMD=$(cat <<EOF
cd "$REMOTE_BASE"
rm -f "$IN_DIR/file_list.txt"
touch "$IN_DIR/file_list.txt"
EOF
)

for pattern in "${PATTERNS[@]}"; do
    REMOTE_CMD+=$'\n'
    REMOTE_CMD+=$(cat <<EOF
find "$REMOTE_DIR" -type f -name "$pattern" -printf "%P\n" | \\
awk -F'[^0-9]*' '{
    num = -1
    for (i=NF; i>0; i--) {
        if (\$i ~ /^[0-9]+\$/) {
            num = \$i
            break
        }
    }
    if (num >= $MIN && num <= $MAX && (num - $MIN) % $STRIDE == 0)
        print \$0
}' >> "$IN_DIR/file_list.txt"
EOF
)
done

echo "Building remote file list..."
ssh mcastel1@abacus "$REMOTE_CMD"

echo "Copying file list locally..."
rsync --stats --size-only -P -v -e ssh mcastel1@abacus:"$IN_DIR/file_list.txt" "$OUT_DIR"

echo "Number of files to copy:"
wc -l "$OUT_DIR/file_list.txt"

echo "Starting recursive copy..."
rsync -avz --files-from="$OUT_DIR/file_list.txt" --relative -e ssh mcastel1@abacus:"$IN_DIR" "$OUT_DIR/$REMOTE_DIR"


echo "✅ Done copying files."
