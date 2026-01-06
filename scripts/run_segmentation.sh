#!/bin/bash
# Wrapper script to run segmentation with data copied to fast storage (RAM)
# This avoids slow network filesystem I/O during normalization computation

set -e

# Configuration
PVC_DATA_ROOT="/pvc"
FAST_STORAGE="/dev/shm"  # tmpfs (RAM) - very fast
DATA_DIRS=("esk/D2A_converted_pose_smpl" "esk/D2A_converted_label_verbs" "esk/traintest_split.txt")

echo "=== Segmentation Runner with Fast Storage ==="
echo "Source: $PVC_DATA_ROOT"
echo "Fast storage: $FAST_STORAGE"

# Check available memory in /dev/shm
SHM_AVAIL=$(df -h /dev/shm | tail -1 | awk '{print $4}')
echo "Available space in /dev/shm: $SHM_AVAIL"

# Copy data to fast storage
echo ""
echo "Copying data to fast storage..."
for item in "${DATA_DIRS[@]}"; do
    src="$PVC_DATA_ROOT/$item"
    dst="$FAST_STORAGE/$item"
    
    if [ -e "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        if [ -d "$src" ]; then
            echo "  Copying directory: $item"
            cp -r "$src" "$dst"
        else
            echo "  Copying file: $item"
            cp "$src" "$dst"
        fi
    else
        echo "  WARNING: $src not found, skipping"
    fi
done

echo ""
echo "Data copy complete. Running segmentation..."
echo ""

# Run segmentation with overridden paths pointing to fast storage
# Hydra allows command-line overrides
uv run scripts/segmentation.py \
    project.data_path="$FAST_STORAGE/esk/D2A_converted_pose_smpl" \
    project.annotation_path="$FAST_STORAGE/esk/D2A_converted_label_verbs" \
    training.split_path="$FAST_STORAGE/esk/traintest_split.txt" \
    "$@"

# Cleanup (optional - comment out if you want to keep the data for debugging)
echo ""
echo "Cleaning up fast storage..."
rm -rf "$FAST_STORAGE/esk"

echo "Done!"
