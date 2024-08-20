#!/bin/bash

# Define the root directory (the top level of the tree)
ROOT_DIR="/nfs/guille/eecs_research/soundbendor/datasets/sounds_and_noise/AudioCommonsTimbre/TimbralModels_v0.2_Development/"
# Define the destination directory where all "Results" contents will be combined
DEST_DIR="$ROOT_DIR/Combined_Results"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Find all subdirectories named "Results" within the root directory
find "$ROOT_DIR" -type d -name "Results" | while read RESULTS_DIR; do
    # Copy the contents of each "Results" directory to the destination directory
    cp -r "$RESULTS_DIR/"* "$DEST_DIR"
done

echo "All 'Results' directories have been combined into $DEST_DIR."
