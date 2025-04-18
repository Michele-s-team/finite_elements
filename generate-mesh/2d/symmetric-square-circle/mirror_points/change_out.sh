#!/bin/bash
# Script to copy output and mesh from ubuntu to windows every time a change is done

# Define source and destination
SOURCE1="output/"
SOURCE2="mesh/"   
DESTINATION="/mnt/c/Users/ferra/OneDrive/Documenti/Tesi/Files"  # Change this to your destination directory

# Copy the file

cp -r "$SOURCE1" "$DESTINATION"
cp -r "$SOURCE2" "$DESTINATION"
echo "Files copied to $DESTINATION"
