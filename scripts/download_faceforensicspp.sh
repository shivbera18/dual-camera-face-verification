#!/usr/bin/env bash
# FaceForensics++ download script
# --------------------------------
# STEP 1: Fill the Google Form to get your personal download link:
#   https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform
#
# STEP 2: You will receive an email with a download script URL like:
#   https://kaldir.vc.in.tum.de/FaceForensics/v3/FaceForensics++.py
#
# STEP 3: Set your download link below and run this script.

DOWNLOAD_SCRIPT_URL=""   # paste your personal URL here
OUTPUT_DIR="data/raw/faceforensicspp"

if [ -z "$DOWNLOAD_SCRIPT_URL" ]; then
    echo "ERROR: Set DOWNLOAD_SCRIPT_URL in this script first."
    echo "  Fill the form: https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform"
    exit 1
fi

# Download the official download script
curl -L "$DOWNLOAD_SCRIPT_URL" -o /tmp/FaceForensics_download.py

# Download faces only (c23 compression, ~3 GB total)
# Deepfakes manipulations + original real videos
python3 /tmp/FaceForensics_download.py "$OUTPUT_DIR" -d Deepfakes -c c23 -t faces
python3 /tmp/FaceForensics_download.py "$OUTPUT_DIR" -d Face2Face -c c23 -t faces
python3 /tmp/FaceForensics_download.py "$OUTPUT_DIR" -d original -c c23 -t faces

echo "Done. Data in: $OUTPUT_DIR"
