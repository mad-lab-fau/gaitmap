# Gaitmap Logos

This folder contains the gaitmap logo source file (`gaitmap_logo_source.svg`).
All versions of the logo are generated from this file using the script `create_all_logos.sh`.

To run the script you must have inkscape and imagemagik installed.

The script relies on correct svg-ids for the important parts of the logo.
The small logo is expected to have the svg-id `logo` and the full logo should have the id `logo-with-text`.

When you modify the logo make sure to run the script and update all files in the `generated` subfolder.
Commit these files.