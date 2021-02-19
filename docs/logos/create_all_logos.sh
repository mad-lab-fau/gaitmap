#!/bin/bash

source="gaitmap_logo_source"
target_dir="../_static/logo/"


inkscape --export-type="png" --export-id="logo-with-text;logo" --export-id-only --export-dpi=250 "${source}.svg"
inkscape --export-type="svg" --export-id="logo-with-text;logo" --export-id-only "${source}.svg"
mkdir -p "${target_dir}"
mv "${source}_logo.png" "${target_dir}gaitmap_logo.png"
mv "${source}_logo.svg" "${target_dir}gaitmap_logo.svg"
mv "${source}_logo-with-text.png" "${target_dir}gaitmap_logo_with_text.png"
mv "${source}_logo-with-text.svg" "${target_dir}gaitmap_logo_with_text.svg"
convert -resize x16 -gravity center -crop 16x16+0+0 "${target_dir}gaitmap_logo.png" -flatten -colors 256 -background transparent "${target_dir}gaitmap.ico"
