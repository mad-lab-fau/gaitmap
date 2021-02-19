#!/bin/bash

source="gaitmap_logo_source"

inkscape --export-type="png" --export-id="logo-with-text;logo" --export-id-only --export-dpi=250 "${source}.svg"

inkscape --export-type="svg" --export-id="logo-with-text;logo" --export-id-only "${source}.svg"
mv "${source}_logo.png" ./generated/gaitmap_logo.png
mv "${source}_logo.svg" ./generated/gaitmap_logo.svg
mv "${source}_logo-with-text.png" ./generated/gaitmap_logo-with-text.png
mv "${source}_logo-with-text.svg" ./generated/gaitmap_logo-with-text.svg
convert -resize x16 -gravity center -crop 16x16+0+0 ./generated/gaitmap_logo.png -flatten -colors 256 -background transparent ./generated/gaitmap.ico
