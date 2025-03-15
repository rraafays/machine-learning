#!/usr/bin/env python3

import kagglehub

# Download latest version
path = kagglehub.dataset_download("thedownhill/art-images-drawings-painting-sculpture-engraving")

print("Path to dataset files:", path)
