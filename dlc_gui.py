"""Launch DLC napari labeler with proper Qt backend and event loop."""
import os
os.environ["QT_API"] = "pyside6"

import napari

# Create viewer
viewer = napari.Viewer()

# Activate the napari-deeplabcut plugin widget
for action in viewer.window.plugins_menu.actions():
    if "deeplabcut" in action.text():
        action.trigger()
        break

# Open the labeled-data folder
folder = "/Users/eg75agon/Downloads/Project_helon/bab_bar_2pts_dlc3-Dani_F-2026-02-24/labeled-data/swept_sine_ready"
config = "/Users/eg75agon/Downloads/Project_helon/bab_bar_2pts_dlc3-Dani_F-2026-02-24/config.yaml"
viewer.open([folder, config], plugin="napari-deeplabcut")

# Block until window is closed
napari.run()
