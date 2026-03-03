"""Launch DLC napari labeler with a proper event loop."""
import sys
from deeplabcut.gui.tabs.label_frames import label_frames

config_path = "/Users/eg75agon/Downloads/Project_helon/bab_bar_2pts_dlc3-Dani_F-2026-02-24/config.yaml"
label_frames(config_path)

# Keep napari open by starting the event loop
import napari
napari.run()
