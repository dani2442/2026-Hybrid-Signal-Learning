"""Open napari-deeplabcut labeler for the video frames."""
import os
os.environ["QT_API"] = "pyside6"

# Delete the broken H5 so napari-deeplabcut reads the CSV instead
h5_path = "/Users/eg75agon/Downloads/Project_helon/bab_bar_2pts_dlc3-Dani_F-2026-02-24/labeled-data/swept_sine_ready/CollectedData_Dani_F.h5"
if os.path.exists(h5_path):
    os.remove(h5_path)

from deeplabcut.gui.tabs.label_frames import label_frames
import napari

config_path = "/Users/eg75agon/Downloads/Project_helon/bab_bar_2pts_dlc3-Dani_F-2026-02-24/config.yaml"
label_frames(config_path)
napari.run()
