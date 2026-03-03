"""
Semi-automatic beam labeler using matplotlib.
- Click beam_left (left endpoint), then beam_right (right endpoint)
- Press 'n' for next frame, 'b' for previous frame
- Press 's' to skip a frame (marks as NaN)
- Press 'z' to undo last click on current frame
- Press 'q' to quit and save
- Progress is auto-saved every 10 frames and on quit
"""
import cv2
import numpy as np
import pandas as pd
import os
os.environ["QT_API"] = "pyside6"
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from pathlib import Path
import json


class BeamLabeler:
    def __init__(self, project_dir, scorer="Dani_F"):
        self.project_dir = Path(project_dir)
        self.scorer = scorer
        self.labeled_dir = self.project_dir / "labeled-data" / "swept_sine_ready"

        self.images = sorted(self.labeled_dir.glob("img*.png"))
        print(f"Found {len(self.images)} frames to label")

        self.progress_file = self.project_dir / "labeling_progress.json"
        self.labels = {}
        self._load_progress()

        self.current_idx = self._find_first_unlabeled()
        self.clicks = []
        self.click_markers = []

        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._show_frame()
        plt.show()

    def _load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                self.labels = json.load(f)
            print(f"Loaded {len(self.labels)} previously labeled frames")

    def _save_progress(self):
        with open(self.progress_file, "w") as f:
            json.dump(self.labels, f, indent=2)

    def _find_first_unlabeled(self):
        for i, img in enumerate(self.images):
            if img.name not in self.labels:
                return i
        return 0

    def _show_frame(self):
        self.ax.clear()
        self.clicks = []
        self.click_markers = []

        img_path = self.images[self.current_idx]
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.ax.imshow(img_rgb)

        fname = img_path.name
        if fname in self.labels:
            lbl = self.labels[fname]
            if lbl.get("beam_left"):
                self.ax.plot(*lbl["beam_left"], "go", markersize=12, label="beam_left")
            if lbl.get("beam_right"):
                self.ax.plot(*lbl["beam_right"], "ro", markersize=12, label="beam_right")
            self.ax.legend(loc="upper right")

        n_labeled = len(self.labels)
        n_total = len(self.images)
        status = "LABELED" if fname in self.labels else "UNLABELED"

        self.ax.set_title(
            f"[{self.current_idx+1}/{n_total}] {fname}  ({status})  |  "
            f"Progress: {n_labeled}/{n_total}\n"
            f"Click: beam_left (green) then beam_right (red)  |  "
            f"Keys: n=next  b=back  s=skip  z=undo  q=save+quit",
            fontsize=11,
        )
        self.fig.canvas.draw()

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return

        x, y = event.xdata, event.ydata

        if len(self.clicks) >= 2:
            return

        self.clicks.append((x, y))

        if len(self.clicks) == 1:
            marker, = self.ax.plot(x, y, "go", markersize=14, markeredgecolor="white", markeredgewidth=2)
            self.ax.annotate("beam_left", (x, y), textcoords="offset points",
                           xytext=(10, 10), color="green", fontsize=12, fontweight="bold")
            self.click_markers.append(marker)
        elif len(self.clicks) == 2:
            marker, = self.ax.plot(x, y, "ro", markersize=14, markeredgecolor="white", markeredgewidth=2)
            self.ax.annotate("beam_right", (x, y), textcoords="offset points",
                           xytext=(10, 10), color="red", fontsize=12, fontweight="bold")
            self.click_markers.append(marker)

            x1, y1 = self.clicks[0]
            x2, y2 = self.clicks[1]
            self.ax.plot([x1, x2], [y1, y2], "y--", linewidth=1.5, alpha=0.7)

            fname = self.images[self.current_idx].name
            self.labels[fname] = {
                "beam_left": [self.clicks[0][0], self.clicks[0][1]],
                "beam_right": [self.clicks[1][0], self.clicks[1][1]],
            }

        self.fig.canvas.draw()

    def _on_key(self, event):
        if event.key == "n":
            if len(self.clicks) == 2:
                if (self.current_idx + 1) % 10 == 0:
                    self._save_progress()
                    print(f"  Auto-saved at frame {self.current_idx + 1}")
            if self.current_idx < len(self.images) - 1:
                self.current_idx += 1
                self._show_frame()
            else:
                print("Last frame reached!")

        elif event.key == "b":
            if self.current_idx > 0:
                self.current_idx -= 1
                self._show_frame()

        elif event.key == "s":
            fname = self.images[self.current_idx].name
            self.labels[fname] = {"beam_left": None, "beam_right": None}
            if self.current_idx < len(self.images) - 1:
                self.current_idx += 1
                self._show_frame()

        elif event.key == "z":
            if self.clicks:
                self.clicks.pop()
                fname = self.images[self.current_idx].name
                if fname in self.labels:
                    del self.labels[fname]
                self._show_frame()

        elif event.key == "q":
            self._save_progress()
            self._export_dlc_format()
            print(f"\nDone! Labeled {len(self.labels)} frames.")
            plt.close(self.fig)

    def _export_dlc_format(self):
        bodyparts = ["beam_left", "beam_right"]
        coords = ["x", "y"]

        columns = pd.MultiIndex.from_product(
            [[self.scorer], bodyparts, coords],
            names=["scorer", "bodyparts", "coords"],
        )

        rows = []
        index_labels = []

        for img_path in self.images:
            fname = img_path.name
            if fname not in self.labels:
                continue

            lbl = self.labels[fname]
            if lbl["beam_left"] is None or lbl["beam_right"] is None:
                continue

            row = [
                lbl["beam_left"][0], lbl["beam_left"][1],
                lbl["beam_right"][0], lbl["beam_right"][1],
            ]
            rows.append(row)
            index_labels.append(f"labeled-data/swept_sine_ready/{fname}")

        if not rows:
            print("No valid labels to export!")
            return

        data = np.array(rows)
        df = pd.DataFrame(data, columns=columns, index=index_labels)

        csv_path = self.labeled_dir / f"CollectedData_{self.scorer}.csv"
        df.to_csv(csv_path)
        print(f"Saved DLC CSV: {csv_path}")

        try:
            h5_path = self.labeled_dir / f"CollectedData_{self.scorer}.h5"
            df.to_hdf(str(h5_path), key="df_with_missing", mode="w")
            print(f"Saved DLC H5: {h5_path}")
        except Exception as e:
            print(f"H5 save skipped: {e}")

        print(f"Exported {len(rows)} labeled frames to DLC format")


if __name__ == "__main__":
    project_dir = "/Users/eg75agon/Downloads/Project_helon/bab_bar_2pts_dlc3-Dani_F-2026-02-24"
    labeler = BeamLabeler(project_dir)
