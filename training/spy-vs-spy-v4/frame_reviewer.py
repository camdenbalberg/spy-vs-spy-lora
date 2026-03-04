#!/usr/bin/env python3
"""Frame Reviewer - PySide6 app for reviewing Spy vs Spy LoRA training frames.

View 1 (Episode List): Table of episodes with reviewed checkboxes. Home screen.
    Double-click -> opens Gallery filtered to that episode.
    Checkbox -> marks episode reviewed, moves to bottom.

View 2 (Gallery): 2x2 grid showing selected frames with full captions.
    Double-click -> open shot detail. Right-click -> exclude/include.
    Shows single episode (from Episode List) or all shots (Show All button).

View 3 (Shot Detail): All candidate frames for a shot + editable caption.
    Double-click candidate -> swap selection, return to gallery.
    Escape -> save caption, return to gallery.

v4 changes:
- Spy type selector (radio buttons)
- Non-standard outfit checkbox
- Character trigger words in caption prefix
- No style suffix
"""

import json
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QFont, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QGridLayout, QLabel, QTextEdit, QPushButton,
    QFrame, QMenu, QSizePolicy, QStackedWidget, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QComboBox, QMessageBox,
)

# Import from spy_factory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spy_factory import (
    find_selected_frame, build_caption_text, build_dataset_for_episode,
    compute_appearance_suffix, FRAMES_DIR, DATASET_DIR,
)

CANDIDATE_W = 450
CANDIDATE_H = 340
PROJECT_ROOT = Path(__file__).resolve().parent
REVIEW_PROGRESS_PATH = PROJECT_ROOT / "review_progress.json"


# -- Caption Parsing Helpers --------------------------------------------------

VIEW_OPTIONS = [
    "full body view",
    "upper body view",
    "close-up",
    "wide shot",
    "medium shot",
    "extreme close-up",
    "small figure in the distance",
    "side view",
    "overhead view",
    "low angle view",
    "split screen",
]


def infer_spy_from_caption(caption_text):
    """Infer spy type from caption text by searching for spy name mentions.

    Looks for 'white spy' and 'black spy' (case-insensitive) in the caption.
    Returns 'White', 'Black', 'Both', or 'None'.
    """
    text_lower = caption_text.lower()
    has_white = "white spy" in text_lower
    has_black = "black spy" in text_lower
    if has_white and has_black:
        return "Both"
    elif has_white:
        return "White"
    elif has_black:
        return "Black"
    return "None"


def parse_view_from_caption(caption_text):
    """Extract the view/framing tag (first comma-separated segment)."""
    comma_idx = caption_text.find(",")
    if comma_idx > 0:
        return caption_text[:comma_idx].strip()
    return ""


def parse_spies_from_caption(caption_text):
    """Find spy1 and spy2 from caption text.

    Returns (spy1, spy2) where each is 'white', 'black', or None.
    spy1 is the first spy mentioned in the text.
    """
    lower = caption_text.lower()
    white_pos = lower.find("white spy")
    black_pos = lower.find("black spy")

    if white_pos >= 0 and black_pos >= 0:
        if white_pos < black_pos:
            return ("white", "black")
        else:
            return ("black", "white")
    elif white_pos >= 0:
        return ("white", None)
    elif black_pos >= 0:
        return ("black", None)
    return (None, None)


def derive_spy_type(spy1, spy2):
    """Derive overall spy type from spy1 and spy2 values.

    Returns 'White', 'Black', 'Both', or 'None'.
    """
    has1 = spy1 is not None
    has2 = spy2 is not None
    if has1 and has2:
        return "Both"
    elif has1:
        return spy1.capitalize()
    elif has2:
        return spy2.capitalize()
    return "None"


def replace_spy_in_caption(caption, old_color, new_color):
    """Replace '{old_color} spy' with '{new_color} spy' case-insensitively."""
    return re.sub(
        rf'{re.escape(old_color)} spy',
        f'{new_color} spy',
        caption,
        flags=re.IGNORECASE,
    )


def swap_spies_in_caption(caption, color_a, color_b):
    """Swap '{color_a} spy' <-> '{color_b} spy' using placeholder approach."""
    temp = re.sub(
        rf'{re.escape(color_a)} spy', '\x00SPY_A\x00',
        caption, flags=re.IGNORECASE,
    )
    temp = re.sub(
        rf'{re.escape(color_b)} spy', '\x00SPY_B\x00',
        temp, flags=re.IGNORECASE,
    )
    result = temp.replace('\x00SPY_A\x00', f'{color_b} spy')
    result = result.replace('\x00SPY_B\x00', f'{color_a} spy')
    return result


def strip_spy_names(caption_text):
    """Remove spy name references from caption text and clean up formatting."""
    cleaned = re.sub(r"white spy'?s?\b", "", caption_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"black spy'?s?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r"^\s*,\s*", "", cleaned)
    cleaned = re.sub(r"\s*,\s*$", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


# -- Data Loading / Config ----------------------------------------------------


def load_all_shots():
    """Scan frames/ and return a flat list of (episode_stem, shot_info) dicts."""
    shots = []
    if not FRAMES_DIR.exists():
        return shots
    for ep_dir in sorted(FRAMES_DIR.iterdir()):
        if not ep_dir.is_dir():
            continue
        manifest_path = ep_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for shot_info in manifest.get("shots", []):
            shots.append({
                "episode": ep_dir.name,
                "shot": shot_info,
            })
    return shots


def group_shots_by_episode(shots):
    """Group flat shot list into OrderedDict {episode_name: [shot_data, ...]}."""
    groups = OrderedDict()
    for s in shots:
        groups.setdefault(s["episode"], []).append(s)
    return groups


def load_review_progress():
    """Load reviewed episode set from config file."""
    if REVIEW_PROGRESS_PATH.exists():
        data = json.loads(REVIEW_PROGRESS_PATH.read_text(encoding="utf-8"))
        return set(data.get("reviewed_episodes", []))
    return set()


def save_review_progress(reviewed):
    """Save reviewed episode set to config file."""
    REVIEW_PROGRESS_PATH.write_text(
        json.dumps({"reviewed_episodes": sorted(reviewed)}, indent=2),
        encoding="utf-8",
    )


def get_candidates(episode_stem, shot_num):
    """Get all candidate frame paths for a shot, sorted by index."""
    ep_dir = FRAMES_DIR / episode_stem
    shot_prefix = f"s{shot_num:02d}"
    candidates = []
    for f in sorted(ep_dir.glob(f"{shot_prefix}_*.jpg")):
        if "scores" in f.name or "caption" in f.name:
            continue
        candidates.append(f)
    return candidates


def get_selected_index(candidates):
    """Find which candidate is the selected one. Returns index or -1."""
    for i, c in enumerate(candidates):
        if "_selected" in c.name:
            return i
    return -1


def candidate_index_str(path):
    """Extract the candidate number string from a frame path."""
    name = path.stem.replace("_selected", "")
    parts = name.split("_")
    return parts[-1] if len(parts) >= 2 else "??"


# -- Gallery Thumbnail (large, with full caption) -----------------------------


class ShotThumbnail(QFrame):
    """A single cell in the 2x2 gallery grid. Shows image + full caption."""
    double_clicked = Signal(int)
    context_menu_requested = Signal(int, object)

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Left: image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self.image_label)

        # Right: info + caption
        right = QVBoxLayout()
        right.setSpacing(4)

        self.info_label = QLabel()
        self.info_label.setFont(QFont("Consolas", 11, QFont.Bold))
        self.info_label.setStyleSheet("color: #ffffff;")
        right.addWidget(self.info_label)

        self.spy_label = QLabel()
        self.spy_label.setFont(QFont("Consolas", 10))
        self.spy_label.setStyleSheet("color: #aaaaaa;")
        right.addWidget(self.spy_label)

        self.caption_label = QLabel()
        self.caption_label.setFont(QFont("Consolas", 9))
        self.caption_label.setStyleSheet("color: #cccccc;")
        self.caption_label.setWordWrap(True)
        self.caption_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.caption_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.addWidget(self.caption_label, stretch=1)

        # Claude-formatted caption (green, shown when available)
        self.claude_label = QLabel()
        self.claude_label.setFont(QFont("Consolas", 9))
        self.claude_label.setStyleSheet("color: #66ff66;")
        self.claude_label.setWordWrap(True)
        self.claude_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.claude_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.addWidget(self.claude_label, stretch=1)

        layout.addLayout(right, stretch=1)

    def set_data(self, pixmap, episode, shot_num, spy, caption, excluded=False):
        if pixmap:
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.minimumWidth(),
                    self.image_label.minimumHeight(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
            )
        else:
            self.image_label.setText("No frame")

        self.info_label.setText(f"{episode}  s{shot_num:02d}")
        self.spy_label.setText(f"Spy: {spy}")
        self.caption_label.setText(caption)
        self.claude_label.hide()

        if excluded:
            self.setStyleSheet(
                "ShotThumbnail { border: 2px solid #ff4444; background: #1a0000; }"
            )
        else:
            self.setStyleSheet(
                "ShotThumbnail { border: 1px solid #333333; background: #111111; }"
            )

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit(self.index)

    def contextMenuEvent(self, event):
        self.context_menu_requested.emit(self.index, event.globalPos())


# -- Candidate Widget ---------------------------------------------------------


class CandidateFrame(QFrame):
    """A single candidate frame in the shot detail view."""
    double_clicked = Signal(int)

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.is_selected = False
        self.setFixedSize(CANDIDATE_W + 10, CANDIDATE_H + 30)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self.image_label = QLabel()
        self.image_label.setFixedSize(CANDIDATE_W, CANDIDATE_H)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.idx_label = QLabel()
        self.idx_label.setAlignment(Qt.AlignCenter)
        self.idx_label.setFont(QFont("Consolas", 9))
        layout.addWidget(self.idx_label)

    def set_data(self, pixmap, label_text, selected=False):
        self.is_selected = selected
        if pixmap:
            self.image_label.setPixmap(
                pixmap.scaled(
                    CANDIDATE_W, CANDIDATE_H,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
            )
        self.idx_label.setText(label_text)
        self._update_style()

    def _update_style(self):
        if self.is_selected:
            self.setStyleSheet(
                "CandidateFrame { border: 3px solid #00ff00; background: #002200; }"
            )
            self.idx_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        else:
            self.setStyleSheet(
                "CandidateFrame { border: 1px solid #444444; background: #111111; }"
            )
            self.idx_label.setStyleSheet("color: #aaaaaa;")

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit(self.index)


# -- Episode List View --------------------------------------------------------


class EpisodeListView(QWidget):
    """Table of episodes with reviewed checkboxes. App home screen."""
    open_episode = Signal(str)
    show_all_gallery = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.episode_groups = OrderedDict()
        self.reviewed = set()
        self._updating_table = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(12, 8, 12, 8)

        self.header_label = QLabel()
        self.header_label.setFont(QFont("Consolas", 12, QFont.Bold))
        self.header_label.setStyleSheet("color: white;")
        header_layout.addWidget(self.header_label, stretch=1)

        show_all_btn = QPushButton("Show All Shots")
        show_all_btn.setFont(QFont("Consolas", 11))
        show_all_btn.setStyleSheet(
            "QPushButton { color: white; background: #333; border: 1px solid #555; "
            "padding: 6px 16px; } QPushButton:hover { background: #555; }"
        )
        show_all_btn.clicked.connect(self.show_all_gallery.emit)
        header_layout.addWidget(show_all_btn)

        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        header_widget.setStyleSheet("background: #1a1a2e;")
        header_widget.setFixedHeight(50)
        layout.addWidget(header_widget)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Episode", "Shots", "Reviewed"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget { background: #111; alternate-background-color: #1a1a1a; "
            "color: #ddd; gridline-color: #333; border: none; }"
            "QTableWidget::item { padding: 8px; }"
            "QTableWidget::item:selected { background: #2a2a4e; }"
            "QHeaderView::section { background: #222; color: #aaa; padding: 6px; "
            "border: 1px solid #333; font-family: Consolas; font-size: 11px; }"
        )
        self.table.setFont(QFont("Consolas", 12))
        self.table.cellDoubleClicked.connect(self._on_double_click)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table)

    def load_episodes(self, shots, reviewed):
        self.episode_groups = group_shots_by_episode(shots)
        self.reviewed = reviewed
        self._rebuild_table()

    def _rebuild_table(self):
        self._updating_table = True
        self.table.setRowCount(0)

        # Sort: unreviewed first, then reviewed, alphabetical within each
        episodes = list(self.episode_groups.keys())
        unreviewed = sorted(e for e in episodes if e not in self.reviewed)
        reviewed_list = sorted(e for e in episodes if e in self.reviewed)
        sorted_episodes = unreviewed + reviewed_list

        self.table.setRowCount(len(sorted_episodes))
        for row, ep_name in enumerate(sorted_episodes):
            ep_shots = self.episode_groups[ep_name]

            # Episode name
            name_item = QTableWidgetItem(ep_name)
            name_item.setData(Qt.UserRole, ep_name)
            self.table.setItem(row, 0, name_item)

            # Shot count
            count_item = QTableWidgetItem(str(len(ep_shots)))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, count_item)

            # Reviewed checkbox
            check_item = QTableWidgetItem()
            check_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check_item.setCheckState(
                Qt.Checked if ep_name in self.reviewed else Qt.Unchecked
            )
            check_item.setData(Qt.UserRole, ep_name)
            self.table.setItem(row, 2, check_item)

            self.table.setRowHeight(row, 40)

        reviewed_count = len(self.reviewed & set(episodes))
        self.header_label.setText(
            f"Spy vs Spy Episode Review (v4)  |  {reviewed_count}/{len(episodes)} "
            f"reviewed  |  Double-click to review episode"
        )

        self._updating_table = False

    def _on_double_click(self, row, col):
        item = self.table.item(row, 0)
        if item:
            self.open_episode.emit(item.data(Qt.UserRole))

    def _on_item_changed(self, item):
        if self._updating_table:
            return
        if item.column() != 2:
            return
        ep_name = item.data(Qt.UserRole)
        if not ep_name:
            return

        if item.checkState() == Qt.Checked:
            self.reviewed.add(ep_name)
        else:
            self.reviewed.discard(ep_name)

        save_review_progress(self.reviewed)
        self._rebuild_table()


# -- Gallery View --------------------------------------------------------------


class GalleryView(QWidget):
    """2x2 grid of shots with full captions. Scroll to see all."""
    open_shot = Signal(int)  # index into self.shots
    back_to_episodes = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shots = []
        self.thumbnails = []
        self.episode_filter = None  # None = show all, str = single episode

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with back button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(12, 4, 12, 4)

        self.back_btn = QPushButton("< Episodes")
        self.back_btn.setFont(QFont("Consolas", 11))
        self.back_btn.setStyleSheet(
            "QPushButton { color: white; background: #333; border: 1px solid #555; "
            "padding: 5px 15px; } QPushButton:hover { background: #555; }"
        )
        self.back_btn.clicked.connect(self.back_to_episodes.emit)
        header_layout.addWidget(self.back_btn)

        self.header = QLabel("Loading...")
        self.header.setFont(QFont("Consolas", 11, QFont.Bold))
        self.header.setStyleSheet("color: white;")
        header_layout.addWidget(self.header, stretch=1)

        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        header_widget.setStyleSheet("background: #1a1a2e;")
        header_widget.setFixedHeight(36)
        layout.addWidget(header_widget)

        # Scroll area with 2-column grid
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: #0a0a0a; }")
        layout.addWidget(self.scroll)

        self.grid_widget = QWidget()
        self.grid_widget.setStyleSheet("background: #0a0a0a;")
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(6)
        self.grid_layout.setContentsMargins(6, 6, 6, 6)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.scroll.setWidget(self.grid_widget)

    def load_shots(self, shots, episode_filter=None):
        self.episode_filter = episode_filter
        if episode_filter:
            self.shots = [s for s in shots if s["episode"] == episode_filter]
        else:
            self.shots = shots
        self._rebuild_grid()

    def _rebuild_grid(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.thumbnails.clear()

        episodes = set()
        for i, shot_data in enumerate(self.shots):
            episodes.add(shot_data["episode"])
            thumb = ShotThumbnail(i)
            thumb.double_clicked.connect(self._on_double_click)
            thumb.context_menu_requested.connect(self._on_context_menu)

            shot_info = shot_data["shot"]
            episode = shot_data["episode"]
            shot_num = shot_info["shot"]
            spy = shot_info.get("spy", "?")
            excluded = shot_info.get("excluded", False)
            shot_prefix = f"s{shot_num:02d}"
            caption = shot_info.get("caption", "")

            selected = find_selected_frame(FRAMES_DIR / episode, shot_prefix)
            pixmap = QPixmap(str(selected)) if selected else None

            thumb.set_data(pixmap, episode, shot_num, spy, caption, excluded)

            row, col = divmod(i, 2)
            self.grid_layout.addWidget(thumb, row, col)
            self.thumbnails.append(thumb)

        if self.episode_filter:
            self.header.setText(
                f"  {self.episode_filter}  |  {len(self.shots)} shots  |  "
                f"Double-click = review shot, Right-click = exclude/include"
            )
        else:
            self.header.setText(
                f"  {len(episodes)} episodes, {len(self.shots)} shots  |  "
                f"Double-click = review shot, Right-click = exclude/include"
            )

    def refresh_shot(self, index):
        """Refresh a single thumbnail after swap/edit."""
        if index < 0 or index >= len(self.shots):
            return
        shot_data = self.shots[index]
        shot_info = shot_data["shot"]
        episode = shot_data["episode"]
        shot_num = shot_info["shot"]
        spy = shot_info.get("spy", "?")
        excluded = shot_info.get("excluded", False)

        shot_prefix = f"s{shot_num:02d}"
        caption_path = FRAMES_DIR / episode / f"{shot_prefix}_caption.txt"

        # Update shot_info caption from file
        if caption_path.exists():
            shot_info["caption"] = caption_path.read_text(encoding="utf-8").strip()

        caption = shot_info.get("caption", "")

        selected = find_selected_frame(FRAMES_DIR / episode, shot_prefix)
        pixmap = QPixmap(str(selected)) if selected else None

        self.thumbnails[index].set_data(
            pixmap, episode, shot_num, spy, caption, excluded
        )

    def _on_double_click(self, index):
        self.open_shot.emit(index)

    def _on_context_menu(self, index, pos):
        shot_data = self.shots[index]
        shot_info = shot_data["shot"]
        excluded = shot_info.get("excluded", False)

        menu = QMenu()
        if excluded:
            action = menu.addAction("Include in dataset")
        else:
            action = menu.addAction("Exclude from dataset")

        result = menu.exec(pos)
        if result == action:
            self._toggle_exclude(index)

    def _toggle_exclude(self, index):
        shot_data = self.shots[index]
        shot_info = shot_data["shot"]
        episode = shot_data["episode"]
        shot_num = shot_info["shot"]
        shot_prefix = f"s{shot_num:02d}"
        excluded = shot_info.get("excluded", False)

        shot_info["excluded"] = not excluded

        manifest_path = FRAMES_DIR / episode / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for s in manifest["shots"]:
            if s["shot"] == shot_num:
                s["excluded"] = not excluded
                break
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        ds_img = DATASET_DIR / f"{episode}_{shot_prefix}.jpg"
        ds_txt = DATASET_DIR / f"{episode}_{shot_prefix}.txt"
        if not excluded:
            ds_img.unlink(missing_ok=True)
            ds_txt.unlink(missing_ok=True)
        else:
            build_dataset_for_episode(episode)

        self.refresh_shot(index)


# -- Shot Detail View ----------------------------------------------------------


class ShotDetailView(QWidget):
    """Detail view: candidate grid + editable caption."""
    go_back = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shot_data = None
        self.shot_index = -1
        self.candidates = []
        self.candidate_widgets = []
        self.original_caption = ""
        self._updating_controls = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(10, 5, 10, 5)

        self.back_btn = QPushButton("< Back")
        self.back_btn.setFont(QFont("Consolas", 11))
        self.back_btn.setStyleSheet(
            "QPushButton { color: white; background: #333; border: 1px solid #555; "
            "padding: 5px 15px; } QPushButton:hover { background: #555; }"
        )
        self.back_btn.clicked.connect(self._save_and_back)
        header_layout.addWidget(self.back_btn)

        self.header_label = QLabel()
        self.header_label.setFont(QFont("Consolas", 12, QFont.Bold))
        self.header_label.setStyleSheet("color: white;")
        header_layout.addWidget(self.header_label, stretch=1)

        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        header_widget.setStyleSheet("background: #1a1a2e;")
        header_widget.setFixedHeight(45)
        layout.addWidget(header_widget)

        # Candidates scroll area
        self.candidates_scroll = QScrollArea()
        self.candidates_scroll.setWidgetResizable(True)
        self.candidates_scroll.setStyleSheet(
            "QScrollArea { border: none; background: #0a0a0a; }"
        )
        layout.addWidget(self.candidates_scroll, stretch=3)

        self.candidates_widget = QWidget()
        self.candidates_widget.setStyleSheet("background: #0a0a0a;")
        self.candidates_grid = QGridLayout(self.candidates_widget)
        self.candidates_grid.setSpacing(6)
        self.candidates_grid.setContentsMargins(10, 10, 10, 10)
        self.candidates_scroll.setWidget(self.candidates_widget)

        # Caption area
        caption_container = QWidget()
        caption_container.setStyleSheet("background: #1a1a2e;")
        caption_layout = QVBoxLayout(caption_container)
        caption_layout.setContentsMargins(10, 5, 10, 10)
        caption_layout.setSpacing(4)

        # Editable caption
        self.caption_header = QLabel("Caption (editable):")
        self.caption_header.setFont(QFont("Consolas", 10))
        self.caption_header.setStyleSheet("color: #aaaaaa;")
        caption_layout.addWidget(self.caption_header)

        self.caption_edit = QTextEdit()
        self.caption_edit.setFont(QFont("Consolas", 11))
        self.caption_edit.setStyleSheet(
            "QTextEdit { background: #222244; color: white; "
            "border: 1px solid #444; padding: 8px; }"
        )
        self.caption_edit.setFixedHeight(100)
        caption_layout.addWidget(self.caption_edit)

        # Controls row: view + spy1/spy2 + hat checkboxes + nonstandard outfit
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 2, 0, 0)
        controls_layout.setSpacing(12)

        combo_style = (
            "QComboBox { background: #2a2a3a; color: #ffffff; "
            "border: 1px solid #555; padding: 3px 8px; min-width: 80px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #2a2a3a; color: #ffffff; "
            "selection-background-color: #4a4a6e; }"
        )

        # View selector (editable)
        view_group = QHBoxLayout()
        view_group.setSpacing(4)
        view_label = QLabel("View:")
        view_label.setFont(QFont("Consolas", 10))
        view_label.setStyleSheet("color: #cccccc;")
        view_group.addWidget(view_label)

        self.view_combo = QComboBox()
        self.view_combo.setEditable(True)
        self.view_combo.addItems(VIEW_OPTIONS)
        self.view_combo.setFont(QFont("Consolas", 10))
        self.view_combo.setStyleSheet(combo_style)
        self.view_combo.setMinimumWidth(160)
        self.view_combo.lineEdit().returnPressed.connect(self._on_view_changed)
        self.view_combo.activated.connect(self._on_view_changed)
        view_group.addWidget(self.view_combo)
        controls_layout.addLayout(view_group)

        # Spy 1 selector
        spy1_group = QHBoxLayout()
        spy1_group.setSpacing(4)
        spy1_label = QLabel("Spy 1:")
        spy1_label.setFont(QFont("Consolas", 10))
        spy1_label.setStyleSheet("color: #cccccc;")
        spy1_group.addWidget(spy1_label)

        self.spy1_combo = QComboBox()
        self.spy1_combo.addItems(["White", "Black", "None"])
        self.spy1_combo.setFont(QFont("Consolas", 10))
        self.spy1_combo.setStyleSheet(combo_style)
        self.spy1_combo.currentTextChanged.connect(self._on_spy1_changed)
        spy1_group.addWidget(self.spy1_combo)
        controls_layout.addLayout(spy1_group)

        # Spy 2 selector
        spy2_group = QHBoxLayout()
        spy2_group.setSpacing(4)
        spy2_label = QLabel("Spy 2:")
        spy2_label.setFont(QFont("Consolas", 10))
        spy2_label.setStyleSheet("color: #cccccc;")
        spy2_group.addWidget(spy2_label)

        self.spy2_combo = QComboBox()
        self.spy2_combo.addItems(["White", "Black", "None"])
        self.spy2_combo.setFont(QFont("Consolas", 10))
        self.spy2_combo.setStyleSheet(combo_style)
        self.spy2_combo.currentTextChanged.connect(self._on_spy2_changed)
        spy2_group.addWidget(self.spy2_combo)
        controls_layout.addLayout(spy2_group)

        # Hat checkboxes
        self.white_hat_off_cb = QCheckBox("White hat off")
        self.white_hat_off_cb.setFont(QFont("Consolas", 10))
        self.white_hat_off_cb.setStyleSheet("color: #cccccc;")
        self.white_hat_off_cb.toggled.connect(self._on_hat_toggled)
        controls_layout.addWidget(self.white_hat_off_cb)

        self.black_hat_off_cb = QCheckBox("Black hat off")
        self.black_hat_off_cb.setFont(QFont("Consolas", 10))
        self.black_hat_off_cb.setStyleSheet("color: #cccccc;")
        self.black_hat_off_cb.toggled.connect(self._on_hat_toggled)
        controls_layout.addWidget(self.black_hat_off_cb)

        # Non-standard outfit checkbox
        self.nonstandard_outfit_cb = QCheckBox("Non-standard outfit")
        self.nonstandard_outfit_cb.setFont(QFont("Consolas", 10))
        self.nonstandard_outfit_cb.setStyleSheet("color: #ff9966;")
        self.nonstandard_outfit_cb.toggled.connect(self._on_nonstandard_toggled)
        controls_layout.addWidget(self.nonstandard_outfit_cb)

        controls_layout.addStretch()
        caption_layout.addLayout(controls_layout)

        # Editable appearance suffix
        self.suffix_header = QLabel("Appearance suffix (editable):")
        self.suffix_header.setFont(QFont("Consolas", 10))
        self.suffix_header.setStyleSheet("color: #aaaaaa;")
        caption_layout.addWidget(self.suffix_header)

        self.suffix_edit = QTextEdit()
        self.suffix_edit.setFont(QFont("Consolas", 11))
        self.suffix_edit.setFixedHeight(50)
        self.suffix_edit.setStyleSheet(
            "QTextEdit { background: #2a2a3a; color: #ffcc66; "
            "border: 1px solid #555; padding: 6px; }"
        )
        caption_layout.addWidget(self.suffix_edit)

        layout.addWidget(caption_container)

    def _get_derived_spy(self):
        """Derive overall spy type from spy1/spy2 dropdowns."""
        s1 = self.spy1_combo.currentText()
        s2 = self.spy2_combo.currentText()
        spy1 = s1.lower() if s1 != "None" else None
        spy2 = s2.lower() if s2 != "None" else None
        return derive_spy_type(spy1, spy2)

    def _update_controls_enabled(self):
        """Enable/disable hat checkboxes and suffix based on spy type and nonstandard_outfit."""
        is_nonstandard = self.nonstandard_outfit_cb.isChecked()
        spy = self._get_derived_spy()

        # Hat checkboxes: disabled by nonstandard outfit OR when spy type doesn't apply
        white_hat_enabled = not is_nonstandard and spy in ("White", "Both")
        black_hat_enabled = not is_nonstandard and spy in ("Black", "Both")
        self.white_hat_off_cb.setEnabled(white_hat_enabled)
        self.black_hat_off_cb.setEnabled(black_hat_enabled)
        self.suffix_edit.setEnabled(not is_nonstandard)

        if is_nonstandard:
            self.suffix_edit.setStyleSheet(
                "QTextEdit { background: #1a1a1a; color: #666666; "
                "border: 1px solid #333; padding: 6px; }"
            )
            self.suffix_header.setStyleSheet("color: #555555;")
        else:
            self.suffix_edit.setStyleSheet(
                "QTextEdit { background: #2a2a3a; color: #ffcc66; "
                "border: 1px solid #555; padding: 6px; }"
            )
            self.suffix_header.setStyleSheet("color: #aaaaaa;")

    def load_shot(self, shot_index, shot_data):
        self._updating_controls = True

        self.shot_index = shot_index
        self.shot_data = shot_data
        shot_info = shot_data["shot"]
        episode = shot_data["episode"]
        shot_num = shot_info["shot"]
        spy = shot_info.get("spy", "?")

        self.candidates = get_candidates(episode, shot_num)
        selected_idx = get_selected_index(self.candidates)

        self.header_label.setText(
            f"  {episode}  |  Shot {shot_num:02d}  |  Spy: {spy}  |  "
            f"{len(self.candidates)} candidates  |  "
            f"Double-click a frame to select it"
        )

        shot_prefix = f"s{shot_num:02d}"
        caption_path = FRAMES_DIR / episode / f"{shot_prefix}_caption.txt"

        if caption_path.exists():
            self.original_caption = caption_path.read_text(encoding="utf-8").strip()
        else:
            self.original_caption = shot_info.get("caption", "")

        self.caption_edit.setPlainText(self.original_caption)

        # Load view selector
        view_val = shot_info.get("view", "")
        if not view_val:
            view_val = parse_view_from_caption(self.original_caption)
        idx = self.view_combo.findText(view_val)
        if idx >= 0:
            self.view_combo.setCurrentIndex(idx)
        else:
            self.view_combo.setEditText(view_val)

        # Load spy 1/spy 2 selectors
        spy1_val = shot_info.get("spy1")
        spy2_val = shot_info.get("spy2")
        self.spy1_combo.setCurrentText(spy1_val.capitalize() if spy1_val else "None")
        self.spy2_combo.setCurrentText(spy2_val.capitalize() if spy2_val else "None")

        # Load hat-off checkboxes
        self.white_hat_off_cb.setChecked(shot_info.get("white_hat_off", False))
        self.black_hat_off_cb.setChecked(shot_info.get("black_hat_off", False))

        # Load non-standard outfit
        self.nonstandard_outfit_cb.setChecked(
            shot_info.get("nonstandard_outfit", False)
        )
        self._update_controls_enabled()

        # Load appearance suffix (use saved override or compute default)
        saved_suffix = shot_info.get("appearance_suffix")
        if saved_suffix is not None:
            self.original_suffix = saved_suffix
        else:
            self.original_suffix = compute_appearance_suffix(
                shot_info.get("spy", "None"),
                shot_info.get("white_hat_off", False),
                shot_info.get("black_hat_off", False),
                shot_info.get("nonstandard_outfit", False),
            )
        self.suffix_edit.setPlainText(self.original_suffix)

        self._updating_controls = False

        # Build candidate grid
        while self.candidates_grid.count():
            item = self.candidates_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.candidate_widgets.clear()

        cols = 4
        for i, cand_path in enumerate(self.candidates):
            widget = CandidateFrame(i)
            widget.double_clicked.connect(self._on_candidate_double_click)

            pixmap = QPixmap(str(cand_path))
            idx_str = candidate_index_str(cand_path)
            is_sel = (i == selected_idx)
            label = f"#{idx_str}" + (" [SELECTED]" if is_sel else "")
            widget.set_data(pixmap, label, is_sel)

            row, col = divmod(i, cols)
            self.candidates_grid.addWidget(widget, row, col)
            self.candidate_widgets.append(widget)

        # Scroll candidates to top
        self.candidates_scroll.verticalScrollBar().setValue(0)

    def _save_caption(self):
        """Save caption if it changed."""
        if self.shot_data is None:
            return
        new_caption = self.caption_edit.toPlainText().strip()
        if new_caption == self.original_caption:
            return

        shot_info = self.shot_data["shot"]
        episode = self.shot_data["episode"]
        shot_num = shot_info["shot"]
        shot_prefix = f"s{shot_num:02d}"

        caption_path = FRAMES_DIR / episode / f"{shot_prefix}_caption.txt"
        caption_path.write_text(new_caption, encoding="utf-8")

        manifest_path = FRAMES_DIR / episode / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for s in manifest["shots"]:
            if s["shot"] == shot_num:
                s["caption"] = new_caption
                break
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        shot_info["caption"] = new_caption
        self.original_caption = new_caption

        if not shot_info.get("excluded", False):
            spy = shot_info.get("spy", "None")
            suffix = self.suffix_edit.toPlainText().strip()
            nonstandard = shot_info.get("nonstandard_outfit", False)
            ds_txt = DATASET_DIR / f"{episode}_{shot_prefix}.txt"
            ds_txt.write_text(build_caption_text(
                new_caption, spy,
                appearance_suffix=suffix if not nonstandard else "",
                nonstandard_outfit=nonstandard,
            ), encoding="utf-8")

    def _save_suffix(self):
        """Save appearance suffix if it changed."""
        if self.shot_data is None:
            return
        new_suffix = self.suffix_edit.toPlainText().strip()
        if new_suffix == self.original_suffix:
            return

        shot_info = self.shot_data["shot"]
        episode = self.shot_data["episode"]
        shot_num = shot_info["shot"]
        shot_prefix = f"s{shot_num:02d}"

        # Update in-memory
        shot_info["appearance_suffix"] = new_suffix
        self.original_suffix = new_suffix

        # Update manifest
        manifest_path = FRAMES_DIR / episode / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for s in manifest["shots"]:
            if s["shot"] == shot_num:
                s["appearance_suffix"] = new_suffix
                break
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Rebuild dataset entry
        if not shot_info.get("excluded", False):
            caption = self.caption_edit.toPlainText().strip()
            spy = shot_info.get("spy", "None")
            nonstandard = shot_info.get("nonstandard_outfit", False)
            ds_txt = DATASET_DIR / f"{episode}_{shot_prefix}.txt"
            ds_txt.write_text(build_caption_text(
                caption, spy,
                appearance_suffix=new_suffix if not nonstandard else "",
                nonstandard_outfit=nonstandard,
            ), encoding="utf-8")

    def _on_view_changed(self):
        """View dropdown changed — replace first caption segment."""
        if self._updating_controls or self.shot_data is None:
            return

        new_view = self.view_combo.currentText().strip()
        if not new_view:
            return

        shot_info = self.shot_data["shot"]
        episode = self.shot_data["episode"]
        shot_num = shot_info["shot"]
        shot_prefix = f"s{shot_num:02d}"
        old_view = shot_info.get("view", "")

        if new_view == old_view:
            return

        # Replace first comma-separated segment in caption
        caption = self.caption_edit.toPlainText().strip()
        comma_idx = caption.find(",")
        if comma_idx > 0:
            new_caption = new_view + caption[comma_idx:]
        else:
            new_caption = new_view

        # Update caption
        self.caption_edit.setPlainText(new_caption)
        self.original_caption = new_caption
        caption_path = FRAMES_DIR / episode / f"{shot_prefix}_caption.txt"
        caption_path.write_text(new_caption, encoding="utf-8")
        shot_info["caption"] = new_caption
        shot_info["view"] = new_view

        # Update manifest
        self._save_spy_to_manifest()

        print(f"[VIEW] {episode} {shot_prefix}: {old_view!r} -> {new_view!r}")

        # Rebuild dataset
        if not shot_info.get("excluded", False):
            build_dataset_for_episode(episode)

    def _on_spy1_changed(self, new_text):
        """Spy 1 dropdown changed — replace spy names in caption, update derived spy."""
        if self._updating_controls or self.shot_data is None:
            return
        self._handle_spy_dropdown_change(1, new_text)

    def _on_spy2_changed(self, new_text):
        """Spy 2 dropdown changed — replace spy names in caption, update derived spy."""
        if self._updating_controls or self.shot_data is None:
            return
        self._handle_spy_dropdown_change(2, new_text)

    def _handle_spy_dropdown_change(self, which, new_text):
        """Shared logic for spy1/spy2 dropdown changes.

        which: 1 or 2 (spy1 or spy2)
        new_text: 'White', 'Black', or 'None'
        """
        shot_info = self.shot_data["shot"]
        episode = self.shot_data["episode"]
        shot_num = shot_info["shot"]
        shot_prefix = f"s{shot_num:02d}"

        field = f"spy{which}"
        other_field = "spy2" if which == 1 else "spy1"
        other_combo = self.spy2_combo if which == 1 else self.spy1_combo

        old_color = shot_info.get(field)  # 'white', 'black', or None
        new_color = new_text.lower() if new_text != "None" else None
        other_color = shot_info.get(other_field)  # the other spy's color

        if new_color == old_color:
            return

        old_spy_type = shot_info.get("spy", "None")
        caption = self.caption_edit.toPlainText().strip()

        # Handle caption text replacement
        if old_color and new_color:
            if new_color == other_color and other_color:
                # Swapping: spy1 and spy2 exchange colors
                caption = swap_spies_in_caption(caption, old_color, other_color)
                # Auto-swap the other dropdown
                self._updating_controls = True
                other_combo.setCurrentText(old_color.capitalize())
                self._updating_controls = False
                shot_info[other_field] = old_color
                print(f"[SPY SWAP] {episode} {shot_prefix}: "
                      f"spy{which} {old_color}->{new_color}, "
                      f"spy{3-which} {other_color}->{old_color}")
            else:
                # Simple replacement
                caption = replace_spy_in_caption(caption, old_color, new_color)
                print(f"[SPY{which}] {episode} {shot_prefix}: {old_color} -> {new_color}")
        elif old_color and not new_color:
            # Spy removed — strip references
            caption = strip_spy_names(caption) if not other_color else re.sub(
                rf'{re.escape(old_color)} spy\'?s?\b', '', caption, flags=re.IGNORECASE,
            )
            # Clean up formatting
            caption = re.sub(r",\s*,", ",", caption)
            caption = re.sub(r"^\s*,\s*", "", caption)
            caption = re.sub(r"\s*,\s*$", "", caption)
            caption = re.sub(r"\s{2,}", " ", caption).strip()
            print(f"[SPY{which}] {episode} {shot_prefix}: {old_color} -> None (stripped)")
        elif not old_color and new_color:
            # No spy was in this slot — just update metadata, no caption change
            print(f"[SPY{which}] {episode} {shot_prefix}: None -> {new_color} (metadata only)")

        # Update caption if changed
        old_caption = self.caption_edit.toPlainText().strip()
        if caption != old_caption:
            self.caption_edit.setPlainText(caption)
            self.original_caption = caption
            caption_path = FRAMES_DIR / episode / f"{shot_prefix}_caption.txt"
            caption_path.write_text(caption, encoding="utf-8")
            shot_info["caption"] = caption

        # Update spy fields
        shot_info[field] = new_color
        new_spy_type = self._get_derived_spy()

        if shot_info.get("spy_original") is None:
            shot_info["spy_original"] = old_spy_type
        shot_info["spy"] = new_spy_type

        # Reset hat checkboxes based on new derived spy type
        self._updating_controls = True
        if new_spy_type == "White":
            self.black_hat_off_cb.setChecked(False)
            shot_info["black_hat_off"] = False
        elif new_spy_type == "Black":
            self.white_hat_off_cb.setChecked(False)
            shot_info["white_hat_off"] = False
        elif new_spy_type == "None":
            self.white_hat_off_cb.setChecked(False)
            self.black_hat_off_cb.setChecked(False)
            shot_info["white_hat_off"] = False
            shot_info["black_hat_off"] = False
            self.nonstandard_outfit_cb.setChecked(True)
            shot_info["nonstandard_outfit"] = True
        self._updating_controls = False

        self._update_controls_enabled()
        self._save_spy_to_manifest()
        self._recompute_suffix()

        # Rebuild dataset
        if not shot_info.get("excluded", False):
            build_dataset_for_episode(episode)

        # Update header
        self.header_label.setText(
            f"  {episode}  |  Shot {shot_num:02d}  |  Spy: {new_spy_type}  |  "
            f"{len(self.candidates)} candidates  |  "
            f"Double-click a frame to select it"
        )

    def _save_spy_to_manifest(self):
        """Save current spy/view/hat/outfit state to manifest."""
        if self.shot_data is None:
            return
        shot_info = self.shot_data["shot"]
        episode = self.shot_data["episode"]
        shot_num = shot_info["shot"]

        manifest_path = FRAMES_DIR / episode / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for s in manifest["shots"]:
            if s["shot"] == shot_num:
                s["spy"] = shot_info.get("spy", "None")
                s["spy1"] = shot_info.get("spy1")
                s["spy2"] = shot_info.get("spy2")
                s["view"] = shot_info.get("view", "")
                s["spy_original"] = shot_info.get("spy_original")
                s["white_hat_off"] = shot_info.get("white_hat_off", False)
                s["black_hat_off"] = shot_info.get("black_hat_off", False)
                s["nonstandard_outfit"] = shot_info.get("nonstandard_outfit", False)
                s["caption"] = shot_info.get("caption", s.get("caption", ""))
                break
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _on_hat_toggled(self):
        """Hat checkbox toggled — save state and recompute suffix field."""
        if self._updating_controls or self.shot_data is None:
            return

        shot_info = self.shot_data["shot"]
        shot_info["white_hat_off"] = self.white_hat_off_cb.isChecked()
        shot_info["black_hat_off"] = self.black_hat_off_cb.isChecked()

        self._save_spy_to_manifest()
        self._recompute_suffix()

    def _on_nonstandard_toggled(self, checked):
        """Non-standard outfit checkbox toggled."""
        if self._updating_controls or self.shot_data is None:
            return

        self.shot_data["shot"]["nonstandard_outfit"] = checked

        self._update_controls_enabled()
        self._save_spy_to_manifest()
        self._recompute_suffix()

        # Rebuild dataset
        if not self.shot_data["shot"].get("excluded", False):
            build_dataset_for_episode(self.shot_data["episode"])

    def _recompute_suffix(self):
        """Recompute and display the appearance suffix from current controls."""
        if self.shot_data is None:
            return
        shot_info = self.shot_data["shot"]
        spy = shot_info.get("spy", "None")
        white_off = self.white_hat_off_cb.isChecked()
        black_off = self.black_hat_off_cb.isChecked()
        nonstandard = self.nonstandard_outfit_cb.isChecked()
        new_suffix = compute_appearance_suffix(spy, white_off, black_off, nonstandard)
        self.suffix_edit.setPlainText(new_suffix)

    def _on_candidate_double_click(self, candidate_index):
        """Swap to this candidate, save caption + suffix, return to gallery."""
        self._save_caption()
        self._save_suffix()

        shot_info = self.shot_data["shot"]
        episode = self.shot_data["episode"]
        shot_num = shot_info["shot"]
        shot_prefix = f"s{shot_num:02d}"
        ep_dir = FRAMES_DIR / episode

        cand_path = self.candidates[candidate_index]

        if "_selected" in cand_path.name:
            self.go_back.emit()
            return

        # Un-rename current selection
        old_selected = find_selected_frame(ep_dir, shot_prefix)
        if old_selected:
            old_plain = old_selected.with_name(
                old_selected.name.replace("_selected", "")
            )
            old_selected.rename(old_plain)

        # Rename new candidate as selected
        new_idx = candidate_index_str(cand_path)
        new_selected = cand_path.with_name(
            f"{shot_prefix}_{new_idx}_selected.jpg"
        )
        cand_path.rename(new_selected)

        # Update manifest
        manifest_path = ep_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for s in manifest["shots"]:
            if s["shot"] == shot_num:
                s["selected"] = new_idx
                break
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        shot_info["selected"] = new_idx

        # Update dataset image
        if not shot_info.get("excluded", False):
            ds_img = DATASET_DIR / f"{episode}_{shot_prefix}.jpg"
            shutil.copy2(new_selected, ds_img)

        self.go_back.emit()

    def _toggle_exclude_current(self):
        """Toggle exclude/include for the current shot."""
        if self.shot_data is None:
            return
        shot_info = self.shot_data["shot"]
        episode = self.shot_data["episode"]
        shot_num = shot_info["shot"]
        shot_prefix = f"s{shot_num:02d}"
        excluded = shot_info.get("excluded", False)

        shot_info["excluded"] = not excluded

        manifest_path = FRAMES_DIR / episode / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for s in manifest["shots"]:
            if s["shot"] == shot_num:
                s["excluded"] = not excluded
                break
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        ds_img = DATASET_DIR / f"{episode}_{shot_prefix}.jpg"
        ds_txt = DATASET_DIR / f"{episode}_{shot_prefix}.txt"
        if not excluded:
            ds_img.unlink(missing_ok=True)
            ds_txt.unlink(missing_ok=True)
        else:
            build_dataset_for_episode(episode)

    def contextMenuEvent(self, event):
        """Right-click anywhere in detail view -> exclude/include this shot."""
        if self.shot_data is None:
            return
        excluded = self.shot_data["shot"].get("excluded", False)
        menu = QMenu(self)
        if excluded:
            action = menu.addAction("Include in dataset")
        else:
            action = menu.addAction("Exclude from dataset")
        result = menu.exec(event.globalPos())
        if result == action:
            self._toggle_exclude_current()

    def _save_and_back(self):
        self._save_caption()
        self._save_suffix()
        self.go_back.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._save_and_back()
        else:
            super().keyPressEvent(event)


# -- Main Window ---------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spy vs Spy Frame Reviewer (v4)")
        self.setMinimumSize(1280, 800)
        self.setStyleSheet("background: #0a0a0a;")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # View 1: Episode List (home)
        self.episode_list = EpisodeListView()
        self.episode_list.open_episode.connect(self._open_episode)
        self.episode_list.show_all_gallery.connect(self._show_all_gallery)
        self.stack.addWidget(self.episode_list)

        # View 2: Gallery (all shots)
        self.gallery = GalleryView()
        self.gallery.open_shot.connect(self._open_shot_from_gallery)
        self.gallery.back_to_episodes.connect(self._show_episode_list)
        self.stack.addWidget(self.gallery)

        # View 3: Shot Detail
        self.detail = ShotDetailView()
        self.detail.go_back.connect(self._back_from_detail)
        self.stack.addWidget(self.detail)

        # Load data
        self.shots = load_all_shots()
        self.reviewed = load_review_progress()
        self.episode_list.load_episodes(self.shots, self.reviewed)
        self.stack.setCurrentWidget(self.episode_list)

    def _open_episode(self, episode_name):
        """Episode list -> filtered gallery for that episode."""
        self.gallery.load_shots(self.shots, episode_filter=episode_name)
        self.stack.setCurrentWidget(self.gallery)

    def _open_shot_from_gallery(self, index):
        """Gallery -> shot detail for a single shot."""
        shot_data = self.gallery.shots[index]
        self.detail.load_shot(index, shot_data)
        self.stack.setCurrentWidget(self.detail)

    def _show_all_gallery(self):
        """Show All button -> gallery with no filter."""
        self.gallery.load_shots(self.shots)
        self.stack.setCurrentWidget(self.gallery)

    def _show_episode_list(self):
        self.stack.setCurrentWidget(self.episode_list)

    def _back_from_detail(self):
        """Detail -> back to gallery, refresh the edited shot."""
        idx = self.detail.shot_index
        if 0 <= idx < len(self.gallery.shots):
            self.gallery.refresh_shot(idx)
        self.stack.setCurrentWidget(self.gallery)
        if 0 <= idx < len(self.gallery.thumbnails):
            thumb = self.gallery.thumbnails[idx]
            self.gallery.scroll.ensureWidgetVisible(thumb)

    def closeEvent(self, event):
        if self.stack.currentWidget() == self.detail:
            self.detail._save_caption()
            self.detail._save_suffix()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            current = self.stack.currentWidget()
            if current == self.detail:
                self.detail._save_and_back()
            elif current == self.gallery:
                self._show_episode_list()
        else:
            super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from PySide6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(10, 10, 10))
    palette.setColor(QPalette.WindowText, QColor(200, 200, 200))
    palette.setColor(QPalette.Base, QColor(15, 15, 15))
    palette.setColor(QPalette.Text, QColor(200, 200, 200))
    palette.setColor(QPalette.Button, QColor(30, 30, 30))
    palette.setColor(QPalette.ButtonText, QColor(200, 200, 200))
    app.setPalette(palette)

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
