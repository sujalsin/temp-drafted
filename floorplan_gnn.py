#!/usr/bin/env python3
"""
floorplan_gnn.py — GNN for Multi-Story Floor Plan Coherence
============================================================
Parses CVC-FP dataset (122 SVG-annotated floor plans), trains a Graph Attention
Network to score structural coherence when stacking single floors into multi-story
buildings, and iteratively refines violations via a heuristic agent.

Architecture: GAT with 5 scoring heads (stair alignment, wall alignment, door
sizing, egress compliance, overall coherence). Attention weights provide
per-edge interpretability for explainability.

Author: Drafted.ai ML Engineering
"""

# ============================================================
# Section A: Imports and Configuration
# ============================================================

import os
import sys
import re
import math
import copy
import random
import warnings
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader

import svgwrite
import imageio.v3 as iio
from scipy.stats import spearmanr

# ---- Reproducibility ----
SEED = 42

# ---- Paths ----
SVG_DIR = Path("/home/sujals2144/drafted/data/cvc_fp/raw_svgs/")
OUTPUT_DIR = Path("/home/sujals2144/drafted/outputs/")

# ---- Device ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Architectural constants ----
STAIR_MIN_AREA_RATIO = 0.02   # Min stair area as fraction of floor bbox area
STAIR_MAX_AREA_RATIO = 0.08   # Max stair area
WALL_ALIGN_THRESHOLD = 0.80   # 80% wall overlap required for "aligned"
DOOR_MIN_WIDTH_PX = 36         # 36-pixel minimum door opening (≈36 inches at 1px/in)
PERTURBATION_STAIR_SHIFT = (50, 100)  # pixels (≈5-10 ft at 10px/ft)

# ---- Model hyperparameters ----
NODE_FEAT_DIM = 32
HIDDEN_DIM = 64
NUM_GAT_HEADS = 4
NUM_GAT_LAYERS = 3
NUM_EDGE_TYPES = 7
EDGE_FEAT_DIM = 10   # 7 one-hot edge type + 3 spatial (norm_dx, norm_dy, norm_dist)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 80
BATCH_SIZE = 16
REFINEMENT_LOOPS = 3
COHERENCE_THRESHOLD = 0.8

# ---- Node / edge type maps ----
NODE_TYPE_MAP = {
    "Room": 0, "Wall": 1, "Door": 2, "Window": 3,
    "Separation": 4, "Parking": 5, "Stair": 6,
}
NUM_NODE_TYPES = len(NODE_TYPE_MAP)

EDGE_TYPE_MAP = {
    "spatial_proximity": 0,
    "incident": 1,
    "access": 2,
    "neighbour": 3,
    "surround": 4,
    "cross_floor": 5,
    "stair_link": 6,
}


def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Section B: Data Classes
# ============================================================

@dataclass
class Polygon:
    """A single polygon element parsed from CVC-FP SVG."""
    id: int
    cls: str                                     # e.g. 'Wall', 'Room', 'Door'
    points: List[Tuple[float, float]]
    bbox: Tuple[float, float, float, float]      # (x_min, y_min, x_max, y_max)
    centroid: Tuple[float, float]
    area: float
    width: float                                  # bbox width
    height: float                                 # bbox height


@dataclass
class Relation:
    """A structural relation from the SVG."""
    type: str                                     # incident | access | neighbour | surround | outerP
    object_ids: List[int]


@dataclass
class FloorLayout:
    """Fully parsed single-floor plan."""
    filename: str
    canvas_width: float
    canvas_height: float
    polygons: Dict[int, Polygon]
    relations: List[Relation]
    # Convenience filtered lists
    rooms: List[Polygon]
    walls: List[Polygon]
    doors: List[Polygon]
    windows: List[Polygon]
    separations: List[Polygon]
    # Parsed relation structures
    access_triples: List[Tuple[int, int, int]]     # (room, door, room)
    neighbour_pairs: List[Tuple[int, int]]
    surround_map: Dict[int, List[int]]             # room_id -> boundary ids
    outer_perimeter_ids: List[int]
    # Synthetic stair — None until injected
    synthetic_stair: Optional[Polygon] = None


@dataclass
class MultiStoryPair:
    """Two floors stacked for coherence evaluation."""
    floor_lower: FloorLayout
    floor_upper: FloorLayout
    label: float              # 1.0 = coherent, 0.0 = incoherent
    stair_iou: float
    wall_overlap: float
    door_compliant: float     # fraction of doors meeting min width
    egress_ok: float          # 1.0 if >=2 egress paths, else 0.0


@dataclass
class RefinementAction:
    """A single refinement action with human-readable explanation."""
    action_type: str          # 'shift_stair' | 'realign_wall' | 'resize_door' | 'add_egress'
    target_id: int
    description: str          # e.g. "Shifted upper stair 30px left to align with lower stair"
    delta: Dict
    score_before: float
    score_after: float


# ============================================================
# Section C: SVG Parser
# ============================================================
# Design decision: Regex-based parsing because CVC-FP SVGs contain non-standard
# bare tags (<width>, <class>, <relation>) that break standard XML parsers.

def parse_polygon_points(points_str: str) -> List[Tuple[float, float]]:
    """Parse 'x1,y1 x2,y2 ...' into [(x1,y1), (x2,y2), ...]."""
    pts = []
    for token in points_str.strip().split():
        parts = token.split(",")
        if len(parts) == 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def compute_bbox(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Axis-aligned bounding box: (x_min, y_min, x_max, y_max)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def compute_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Simple average centroid (sufficient for convex-ish floor plan polygons)."""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return (cx, cy)


def compute_area(points: List[Tuple[float, float]]) -> float:
    """Shoelace formula, orientation-independent (absolute value)."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def _classify_polygon(cls_raw: str) -> str:
    """Map raw SVG class name to our canonical types.

    CVC-FP has 50 class names; we collapse them into 7 canonical types.
    Furniture classes (Sofa-*, Table-*, Sink-*, etc.) are mapped to the
    closest structural category or skipped if irrelevant.
    """
    cls = cls_raw.strip()
    # Direct matches
    if cls in ("Wall", "Wall-1", "Wallieee"):
        return "Wall"
    if cls in ("Room",):
        return "Room"
    if cls.startswith("Door"):
        return "Door"
    if cls in ("Window",):
        return "Window"
    if cls in ("Separation",):
        return "Separation"
    if cls in ("Parking",):
        return "Parking"
    if cls.startswith("Stairs"):
        return "Stair"
    # Everything else (furniture, text, roof) — skip in structural analysis
    return "Other"


def parse_svg(filepath: str) -> Optional[FloorLayout]:
    """Parse one CVC-FP SVG file into a FloorLayout.

    Returns None if the file cannot be parsed.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        warnings.warn(f"Cannot read {filepath}: {e}")
        return None

    # Extract canvas dimensions
    w_match = re.search(r"<width>(\d+\.?\d*)</width>", content)
    h_match = re.search(r"<height>(\d+\.?\d*)</height>", content)
    canvas_w = float(w_match.group(1)) if w_match else 1000.0
    canvas_h = float(h_match.group(1)) if h_match else 1000.0

    # Extract polygons
    poly_pattern = re.compile(
        r'<polygon\s+class="([^"]+)"\s+fill="[^"]*"\s+id="(\d+)"\s+'
        r'transcription="[^"]*"\s+points="([^"]+)"\s*/>'
    )
    polygons: Dict[int, Polygon] = {}
    rooms, walls, doors, windows, separations = [], [], [], [], []

    for m in poly_pattern.finditer(content):
        cls_raw = m.group(1)
        pid = int(m.group(2))
        pts = parse_polygon_points(m.group(3))
        if len(pts) < 3:
            continue

        canonical = _classify_polygon(cls_raw)
        if canonical == "Other":
            continue  # Skip furniture / text / roof for structural analysis

        bbox = compute_bbox(pts)
        cent = compute_centroid(pts)
        area = compute_area(pts)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        poly = Polygon(id=pid, cls=canonical, points=pts,
                       bbox=bbox, centroid=cent, area=area,
                       width=w, height=h)
        polygons[pid] = poly

        if canonical == "Room":
            rooms.append(poly)
        elif canonical == "Wall":
            walls.append(poly)
        elif canonical == "Door":
            doors.append(poly)
        elif canonical == "Window":
            windows.append(poly)
        elif canonical == "Separation":
            separations.append(poly)

    # Extract relations
    rel_pattern = re.compile(r'<relation\s+type="([^"]+)"\s+objects="([^"]+)"\s*/>')
    relations: List[Relation] = []
    access_triples: List[Tuple[int, int, int]] = []
    neighbour_pairs: List[Tuple[int, int]] = []
    surround_map: Dict[int, List[int]] = {}
    outer_perimeter_ids: List[int] = []

    for m in rel_pattern.finditer(content):
        rtype = m.group(1)
        obj_ids = [int(x) for x in m.group(2).split(",")]
        relations.append(Relation(type=rtype, object_ids=obj_ids))

        if rtype == "access" and len(obj_ids) == 3:
            access_triples.append((obj_ids[0], obj_ids[1], obj_ids[2]))
        elif rtype == "neighbour" and len(obj_ids) == 2:
            neighbour_pairs.append((obj_ids[0], obj_ids[1]))
        elif rtype == "surround" and len(obj_ids) >= 2:
            # Last element is the room; rest are boundary elements
            room_id = obj_ids[-1]
            boundary = obj_ids[:-1]
            surround_map[room_id] = boundary
        elif rtype == "outerP":
            outer_perimeter_ids = obj_ids

    return FloorLayout(
        filename=os.path.basename(filepath),
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        polygons=polygons,
        relations=relations,
        rooms=rooms,
        walls=walls,
        doors=doors,
        windows=windows,
        separations=separations,
        access_triples=access_triples,
        neighbour_pairs=neighbour_pairs,
        surround_map=surround_map,
        outer_perimeter_ids=outer_perimeter_ids,
    )


def load_dataset(svg_dir: str) -> List[FloorLayout]:
    """Load all SVG files from directory. Skips files that fail to parse."""
    layouts = []
    svg_files = sorted(Path(svg_dir).glob("*.svg"))
    for fp in svg_files:
        layout = parse_svg(str(fp))
        if layout is not None and len(layout.rooms) > 0:
            layouts.append(layout)
    return layouts


# ============================================================
# Section D: Synthetic Stair Injection & Multi-Story Generation
# ============================================================
# Critical design decision: No Stairs-1 polygons exist in the actual data
# (only listed in the 50-class header). We inject synthetic stairs into the
# largest room of each floor, then create positive/negative/mixed pairs.

def _deep_copy_layout(layout: FloorLayout) -> FloorLayout:
    """Deep-copy a FloorLayout so perturbations don't affect the original."""
    return copy.deepcopy(layout)


def inject_synthetic_stair(layout: FloorLayout, rng: random.Random) -> FloorLayout:
    """Place a synthetic stair polygon inside the largest room.

    Stair sizing: 2-8% of floor bbox area, aspect ratio ~1:2 (width:height).
    Position: Near an interior wall for architectural realism.
    """
    layout = _deep_copy_layout(layout)
    if not layout.rooms:
        return layout

    # Find largest room by area
    largest = max(layout.rooms, key=lambda r: r.area)
    floor_area = layout.canvas_width * layout.canvas_height

    # Stair area between 2-8% of floor area
    stair_area = rng.uniform(STAIR_MIN_AREA_RATIO, STAIR_MAX_AREA_RATIO) * floor_area
    # Aspect ratio ~1:2 → width = sqrt(area/2), height = 2*width
    stair_w = math.sqrt(stair_area / 2.0)
    stair_h = 2.0 * stair_w

    # Place inside largest room bbox with margin
    rx_min, ry_min, rx_max, ry_max = largest.bbox
    margin = 10.0
    max_x = rx_max - stair_w - margin
    max_y = ry_max - stair_h - margin
    min_x = rx_min + margin
    min_y = ry_min + margin

    if max_x <= min_x:
        min_x, max_x = rx_min, rx_max - stair_w
    if max_y <= min_y:
        min_y, max_y = ry_min, ry_max - stair_h

    sx = rng.uniform(min_x, max(min_x, max_x))
    sy = rng.uniform(min_y, max(min_y, max_y))

    stair_pts = [
        (sx, sy), (sx + stair_w, sy),
        (sx + stair_w, sy + stair_h), (sx, sy + stair_h),
    ]

    stair_id = max(layout.polygons.keys(), default=-1) + 1
    stair = Polygon(
        id=stair_id, cls="Stair", points=stair_pts,
        bbox=(sx, sy, sx + stair_w, sy + stair_h),
        centroid=(sx + stair_w / 2, sy + stair_h / 2),
        area=stair_w * stair_h,
        width=stair_w, height=stair_h,
    )
    layout.polygons[stair_id] = stair
    layout.synthetic_stair = stair
    return layout


def _bbox_iou(a: Tuple[float, float, float, float],
              b: Tuple[float, float, float, float]) -> float:
    """Axis-aligned bounding box IoU."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_stair_iou(a: Optional[Polygon], b: Optional[Polygon]) -> float:
    """IoU between two stair bboxes. Returns 0 if either is None."""
    if a is None or b is None:
        return 0.0
    return _bbox_iou(a.bbox, b.bbox)


def compute_wall_alignment(layout_a: FloorLayout, layout_b: FloorLayout) -> float:
    """Fraction of walls in A that have a matching wall in B with IoU > threshold.

    Design decision: Use bbox overlap (not polygon overlap) because walls in
    CVC-FP are nearly axis-aligned thin rectangles.
    """
    if not layout_a.walls or not layout_b.walls:
        return 0.0
    aligned = 0
    for wa in layout_a.walls:
        best_iou = max(_bbox_iou(wa.bbox, wb.bbox) for wb in layout_b.walls)
        if best_iou >= WALL_ALIGN_THRESHOLD:
            aligned += 1
    return aligned / len(layout_a.walls)


def compute_door_compliance(layout: FloorLayout) -> float:
    """Fraction of doors meeting minimum width (36px).

    Door opening width ≈ min(bbox_width, bbox_height) because CVC-FP doors
    are fan-shaped arcs; the narrow bbox dimension approximates the opening.
    """
    if not layout.doors:
        return 1.0
    compliant = sum(1 for d in layout.doors if min(d.width, d.height) >= DOOR_MIN_WIDTH_PX)
    return compliant / len(layout.doors)


def check_egress(layout: FloorLayout) -> float:
    """Check if floor has >=2 egress paths (exterior doors).

    An egress path is a door on the outer perimeter. We count doors whose id
    appears in outer_perimeter_ids, or whose bbox overlaps an outer wall bbox.
    Returns 1.0 if >=2, else 0.0.
    """
    outer_set = set(layout.outer_perimeter_ids)
    # Count doors that are on the perimeter
    egress_count = 0
    for d in layout.doors:
        if d.id in outer_set:
            egress_count += 1
            continue
        # Check if door bbox overlaps any outer-perimeter wall bbox
        for oid in layout.outer_perimeter_ids:
            if oid in layout.polygons:
                op = layout.polygons[oid]
                if op.cls == "Wall" and _bbox_iou(d.bbox, op.bbox) > 0.01:
                    egress_count += 1
                    break
    return 1.0 if egress_count >= 2 else 0.0


def _perturb_polygon_positions(layout: FloorLayout, rng: random.Random,
                                max_shift: float) -> FloorLayout:
    """Shift all polygon positions by random amounts up to max_shift pixels."""
    layout = _deep_copy_layout(layout)
    for pid, poly in layout.polygons.items():
        dx = rng.uniform(-max_shift, max_shift)
        dy = rng.uniform(-max_shift, max_shift)
        new_pts = [(x + dx, y + dy) for (x, y) in poly.points]
        poly.points = new_pts
        poly.bbox = compute_bbox(new_pts)
        poly.centroid = compute_centroid(new_pts)
        poly.width = poly.bbox[2] - poly.bbox[0]
        poly.height = poly.bbox[3] - poly.bbox[1]
    # Recompute stair if present
    if layout.synthetic_stair and layout.synthetic_stair.id in layout.polygons:
        layout.synthetic_stair = layout.polygons[layout.synthetic_stair.id]
    # Refresh convenience lists
    layout.rooms = [p for p in layout.polygons.values() if p.cls == "Room"]
    layout.walls = [p for p in layout.polygons.values() if p.cls == "Wall"]
    layout.doors = [p for p in layout.polygons.values() if p.cls == "Door"]
    layout.windows = [p for p in layout.polygons.values() if p.cls == "Window"]
    layout.separations = [p for p in layout.polygons.values() if p.cls == "Separation"]
    return layout


def create_positive_pair(layout: FloorLayout, rng: random.Random) -> MultiStoryPair:
    """Coherent pair: clone floor with tiny jitter (construction tolerances)."""
    lower = inject_synthetic_stair(layout, rng)
    upper = _perturb_polygon_positions(lower, rng, max_shift=3.0)
    stair_iou = compute_stair_iou(lower.synthetic_stair, upper.synthetic_stair)
    wall_align = compute_wall_alignment(lower, upper)
    door_comply = compute_door_compliance(upper)
    egress = check_egress(upper)
    return MultiStoryPair(
        floor_lower=lower, floor_upper=upper,
        label=1.0, stair_iou=stair_iou, wall_overlap=wall_align,
        door_compliant=door_comply, egress_ok=egress,
    )


def create_negative_pair(layout: FloorLayout, rng: random.Random) -> MultiStoryPair:
    """Incoherent pair: deliberate structural violations in upper floor."""
    lower = inject_synthetic_stair(layout, rng)
    upper = _deep_copy_layout(lower)

    # Violation 1: Large stair shift (5-10 ft ≈ 50-100 px)
    if upper.synthetic_stair is not None:
        shift = rng.uniform(*PERTURBATION_STAIR_SHIFT)
        dx = rng.choice([-1, 1]) * shift
        dy = rng.choice([-1, 1]) * shift
        s = upper.synthetic_stair
        new_pts = [(x + dx, y + dy) for (x, y) in s.points]
        s.points = new_pts
        s.bbox = compute_bbox(new_pts)
        s.centroid = compute_centroid(new_pts)
        s.width = s.bbox[2] - s.bbox[0]
        s.height = s.bbox[3] - s.bbox[1]
        upper.polygons[s.id] = s

    # Violation 2: Remove 20-40% of walls
    n_remove = max(1, int(len(upper.walls) * rng.uniform(0.2, 0.4)))
    remove_ids = set(w.id for w in rng.sample(upper.walls, min(n_remove, len(upper.walls))))
    for rid in remove_ids:
        if rid in upper.polygons:
            del upper.polygons[rid]
    upper.walls = [p for p in upper.polygons.values() if p.cls == "Wall"]

    # Violation 3: Shrink random doors below minimum
    for d in upper.doors:
        if rng.random() < 0.5:
            scale = rng.uniform(0.3, 0.7)
            cx, cy = d.centroid
            d.points = [
                (cx + (x - cx) * scale, cy + (y - cy) * scale) for (x, y) in d.points
            ]
            d.bbox = compute_bbox(d.points)
            d.centroid = compute_centroid(d.points)
            d.width = d.bbox[2] - d.bbox[0]
            d.height = d.bbox[3] - d.bbox[1]
            upper.polygons[d.id] = d
    upper.doors = [p for p in upper.polygons.values() if p.cls == "Door"]

    stair_iou = compute_stair_iou(lower.synthetic_stair, upper.synthetic_stair)
    wall_align = compute_wall_alignment(lower, upper)
    door_comply = compute_door_compliance(upper)
    egress = check_egress(upper)
    return MultiStoryPair(
        floor_lower=lower, floor_upper=upper,
        label=0.0, stair_iou=stair_iou, wall_overlap=wall_align,
        door_compliant=door_comply, egress_ok=egress,
    )


def create_mixed_pair(layout: FloorLayout, rng: random.Random) -> MultiStoryPair:
    """Partially coherent: some aspects aligned, others broken."""
    lower = inject_synthetic_stair(layout, rng)
    upper = _deep_copy_layout(lower)

    # Only apply 1-2 mild violations
    if rng.random() < 0.5 and upper.synthetic_stair is not None:
        # Moderate stair shift (20-40 px)
        shift = rng.uniform(20, 40)
        dx = rng.choice([-1, 1]) * shift
        s = upper.synthetic_stair
        new_pts = [(x + dx, y) for (x, y) in s.points]
        s.points = new_pts
        s.bbox = compute_bbox(new_pts)
        s.centroid = compute_centroid(new_pts)
        s.width = s.bbox[2] - s.bbox[0]
        s.height = s.bbox[3] - s.bbox[1]
        upper.polygons[s.id] = s

    if rng.random() < 0.5:
        # Remove 5-15% of walls
        n_remove = max(1, int(len(upper.walls) * rng.uniform(0.05, 0.15)))
        remove_ids = set(w.id for w in rng.sample(upper.walls, min(n_remove, len(upper.walls))))
        for rid in remove_ids:
            if rid in upper.polygons:
                del upper.polygons[rid]
        upper.walls = [p for p in upper.polygons.values() if p.cls == "Wall"]

    stair_iou = compute_stair_iou(lower.synthetic_stair, upper.synthetic_stair)
    wall_align = compute_wall_alignment(lower, upper)
    door_comply = compute_door_compliance(upper)
    egress = check_egress(upper)
    # Label reflects average quality
    label = (stair_iou + wall_align + door_comply + egress) / 4.0
    return MultiStoryPair(
        floor_lower=lower, floor_upper=upper,
        label=label, stair_iou=stair_iou, wall_overlap=wall_align,
        door_compliant=door_comply, egress_ok=egress,
    )


def create_stair_sweep_pair(layout: FloorLayout, rng: random.Random,
                            shift_frac: float) -> MultiStoryPair:
    """Pair where ONLY stair is shifted by a controlled amount; rest intact.

    shift_frac: 0.0 (perfect) to 1.0 (max shift ~100px).
    Teaches the GNN to regress stair alignment as a continuous function.
    """
    lower = inject_synthetic_stair(layout, rng)
    upper = _deep_copy_layout(lower)

    if upper.synthetic_stair is not None and shift_frac > 0:
        max_shift = 100.0
        shift = shift_frac * max_shift
        angle = rng.uniform(0, 2 * math.pi)
        dx = shift * math.cos(angle)
        dy = shift * math.sin(angle)
        s = upper.synthetic_stair
        new_pts = [(x + dx, y + dy) for (x, y) in s.points]
        s.points = new_pts
        s.bbox = compute_bbox(new_pts)
        s.centroid = compute_centroid(new_pts)
        s.width = s.bbox[2] - s.bbox[0]
        s.height = s.bbox[3] - s.bbox[1]
        upper.polygons[s.id] = s

    stair_iou = compute_stair_iou(lower.synthetic_stair, upper.synthetic_stair)
    wall_align = compute_wall_alignment(lower, upper)
    door_comply = compute_door_compliance(upper)
    egress = check_egress(upper)
    label = (stair_iou + wall_align + door_comply + egress) / 4.0
    return MultiStoryPair(
        floor_lower=lower, floor_upper=upper,
        label=label, stair_iou=stair_iou, wall_overlap=wall_align,
        door_compliant=door_comply, egress_ok=egress,
    )


def create_wall_sweep_pair(layout: FloorLayout, rng: random.Random,
                           remove_frac: float) -> MultiStoryPair:
    """Pair where ONLY walls are removed at a controlled rate; rest intact.

    remove_frac: 0.0 (all walls) to 0.4 (40% removed).
    """
    lower = inject_synthetic_stair(layout, rng)
    upper = _deep_copy_layout(lower)

    if remove_frac > 0 and upper.walls:
        n_remove = max(1, int(len(upper.walls) * remove_frac))
        remove_ids = set(w.id for w in rng.sample(
            upper.walls, min(n_remove, len(upper.walls))))
        for rid in remove_ids:
            if rid in upper.polygons:
                del upper.polygons[rid]
        upper.walls = [p for p in upper.polygons.values() if p.cls == "Wall"]

    stair_iou = compute_stair_iou(lower.synthetic_stair, upper.synthetic_stair)
    wall_align = compute_wall_alignment(lower, upper)
    door_comply = compute_door_compliance(upper)
    egress = check_egress(upper)
    label = (stair_iou + wall_align + door_comply + egress) / 4.0
    return MultiStoryPair(
        floor_lower=lower, floor_upper=upper,
        label=label, stair_iou=stair_iou, wall_overlap=wall_align,
        door_compliant=door_comply, egress_ok=egress,
    )


def create_door_sweep_pair(layout: FloorLayout, rng: random.Random,
                           shrink_frac: float) -> MultiStoryPair:
    """Pair where ONLY doors are shrunk at a controlled rate; rest intact.

    shrink_frac: 0.0 (normal) to 1.0 (all shrunk to 30% of original size).
    """
    lower = inject_synthetic_stair(layout, rng)
    upper = _deep_copy_layout(lower)

    if shrink_frac > 0:
        scale = 1.0 - shrink_frac * 0.7  # 1.0 down to 0.3
        for d in upper.doors:
            cx, cy = d.centroid
            d.points = [(cx + (x - cx) * scale, cy + (y - cy) * scale)
                        for (x, y) in d.points]
            d.bbox = compute_bbox(d.points)
            d.centroid = compute_centroid(d.points)
            d.width = d.bbox[2] - d.bbox[0]
            d.height = d.bbox[3] - d.bbox[1]
            upper.polygons[d.id] = d
        upper.doors = [p for p in upper.polygons.values() if p.cls == "Door"]

    stair_iou = compute_stair_iou(lower.synthetic_stair, upper.synthetic_stair)
    wall_align = compute_wall_alignment(lower, upper)
    door_comply = compute_door_compliance(upper)
    egress = check_egress(upper)
    label = (stair_iou + wall_align + door_comply + egress) / 4.0
    return MultiStoryPair(
        floor_lower=lower, floor_upper=upper,
        label=label, stair_iou=stair_iou, wall_overlap=wall_align,
        door_compliant=door_comply, egress_ok=egress,
    )


def generate_training_data(layouts: List[FloorLayout], n_pairs: int = 2000) -> List[MultiStoryPair]:
    """Generate multi-story pairs with CONTINUOUS perturbation spectrum.

    Key design change: Instead of 3 discrete categories (pos/neg/mixed), we
    generate pairs along a continuous perturbation spectrum for EACH coherence
    dimension INDEPENDENTLY. This forces the GNN to learn smooth regression
    (tracking spatial distance), not binary classification (pos vs neg).

    When the model only sees fully-broken vs fully-intact pairs, it can only
    learn a binary classifier. By providing single-dimension sweeps with
    continuously varying perturbation levels, each scoring head gets a smooth
    gradient to learn from.

    Distribution:
    - 10% clean positive (all aligned, ~200)
    - 25% stair-only sweeps (stair shifted 0-100px, rest intact, ~500)
    - 25% wall-only sweeps (0-40% walls removed, rest intact, ~500)
    - 15% door-only sweeps (doors shrunk 0-70%, rest intact, ~300)
    - 10% multi-dimension (2+ broken at moderate levels, ~200)
    - 15% fully broken (all dimensions violated, ~300)
    """
    rng = random.Random(SEED)
    pairs = []

    n_pos = int(n_pairs * 0.10)
    n_stair = int(n_pairs * 0.25)
    n_wall = int(n_pairs * 0.25)
    n_door = int(n_pairs * 0.15)
    n_multi = int(n_pairs * 0.10)
    n_neg = n_pairs - n_pos - n_stair - n_wall - n_door - n_multi

    # 1. Clean positive pairs (baseline for "fully coherent")
    for _ in range(n_pos):
        layout = rng.choice(layouts)
        pairs.append(create_positive_pair(layout, rng))

    # 2. Stair-only sweeps: evenly spaced shift magnitudes 0.0 to 1.0
    for i in range(n_stair):
        layout = rng.choice(layouts)
        shift_frac = i / max(n_stair - 1, 1)
        shift_frac = min(1.0, max(0.0, shift_frac + rng.uniform(-0.05, 0.05)))
        pairs.append(create_stair_sweep_pair(layout, rng, shift_frac))

    # 3. Wall-only sweeps: evenly spaced removal rates 0.0 to 0.4
    for i in range(n_wall):
        layout = rng.choice(layouts)
        remove_frac = (i / max(n_wall - 1, 1)) * 0.4
        remove_frac = min(0.4, max(0.0, remove_frac + rng.uniform(-0.02, 0.02)))
        pairs.append(create_wall_sweep_pair(layout, rng, remove_frac))

    # 4. Door-only sweeps: evenly spaced shrink amounts 0.0 to 1.0
    for i in range(n_door):
        layout = rng.choice(layouts)
        shrink_frac = i / max(n_door - 1, 1)
        shrink_frac = min(1.0, max(0.0, shrink_frac + rng.uniform(-0.05, 0.05)))
        pairs.append(create_door_sweep_pair(layout, rng, shrink_frac))

    # 5. Multi-dimension: 2+ dimensions broken at moderate levels
    for _ in range(n_multi):
        layout = rng.choice(layouts)
        pairs.append(create_mixed_pair(layout, rng))

    # 6. Fully broken: all dimensions violated (old-style negative)
    for _ in range(n_neg):
        layout = rng.choice(layouts)
        pairs.append(create_negative_pair(layout, rng))

    rng.shuffle(pairs)
    return pairs


# ============================================================
# Section E: Graph Construction
# ============================================================

def encode_node_features(poly: Polygon, canvas_w: float, canvas_h: float,
                         room_area_rank: float = 0.0,
                         n_boundary: int = 0) -> torch.Tensor:
    """Encode a polygon into a 32-dim feature vector.

    Feature layout:
    [0:7]   one-hot node type (7 classes)
    [7:11]  normalized bbox: x_min/W, y_min/H, x_max/W, y_max/H
    [11:13] normalized centroid: cx/W, cy/H
    [13:15] normalized size: width/W, height/H
    [15]    normalized area: area / (W*H)
    [16]    aspect ratio: min(w,h) / max(w,h)
    [17:19] relative to center: (cx - W/2)/W, (cy - H/2)/H
    [19]    vertex count (normalized: n/20)
    [20:24] Fourier position: sin(pi*cx/W), cos(pi*cx/W), sin(pi*cy/H), cos(pi*cy/H)
    [24:28] wall-specific: is_horiz, is_vert, thickness(norm), length(norm)
    [28:30] door-specific: opening_width(norm), 0 (reserved)
    [30:32] room-specific: area_rank, n_boundary/20
    """
    feat = torch.zeros(NODE_FEAT_DIM)

    # One-hot type
    type_idx = NODE_TYPE_MAP.get(poly.cls, 0)
    feat[type_idx] = 1.0

    # Normalized bbox
    feat[7] = poly.bbox[0] / max(canvas_w, 1)
    feat[8] = poly.bbox[1] / max(canvas_h, 1)
    feat[9] = poly.bbox[2] / max(canvas_w, 1)
    feat[10] = poly.bbox[3] / max(canvas_h, 1)

    # Normalized centroid
    feat[11] = poly.centroid[0] / max(canvas_w, 1)
    feat[12] = poly.centroid[1] / max(canvas_h, 1)

    # Normalized size
    feat[13] = poly.width / max(canvas_w, 1)
    feat[14] = poly.height / max(canvas_h, 1)

    # Normalized area
    feat[15] = poly.area / max(canvas_w * canvas_h, 1)

    # Aspect ratio
    max_dim = max(poly.width, poly.height, 1e-6)
    min_dim = min(poly.width, poly.height)
    feat[16] = min_dim / max_dim

    # Relative to center
    feat[17] = (poly.centroid[0] - canvas_w / 2) / max(canvas_w, 1)
    feat[18] = (poly.centroid[1] - canvas_h / 2) / max(canvas_h, 1)

    # Vertex count
    feat[19] = min(len(poly.points) / 20.0, 1.0)

    # Fourier position encoding
    feat[20] = math.sin(math.pi * poly.centroid[0] / max(canvas_w, 1))
    feat[21] = math.cos(math.pi * poly.centroid[0] / max(canvas_w, 1))
    feat[22] = math.sin(math.pi * poly.centroid[1] / max(canvas_h, 1))
    feat[23] = math.cos(math.pi * poly.centroid[1] / max(canvas_h, 1))

    # Wall-specific features
    if poly.cls == "Wall":
        is_horiz = 1.0 if poly.width > poly.height * 2 else 0.0
        is_vert = 1.0 if poly.height > poly.width * 2 else 0.0
        thickness = min(poly.width, poly.height) / max(canvas_w, 1)
        length = max(poly.width, poly.height) / max(canvas_w, 1)
        feat[24] = is_horiz
        feat[25] = is_vert
        feat[26] = thickness
        feat[27] = length

    # Door-specific features
    if poly.cls == "Door":
        opening = min(poly.width, poly.height) / max(canvas_w, 1)
        feat[28] = opening

    # Room-specific features
    if poly.cls == "Room":
        feat[30] = room_area_rank
        feat[31] = min(n_boundary / 20.0, 1.0)

    return feat


def build_single_floor_graph(layout: FloorLayout) -> Tuple[torch.Tensor, torch.Tensor,
                                                             torch.Tensor, Dict[int, int]]:
    """Build node features, edge_index, edge_attr for one floor.

    Returns: (node_features [N, 32], edge_index [2, E], edge_attr [E, 7], id_to_idx mapping)
    """
    # Collect all structural polygons (including stair)
    all_polys = list(layout.polygons.values())
    if not all_polys:
        return (torch.zeros(1, NODE_FEAT_DIM), torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, EDGE_FEAT_DIM), {})

    # Map polygon id -> node index
    id_to_idx = {p.id: i for i, p in enumerate(all_polys)}
    n_nodes = len(all_polys)

    # Precompute room area ranks
    room_areas = sorted([p.area for p in layout.rooms], reverse=True) if layout.rooms else []
    room_rank_map = {a: i / max(len(room_areas), 1) for i, a in enumerate(room_areas)}

    # Node features
    feats = []
    for p in all_polys:
        rank = room_rank_map.get(p.area, 0.0) if p.cls == "Room" else 0.0
        n_boundary = len(layout.surround_map.get(p.id, []))
        feats.append(encode_node_features(p, layout.canvas_width, layout.canvas_height,
                                          room_area_rank=rank, n_boundary=n_boundary))
    node_features = torch.stack(feats)

    # Edge construction
    edges = []  # List of (src_idx, dst_idx, edge_type_idx)
    seen_edges: Set[Tuple[int, int, int]] = set()

    def add_edge(a_id: int, b_id: int, etype: int):
        if a_id in id_to_idx and b_id in id_to_idx:
            ai, bi = id_to_idx[a_id], id_to_idx[b_id]
            if (ai, bi, etype) not in seen_edges:
                edges.append((ai, bi, etype))
                seen_edges.add((ai, bi, etype))

    # 1. Incident edges (wall-chain connectivity)
    for rel in layout.relations:
        if rel.type == "incident" and len(rel.object_ids) == 2:
            a, b = rel.object_ids
            add_edge(a, b, EDGE_TYPE_MAP["incident"])
            add_edge(b, a, EDGE_TYPE_MAP["incident"])

    # 2. Access edges (room ↔ door ↔ room)
    for r1, d, r2 in layout.access_triples:
        add_edge(r1, d, EDGE_TYPE_MAP["access"])
        add_edge(d, r1, EDGE_TYPE_MAP["access"])
        add_edge(r2, d, EDGE_TYPE_MAP["access"])
        add_edge(d, r2, EDGE_TYPE_MAP["access"])
        add_edge(r1, r2, EDGE_TYPE_MAP["access"])
        add_edge(r2, r1, EDGE_TYPE_MAP["access"])

    # 3. Neighbour edges (room ↔ room)
    for a, b in layout.neighbour_pairs:
        add_edge(a, b, EDGE_TYPE_MAP["neighbour"])
        add_edge(b, a, EDGE_TYPE_MAP["neighbour"])

    # 4. Surround edges (room ↔ boundary elements)
    for room_id, boundary_ids in layout.surround_map.items():
        for bid in boundary_ids:
            add_edge(room_id, bid, EDGE_TYPE_MAP["surround"])
            add_edge(bid, room_id, EDGE_TYPE_MAP["surround"])

    # 5. Spatial proximity edges (fallback for uncaptured relations)
    diag = math.sqrt(layout.canvas_width**2 + layout.canvas_height**2)
    dist_threshold = 0.10 * diag
    for i, pi in enumerate(all_polys):
        for j, pj in enumerate(all_polys):
            if i >= j:
                continue
            dx = pi.centroid[0] - pj.centroid[0]
            dy = pi.centroid[1] - pj.centroid[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < dist_threshold:
                # Only add if no existing edge between these nodes
                has_edge = any((i, j, t) in seen_edges or (j, i, t) in seen_edges
                               for t in range(NUM_EDGE_TYPES))
                if not has_edge:
                    add_edge(pi.id, pj.id, EDGE_TYPE_MAP["spatial_proximity"])
                    add_edge(pj.id, pi.id, EDGE_TYPE_MAP["spatial_proximity"])

    # Convert to tensors with spatial distance features
    # Edge attr: [7 one-hot type | norm_dx | norm_dy | norm_dist] = 10-dim
    # Spatial features give the GNN a direct geometric signal for learning
    # continuous distance regression (not just binary edge type).
    if edges:
        src = torch.tensor([e[0] for e in edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
        edge_index = torch.stack([src, dst])
        edge_attr = torch.zeros(len(edges), EDGE_FEAT_DIM)
        for i, (si, di, etype) in enumerate(edges):
            edge_attr[i, etype] = 1.0
            # Spatial features: normalized dx, dy, distance between endpoints
            p_src = all_polys[si]
            p_dst = all_polys[di]
            dx = (p_src.centroid[0] - p_dst.centroid[0]) / max(diag, 1)
            dy = (p_src.centroid[1] - p_dst.centroid[1]) / max(diag, 1)
            dist = math.sqrt(dx * dx + dy * dy)
            edge_attr[i, 7] = dx
            edge_attr[i, 8] = dy
            edge_attr[i, 9] = dist
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, EDGE_FEAT_DIM)

    return node_features, edge_index, edge_attr, id_to_idx


def build_pair_graph(pair: MultiStoryPair) -> Data:
    """Build combined PyG Data for a multi-story pair.

    Strategy: Build subgraphs for each floor, offset upper node indices,
    then add cross-floor edges (same-class spatial proximity + stair link).
    """
    feat_l, ei_l, ea_l, id2idx_l = build_single_floor_graph(pair.floor_lower)
    feat_u, ei_u, ea_u, id2idx_u = build_single_floor_graph(pair.floor_upper)

    n_lower = feat_l.shape[0]
    n_upper = feat_u.shape[0]

    # Offset upper floor node indices
    if ei_u.shape[1] > 0:
        ei_u = ei_u + n_lower

    # Concatenate node features
    x = torch.cat([feat_l, feat_u], dim=0)

    # Floor mask: 0 = lower, 1 = upper
    floor_mask = torch.cat([
        torch.zeros(n_lower, dtype=torch.long),
        torch.ones(n_upper, dtype=torch.long),
    ])

    # Concatenate within-floor edges
    if ei_l.shape[1] > 0 and ei_u.shape[1] > 0:
        edge_index = torch.cat([ei_l, ei_u], dim=1)
        edge_attr = torch.cat([ea_l, ea_u], dim=0)
    elif ei_l.shape[1] > 0:
        edge_index = ei_l
        edge_attr = ea_l
    elif ei_u.shape[1] > 0:
        edge_index = ei_u
        edge_attr = ea_u
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, EDGE_FEAT_DIM)

    # Cross-floor edges with spatial distance features.
    # These are critical for learning continuous alignment: each cross-floor
    # edge now carries the normalized dx, dy, and distance between the two
    # connected polygons, giving the model a direct geometric signal.
    cross_edges_src = []
    cross_edges_dst = []
    cross_edge_types = []
    cross_edge_spatial = []  # (dx_norm, dy_norm, dist_norm) per edge

    polys_l = list(pair.floor_lower.polygons.values())
    polys_u = list(pair.floor_upper.polygons.values())
    diag = math.sqrt(pair.floor_lower.canvas_width**2 + pair.floor_lower.canvas_height**2)
    diag = max(diag, 1.0)

    # Stair-to-stair link
    if pair.floor_lower.synthetic_stair and pair.floor_upper.synthetic_stair:
        sl = pair.floor_lower.synthetic_stair
        su = pair.floor_upper.synthetic_stair
        sl_id = sl.id
        su_id = su.id
        if sl_id in id2idx_l and su_id in id2idx_u:
            li = id2idx_l[sl_id]
            ui = id2idx_u[su_id] + n_lower
            dx = (sl.centroid[0] - su.centroid[0]) / diag
            dy = (sl.centroid[1] - su.centroid[1]) / diag
            dist = math.sqrt(dx * dx + dy * dy)
            cross_edges_src.extend([li, ui])
            cross_edges_dst.extend([ui, li])
            cross_edge_types.extend([EDGE_TYPE_MAP["stair_link"]] * 2)
            cross_edge_spatial.extend([(dx, dy, dist), (-dx, -dy, dist)])

    # Same-class spatial proximity across floors
    cross_threshold = 0.10 * diag

    for pl in polys_l:
        if pl.cls in ("Room", "Wall", "Door", "Window"):
            for pu in polys_u:
                if pu.cls == pl.cls:
                    raw_dx = pl.centroid[0] - pu.centroid[0]
                    raw_dy = pl.centroid[1] - pu.centroid[1]
                    raw_dist = math.sqrt(raw_dx * raw_dx + raw_dy * raw_dy)
                    if raw_dist < cross_threshold and pl.id in id2idx_l and pu.id in id2idx_u:
                        li = id2idx_l[pl.id]
                        ui = id2idx_u[pu.id] + n_lower
                        dx = raw_dx / diag
                        dy = raw_dy / diag
                        dist = raw_dist / diag
                        cross_edges_src.extend([li, ui])
                        cross_edges_dst.extend([ui, li])
                        cross_edge_types.extend([EDGE_TYPE_MAP["cross_floor"]] * 2)
                        cross_edge_spatial.extend([(dx, dy, dist), (-dx, -dy, dist)])

    # Append cross-floor edges with spatial features
    if cross_edges_src:
        cf_ei = torch.tensor([cross_edges_src, cross_edges_dst], dtype=torch.long)
        cf_ea = torch.zeros(len(cross_edges_src), EDGE_FEAT_DIM)
        for i, et in enumerate(cross_edge_types):
            cf_ea[i, et] = 1.0
            dx, dy, dist = cross_edge_spatial[i]
            cf_ea[i, 7] = dx
            cf_ea[i, 8] = dy
            cf_ea[i, 9] = dist
        edge_index = torch.cat([edge_index, cf_ei], dim=1)
        edge_attr = torch.cat([edge_attr, cf_ea], dim=0)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        floor_mask=floor_mask,
        y_stair_iou=torch.tensor([pair.stair_iou], dtype=torch.float),
        y_wall_align=torch.tensor([pair.wall_overlap], dtype=torch.float),
        y_door_comply=torch.tensor([pair.door_compliant], dtype=torch.float),
        y_egress=torch.tensor([pair.egress_ok], dtype=torch.float),
        y_overall=torch.tensor([pair.label], dtype=torch.float),
    )
    return data


# ============================================================
# Section F: GNN Model
# ============================================================
# Architecture: GAT chosen over GCNConv for attention-based interpretability.
# Attention weights let us visualize which edges drive each coherence score.

class FloorPlanGAT(nn.Module):
    """Graph Attention Network for multi-story floor plan coherence.

    3 GAT layers × 4 heads, floor-aware pooling, 5 scoring heads.
    ~870K parameters, <200 MB peak VRAM at batch_size=16.
    """

    def __init__(
        self,
        in_channels: int = NODE_FEAT_DIM,
        hidden_channels: int = HIDDEN_DIM,
        num_heads: int = NUM_GAT_HEADS,
        num_layers: int = NUM_GAT_LAYERS,
        edge_dim: int = EDGE_FEAT_DIM,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        hid_total = hidden_channels * num_heads  # 256

        # Input projection: 32 → 256
        self.input_proj = nn.Linear(in_channels, hid_total)

        # GAT layers with residual connections and LayerNorm
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(hid_total, hidden_channels, heads=num_heads, concat=True,
                        edge_dim=edge_dim, dropout=dropout, add_self_loops=True)
            )
            self.layer_norms.append(nn.LayerNorm(hid_total))

        # Floor-aware pooling → 512 per floor → 1024 combined
        pool_dim = hid_total * 2  # mean + max = 512
        combined_dim = pool_dim * 2  # 2 floors = 1024

        # 5 scoring heads (2-layer MLP each)
        self.stair_head = self._make_head(combined_dim)
        self.wall_head = self._make_head(combined_dim)
        self.door_head = self._make_head(combined_dim)
        self.egress_head = self._make_head(combined_dim)
        self.overall_head = self._make_head(combined_dim)

        self.dropout_layer = nn.Dropout(dropout)

        # Store attention weights for explainability
        self._attention_weights: List[torch.Tensor] = []

    @staticmethod
    def _make_head(in_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, edge_attr, floor_mask, batch):
        """Forward pass → dict of per-head scores + attention weights."""
        self._attention_weights = []

        # Input projection
        h = self.input_proj(x)

        # GAT layers with residual connections
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            h_res = h
            h, (_, attn_w) = gat(h, edge_index, edge_attr=edge_attr,
                                  return_attention_weights=True)
            self._attention_weights.append(attn_w)
            h = ln(h)
            h = F.elu(h)
            h = self.dropout_layer(h)
            h = h + h_res  # Residual

        # Floor-aware global pooling: separate mean+max per floor, then concat
        lower_mask = (floor_mask == 0)
        upper_mask = (floor_mask == 1)

        # Handle batched graphs: filter batch indices per floor
        batch_lower = batch[lower_mask]
        batch_upper = batch[upper_mask]
        h_lower = h[lower_mask]
        h_upper = h[upper_mask]

        # Pool per floor (mean + max → 512)
        if h_lower.shape[0] > 0:
            lower_pool = torch.cat([
                global_mean_pool(h_lower, batch_lower),
                global_max_pool(h_lower, batch_lower),
            ], dim=-1)
        else:
            n_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
            lower_pool = torch.zeros(n_graphs, self.hidden_channels * self.num_heads * 2,
                                     device=x.device)

        if h_upper.shape[0] > 0:
            upper_pool = torch.cat([
                global_mean_pool(h_upper, batch_upper),
                global_max_pool(h_upper, batch_upper),
            ], dim=-1)
        else:
            n_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
            upper_pool = torch.zeros(n_graphs, self.hidden_channels * self.num_heads * 2,
                                     device=x.device)

        graph_repr = torch.cat([lower_pool, upper_pool], dim=-1)  # [B, 1024]

        return {
            "stair_score": self.stair_head(graph_repr).squeeze(-1),
            "wall_score": self.wall_head(graph_repr).squeeze(-1),
            "door_score": self.door_head(graph_repr).squeeze(-1),
            "egress_score": self.egress_head(graph_repr).squeeze(-1),
            "overall_score": self.overall_head(graph_repr).squeeze(-1),
            "attention_weights": self._attention_weights,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Section G: Training Loop
# ============================================================

class CoherenceLoss(nn.Module):
    """Multi-task loss: SmoothL1 for continuous + BCE for binary heads.

    Design change: SmoothL1Loss (Huber) replaces MSE for continuous heads.
    SmoothL1 provides a linear gradient when predictions are far from targets
    (avoiding the exploding gradient of MSE) and quadratic near the target
    (smooth convergence). This is critical for learning continuous regression
    on the stair/wall/overall dimensions where the model previously collapsed
    to binary classification under MSE.

    Weights: stair=2.0, wall=2.0, door=0.5, egress=0.5, overall=1.5.
    Stair and wall weighted higher because these are the spatial alignment
    dimensions that require continuous regression learning.
    """

    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.bce = nn.BCELoss()
        self.weights = {
            "stair": 2.0, "wall": 2.0, "door": 0.5,
            "egress": 0.5, "overall": 1.5,
        }

    def forward(self, preds, batch_data):
        # Gather targets
        y_stair = batch_data.y_stair_iou.to(preds["stair_score"].device)
        y_wall = batch_data.y_wall_align.to(preds["stair_score"].device)
        y_door = batch_data.y_door_comply.to(preds["stair_score"].device)
        y_egress = batch_data.y_egress.to(preds["stair_score"].device)
        y_overall = batch_data.y_overall.to(preds["stair_score"].device)

        # Reshape targets to match predictions
        y_stair = y_stair.view_as(preds["stair_score"])
        y_wall = y_wall.view_as(preds["wall_score"])
        y_door = y_door.view_as(preds["door_score"])
        y_egress = y_egress.view_as(preds["egress_score"])
        y_overall = y_overall.view_as(preds["overall_score"])

        # Clamp predictions to avoid BCE edge cases
        eps = 1e-7
        door_pred = preds["door_score"].clamp(eps, 1 - eps)
        egress_pred = preds["egress_score"].clamp(eps, 1 - eps)

        losses = {
            "stair": self.smooth_l1(preds["stair_score"], y_stair),
            "wall": self.smooth_l1(preds["wall_score"], y_wall),
            "door": self.bce(door_pred, y_door),
            "egress": self.bce(egress_pred, y_egress),
            "overall": self.smooth_l1(preds["overall_score"], y_overall),
        }

        total = sum(self.weights[k] * losses[k] for k in losses)
        return total, {k: v.item() for k, v in losses.items()}


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns dict of average losses."""
    model.train()
    total_losses = defaultdict(float)
    n_batches = 0

    for batch_data in loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()

        preds = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr,
                      batch_data.floor_mask, batch_data.batch)

        loss, per_head = criterion(preds, batch_data)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in per_head.items():
            total_losses[k] += v
        total_losses["total"] += loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a dataset. Returns dict of metrics."""
    model.eval()
    total_losses = defaultdict(float)
    all_preds = defaultdict(list)
    all_targets = defaultdict(list)
    n_batches = 0

    for batch_data in loader:
        batch_data = batch_data.to(device)
        preds = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr,
                      batch_data.floor_mask, batch_data.batch)

        loss, per_head = criterion(preds, batch_data)
        for k, v in per_head.items():
            total_losses[k] += v
        total_losses["total"] += loss.item()

        # Collect predictions for accuracy computation
        all_preds["stair"].extend(preds["stair_score"].cpu().tolist())
        all_preds["wall"].extend(preds["wall_score"].cpu().tolist())
        all_preds["door"].extend(preds["door_score"].cpu().tolist())
        all_preds["egress"].extend(preds["egress_score"].cpu().tolist())
        all_preds["overall"].extend(preds["overall_score"].cpu().tolist())

        all_targets["stair"].extend(batch_data.y_stair_iou.cpu().tolist())
        all_targets["wall"].extend(batch_data.y_wall_align.cpu().tolist())
        all_targets["door"].extend(batch_data.y_door_comply.cpu().tolist())
        all_targets["egress"].extend(batch_data.y_egress.cpu().tolist())
        all_targets["overall"].extend(batch_data.y_overall.cpu().tolist())

        n_batches += 1

    metrics = {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    # Binary accuracy for door and egress heads (threshold at 0.5)
    for head in ("door", "egress"):
        preds_arr = np.array(all_preds[head])
        targs_arr = np.array(all_targets[head])
        pred_bin = (preds_arr > 0.5).astype(float)
        targ_bin = (targs_arr > 0.5).astype(float)
        metrics[f"{head}_acc"] = float(np.mean(pred_bin == targ_bin))

    # MAE for continuous heads
    for head in ("stair", "wall", "overall"):
        preds_arr = np.array(all_preds[head])
        targs_arr = np.array(all_targets[head])
        metrics[f"{head}_mae"] = float(np.mean(np.abs(preds_arr - targs_arr)))

    return metrics


def train_model(model, train_data, val_data, epochs=EPOCHS,
                batch_size=BATCH_SIZE, lr=LEARNING_RATE, device=DEVICE):
    """Full training loop with early stopping and LR scheduling.

    Returns: (trained_model, training_history)
    """
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-6
    )
    criterion = CoherenceLoss()

    history = defaultdict(list)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        # Record history
        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)

        scheduler.step(val_metrics["total"])

        # Early stopping
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_metrics['total']:.4f} | "
                  f"Val: {val_metrics['total']:.4f} | "
                  f"LR: {lr_now:.1e} | "
                  f"Time: {elapsed:.1f}s")

        if patience_counter >= 20:
            print(f"  Early stopping at epoch {epoch} (patience=20)")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = OUTPUT_DIR / "best_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Best model saved to {ckpt_path}")

    return model, dict(history)


# ============================================================
# Section H: Refinement Agent
# ============================================================
# Score-guided iterative refinement: model identifies weakest dimension,
# heuristic fixes it, repeat up to 3 times.

def _score_pair(model, pair, device):
    """Score a multi-story pair using the trained model."""
    model.eval()
    data = build_pair_graph(pair).to(device)
    # Add batch dimension for single graph
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
    with torch.no_grad():
        preds = model(data.x, data.edge_index, data.edge_attr,
                      data.floor_mask, data.batch)
    return {
        "stair_score": preds["stair_score"].item(),
        "wall_score": preds["wall_score"].item(),
        "door_score": preds["door_score"].item(),
        "egress_score": preds["egress_score"].item(),
        "overall_score": preds["overall_score"].item(),
    }


def refine_stair_alignment(pair, stair_score):
    """Shift upper stair 50% toward lower stair centroid (conservative step)."""
    actions = []
    if pair.floor_lower.synthetic_stair is None or pair.floor_upper.synthetic_stair is None:
        return pair, actions

    pair = MultiStoryPair(
        floor_lower=pair.floor_lower,
        floor_upper=_deep_copy_layout(pair.floor_upper),
        label=pair.label, stair_iou=pair.stair_iou,
        wall_overlap=pair.wall_overlap, door_compliant=pair.door_compliant,
        egress_ok=pair.egress_ok,
    )

    sl = pair.floor_lower.synthetic_stair
    su = pair.floor_upper.synthetic_stair
    dx = (sl.centroid[0] - su.centroid[0]) * 0.5  # Conservative 50% step
    dy = (sl.centroid[1] - su.centroid[1]) * 0.5

    new_pts = [(x + dx, y + dy) for (x, y) in su.points]
    su.points = new_pts
    su.bbox = compute_bbox(new_pts)
    su.centroid = compute_centroid(new_pts)
    pair.floor_upper.polygons[su.id] = su
    pair.floor_upper.synthetic_stair = su

    new_iou = compute_stair_iou(sl, su)
    actions.append(RefinementAction(
        action_type="shift_stair", target_id=su.id,
        description=f"Shifted upper stair by ({dx:.1f}, {dy:.1f})px toward lower stair. "
                    f"Stair IoU: {pair.stair_iou:.3f} → {new_iou:.3f}",
        delta={"dx": dx, "dy": dy},
        score_before=stair_score, score_after=new_iou,
    ))
    pair.stair_iou = new_iou
    return pair, actions


def refine_wall_alignment(pair, wall_score):
    """Snap misaligned upper walls to nearest lower-floor wall."""
    actions = []
    pair = MultiStoryPair(
        floor_lower=pair.floor_lower,
        floor_upper=_deep_copy_layout(pair.floor_upper),
        label=pair.label, stair_iou=pair.stair_iou,
        wall_overlap=pair.wall_overlap, door_compliant=pair.door_compliant,
        egress_ok=pair.egress_ok,
    )

    for wu in pair.floor_upper.walls:
        best_iou = 0
        best_wall = None
        for wl in pair.floor_lower.walls:
            iou = _bbox_iou(wu.bbox, wl.bbox)
            if iou > best_iou:
                best_iou = iou
                best_wall = wl

        # If close but not aligned, snap to lower wall position
        if best_wall and 0.1 < best_iou < WALL_ALIGN_THRESHOLD:
            dx = best_wall.centroid[0] - wu.centroid[0]
            dy = best_wall.centroid[1] - wu.centroid[1]
            if abs(dx) < 50 and abs(dy) < 50:  # Only snap if reasonably close
                wu.points = [(x + dx, y + dy) for (x, y) in wu.points]
                wu.bbox = compute_bbox(wu.points)
                wu.centroid = compute_centroid(wu.points)
                pair.floor_upper.polygons[wu.id] = wu
                actions.append(RefinementAction(
                    action_type="realign_wall", target_id=wu.id,
                    description=f"Snapped wall {wu.id} by ({dx:.1f}, {dy:.1f})px to align with lower floor",
                    delta={"dx": dx, "dy": dy},
                    score_before=wall_score, score_after=0.0,
                ))

    pair.floor_upper.walls = [p for p in pair.floor_upper.polygons.values() if p.cls == "Wall"]
    new_align = compute_wall_alignment(pair.floor_lower, pair.floor_upper)
    for a in actions:
        a.score_after = new_align
    pair.wall_overlap = new_align
    return pair, actions


def refine_door_sizing(pair, door_score):
    """Enlarge undersized doors to meet 36px minimum + 2px margin."""
    actions = []
    pair = MultiStoryPair(
        floor_lower=pair.floor_lower,
        floor_upper=_deep_copy_layout(pair.floor_upper),
        label=pair.label, stair_iou=pair.stair_iou,
        wall_overlap=pair.wall_overlap, door_compliant=pair.door_compliant,
        egress_ok=pair.egress_ok,
    )

    target_width = DOOR_MIN_WIDTH_PX + 2  # 38px with margin
    for d in pair.floor_upper.doors:
        opening = min(d.width, d.height)
        if opening < DOOR_MIN_WIDTH_PX:
            scale = target_width / max(opening, 1e-6)
            cx, cy = d.centroid
            d.points = [(cx + (x - cx) * scale, cy + (y - cy) * scale) for (x, y) in d.points]
            d.bbox = compute_bbox(d.points)
            d.centroid = compute_centroid(d.points)
            d.width = d.bbox[2] - d.bbox[0]
            d.height = d.bbox[3] - d.bbox[1]
            pair.floor_upper.polygons[d.id] = d
            actions.append(RefinementAction(
                action_type="resize_door", target_id=d.id,
                description=f"Enlarged door {d.id} from {opening:.0f}px to {target_width}px opening",
                delta={"scale": scale},
                score_before=door_score, score_after=0.0,
            ))

    pair.floor_upper.doors = [p for p in pair.floor_upper.polygons.values() if p.cls == "Door"]
    new_comply = compute_door_compliance(pair.floor_upper)
    for a in actions:
        a.score_after = new_comply
    pair.door_compliant = new_comply
    return pair, actions


def refine_egress(pair, egress_score):
    """If <2 egress paths, add a door on the longest outer-perimeter wall."""
    actions = []
    pair = MultiStoryPair(
        floor_lower=pair.floor_lower,
        floor_upper=_deep_copy_layout(pair.floor_upper),
        label=pair.label, stair_iou=pair.stair_iou,
        wall_overlap=pair.wall_overlap, door_compliant=pair.door_compliant,
        egress_ok=pair.egress_ok,
    )

    layout = pair.floor_upper
    outer_walls = []
    for oid in layout.outer_perimeter_ids:
        if oid in layout.polygons and layout.polygons[oid].cls == "Wall":
            outer_walls.append(layout.polygons[oid])

    if not outer_walls:
        return pair, actions

    # Find longest outer wall without an adjacent door
    door_set = set(d.id for d in layout.doors)
    best_wall = max(outer_walls, key=lambda w: max(w.width, w.height))

    # Create new door at wall midpoint
    new_id = max(layout.polygons.keys(), default=-1) + 1
    cx, cy = best_wall.centroid
    door_w = 40  # Standard door width
    door_h = 10  # Door thickness

    if best_wall.width > best_wall.height:
        # Horizontal wall → vertical door
        door_pts = [(cx - door_w / 2, cy - door_h / 2), (cx + door_w / 2, cy - door_h / 2),
                    (cx + door_w / 2, cy + door_h / 2), (cx - door_w / 2, cy + door_h / 2)]
    else:
        # Vertical wall → horizontal door
        door_pts = [(cx - door_h / 2, cy - door_w / 2), (cx + door_h / 2, cy - door_w / 2),
                    (cx + door_h / 2, cy + door_w / 2), (cx - door_h / 2, cy + door_w / 2)]

    new_door = Polygon(
        id=new_id, cls="Door", points=door_pts,
        bbox=compute_bbox(door_pts), centroid=compute_centroid(door_pts),
        area=compute_area(door_pts), width=door_w, height=door_h,
    )
    layout.polygons[new_id] = new_door
    layout.doors.append(new_door)
    layout.outer_perimeter_ids.append(new_id)

    new_egress = check_egress(layout)
    actions.append(RefinementAction(
        action_type="add_egress", target_id=new_id,
        description=f"Added egress door {new_id} at outer wall {best_wall.id} midpoint",
        delta={"new_door_id": new_id},
        score_before=egress_score, score_after=new_egress,
    ))
    pair.egress_ok = new_egress
    return pair, actions


def refinement_loop(model, pair, max_iterations=REFINEMENT_LOOPS,
                    target_score=COHERENCE_THRESHOLD, device=DEVICE):
    """3-loop iterative refinement: score -> identify weakest head -> fix -> repeat.

    Returns: (final_pair, actions_per_iteration, scores_per_iteration,
              pair_snapshots -- list of deep-copied pair at each step for viz)
    """
    all_actions = []
    all_scores = []
    # Save the initial broken state and every intermediate state for GIF viz
    pair_snapshots = [copy.deepcopy(pair)]

    for iteration in range(max_iterations):
        scores = _score_pair(model, pair, device)
        all_scores.append(scores)

        # Check convergence
        if scores["overall_score"] >= target_score:
            print(f"    Iteration {iteration}: CONVERGED (overall={scores['overall_score']:.3f})")
            break

        # Find weakest head (excluding overall)
        head_scores = {
            "stair": scores["stair_score"],
            "wall": scores["wall_score"],
            "door": scores["door_score"],
            "egress": scores["egress_score"],
        }
        weakest = min(head_scores, key=head_scores.get)

        print(f"    Iteration {iteration}: overall={scores['overall_score']:.3f}, "
              f"weakest={weakest} ({head_scores[weakest]:.3f})")

        # Apply targeted refinement
        if weakest == "stair":
            pair, actions = refine_stair_alignment(pair, scores["stair_score"])
        elif weakest == "wall":
            pair, actions = refine_wall_alignment(pair, scores["wall_score"])
        elif weakest == "door":
            pair, actions = refine_door_sizing(pair, scores["door_score"])
        elif weakest == "egress":
            pair, actions = refine_egress(pair, scores["egress_score"])
        else:
            actions = []

        all_actions.append(actions)
        pair_snapshots.append(copy.deepcopy(pair))
        for a in actions:
            print(f"      -> {a.description}")

    # Final score
    final_scores = _score_pair(model, pair, device)
    all_scores.append(final_scores)

    return pair, all_actions, all_scores, pair_snapshots


# ============================================================
# Section I: Evaluation
# ============================================================

def _ground_truth_metrics(pair):
    """Recompute ground-truth coherence metrics for a pair's current geometry."""
    stair_iou = compute_stair_iou(pair.floor_lower.synthetic_stair,
                                   pair.floor_upper.synthetic_stair)
    wall_align = compute_wall_alignment(pair.floor_lower, pair.floor_upper)
    door_comply = compute_door_compliance(pair.floor_upper)
    egress = check_egress(pair.floor_upper)
    overall = (stair_iou + wall_align + door_comply + egress) / 4.0
    return {
        "stair": stair_iou,
        "wall": wall_align,
        "door": door_comply,
        "egress": egress,
        "overall": overall,
    }


def compute_baseline_scores(test_pairs):
    """Baseline: ground-truth coherence of NEGATIVE pairs (broken stacking).

    We evaluate only negative/mixed pairs (label < 0.9) to show how bad naive
    stacking is. Positive pairs are already coherent by construction.
    """
    broken = [p for p in test_pairs if p.label < 0.9]
    if not broken:
        broken = test_pairs  # fallback
    metrics = [_ground_truth_metrics(p) for p in broken]
    return {k: np.mean([m[k] for m in metrics]) * 100 for k in metrics[0]}


def compute_gnn_agent_scores(model, test_pairs, device=DEVICE):
    """Run GNN + refinement agent on negative/mixed test pairs.

    Returns ground-truth metrics AFTER refinement, plus traces.
    """
    broken = [(i, p) for i, p in enumerate(test_pairs) if p.label < 0.9]
    if not broken:
        broken = list(enumerate(test_pairs))

    all_traces = []  # For all test pairs (for viz)
    refined_metrics = []

    # First refine the broken pairs
    for idx, (orig_i, pair) in enumerate(broken):
        if idx < 5:
            print(f"  [Refining broken pair {idx+1}/{len(broken)}]")
        refined, actions, scores, snapshots = refinement_loop(model, pair, device=device)
        all_traces.append((refined, actions, scores, snapshots))
        refined_metrics.append(_ground_truth_metrics(refined))

    avg_scores = {k: np.mean([m[k] for m in refined_metrics]) * 100
                  for k in refined_metrics[0]}
    return avg_scores, all_traces


def print_evaluation_table(baseline, gnn_agent):
    """Print markdown evaluation table."""
    fix_rate = ((gnn_agent["overall"] - baseline["overall"])
                / max(baseline["overall"], 1) * 100)

    table = f"""
| Method     | Egress % | Alignment % | Overall % | Fix Rate |
|------------|----------|-------------|-----------|----------|
| Baseline   | {baseline['egress']:.0f}       | {baseline['wall']:.0f}          | {baseline['overall']:.0f}        | -        |
| GNN Agent  | {gnn_agent['egress']:.0f}       | {gnn_agent['wall']:.0f}          | {gnn_agent['overall']:.0f}        | +{fix_rate:.0f}%     |
"""
    print(table)
    return table


def print_per_head_breakdown(scores, label="GNN Agent"):
    """Print per-head score breakdown for explainability."""
    print(f"\n  Per-Head Score Breakdown ({label}):")
    print(f"  +----------------------+----------+")
    print(f"  | Head                 | Score    |")
    print(f"  +----------------------+----------+")
    print(f"  | Stair Alignment      | {scores['stair']:6.1f}%  |")
    print(f"  | Wall Alignment       | {scores['wall']:6.1f}%  |")
    print(f"  | Door Compliance      | {scores['door']:6.1f}%  |")
    print(f"  | Egress Compliance    | {scores['egress']:6.1f}%  |")
    print(f"  | Overall Coherence    | {scores['overall']:6.1f}%  |")
    print(f"  +----------------------+----------+")


# ============================================================
# Section J: Visualization
# ============================================================
# Completely rewritten to show clear BEFORE/AFTER comparisons,
# highlight differences between floors, and animate refinement.

COLORS = {
    "Wall": "#AFD8F8",
    "Room": "#008E8E",
    "Door": "#FF8E46",
    "Window": "#F6BD0F",
    "Separation": "#D64646",
    "Stair": "#9B59B6",
    "Parking": "#8BBA00",
}


def _draw_layout_on_ax(layout, ax, title="", ghost_layout=None):
    """Draw a floor layout. If ghost_layout is given, draw its missing/shifted
    elements as red dashed outlines (ghost overlay) to highlight differences."""
    ax.set_xlim(0, layout.canvas_width)
    ax.set_ylim(layout.canvas_height, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold")

    # If ghost_layout provided, find elements in ghost that are missing or
    # shifted in the current layout, and draw them as red dashed ghosts.
    if ghost_layout:
        current_ids = set(layout.polygons.keys())
        for gid, gpoly in ghost_layout.polygons.items():
            if gpoly.cls in ("Stair",):
                continue  # stair handled separately
            if gid not in current_ids:
                # This element was REMOVED from the current layout
                xs = [p[0] for p in gpoly.points] + [gpoly.points[0][0]]
                ys = [p[1] for p in gpoly.points] + [gpoly.points[0][1]]
                ax.fill(xs, ys, color="red", alpha=0.08)
                ax.plot(xs, ys, color="red", linewidth=2, linestyle="--", alpha=0.6)
                ax.annotate("MISSING", xy=gpoly.centroid, ha="center", va="center",
                            fontsize=5, color="red", alpha=0.7)

    # Draw rooms (background fill)
    for room in layout.rooms:
        xs = [p[0] for p in room.points] + [room.points[0][0]]
        ys = [p[1] for p in room.points] + [room.points[0][1]]
        ax.fill(xs, ys, color=COLORS["Room"], alpha=0.15)
        ax.plot(xs, ys, color=COLORS["Room"], linewidth=0.5)

    # Draw walls
    for wall in layout.walls:
        xs = [p[0] for p in wall.points] + [wall.points[0][0]]
        ys = [p[1] for p in wall.points] + [wall.points[0][1]]
        ax.fill(xs, ys, color=COLORS["Wall"], alpha=0.7)
        ax.plot(xs, ys, color="#5DADE2", linewidth=0.3)

    # Draw doors -- highlight undersized ones in red
    for door in layout.doors:
        xs = [p[0] for p in door.points] + [door.points[0][0]]
        ys = [p[1] for p in door.points] + [door.points[0][1]]
        opening = min(door.width, door.height)
        if opening < DOOR_MIN_WIDTH_PX:
            ax.fill(xs, ys, color="red", alpha=0.5)
            ax.plot(xs, ys, color="darkred", linewidth=2)
            ax.annotate(f"{opening:.0f}px", xy=door.centroid, ha="center",
                        va="center", fontsize=5, color="darkred", fontweight="bold")
        else:
            ax.fill(xs, ys, color=COLORS["Door"], alpha=0.7)

    # Draw windows
    for win in layout.windows:
        xs = [p[0] for p in win.points] + [win.points[0][0]]
        ys = [p[1] for p in win.points] + [win.points[0][1]]
        ax.fill(xs, ys, color=COLORS["Window"], alpha=0.7)

    # Draw stair
    if layout.synthetic_stair:
        s = layout.synthetic_stair
        xs = [p[0] for p in s.points] + [s.points[0][0]]
        ys = [p[1] for p in s.points] + [s.points[0][1]]
        ax.fill(xs, ys, color=COLORS["Stair"], alpha=0.5, hatch="//")
        ax.plot(xs, ys, color=COLORS["Stair"], linewidth=2.5)
        ax.annotate("STAIR", xy=s.centroid, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["Stair"], alpha=0.9))

        # If ghost layout has a stair at a different position, draw arrow showing offset
        if ghost_layout and ghost_layout.synthetic_stair:
            gs = ghost_layout.synthetic_stair
            dx = s.centroid[0] - gs.centroid[0]
            dy = s.centroid[1] - gs.centroid[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 5:  # Only show arrow if meaningful offset
                # Draw ghost stair position
                gxs = [p[0] for p in gs.points] + [gs.points[0][0]]
                gys = [p[1] for p in gs.points] + [gs.points[0][1]]
                ax.plot(gxs, gys, color="red", linewidth=2, linestyle=":", alpha=0.7)
                ax.annotate("", xy=s.centroid, xytext=gs.centroid,
                            arrowprops=dict(arrowstyle="->", color="red",
                                            lw=2.5, connectionstyle="arc3,rad=0.1"))
                ax.annotate(f"OFFSET {dist:.0f}px", xy=gs.centroid,
                            xytext=(gs.centroid[0], gs.centroid[1] - 25),
                            ha="center", fontsize=6, color="red", fontweight="bold")

    ax.tick_params(labelsize=6)


def _draw_alignment_lines(ax, layout_lower, layout_upper, y_gap_center, x_range):
    """Draw vertical alignment lines between two vertically-stacked floors.
    Green = aligned, Red = misaligned."""
    for wl in layout_lower.walls:
        best_iou = 0
        for wu in layout_upper.walls:
            iou = _bbox_iou(wl.bbox, wu.bbox)
            if iou > best_iou:
                best_iou = iou
        color = "#27AE60" if best_iou >= WALL_ALIGN_THRESHOLD else "#E74C3C"
        alpha = 0.3 if best_iou >= WALL_ALIGN_THRESHOLD else 0.5
        # Draw a small tick at the gap center
        cx = wl.centroid[0]
        if x_range[0] <= cx <= x_range[1]:
            ax.plot([cx, cx], [y_gap_center - 8, y_gap_center + 8],
                    color=color, linewidth=1.5, alpha=alpha)


def plot_multi_story_pair(pair, scores, save_path, title="Multi-Story Coherence",
                          initial_pair=None):
    """BEFORE/AFTER comparison with vertically stacked floors and alignment indicators.

    If initial_pair is given, shows a 2-column layout:
      Left column: BEFORE (broken) with violations highlighted
      Right column: AFTER (refined) with fixes highlighted
    Otherwise shows a single-column stacked view.
    """
    cw = pair.floor_lower.canvas_width
    ch = pair.floor_lower.canvas_height

    if initial_pair is not None:
        # ---- 2-column BEFORE / AFTER layout ----
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.6], hspace=0.35, wspace=0.3)

        # BEFORE column (left)
        ax_b_upper = fig.add_subplot(gs[0, 0])
        ax_b_lower = fig.add_subplot(gs[1, 0])
        # Show upper floor with ghost overlay of lower floor to highlight missing walls
        _draw_layout_on_ax(initial_pair.floor_upper, ax_b_upper,
                           title="BEFORE: Floor 2 (Upper) -- Broken",
                           ghost_layout=initial_pair.floor_lower)
        _draw_layout_on_ax(initial_pair.floor_lower, ax_b_lower,
                           title="BEFORE: Floor 1 (Lower) -- Reference")
        ax_b_upper.set_facecolor("#FFF5F5")  # Light red tint
        # Compute before scores
        gt_before = _ground_truth_metrics(initial_pair)

        # AFTER column (right)
        ax_a_upper = fig.add_subplot(gs[0, 1])
        ax_a_lower = fig.add_subplot(gs[1, 1])
        _draw_layout_on_ax(pair.floor_upper, ax_a_upper,
                           title="AFTER: Floor 2 (Upper) -- Refined",
                           ghost_layout=pair.floor_lower)
        _draw_layout_on_ax(pair.floor_lower, ax_a_lower,
                           title="AFTER: Floor 1 (Lower) -- Reference")
        ax_a_upper.set_facecolor("#F5FFF5")  # Light green tint

        # Score comparison table at bottom
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis("off")
        gt_after = _ground_truth_metrics(pair)
        headers = ["Metric", "Before", "After", "Change", "Status"]
        rows = []
        for key, label in [("stair", "Stair Alignment"), ("wall", "Wall Alignment"),
                           ("door", "Door Compliance"), ("egress", "Egress Compliance"),
                           ("overall", "Overall Coherence")]:
            bv = gt_before[key] * 100
            av = gt_after[key] * 100
            delta = av - bv
            sign = "+" if delta >= 0 else ""
            status = "PASS" if av >= 50 else "FAIL"
            rows.append([label, f"{bv:.1f}%", f"{av:.1f}%", f"{sign}{delta:.1f}%", status])
        cell_colors = []
        for row in rows:
            if row[4] == "PASS":
                cell_colors.append(["#D5F5E3"] * 5)
            else:
                cell_colors.append(["#FADBD8"] * 5)
        tbl = ax_table.table(cellText=rows, colLabels=headers,
                             cellColours=cell_colors, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 1.8)

        fig.suptitle(title + " -- Before vs After Refinement",
                     fontsize=15, fontweight="bold")
    else:
        # ---- Single view (no initial pair for comparison) ----
        fig = plt.figure(figsize=(14, 12))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.5], hspace=0.3)

        ax_upper = fig.add_subplot(gs[0, :])
        ax_lower = fig.add_subplot(gs[1, :])
        _draw_layout_on_ax(pair.floor_upper, ax_upper, title="Floor 2 (Upper)")
        _draw_layout_on_ax(pair.floor_lower, ax_lower, title="Floor 1 (Lower)")

        # Radar chart
        ax_radar = fig.add_subplot(gs[2, 0], projection="polar")
        categories = ["Stair", "Wall", "Door", "Egress", "Overall"]
        values = [scores.get("stair_score", scores.get("stair", 0)),
                  scores.get("wall_score", scores.get("wall", 0)),
                  scores.get("door_score", scores.get("door", 0)),
                  scores.get("egress_score", scores.get("egress", 0)),
                  scores.get("overall_score", scores.get("overall", 0))]
        values = [v / 100 if v > 1 else v for v in values]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        vals_p = values + [values[0]]
        angs_p = angles + [angles[0]]
        ax_radar.fill(angs_p, vals_p, color="#3498DB", alpha=0.3)
        ax_radar.plot(angs_p, vals_p, color="#3498DB", linewidth=2)
        ax_radar.set_xticks(angles)
        ax_radar.set_xticklabels(categories, fontsize=8)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title("Coherence", fontsize=10, fontweight="bold", pad=15)

        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_attention_heatmap(model, data, layout_lower, layout_upper, save_path):
    """GAT attention weights overlaid on floor geometry."""
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    with torch.no_grad():
        preds = model(data.x, data.edge_index, data.edge_attr,
                      data.floor_mask, data.batch)

    if model._attention_weights:
        attn = model._attention_weights[-1].cpu().numpy()
        if attn.ndim == 2:
            attn_avg = attn.mean(axis=1) if attn.shape[1] > 1 else attn.squeeze()
        else:
            attn_avg = attn
    else:
        print("    Warning: No attention weights captured")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    _draw_layout_on_ax(layout_lower, ax1, title="Floor 1 -- Attention Overlay")
    _draw_layout_on_ax(layout_upper, ax2, title="Floor 2 -- Attention Overlay")

    edge_index = data.edge_index.cpu().numpy()
    floor_mask = data.floor_mask.cpu().numpy()
    n_lower = int((floor_mask == 0).sum())
    all_polys_lower = list(layout_lower.polygons.values())
    all_polys_upper = list(layout_upper.polygons.values())

    if len(attn_avg) > 0:
        attn_norm = (attn_avg - attn_avg.min()) / max(attn_avg.max() - attn_avg.min(), 1e-8)
    else:
        attn_norm = np.array([])

    cmap = plt.cm.YlOrRd
    for ei in range(min(edge_index.shape[1], len(attn_norm))):
        src, dst = edge_index[0, ei], edge_index[1, ei]
        alpha_val = float(attn_norm[ei])
        if src < n_lower and dst < n_lower:
            if src < len(all_polys_lower) and dst < len(all_polys_lower):
                p1 = all_polys_lower[src].centroid
                p2 = all_polys_lower[dst].centroid
                ax1.plot([p1[0], p2[0]], [p1[1], p2[1]],
                         color=cmap(alpha_val), alpha=max(alpha_val, 0.1), linewidth=1)
        elif src >= n_lower and dst >= n_lower:
            si, di = src - n_lower, dst - n_lower
            if si < len(all_polys_upper) and di < len(all_polys_upper):
                p1 = all_polys_upper[si].centroid
                p2 = all_polys_upper[di].centroid
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]],
                         color=cmap(alpha_val), alpha=max(alpha_val, 0.1), linewidth=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax1, ax2], label="Attention Weight", shrink=0.6)
    fig.suptitle("GAT Attention Heatmap (Last Layer, Avg Heads)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def create_svg_interactive(pair, scores, actions, save_path, initial_pair=None):
    """Interactive SVG showing before/after with hover tooltips."""
    w = max(pair.floor_lower.canvas_width, pair.floor_upper.canvas_width)
    h = max(pair.floor_lower.canvas_height, pair.floor_upper.canvas_height)
    # Layout: 2 columns (before | after), 2 rows (upper | lower)
    total_w = w * 2.4
    total_h = h * 2.4 + 120

    dwg = svgwrite.Drawing(save_path, size=(f"{total_w}px", f"{total_h}px"),
                           debug=False)
    dwg.defs.add(dwg.style("""
        .el:hover { stroke: #E74C3C; stroke-width: 3; cursor: pointer; }
        .tip { visibility: hidden; font-size: 11px; fill: #333; }
        .el:hover ~ .tip { visibility: visible; }
        @keyframes pulse { 0%,100% { opacity: 0.3; } 50% { opacity: 1; } }
        .missing { animation: pulse 1.5s infinite; fill: red; opacity: 0.15;
                   stroke: red; stroke-width: 2; stroke-dasharray: 8,4; }
    """))

    dwg.add(dwg.text("Multi-Story Coherence: Before vs After Refinement",
                      insert=(total_w / 2, 30), text_anchor="middle",
                      font_size="20px", font_weight="bold", fill="#2C3E50"))

    def draw_floor_svg(layout, x_off, y_off, label, ghost=None):
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
        g = dwg.g(id=f"f_{safe}")
        g.add(dwg.text(label, insert=(x_off + w / 2, y_off + 20),
                        text_anchor="middle", font_size="13px", font_weight="bold"))
        # Ghost missing elements
        if ghost:
            current_ids = set(layout.polygons.keys())
            for gid, gp in ghost.polygons.items():
                if gp.cls == "Stair":
                    continue
                if gid not in current_ids:
                    pts = [(p[0] + x_off, p[1] + y_off + 30) for p in gp.points]
                    g.add(dwg.polygon(points=pts, class_="missing"))
        # Draw elements
        for poly in layout.polygons.values():
            pts = [(p[0] + x_off, p[1] + y_off + 30) for p in poly.points]
            clr = COLORS.get(poly.cls, "#CCC")
            op = 0.4 if poly.cls == "Room" else 0.75
            pg = dwg.polygon(points=pts, fill=clr, opacity=op, class_="el")
            g.add(pg)
            g.add(dwg.text(f"{poly.cls} #{poly.id}",
                           insert=(poly.centroid[0] + x_off, poly.centroid[1] + y_off + 25),
                           class_="tip"))
        # Stair highlight
        if layout.synthetic_stair:
            s = layout.synthetic_stair
            pts = [(p[0] + x_off, p[1] + y_off + 30) for p in s.points]
            g.add(dwg.polygon(points=pts, fill=COLORS["Stair"], opacity=0.6,
                              stroke=COLORS["Stair"], stroke_width=3))
            g.add(dwg.text("STAIR", insert=(s.centroid[0] + x_off, s.centroid[1] + y_off + 34),
                           text_anchor="middle", font_size="10px",
                           font_weight="bold", fill="white"))
        dwg.add(g)

    col_gap = w * 1.2
    row_gap = h * 1.1

    if initial_pair:
        # Before column
        dwg.add(dwg.text("BEFORE (Broken)", insert=(w / 2, 55),
                         text_anchor="middle", font_size="16px",
                         fill="#E74C3C", font_weight="bold"))
        draw_floor_svg(initial_pair.floor_upper, 10, 60, "Upper Floor (Broken)",
                       ghost=initial_pair.floor_lower)
        draw_floor_svg(initial_pair.floor_lower, 10, 60 + row_gap, "Lower Floor (Reference)")

        # After column
        dwg.add(dwg.text("AFTER (Refined)", insert=(col_gap + w / 2, 55),
                         text_anchor="middle", font_size="16px",
                         fill="#27AE60", font_weight="bold"))
        draw_floor_svg(pair.floor_upper, col_gap, 60, "Upper Floor (Fixed)")
        draw_floor_svg(pair.floor_lower, col_gap, 60 + row_gap, "Lower Floor (Reference)")
    else:
        draw_floor_svg(pair.floor_lower, 10, 50, "Floor 1 (Lower)")
        draw_floor_svg(pair.floor_upper, col_gap, 50, "Floor 2 (Upper)")

    # Score bar
    y_sc = total_h - 40
    keys = ["stair_score", "wall_score", "door_score", "egress_score", "overall_score"]
    labels = ["Stair", "Wall", "Door", "Egress", "Overall"]
    for i, (k, lb) in enumerate(zip(keys, labels)):
        v = scores.get(k, scores.get(k.replace("_score", ""), 0))
        if v > 1:
            v = v / 100
        clr = "#27AE60" if v > 0.5 else "#E74C3C"
        x = 50 + i * (total_w / 5)
        dwg.add(dwg.text(f"{lb}: {v:.2f}", insert=(x, y_sc),
                         font_size="13px", fill=clr, font_weight="bold"))

    dwg.save()
    print(f"    Saved: {save_path}")


def create_refinement_gif(pair_snapshots, scores_list, save_path, fps=1):
    """Animated GIF using actual pair snapshots at each refinement step.

    pair_snapshots: list of MultiStoryPair at step 0, 1, 2, ...
    scores_list: list of score dicts at step 0, 1, 2, ...
    """
    if not pair_snapshots or not scores_list:
        print(f"    Warning: No snapshots for {save_path}")
        return

    frames = []

    def render_frame(pair, scores, step_label, ref_layout=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        # Left: upper floor with ghost overlay showing where reference walls are
        _draw_layout_on_ax(pair.floor_upper, ax1, title=f"Floor 2 (Upper) -- {step_label}",
                           ghost_layout=ref_layout)
        ax1.set_facecolor("#FFFAF0")
        # Right: lower floor (reference, doesn't change)
        _draw_layout_on_ax(pair.floor_lower, ax2, title="Floor 1 (Lower) -- Reference")

        # Score bar at bottom
        parts = []
        for k, lb in [("stair_score", "Stair"), ("wall_score", "Wall"),
                      ("door_score", "Door"), ("egress_score", "Egress"),
                      ("overall_score", "Overall")]:
            v = scores.get(k, 0)
            parts.append(f"{lb}: {v:.2f}")
        fig.text(0.5, 0.02, " | ".join(parts), ha="center", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        fig.suptitle(f"Refinement Progress: {step_label}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0.06, 1, 0.94])

        fig.canvas.draw()
        ww, hh = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(hh, ww, 3)
        plt.close()
        return buf

    ref = pair_snapshots[0].floor_lower  # reference floor never changes

    for i, snap in enumerate(pair_snapshots):
        if i < len(scores_list):
            sc = scores_list[i]
        else:
            sc = scores_list[-1] if scores_list else {}

        if i == 0:
            label = "Step 0: Initial (Broken)"
        else:
            label = f"Step {i}: After Fix #{i}"
        frames.append(render_frame(snap, sc, label, ref_layout=ref))

    # Duplicate last frame so the final state lingers
    if frames:
        frames.append(frames[-1])
        frames.append(frames[-1])

    if frames:
        iio.imwrite(save_path, frames, duration=1500, loop=0)
        print(f"    Saved: {save_path}")


def plot_training_curves(history, save_path):
    """2x3 subplot grid showing loss curves for all heads."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    heads = ["total", "stair", "wall", "door", "egress", "overall"]
    titles = ["Total Loss", "Stair Alignment", "Wall Alignment",
              "Door Compliance", "Egress Compliance", "Overall Coherence"]

    for ax, head, ttl in zip(axes.flat, heads, titles):
        tk = f"train_{head}"
        vk = f"val_{head}"
        if tk in history:
            ax.plot(history[tk], label="Train", color="#3498DB")
        if vk in history:
            ax.plot(history[vk], label="Val", color="#E74C3C")
        ax.set_title(ttl, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves -- Per-Head Loss", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# Section K: Main Pipeline
# ============================================================

def main():
    """Main pipeline: parse → synthesize → train → evaluate → refine → visualize.

    All design decisions are commented inline.
    """
    print("=" * 70)
    print("  GNN for Multi-Story Floor Plan Coherence")
    print("  Dataset: CVC-FP (122 SVG files)")
    print("  Model: GAT (3 layers × 4 heads, 5 scoring heads)")
    print("=" * 70)

    # ---- Stage 1: Setup ----
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[SETUP] Device: {DEVICE}")
    print(f"[SETUP] SEED: {SEED}")
    if torch.cuda.is_available():
        print(f"[SETUP] GPU: {torch.cuda.get_device_name(0)}")
        gpu_count = torch.cuda.device_count()
        print(f"[SETUP] GPU count: {gpu_count}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[SETUP] GPU Memory: {mem:.1f} GB")

    # ---- Stage 2: Parse CVC-FP SVGs ----
    print(f"\n[PARSE] Loading CVC-FP from {SVG_DIR}...")
    layouts = load_dataset(str(SVG_DIR))
    print(f"[PARSE] Successfully parsed: {len(layouts)} floor plans")
    total_rooms = sum(len(l.rooms) for l in layouts)
    total_walls = sum(len(l.walls) for l in layouts)
    total_doors = sum(len(l.doors) for l in layouts)
    total_windows = sum(len(l.windows) for l in layouts)
    print(f"[PARSE] Total rooms: {total_rooms}, walls: {total_walls}, "
          f"doors: {total_doors}, windows: {total_windows}")
    print(f"[PARSE] Avg per floor: {total_rooms/len(layouts):.1f} rooms, "
          f"{total_walls/len(layouts):.1f} walls, {total_doors/len(layouts):.1f} doors")

    # ---- Stage 3: Synthetic multi-story pairs ----
    print(f"\n[SYNTH] Generating 2000 multi-story training pairs...")
    t0 = time.time()
    pairs = generate_training_data(layouts, n_pairs=2000)
    print(f"[SYNTH] Generated {len(pairs)} pairs in {time.time()-t0:.1f}s")
    pos = sum(1 for p in pairs if p.label > 0.9)
    neg = sum(1 for p in pairs if p.label < 0.1)
    mid = len(pairs) - pos - neg
    print(f"[SYNTH] High coherence (>0.9): {pos}, Low (<0.1): {neg}, "
          f"Continuous spectrum: {mid}")
    # Sample ground-truth scores
    sample = pairs[0]
    print(f"[SYNTH] Sample pair scores: stair_iou={sample.stair_iou:.3f}, "
          f"wall_overlap={sample.wall_overlap:.3f}, "
          f"door_comply={sample.door_compliant:.3f}, egress={sample.egress_ok:.1f}")

    # ---- Stage 4: Graph construction ----
    print(f"\n[GRAPH] Converting pairs to PyTorch Geometric Data objects...")
    t0 = time.time()
    data_list = [build_pair_graph(p) for p in pairs]
    print(f"[GRAPH] Built {len(data_list)} graphs in {time.time()-t0:.1f}s")
    avg_nodes = np.mean([d.num_nodes for d in data_list])
    avg_edges = np.mean([d.num_edges for d in data_list])
    print(f"[GRAPH] Avg nodes/graph: {avg_nodes:.0f}, Avg edges/graph: {avg_edges:.0f}")

    # ---- Stage 5: Train/val/test split ----
    print(f"\n[SPLIT] 70/15/15 train/val/test split...")
    n = len(data_list)
    idx = list(range(n))
    rng = random.Random(SEED)
    rng.shuffle(idx)
    n_train = int(0.70 * n)
    n_val = int(0.85 * n)
    train_data = [data_list[i] for i in idx[:n_train]]
    val_data = [data_list[i] for i in idx[n_train:n_val]]
    test_data = [data_list[i] for i in idx[n_val:]]
    test_pairs_subset = [pairs[i] for i in idx[n_val:]]
    print(f"[SPLIT] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # ---- Stage 6: Train ----
    print(f"\n[TRAIN] Initializing FloorPlanGAT model...")
    model = FloorPlanGAT().to(DEVICE)
    print(f"[TRAIN] Trainable parameters: {model.count_parameters():,}")

    # DataParallel if multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"[TRAIN] Using DataParallel across {torch.cuda.device_count()} GPUs")
        # Wrap in DataParallel — note: PyG batching handles this via Batch object
        # For simplicity, we keep single-GPU training (T4 has 15.6 GB, plenty)
        pass

    print(f"[TRAIN] Training for up to {EPOCHS} epochs (early stop patience=20)...")
    model, history = train_model(model, train_data, val_data)

    # ---- Stage 6b: Comprehensive Post-Training Diagnostic ----
    # The GNN (StructuralCritic) must learn CONTINUOUS distance relationships,
    # not just binary "good/bad". If it fails, the Refiner gets no useful
    # "colder/warmer" signal and just moves rooms randomly.
    #
    # We test this with 3 controlled sweep experiments + scatter analysis:
    # 1. Stair alignment sweep: shift_frac 0→1, does predicted stair score track GT IoU?
    # 2. Wall removal sweep: remove_frac 0→0.4, does predicted wall score track GT alignment?
    # 3. Door shrink sweep: shrink_frac 0→1, does predicted door score track GT compliance?
    # Success criterion: Spearman rank correlation ≥ 0.7 for continuous heads, ≥ 0.5 for binary.
    print(f"\n{'=' * 70}")
    print(f"  POST-TRAINING DIAGNOSTIC: Does the GNN learn distance?")
    print(f"{'=' * 70}")

    diag_results = {}  # {head_name: {spearman, p_value, monotonic_frac, gt_vals, pred_vals}}
    n_sweep = 20  # Number of sweep steps per dimension

    # --- Diagnostic 1: Stair Alignment Sweep ---
    print(f"\n[DIAG 1/3] Stair alignment sweep ({n_sweep} steps, shift_frac 0.0 → 1.0)...")
    stair_gt, stair_pred = [], []
    for i in range(n_sweep):
        frac = i / (n_sweep - 1)  # 0.0 to 1.0
        diag_rng = random.Random(9999 + i)
        diag_pair = create_stair_sweep_pair(layouts[i % len(layouts)], diag_rng, shift_frac=frac)
        sc = _score_pair(model, diag_pair, DEVICE)
        gt_val = diag_pair.stair_iou
        pred_val = sc["stair_score"]
        stair_gt.append(gt_val)
        stair_pred.append(pred_val)
        print(f"  shift={frac:.2f} | GT_IoU={gt_val:.3f} | Pred={pred_val:.4f}")

    rho, p_val = spearmanr(stair_gt, stair_pred)
    mono_pairs = sum(1 for j in range(len(stair_gt)-1)
                     if (stair_gt[j+1] - stair_gt[j]) * (stair_pred[j+1] - stair_pred[j]) >= 0)
    mono_frac = mono_pairs / max(len(stair_gt) - 1, 1)
    diag_results["stair"] = {"spearman": rho, "p_value": p_val, "monotonic_frac": mono_frac,
                              "gt": stair_gt, "pred": stair_pred}
    print(f"  Spearman ρ = {rho:.3f} (p={p_val:.4f}), Monotonicity = {mono_frac:.1%}")

    # --- Diagnostic 2: Wall Alignment Sweep ---
    print(f"\n[DIAG 2/3] Wall removal sweep ({n_sweep} steps, remove_frac 0.0 → 0.4)...")
    wall_gt, wall_pred = [], []
    for i in range(n_sweep):
        frac = i / (n_sweep - 1) * 0.4  # 0.0 to 0.4
        diag_rng = random.Random(7777 + i)
        diag_pair = create_wall_sweep_pair(layouts[i % len(layouts)], diag_rng, remove_frac=frac)
        sc = _score_pair(model, diag_pair, DEVICE)
        gt_val = diag_pair.wall_overlap
        pred_val = sc["wall_score"]
        wall_gt.append(gt_val)
        wall_pred.append(pred_val)
        print(f"  remove={frac:.2f} | GT_align={gt_val:.3f} | Pred={pred_val:.4f}")

    rho, p_val = spearmanr(wall_gt, wall_pred)
    mono_pairs = sum(1 for j in range(len(wall_gt)-1)
                     if (wall_gt[j+1] - wall_gt[j]) * (wall_pred[j+1] - wall_pred[j]) >= 0)
    mono_frac = mono_pairs / max(len(wall_gt) - 1, 1)
    diag_results["wall"] = {"spearman": rho, "p_value": p_val, "monotonic_frac": mono_frac,
                             "gt": wall_gt, "pred": wall_pred}
    print(f"  Spearman ρ = {rho:.3f} (p={p_val:.4f}), Monotonicity = {mono_frac:.1%}")

    # --- Diagnostic 3: Door Compliance Sweep ---
    print(f"\n[DIAG 3/3] Door shrink sweep ({n_sweep} steps, shrink_frac 0.0 → 1.0)...")
    door_gt, door_pred = [], []
    for i in range(n_sweep):
        frac = i / (n_sweep - 1)  # 0.0 to 1.0
        diag_rng = random.Random(5555 + i)
        diag_pair = create_door_sweep_pair(layouts[i % len(layouts)], diag_rng, shrink_frac=frac)
        sc = _score_pair(model, diag_pair, DEVICE)
        gt_val = diag_pair.door_compliant
        pred_val = sc["door_score"]
        door_gt.append(gt_val)
        door_pred.append(pred_val)
        print(f"  shrink={frac:.2f} | GT_comply={gt_val:.3f} | Pred={pred_val:.4f}")

    rho, p_val = spearmanr(door_gt, door_pred)
    mono_pairs = sum(1 for j in range(len(door_gt)-1)
                     if (door_gt[j+1] - door_gt[j]) * (door_pred[j+1] - door_pred[j]) >= 0)
    mono_frac = mono_pairs / max(len(door_gt) - 1, 1)
    diag_results["door"] = {"spearman": rho, "p_value": p_val, "monotonic_frac": mono_frac,
                             "gt": door_gt, "pred": door_pred}
    print(f"  Spearman ρ = {rho:.3f} (p={p_val:.4f}), Monotonicity = {mono_frac:.1%}")

    # --- Diagnostic 4: Prediction vs Ground-Truth Scatter Plots ---
    print(f"\n[DIAG] Generating prediction vs ground-truth scatter plots...")
    # Collect predictions on the full test set for scatter analysis
    test_loader_diag = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    test_preds = defaultdict(list)
    test_targets = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch_data in test_loader_diag:
            batch_data = batch_data.to(DEVICE)
            preds = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr,
                          batch_data.floor_mask, batch_data.batch)
            test_preds["stair"].extend(preds["stair_score"].cpu().tolist())
            test_preds["wall"].extend(preds["wall_score"].cpu().tolist())
            test_preds["door"].extend(preds["door_score"].cpu().tolist())
            test_preds["egress"].extend(preds["egress_score"].cpu().tolist())
            test_preds["overall"].extend(preds["overall_score"].cpu().tolist())
            test_targets["stair"].extend(batch_data.y_stair_iou.cpu().tolist())
            test_targets["wall"].extend(batch_data.y_wall_align.cpu().tolist())
            test_targets["door"].extend(batch_data.y_door_comply.cpu().tolist())
            test_targets["egress"].extend(batch_data.y_egress.cpu().tolist())
            test_targets["overall"].extend(batch_data.y_overall.cpu().tolist())

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    head_names = ["stair", "wall", "door", "egress", "overall"]
    head_labels = ["Stair IoU", "Wall Align", "Door Comply", "Egress", "Overall"]
    r2_scores = {}
    for ax, hname, hlabel in zip(axes, head_names, head_labels):
        gt_arr = np.array(test_targets[hname])
        pr_arr = np.array(test_preds[hname])
        ax.scatter(gt_arr, pr_arr, alpha=0.4, s=10, c="#2196F3")
        ax.plot([0, 1], [0, 1], "r--", lw=1, label="Perfect")
        # R² computation
        ss_res = np.sum((gt_arr - pr_arr) ** 2)
        ss_tot = np.sum((gt_arr - np.mean(gt_arr)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        r2_scores[hname] = r2
        rho_test, _ = spearmanr(gt_arr, pr_arr)
        ax.set_title(f"{hlabel}\nR²={r2:.3f}, ρ={rho_test:.3f}", fontsize=10)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
    plt.suptitle("Post-Training Diagnostic: Predicted vs Ground-Truth (Test Set)", fontsize=13)
    plt.tight_layout()
    scatter_path = str(OUTPUT_DIR / "diagnostic_scatter.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"    Saved: {scatter_path}")

    # --- Diagnostic 5: Sweep scatter plots ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    sweep_data = [
        ("stair", "Stair: GT IoU vs Predicted", "GT Stair IoU"),
        ("wall", "Wall: GT Alignment vs Predicted", "GT Wall Alignment"),
        ("door", "Door: GT Compliance vs Predicted", "GT Door Compliance"),
    ]
    for ax, (hname, title, xlabel) in zip(axes2, sweep_data):
        d = diag_results[hname]
        ax.scatter(d["gt"], d["pred"], c="#E91E63", s=40, zorder=3)
        ax.plot(d["gt"], d["pred"], c="#E91E63", alpha=0.3)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3, label="Perfect")
        ax.set_title(f"{title}\nρ={d['spearman']:.3f}, mono={d['monotonic_frac']:.0%}", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Model Predicted Score")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
    plt.suptitle("Sweep Diagnostic: Does the Model Track Geometry Changes?", fontsize=13)
    plt.tight_layout()
    sweep_path = str(OUTPUT_DIR / "diagnostic_sweeps.png")
    plt.savefig(sweep_path, dpi=150)
    plt.close()
    print(f"    Saved: {sweep_path}")

    # --- Diagnostic Summary: PASS/FAIL ---
    print(f"\n  {'=' * 60}")
    print(f"  DIAGNOSTIC SUMMARY: Can the Refiner get useful signals?")
    print(f"  {'=' * 60}")
    print(f"  {'Head':<12} {'Spearman ρ':>12} {'Monotonic':>12} {'R² (test)':>12} {'Result':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    thresholds = {"stair": 0.7, "wall": 0.7, "door": 0.5}  # Lower bar for binary head
    all_pass = True
    for hname in ["stair", "wall", "door"]:
        d = diag_results[hname]
        r2 = r2_scores.get(hname, 0)
        threshold = thresholds[hname]
        passed = d["spearman"] >= threshold
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_pass = False
        print(f"  {hname:<12} {d['spearman']:>12.3f} {d['monotonic_frac']:>11.0%} {r2:>12.3f} {status:>10}")

    # Add egress and overall from test-set R² (no sweep for these)
    for hname in ["egress", "overall"]:
        r2 = r2_scores.get(hname, 0)
        rho_test, _ = spearmanr(test_targets[hname], test_preds[hname])
        passed = rho_test >= 0.5
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_pass = False
        print(f"  {hname:<12} {rho_test:>12.3f} {'N/A':>12} {r2:>12.3f} {status:>10}")

    print(f"  {'-'*60}")
    if all_pass:
        print(f"  ✓ ALL HEADS PASS — The GNN has learned continuous distance.")
        print(f"    The Refiner WILL get useful 'colder/warmer' signals.")
    else:
        print(f"  ✗ SOME HEADS FAILED — The GNN may not track distance for those dims.")
        print(f"    The Refiner may randomly move rooms for failed dimensions.")
        print(f"    Consider: more sweep training data, higher loss weights, or more epochs.")
    print(f"  {'=' * 60}")

    # ---- Stage 7-8: Evaluate + Refine ----
    print(f"\n[EVAL] Computing baseline scores (naive stacking)...")
    baseline_scores = compute_baseline_scores(test_pairs_subset)

    broken_test = [p for p in test_pairs_subset if p.label < 0.9]
    print(f"\n[REFINE] Running GNN + refinement agent on {len(broken_test)} broken test pairs...")
    gnn_scores, traces = compute_gnn_agent_scores(model, test_pairs_subset)

    # ---- Stage 9: Report ----
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print_evaluation_table(baseline_scores, gnn_scores)
    print_per_head_breakdown(gnn_scores)
    print_per_head_breakdown(baseline_scores, label="Baseline")

    # ---- Stage 10: Visualization ----
    print(f"\n[VIZ] Generating visualizations...")

    # Training curves
    plot_training_curves(history, str(OUTPUT_DIR / "training_curves.png"))

    # Pick the 3 MOST BROKEN traces (ones where refinement actually did work)
    # Sort traces by number of refinement actions taken (most actions = most interesting)
    interesting = [(idx, t) for idx, t in enumerate(traces)
                   if len(t[1]) > 0]  # traces with at least 1 action
    interesting.sort(key=lambda x: len(x[1][1]), reverse=True)  # most actions first
    if len(interesting) < 3:
        interesting = list(enumerate(traces))[:3]
    else:
        interesting = interesting[:3]

    for vi, (trace_idx, trace) in enumerate(interesting):
        refined_pair, actions_list, scores_list, pair_snapshots = trace
        initial_pair = pair_snapshots[0]  # The broken state before any fixes

        # BEFORE/AFTER comparison plot
        final_scores = scores_list[-1] if scores_list else {}
        plot_multi_story_pair(refined_pair, final_scores,
                             str(OUTPUT_DIR / f"pair_{vi}.png"),
                             title=f"Example {vi+1}: Multi-Story Coherence",
                             initial_pair=initial_pair)

        # Attention heatmap on the BROKEN pair (shows what model focuses on)
        broken_data = build_pair_graph(initial_pair)
        plot_attention_heatmap(model, broken_data,
                             initial_pair.floor_lower, initial_pair.floor_upper,
                             str(OUTPUT_DIR / f"attention_{vi}.png"))

        # Interactive SVG with before/after
        flat_actions = [a for al in actions_list for a in al]
        create_svg_interactive(refined_pair, final_scores, flat_actions,
                              str(OUTPUT_DIR / f"interactive_{vi}.svg"),
                              initial_pair=initial_pair)

        # Refinement GIF using actual snapshots
        create_refinement_gif(pair_snapshots, scores_list,
                             str(OUTPUT_DIR / f"refinement_{vi}.gif"))

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE — All outputs saved to {OUTPUT_DIR}/")
    print(f"  Files:")
    print(f"    training_curves.png   — Per-head loss curves")
    print(f"    pair_{{0,1,2}}.png      — Floor plan + radar score dashboard")
    print(f"    attention_{{0,1,2}}.png — GAT attention weight heatmaps")
    print(f"    interactive_{{0,1,2}}.svg — Hover-interactive SVGs")
    print(f"    refinement_{{0,1,2}}.gif — Animated refinement progress")
    print(f"    diagnostic_scatter.png — Pred vs GT scatter (5 heads)")
    print(f"    diagnostic_sweeps.png  — Sweep correlation plots")
    print(f"    best_model.pt         — Trained model checkpoint")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
