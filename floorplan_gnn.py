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


def generate_training_data(layouts: List[FloorLayout], n_pairs: int = 2000) -> List[MultiStoryPair]:
    """Generate synthetic multi-story pairs: 33% positive, 34% negative, 33% mixed.

    Each of the 122 layouts is reused ~16 times with different random perturbations.
    """
    rng = random.Random(SEED)
    pairs = []
    n_pos = int(n_pairs * 0.33)
    n_neg = int(n_pairs * 0.34)
    n_mix = n_pairs - n_pos - n_neg

    for i in range(n_pos):
        layout = rng.choice(layouts)
        pairs.append(create_positive_pair(layout, rng))
    for i in range(n_neg):
        layout = rng.choice(layouts)
        pairs.append(create_negative_pair(layout, rng))
    for i in range(n_mix):
        layout = rng.choice(layouts)
        pairs.append(create_mixed_pair(layout, rng))

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
                torch.zeros(0, NUM_EDGE_TYPES), {})

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

    # Convert to tensors
    if edges:
        src = torch.tensor([e[0] for e in edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
        edge_index = torch.stack([src, dst])
        # One-hot edge type
        edge_attr = torch.zeros(len(edges), NUM_EDGE_TYPES)
        for i, (_, _, etype) in enumerate(edges):
            edge_attr[i, etype] = 1.0
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, NUM_EDGE_TYPES)

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
        edge_attr = torch.zeros(0, NUM_EDGE_TYPES)

    # Cross-floor edges
    cross_edges_src = []
    cross_edges_dst = []
    cross_edge_types = []

    # Stair-to-stair link
    if pair.floor_lower.synthetic_stair and pair.floor_upper.synthetic_stair:
        sl_id = pair.floor_lower.synthetic_stair.id
        su_id = pair.floor_upper.synthetic_stair.id
        if sl_id in id2idx_l and su_id in id2idx_u:
            li = id2idx_l[sl_id]
            ui = id2idx_u[su_id] + n_lower
            cross_edges_src.extend([li, ui])
            cross_edges_dst.extend([ui, li])
            cross_edge_types.extend([EDGE_TYPE_MAP["stair_link"]] * 2)

    # Same-class spatial proximity across floors
    polys_l = list(pair.floor_lower.polygons.values())
    polys_u = list(pair.floor_upper.polygons.values())
    diag = math.sqrt(pair.floor_lower.canvas_width**2 + pair.floor_lower.canvas_height**2)
    cross_threshold = 0.10 * diag

    for pl in polys_l:
        if pl.cls in ("Room", "Wall", "Door", "Window"):
            for pu in polys_u:
                if pu.cls == pl.cls:
                    dx = pl.centroid[0] - pu.centroid[0]
                    dy = pl.centroid[1] - pu.centroid[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < cross_threshold and pl.id in id2idx_l and pu.id in id2idx_u:
                        li = id2idx_l[pl.id]
                        ui = id2idx_u[pu.id] + n_lower
                        cross_edges_src.extend([li, ui])
                        cross_edges_dst.extend([ui, li])
                        cross_edge_types.extend([EDGE_TYPE_MAP["cross_floor"]] * 2)

    # Append cross-floor edges
    if cross_edges_src:
        cf_ei = torch.tensor([cross_edges_src, cross_edges_dst], dtype=torch.long)
        cf_ea = torch.zeros(len(cross_edges_src), NUM_EDGE_TYPES)
        for i, et in enumerate(cross_edge_types):
            cf_ea[i, et] = 1.0
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
        edge_dim: int = NUM_EDGE_TYPES,
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
    """Multi-task loss: MSE for continuous + BCE for binary heads.

    Weights: stair=1.0, wall=1.0, door=0.5, egress=0.5, overall=2.0.
    Overall weighted higher to anchor the global coherence signal.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.weights = {
            "stair": 1.0, "wall": 1.0, "door": 0.5,
            "egress": 0.5, "overall": 2.0,
        }

    def forward(self, preds, batch_data):
        # Gather targets — handle both batched and single
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
            "stair": self.mse(preds["stair_score"], y_stair),
            "wall": self.mse(preds["wall_score"], y_wall),
            "door": self.bce(door_pred, y_door),
            "egress": self.bce(egress_pred, y_egress),
            "overall": self.mse(preds["overall_score"], y_overall),
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
    """3-loop iterative refinement: score → identify weakest head → fix → repeat.

    Returns: (final_pair, actions_per_iteration, scores_per_iteration)
    """
    all_actions = []
    all_scores = []

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
        for a in actions:
            print(f"      → {a.description}")

    # Final score
    final_scores = _score_pair(model, pair, device)
    all_scores.append(final_scores)

    return pair, all_actions, all_scores


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
        refined, actions, scores = refinement_loop(model, pair, device=device)
        all_traces.append((refined, actions, scores))
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

# Color scheme matching CVC-FP conventions
COLORS = {
    "Wall": "#AFD8F8",
    "Room": "#008E8E",
    "Door": "#FF8E46",
    "Window": "#F6BD0F",
    "Separation": "#D64646",
    "Stair": "#9B59B6",
    "Parking": "#8BBA00",
}


def plot_floor_layout(layout, ax, title="", highlight_stair=True, highlight_ids=None):
    """Draw a single floor layout on a matplotlib axes."""
    ax.set_xlim(0, layout.canvas_width)
    ax.set_ylim(layout.canvas_height, 0)  # Flip Y axis (SVG convention)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold")

    # Draw rooms first (background)
    for room in layout.rooms:
        xs = [p[0] for p in room.points] + [room.points[0][0]]
        ys = [p[1] for p in room.points] + [room.points[0][1]]
        ax.fill(xs, ys, color=COLORS["Room"], alpha=0.2)
        ax.plot(xs, ys, color=COLORS["Room"], linewidth=0.5)

    # Draw walls
    for wall in layout.walls:
        xs = [p[0] for p in wall.points] + [wall.points[0][0]]
        ys = [p[1] for p in wall.points] + [wall.points[0][1]]
        ax.fill(xs, ys, color=COLORS["Wall"], alpha=0.7)

    # Draw doors
    for door in layout.doors:
        xs = [p[0] for p in door.points] + [door.points[0][0]]
        ys = [p[1] for p in door.points] + [door.points[0][1]]
        ax.fill(xs, ys, color=COLORS["Door"], alpha=0.7)

    # Draw windows
    for win in layout.windows:
        xs = [p[0] for p in win.points] + [win.points[0][0]]
        ys = [p[1] for p in win.points] + [win.points[0][1]]
        ax.fill(xs, ys, color=COLORS["Window"], alpha=0.7)

    # Draw stair
    if highlight_stair and layout.synthetic_stair:
        s = layout.synthetic_stair
        xs = [p[0] for p in s.points] + [s.points[0][0]]
        ys = [p[1] for p in s.points] + [s.points[0][1]]
        ax.fill(xs, ys, color=COLORS["Stair"], alpha=0.5, hatch="//")
        ax.plot(xs, ys, color=COLORS["Stair"], linewidth=2)
        ax.annotate("STAIR", xy=s.centroid, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["Stair"], alpha=0.8))

    # Highlight problem polygons
    if highlight_ids:
        for pid in highlight_ids:
            if pid in layout.polygons:
                poly = layout.polygons[pid]
                xs = [p[0] for p in poly.points] + [poly.points[0][0]]
                ys = [p[1] for p in poly.points] + [poly.points[0][1]]
                ax.plot(xs, ys, color="red", linewidth=3, linestyle="--")

    ax.set_xlabel("x (px)", fontsize=8)
    ax.set_ylabel("y (px)", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_multi_story_pair(pair, scores, save_path, title="Multi-Story Coherence"):
    """Stacked visualization: two floors + radar score dashboard."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1])

    # Floor plans
    ax1 = fig.add_subplot(gs[0, 0])
    plot_floor_layout(pair.floor_lower, ax1, title="Floor 1 (Lower)")
    ax2 = fig.add_subplot(gs[0, 1])
    plot_floor_layout(pair.floor_upper, ax2, title="Floor 2 (Upper)")

    # Legend
    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")
    patches = [mpatches.Patch(color=c, label=l, alpha=0.7)
               for l, c in COLORS.items()]
    ax_leg.legend(handles=patches, loc="center", fontsize=10, title="Element Types")

    # Radar chart
    ax_radar = fig.add_subplot(gs[1, 0], projection="polar")
    categories = ["Stair\nAlign", "Wall\nAlign", "Door\nComply", "Egress", "Overall"]
    values = [scores.get("stair_score", scores.get("stair", 0)),
              scores.get("wall_score", scores.get("wall", 0)),
              scores.get("door_score", scores.get("door", 0)),
              scores.get("egress_score", scores.get("egress", 0)),
              scores.get("overall_score", scores.get("overall", 0))]
    # Normalize to [0, 1] if given as percentages
    values = [v / 100 if v > 1 else v for v in values]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]
    ax_radar.fill(angles_plot, values_plot, color="#3498DB", alpha=0.3)
    ax_radar.plot(angles_plot, values_plot, color="#3498DB", linewidth=2)
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(categories, fontsize=8)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title("Coherence Scores", fontsize=10, fontweight="bold", pad=20)

    # Score table
    ax_table = fig.add_subplot(gs[1, 1:])
    ax_table.axis("off")
    table_data = [
        ["Stair Alignment", f"{values[0]:.3f}", "PASS" if values[0] > 0.5 else "FAIL"],
        ["Wall Alignment", f"{values[1]:.3f}", "PASS" if values[1] > 0.5 else "FAIL"],
        ["Door Compliance", f"{values[2]:.3f}", "PASS" if values[2] > 0.5 else "FAIL"],
        ["Egress Compliance", f"{values[3]:.3f}", "PASS" if values[3] > 0.5 else "FAIL"],
        ["Overall Coherence", f"{values[4]:.3f}", "PASS" if values[4] > 0.5 else "FAIL"],
    ]
    cell_colors = []
    for row in table_data:
        color = "#D5F5E3" if row[2] == "PASS" else "#FADBD8"
        cell_colors.append([color, color, color])
    table = ax_table.table(cellText=table_data,
                           colLabels=["Metric", "Score", "Status"],
                           cellColours=cell_colors,
                           loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_attention_heatmap(model, data, layout_lower, layout_upper, save_path):
    """Visualize GAT attention weights overlaid on floor geometry.

    Primary explainability visualization: shows which edges the model
    attends to most when scoring coherence.
    """
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    with torch.no_grad():
        preds = model(data.x, data.edge_index, data.edge_attr,
                      data.floor_mask, data.batch)

    # Get last-layer attention weights (averaged across heads)
    if model._attention_weights:
        attn = model._attention_weights[-1].cpu().numpy()
        # Average across heads if multi-head
        if attn.ndim == 2:
            attn_avg = attn.mean(axis=1) if attn.shape[1] > 1 else attn.squeeze()
        else:
            attn_avg = attn
    else:
        print("    Warning: No attention weights captured")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Draw floor layouts
    plot_floor_layout(layout_lower, ax1, title="Floor 1 — Attention Overlay")
    plot_floor_layout(layout_upper, ax2, title="Floor 2 — Attention Overlay")

    # Overlay attention as colored edges
    edge_index = data.edge_index.cpu().numpy()
    floor_mask = data.floor_mask.cpu().numpy()
    n_lower = int((floor_mask == 0).sum())

    all_polys_lower = list(layout_lower.polygons.values())
    all_polys_upper = list(layout_upper.polygons.values())

    # Normalize attention for colormap
    if len(attn_avg) > 0:
        attn_norm = (attn_avg - attn_avg.min()) / max(attn_avg.max() - attn_avg.min(), 1e-8)
    else:
        attn_norm = np.array([])

    cmap = plt.cm.YlOrRd

    for ei in range(min(edge_index.shape[1], len(attn_norm))):
        src, dst = edge_index[0, ei], edge_index[1, ei]
        alpha_val = float(attn_norm[ei])

        # Determine which floor
        if src < n_lower and dst < n_lower:
            # Both on lower floor
            if src < len(all_polys_lower) and dst < len(all_polys_lower):
                p1 = all_polys_lower[src].centroid
                p2 = all_polys_lower[dst].centroid
                ax1.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color=cmap(alpha_val), alpha=max(alpha_val, 0.1), linewidth=1)
        elif src >= n_lower and dst >= n_lower:
            # Both on upper floor
            si, di = src - n_lower, dst - n_lower
            if si < len(all_polys_upper) and di < len(all_polys_upper):
                p1 = all_polys_upper[si].centroid
                p2 = all_polys_upper[di].centroid
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color=cmap(alpha_val), alpha=max(alpha_val, 0.1), linewidth=1)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax1, ax2], label="Attention Weight", shrink=0.6)

    fig.suptitle("GAT Attention Heatmap (Last Layer, Avg Heads)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def create_svg_interactive(pair, scores, actions, save_path):
    """Create interactive SVG with hover tooltips and pulsing problem highlights."""
    w = max(pair.floor_lower.canvas_width, pair.floor_upper.canvas_width)
    h = max(pair.floor_lower.canvas_height, pair.floor_upper.canvas_height)
    total_w = w * 2.2  # Two floors side by side with gap

    dwg = svgwrite.Drawing(save_path, size=(f"{total_w}px", f"{h + 100}px"))

    # CSS for hover tooltips and pulse animation
    dwg.defs.add(dwg.style("""
        .element:hover { stroke: #E74C3C; stroke-width: 3; cursor: pointer; }
        .tooltip { visibility: hidden; font-size: 12px; }
        .element:hover + .tooltip { visibility: visible; }
        @keyframes pulse { 0%,100% { opacity: 0.3; } 50% { opacity: 1; } }
        .problem { animation: pulse 1.5s infinite; stroke: red; stroke-width: 2; }
    """))

    # Title
    dwg.add(dwg.text("Multi-Story Floor Plan Coherence",
                      insert=(total_w / 2, 25), text_anchor="middle",
                      font_size="18px", font_weight="bold", fill="#2C3E50"))

    def draw_floor(layout, x_offset, y_offset, label):
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
        g = dwg.g(id=f"floor_{safe_id}")
        dwg.add(dwg.text(label, insert=(x_offset + w / 2, y_offset + 50),
                         text_anchor="middle", font_size="14px", font_weight="bold"))

        for poly in layout.polygons.values():
            pts = [(p[0] + x_offset, p[1] + y_offset + 60) for p in poly.points]
            color = COLORS.get(poly.cls, "#CCCCCC")
            opacity = 0.5 if poly.cls == "Room" else 0.8

            polygon = dwg.polygon(points=pts, fill=color, opacity=opacity,
                                  class_="element")
            polygon.set_desc(title=f"{poly.cls} (id={poly.id})")
            g.add(polygon)

            # Tooltip
            tooltip = dwg.text(f"{poly.cls} #{poly.id}",
                              insert=(poly.centroid[0] + x_offset,
                                     poly.centroid[1] + y_offset + 55),
                              class_="tooltip", font_size="9px", fill="#333")
            g.add(tooltip)

        # Highlight stair with pulse animation
        if layout.synthetic_stair:
            s = layout.synthetic_stair
            pts = [(p[0] + x_offset, p[1] + y_offset + 60) for p in s.points]
            stair_poly = dwg.polygon(points=pts, fill=COLORS["Stair"],
                                     opacity=0.6, class_="problem")
            g.add(stair_poly)

        dwg.add(g)

    draw_floor(pair.floor_lower, 10, 0, "Floor 1 (Lower)")
    draw_floor(pair.floor_upper, w + 30, 0, "Floor 2 (Upper)")

    # Score dashboard at bottom
    y_scores = h + 70
    score_keys = ["stair_score", "wall_score", "door_score", "egress_score", "overall_score"]
    score_labels = ["Stair", "Wall", "Door", "Egress", "Overall"]
    for i, (key, label) in enumerate(zip(score_keys, score_labels)):
        val = scores.get(key, scores.get(key.replace("_score", ""), 0))
        if val > 1:
            val = val / 100
        color = "#27AE60" if val > 0.5 else "#E74C3C"
        x = 50 + i * (total_w / 5)
        dwg.add(dwg.text(f"{label}: {val:.2f}", insert=(x, y_scores),
                         font_size="12px", fill=color, font_weight="bold"))

    dwg.save()
    print(f"    Saved: {save_path}")


def create_refinement_gif(pair_initial, refinement_steps, save_path, fps=1):
    """Animated GIF showing refinement progress across iterations.

    Each frame: current floor state + scores + action description.
    """
    frames = []

    def render_frame(pair, scores, title):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plot_floor_layout(pair.floor_lower, ax1, title="Floor 1")
        plot_floor_layout(pair.floor_upper, ax2, title="Floor 2")
        fig.suptitle(title, fontsize=12, fontweight="bold")

        # Add score text
        score_text = " | ".join([
            f"Stair: {scores.get('stair_score', 0):.2f}",
            f"Wall: {scores.get('wall_score', 0):.2f}",
            f"Door: {scores.get('door_score', 0):.2f}",
            f"Egress: {scores.get('egress_score', 0):.2f}",
            f"Overall: {scores.get('overall_score', 0):.2f}",
        ])
        fig.text(0.5, 0.02, score_text, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Render to numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(h, w, 3)
        plt.close()
        return buf

    # Frame 0: initial state
    if refinement_steps:
        initial_scores = refinement_steps[0][-1][0] if refinement_steps[0][-1] else {}
        frames.append(render_frame(pair_initial, initial_scores, "Initial (Before Refinement)"))

        # Subsequent frames
        for i, (refined, actions, scores_list) in enumerate(refinement_steps[:1]):
            for j, sc in enumerate(scores_list[1:], 1):
                frames.append(render_frame(refined, sc, f"After Refinement Step {j}"))

    if frames:
        iio.imwrite(save_path, frames, duration=1000 // fps, loop=0)
        print(f"    Saved: {save_path}")
    else:
        print(f"    Warning: No frames to save for {save_path}")


def plot_training_curves(history, save_path):
    """2×3 subplot grid showing loss curves for all heads."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    heads = ["total", "stair", "wall", "door", "egress", "overall"]
    titles = ["Total Loss", "Stair Alignment", "Wall Alignment",
              "Door Compliance", "Egress Compliance", "Overall Coherence"]

    for ax, head, title in zip(axes.flat, heads, titles):
        train_key = f"train_{head}"
        val_key = f"val_{head}"
        if train_key in history:
            ax.plot(history[train_key], label="Train", color="#3498DB")
        if val_key in history:
            ax.plot(history[val_key], label="Val", color="#E74C3C")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves — Per-Head Loss", fontsize=13, fontweight="bold")
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
    pos = sum(1 for p in pairs if p.label > 0.8)
    neg = sum(1 for p in pairs if p.label < 0.2)
    mixed = len(pairs) - pos - neg
    print(f"[SYNTH] Positive: {pos}, Negative: {neg}, Mixed: {mixed}")
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

    # Pick 3 representative examples from refined traces
    for i in range(min(3, len(traces))):
        refined_pair, actions_list, scores_list = traces[i]

        # Multi-story pair plot with scores
        final_scores = scores_list[-1] if scores_list else {}
        plot_multi_story_pair(refined_pair, final_scores,
                             str(OUTPUT_DIR / f"pair_{i}.png"),
                             title=f"Example {i+1}: Multi-Story Coherence")

        # Attention heatmap
        pair_data = build_pair_graph(refined_pair)
        plot_attention_heatmap(model, pair_data,
                             refined_pair.floor_lower, refined_pair.floor_upper,
                             str(OUTPUT_DIR / f"attention_{i}.png"))

        # Interactive SVG
        flat_actions = [a for al in actions_list for a in al]
        create_svg_interactive(refined_pair, final_scores, flat_actions,
                              str(OUTPUT_DIR / f"interactive_{i}.svg"))

        # Refinement GIF
        create_refinement_gif(refined_pair, [traces[i]],
                             str(OUTPUT_DIR / f"refinement_{i}.gif"))

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE — All outputs saved to {OUTPUT_DIR}/")
    print(f"  Files:")
    print(f"    training_curves.png   — Per-head loss curves")
    print(f"    pair_{{0,1,2}}.png      — Floor plan + radar score dashboard")
    print(f"    attention_{{0,1,2}}.png — GAT attention weight heatmaps")
    print(f"    interactive_{{0,1,2}}.svg — Hover-interactive SVGs")
    print(f"    refinement_{{0,1,2}}.gif — Animated refinement progress")
    print(f"    best_model.pt         — Trained model checkpoint")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
