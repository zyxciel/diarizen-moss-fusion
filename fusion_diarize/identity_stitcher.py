from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from numbers import Integral, Real
import re
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from fusion_diarize.types import ChunkWindow, Turn


@dataclass(frozen=True)
class StitchConfig:
    frame_hop: float = 0.02
    min_overlap_activity: float = 3.0
    min_intersection: float = 1.5
    min_intersection_ratio: float = 0.3
    overlap_mask_dilation: float = 0.2
    overlap_link_score: float = 0.70
    assignment_margin: float = 0.10
    embedding_cosine: float = 0.80
    embedding_margin: float = 0.05
    matching_anchor_cosine: float = 0.75
    conflicting_anchor_cosine: float = 0.85
    asymmetric_anchor_cosine: float = 0.95
    long_distance_chunk_span: int = 2
    cluster_max_distance: float = 0.25
    centroid_outlier_distance: float = 0.15
    anchor_purity: float = 0.70
    anchor_min_overlap: float = 3.0
    anchor_activity_ratio: float = 0.6
    anchor_min_cosine: float = 0.40
    explosion_abs_cap: int = 12
    explosion_ratio: float = 2.0

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

        if self.frame_hop <= 0:
            raise ValueError("frame_hop must be positive")
        for name in (
            "min_overlap_activity",
            "min_intersection",
            "min_intersection_ratio",
            "overlap_mask_dilation",
            "assignment_margin",
            "embedding_margin",
            "cluster_max_distance",
            "centroid_outlier_distance",
            "anchor_min_overlap",
            "anchor_activity_ratio",
        ):
            if values[name] < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "overlap_link_score",
            "anchor_purity",
            "min_intersection_ratio",
            "anchor_activity_ratio",
        ):
            if not 0 <= values[name] <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in (
            "embedding_cosine",
            "matching_anchor_cosine",
            "conflicting_anchor_cosine",
            "asymmetric_anchor_cosine",
            "anchor_min_cosine",
        ):
            if not -1 <= values[name] <= 1:
                raise ValueError(f"{name} must be between -1 and 1")
        if self.assignment_margin > 1:
            raise ValueError("assignment_margin must not exceed 1")
        if self.embedding_margin > 2:
            raise ValueError("embedding_margin must not exceed 2")
        if self.cluster_max_distance > 2:
            raise ValueError("cluster_max_distance must not exceed 2")
        if self.centroid_outlier_distance > 2:
            raise ValueError("centroid_outlier_distance must not exceed 2")
        for name in ("explosion_abs_cap", "long_distance_chunk_span"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.explosion_ratio <= 0:
            raise ValueError("explosion_ratio must be positive")
        for name in values:
            normalized = (
                int(getattr(self, name))
                if name in ("explosion_abs_cap", "long_distance_chunk_span")
                else float(getattr(self, name))
            )
            object.__setattr__(self, name, normalized)


@dataclass
class LocalSpeakerNode:
    local_id: str
    chunk_index: int
    turns: list[Turn]
    embedding: np.ndarray | None
    anchor: str | None = None
    anchor_purity: float = 0.0

    @property
    def total_activity_duration(self) -> float:
        return float(sum(max(0.0, item.end - item.start) for item in self.turns))


@dataclass(frozen=True)
class LinkEdge:
    left: str
    right: str
    strength: float
    method: str
    temporal_iou: float | None = None
    embedding_cosine: float | None = None
    assignment_margin: float | None = None


@dataclass
class StitchResult:
    turns: list[Turn]
    mapping: dict[str, str]
    metadata: dict[str, Any]
    exploded_spans: list[tuple[float, float]]


_LOCAL_ID = re.compile(r"^c(\d+):.+$")


def parse_chunk_index(local_id: str) -> int:
    match = _LOCAL_ID.match(local_id)
    if match is None:
        raise ValueError(f"local speaker ID must be namespaced as cN:name: {local_id!r}")
    return int(match.group(1))


def _paint(
    turns: list[Turn], window_start: float, window_end: float, frame_hop: float
) -> np.ndarray:
    n_frames = max(0, int(math.ceil((window_end - window_start) / frame_hop)))
    mask = np.zeros(n_frames, dtype=bool)
    for current in turns:
        start = max(window_start, current.start)
        end = min(window_end, current.end)
        if end <= start:
            continue
        first = max(0, int(math.floor((start - window_start) / frame_hop + 1e-9)))
        last = min(n_frames, int(math.ceil((end - window_start) / frame_hop - 1e-9)))
        mask[first:last] = True
    return mask


def _dilate_mask(mask: np.ndarray, radius_frames: int) -> np.ndarray:
    """Expand True runs by ``radius_frames`` on each side (edge jitter tolerance)."""
    if radius_frames <= 0 or mask.size == 0 or not np.any(mask):
        return mask
    padded = np.pad(mask, radius_frames, mode="constant", constant_values=False)
    window = 2 * radius_frames + 1
    # Any-True in a sliding window == binary dilation with a flat kernel.
    cumulative = np.cumsum(padded.astype(np.int32))
    totals = cumulative[window - 1 :] - np.concatenate(
        ([0], cumulative[: len(padded) - window])
    )
    return totals > 0


def activity_iou(
    left_turns: list[Turn],
    right_turns: list[Turn],
    window_start: float,
    window_end: float,
    frame_hop: float = 0.02,
    dilation_seconds: float = 0.0,
) -> tuple[float, float, float, float]:
    if window_end <= window_start:
        return (0.0, 0.0, 0.0, 0.0)
    left = _paint(left_turns, window_start, window_end, frame_hop)
    right = _paint(right_turns, window_start, window_end, frame_hop)
    if dilation_seconds > 0:
        radius = max(0, int(round(dilation_seconds / frame_hop)))
        left = _dilate_mask(left, radius)
        right = _dilate_mask(right, radius)
    union = int(np.count_nonzero(left | right))
    if union == 0:
        return (0.0, 0.0, 0.0, 0.0)
    intersection = int(np.count_nonzero(left & right))
    return (
        intersection / union,
        intersection * frame_hop,
        int(np.count_nonzero(left)) * frame_hop,
        int(np.count_nonzero(right)) * frame_hop,
    )


def _embedding(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
        return None
    return vector


def _cosine(left: Any, right: Any) -> float | None:
    left_vector = _embedding(left)
    right_vector = _embedding(right)
    if (
        left_vector is None
        or right_vector is None
        or left_vector.shape != right_vector.shape
    ):
        return None
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator == 0:
        return None
    value = float(np.dot(left_vector, right_vector) / denominator)
    return max(-1.0, min(1.0, value)) if math.isfinite(value) else None


def build_nodes(
    turns: list[Turn], embeddings: dict[str, list[float] | np.ndarray]
) -> list[LocalSpeakerNode]:
    grouped: dict[str, list[Turn]] = {}
    for current in turns:
        grouped.setdefault(current.speaker_id, []).append(current)
    return [
        LocalSpeakerNode(
            local_id,
            parse_chunk_index(local_id),
            sorted(grouped[local_id], key=lambda item: (item.start, item.end)),
            _embedding(embeddings.get(local_id)),
        )
        for local_id in sorted(grouped, key=lambda item: (parse_chunk_index(item), item))
    ]


def compute_diarizen_anchors(
    nodes: list[LocalSpeakerNode],
    diarizen_turns: list[Turn],
    diarizen_centroids: dict[str, list[float] | np.ndarray],
    config: StitchConfig = StitchConfig(),
) -> list[LocalSpeakerNode]:
    by_speaker: dict[str, list[Turn]] = {}
    for current in diarizen_turns:
        by_speaker.setdefault(current.speaker_id, []).append(current)

    for node in nodes:
        node.anchor = None
        node.anchor_purity = 0.0
        if not node.turns or not by_speaker:
            continue
        start = min(item.start for item in node.turns)
        end = max(item.end for item in node.turns)
        active = int(np.count_nonzero(_paint(node.turns, start, end, config.frame_hop)))
        if active == 0:
            continue
        overlaps: list[tuple[float, str]] = []
        for speaker, turns in by_speaker.items():
            _, intersection, _, _ = activity_iou(
                node.turns, turns, start, end, config.frame_hop
            )
            overlaps.append((intersection, speaker))
        total_overlap = sum(intersection for intersection, _ in overlaps)
        if total_overlap <= 0:
            continue
        intersection, speaker = max(
            overlaps, key=lambda item: (item[0], item[1])
        )
        purity = intersection / total_overlap
        activity_duration = max(
            active * config.frame_hop, node.total_activity_duration
        )
        dynamic_min_overlap = min(
            config.anchor_min_overlap,
            activity_duration * config.anchor_activity_ratio,
        )
        if intersection + 1e-9 < dynamic_min_overlap:
            continue
        if purity + 1e-9 < config.anchor_purity:
            continue
        centroid = _embedding(diarizen_centroids.get(speaker))
        similarity = _cosine(node.embedding, centroid)
        if similarity is not None and similarity + 1e-9 < config.anchor_min_cosine:
            continue
        node.anchor = speaker
        node.anchor_purity = float(purity)
    return nodes


def _edge_dict(edge: LinkEdge) -> dict[str, Any]:
    return asdict(edge)


def match_adjacent_chunks(
    nodes: list[LocalSpeakerNode],
    chunks: list[ChunkWindow],
    config: StitchConfig = StitchConfig(),
) -> tuple[list[LinkEdge], list[dict[str, Any]]]:
    by_chunk: dict[int, list[LocalSpeakerNode]] = {}
    for node in nodes:
        by_chunk.setdefault(node.chunk_index, []).append(node)
    accepted: list[LinkEdge] = []
    rejected: list[dict[str, Any]] = []

    for chunk_index in range(len(chunks) - 1):
        start = max(chunks[chunk_index].start, chunks[chunk_index + 1].start)
        end = min(chunks[chunk_index].end, chunks[chunk_index + 1].end)
        left_nodes = sorted(by_chunk.get(chunk_index, []), key=lambda item: item.local_id)
        right_nodes = sorted(
            by_chunk.get(chunk_index + 1, []), key=lambda item: item.local_id
        )
        if start >= end or not left_nodes or not right_nodes:
            continue
        overlap_window_duration = end - start
        dynamic_min_intersection = max(
            config.min_intersection,
            config.min_intersection_ratio * overlap_window_duration,
        )
        scores = np.full((len(left_nodes), len(right_nodes)), -1e6, dtype=float)
        details: dict[tuple[int, int], tuple[float, float | None]] = {}
        assignable: set[tuple[int, int]] = set()
        for row, left in enumerate(left_nodes):
            for column, right in enumerate(right_nodes):
                iou, intersection, left_active, right_active = activity_iou(
                    left.turns,
                    right.turns,
                    start,
                    end,
                    config.frame_hop,
                    dilation_seconds=config.overlap_mask_dilation,
                )
                diagnostic = {
                    "left": left.local_id,
                    "right": right.local_id,
                    "chunk_pair": [chunk_index, chunk_index + 1],
                    "temporal_iou": iou,
                    "intersection_seconds": intersection,
                    "left_seconds": left_active,
                    "right_seconds": right_active,
                    "dynamic_min_intersection": dynamic_min_intersection,
                }
                if (
                    left_active + 1e-9 < config.min_overlap_activity
                    or right_active + 1e-9 < config.min_overlap_activity
                    or intersection + 1e-9 < dynamic_min_intersection
                ):
                    diagnostic["reason"] = "insufficient_overlap_activity"
                    rejected.append(diagnostic)
                    continue
                cosine = _cosine(left.embedding, right.embedding)
                score = iou if cosine is None else 0.7 * iou + 0.3 * ((1 + cosine) / 2)
                scores[row, column] = score
                details[(row, column)] = (iou, cosine)
                if score + 1e-9 >= config.overlap_link_score:
                    assignable.add((row, column))
                else:
                    rejected.append(
                        {
                            "left": left.local_id,
                            "right": right.local_id,
                            "score": score,
                            "assignment_margin": None,
                            "reason": "below_link_score",
                        }
                    )

        if not assignable:
            continue
        n_left, n_right = len(left_nodes), len(right_nodes)
        assignment_scores = np.zeros(
            (n_left + n_right, n_right + n_left), dtype=float
        )
        assignment_scores[:n_left, :n_right] = -1e6
        for row, column in assignable:
            assignment_scores[row, column] = scores[row, column] + 1e-12
        rows, columns = linear_sum_assignment(-assignment_scores)
        assigned = {
            (row, column)
            for row, column in zip(rows.tolist(), columns.tolist())
            if row < n_left and column < n_right and (row, column) in assignable
        }
        for row, column in sorted(details):
            score = float(scores[row, column])
            if (row, column) not in assignable:
                continue
            if (row, column) not in assigned:
                rejected.append(
                    {
                        "left": left_nodes[row].local_id,
                        "right": right_nodes[column].local_id,
                        "score": score,
                        "reason": "not_selected_by_assignment",
                    }
                )
                continue
            row_alternatives = [
                float(scores[row, other])
                for other in range(len(right_nodes))
                if other != column and (row, other) in details
            ]
            column_alternatives = [
                float(scores[other, column])
                for other in range(len(left_nodes))
                if other != row and (other, column) in details
            ]
            margins = [
                score - max(alternatives)
                for alternatives in (row_alternatives, column_alternatives)
                if alternatives
            ]
            margin = min(margins) if margins else None
            reason = None
            if margin is not None and margin + 1e-9 < config.assignment_margin:
                reason = "ambiguous_assignment"
            if reason is not None:
                rejected.append(
                    {
                        "left": left_nodes[row].local_id,
                        "right": right_nodes[column].local_id,
                        "score": score,
                        "assignment_margin": margin,
                        "reason": reason,
                    }
                )
                continue
            iou, cosine = details[(row, column)]
            accepted.append(
                LinkEdge(
                    left_nodes[row].local_id,
                    right_nodes[column].local_id,
                    score,
                    "moss_overlap",
                    temporal_iou=iou,
                    embedding_cosine=cosine,
                    assignment_margin=margin,
                )
            )
    return accepted, rejected


def _embedding_cosine_threshold(
    left: LocalSpeakerNode, right: LocalSpeakerNode, config: StitchConfig
) -> float:
    if left.anchor is not None and right.anchor is not None:
        if left.anchor == right.anchor:
            return config.matching_anchor_cosine
        return config.conflicting_anchor_cosine
    return config.embedding_cosine


def build_embedding_edges(
    nodes: list[LocalSpeakerNode],
    config: StitchConfig = StitchConfig(),
) -> tuple[list[LinkEdge], list[dict[str, Any]]]:
    eligible: dict[tuple[int, int], float] = {}
    rejected: list[dict[str, Any]] = []
    for left_index, left in enumerate(nodes):
        for right_index in range(left_index + 1, len(nodes)):
            right = nodes[right_index]
            if left.chunk_index == right.chunk_index:
                continue
            cosine = _cosine(left.embedding, right.embedding)
            diagnostic = {
                "left": left.local_id,
                "right": right.local_id,
                "embedding_cosine": cosine,
            }
            if abs(left.chunk_index - right.chunk_index) == 1:
                diagnostic["reason"] = "adjacent_chunks_use_overlap"
                rejected.append(diagnostic)
                continue
            if cosine is None:
                diagnostic["reason"] = "missing_embedding"
                rejected.append(diagnostic)
                continue
            chunk_span = abs(right.chunk_index - left.chunk_index)
            if chunk_span > config.long_distance_chunk_span and (
                left.anchor is None
                or right.anchor is None
                or left.anchor != right.anchor
            ):
                diagnostic["reason"] = "long_distance_without_matching_anchors"
                rejected.append(diagnostic)
                continue
            if left.anchor is None and right.anchor is None:
                diagnostic["reason"] = "both_no_anchor"
                rejected.append(diagnostic)
                continue
            if (left.anchor is None) != (right.anchor is None):
                if cosine + 1e-9 < config.asymmetric_anchor_cosine:
                    diagnostic["reason"] = "asymmetric_anchor_low_confidence"
                    rejected.append(diagnostic)
                    continue
            threshold = _embedding_cosine_threshold(left, right, config)
            if cosine + 1e-9 < threshold:
                diagnostic["threshold"] = threshold
                diagnostic["reason"] = (
                    "conflicting_anchors"
                    if (
                        left.anchor is not None
                        and right.anchor is not None
                        and left.anchor != right.anchor
                    )
                    else "below_dynamic_cosine_threshold"
                )
                rejected.append(diagnostic)
                continue
            eligible[(left_index, right_index)] = cosine

    incident_best: dict[int, list[tuple[tuple[int, int], float]]] = {}
    for pair, cosine in eligible.items():
        for node_index in pair:
            best = incident_best.setdefault(node_index, [])
            best.append((pair, cosine))
            best.sort(key=lambda item: (-item[1], item[0]))
            del best[2:]

    incident: dict[int, list[tuple[tuple[int, int], float]]] = {}
    for pair, cosine in eligible.items():
        incident.setdefault(pair[0], []).append((pair, cosine))
        incident.setdefault(pair[1], []).append((pair, cosine))

    preferred: dict[int, tuple[int, int]] = {}
    for node_index, candidates in incident.items():
        highest = max(cosine for _, cosine in candidates)
        compatible = [
            (pair, cosine)
            for pair, cosine in candidates
            if cosine + 0.03 + 1e-9 >= highest
            and nodes[pair[0]].anchor is not None
            and nodes[pair[0]].anchor == nodes[pair[1]].anchor
        ]
        pool = compatible or candidates
        preferred[node_index] = min(
            pool,
            key=lambda item: (
                -item[1],
                nodes[item[0][1] if item[0][0] == node_index else item[0][0]].local_id,
            ),
        )[0]

    accepted: list[LinkEdge] = []
    for (left_index, right_index), cosine in eligible.items():
        left, right = nodes[left_index], nodes[right_index]
        alternatives: list[float] = []
        for node_index in (left_index, right_index):
            alternative = next(
                (
                    other_cosine
                    for other_pair, other_cosine in incident_best.get(node_index, [])
                    if other_pair != (left_index, right_index)
                ),
                None,
            )
            if alternative is not None:
                alternatives.append(alternative)
        margin = cosine - max(alternatives) if alternatives else cosine + 1.0
        compatible_near_tie = (
            bool(alternatives)
            and abs(margin) <= 0.03 + 1e-9
            and left.anchor is not None
            and left.anchor == right.anchor
        )
        mutual_winner = (
            preferred.get(left_index) == (left_index, right_index)
            and preferred.get(right_index) == (left_index, right_index)
        )
        if not mutual_winner or (
            margin is not None
            and margin + 1e-9 < config.embedding_margin
            and not compatible_near_tie
        ):
            rejected.append(
                {
                    "left": left.local_id,
                    "right": right.local_id,
                    "embedding_cosine": cosine,
                    "margin": margin,
                    "reason": "ambiguous_embedding",
                }
            )
            continue
        accepted.append(
            LinkEdge(
                left.local_id,
                right.local_id,
                cosine,
                "embedding",
                embedding_cosine=cosine,
                assignment_margin=margin,
            )
        )
    return accepted, rejected


class _Components:
    def __init__(self, identifiers: list[str]) -> None:
        self.parent = {identifier: identifier for identifier in identifiers}

    def find(self, identifier: str) -> str:
        while self.parent[identifier] != identifier:
            self.parent[identifier] = self.parent[self.parent[identifier]]
            identifier = self.parent[identifier]
        return identifier

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def members(self, identifier: str) -> set[str]:
        root = self.find(identifier)
        return {item for item in self.parent if self.find(item) == root}


def _component_centroid(
    identifiers: set[str],
    by_id: dict[str, LocalSpeakerNode],
) -> np.ndarray | None:
    vectors = []
    for identifier in sorted(identifiers):
        vector = _embedding(by_id[identifier].embedding)
        if vector is None:
            continue
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            continue
        vectors.append(vector / norm)
    if not vectors:
        return None
    centroid = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0:
        return None
    return centroid / norm


def _component_centroid_outlier(
    identifiers: set[str],
    by_id: dict[str, LocalSpeakerNode],
    config: StitchConfig,
) -> tuple[str, float] | None:
    if len(identifiers) < 2:
        return None
    centroid = _component_centroid(identifiers, by_id)
    if centroid is None:
        return None
    worst_id: str | None = None
    worst_distance = -1.0
    for identifier in sorted(identifiers):
        cosine = _cosine(by_id[identifier].embedding, centroid)
        if cosine is None:
            continue
        distance = 1.0 - cosine
        if distance > worst_distance + 1e-12 or (
            abs(distance - worst_distance) <= 1e-12
            and (worst_id is None or identifier < worst_id)
        ):
            worst_distance = distance
            worst_id = identifier
    if worst_id is None or worst_distance <= config.centroid_outlier_distance + 1e-9:
        return None
    return worst_id, worst_distance


def _component_violation(
    identifiers: set[str],
    by_id: dict[str, LocalSpeakerNode],
    config: StitchConfig,
) -> tuple[str, dict[str, Any]] | None:
    chunk_indices = [by_id[item].chunk_index for item in identifiers]
    if len(set(chunk_indices)) != len(chunk_indices):
        return "same_chunk_cannot_link", {}
    maximum, _, pair = _component_distances(identifiers, by_id)
    if maximum > config.cluster_max_distance + 1e-9:
        return "cluster_max_distance", {"max_distance": maximum, "pair": pair}
    outlier = _component_centroid_outlier(identifiers, by_id, config)
    if outlier is not None:
        outlier_id, distance = outlier
        return "centroid_outlier", {
            "outlier": outlier_id,
            "centroid_distance": distance,
        }
    return None


def _component_distances(
    identifiers: set[str],
    by_id: dict[str, LocalSpeakerNode],
) -> tuple[float, float, list[str] | None]:
    maximum = 0.0
    pair: list[str] | None = None
    distances_by_node: dict[str, list[float]] = {
        identifier: [] for identifier in identifiers
    }
    ordered = sorted(identifiers)
    for index, left_id in enumerate(ordered):
        for right_id in ordered[index + 1 :]:
            cosine = _cosine(by_id[left_id].embedding, by_id[right_id].embedding)
            if cosine is None:
                continue
            distance = 1.0 - cosine
            distances_by_node[left_id].append(distance)
            distances_by_node[right_id].append(distance)
            if distance > maximum:
                maximum, pair = distance, [left_id, right_id]
    medoid_candidates = [
        (sum(distances), identifier, max(distances))
        for identifier, distances in distances_by_node.items()
        if distances
    ]
    medoid_distance = (
        min(medoid_candidates, key=lambda item: (item[0], item[1]))[2]
        if medoid_candidates
        else 0.0
    )
    return maximum, medoid_distance, pair


def cluster_nodes(
    nodes: list[LocalSpeakerNode],
    edges: list[LinkEdge],
    config: StitchConfig = StitchConfig(),
) -> tuple[dict[str, str], dict[str, Any]]:
    by_id = {node.local_id: node for node in nodes}
    components = _Components(list(by_id))
    accepted: list[LinkEdge] = []
    rejected: list[dict[str, Any]] = []
    for edge in sorted(
        edges, key=lambda item: (-item.strength, item.method, item.left, item.right)
    ):
        if edge.left not in by_id or edge.right not in by_id:
            rejected.append({**_edge_dict(edge), "reason": "unknown_node"})
            continue
        if components.find(edge.left) == components.find(edge.right):
            continue
        proposed = components.members(edge.left) | components.members(edge.right)
        violation = _component_violation(proposed, by_id, config)
        if violation is not None:
            reason, diagnostic = violation
            rejected.append(
                {
                    **_edge_dict(edge),
                    "reason": reason,
                    "removed_edge": {"left": edge.left, "right": edge.right},
                    **diagnostic,
                }
            )
            continue
        components.union(edge.left, edge.right)
        accepted.append(edge)

    # Safety validation: drop the weakest edge that sustains a diameter/outlier trap.
    while True:
        groups = {
            frozenset(components.members(identifier)) for identifier in sorted(by_id)
        }
        offending_group = None
        offending_violation: tuple[str, dict[str, Any]] | None = None
        for group in groups:
            violation = _component_violation(set(group), by_id, config)
            if violation is not None:
                offending_group = group
                offending_violation = violation
                break
        if offending_group is None or offending_violation is None:
            break
        reason, diagnostic = offending_violation
        candidates = [
            edge
            for edge in accepted
            if edge.left in offending_group and edge.right in offending_group
        ]
        outlier_id = diagnostic.get("outlier")
        if reason == "centroid_outlier" and outlier_id is not None:
            outlier_edges = [
                edge
                for edge in candidates
                if outlier_id in (edge.left, edge.right)
            ]
            if outlier_edges:
                candidates = outlier_edges
        if not candidates:
            break
        weakest = min(
            candidates, key=lambda item: (item.strength, item.method, item.left, item.right)
        )
        accepted.remove(weakest)
        rejected.append(
            {
                **_edge_dict(weakest),
                "reason": "safety_edge_removed",
                "violation": reason,
                "removed_edge": {
                    "left": weakest.left,
                    "right": weakest.right,
                    "strength": weakest.strength,
                },
                **diagnostic,
            }
        )
        components = _Components(list(by_id))
        for edge in accepted:
            components.union(edge.left, edge.right)

    groups = [
        sorted(group)
        for group in {
            frozenset(components.members(identifier)) for identifier in sorted(by_id)
        }
    ]
    groups.sort(
        key=lambda group: (
            min(
                (current.start for item in group for current in by_id[item].turns),
                default=math.inf,
            ),
            min(group),
        )
    )
    mapping = {
        local_id: f"speaker_{speaker_index}"
        for speaker_index, group in enumerate(groups)
        for local_id in group
    }
    diagnostics = []
    for group in groups:
        violation = _component_violation(set(group), by_id, config)
        maximum_pairwise, maximum_medoid, _ = _component_distances(
            set(group), by_id
        )
        diagnostics.append(
            {
                "speaker_id": mapping[group[0]],
                "local_ids": group,
                "valid": violation is None,
                "maximum_pairwise_distance": maximum_pairwise,
                "maximum_medoid_distance": maximum_medoid,
            }
        )
    return mapping, {
        "assignment_method": "scipy_linear_sum_assignment",
        "accepted_edges": [_edge_dict(edge) for edge in accepted],
        "rejected_edges": rejected,
        "clusters": diagnostics,
    }


def _longest_silence_gap(
    activity: np.ndarray, frame_hop: float
) -> tuple[float, float] | None:
    """Return (gap_start_frames, gap_end_frames) for the longest False run."""
    if activity.size == 0 or np.all(activity):
        return None
    best_start = best_end = -1
    cursor = 0
    n_frames = int(activity.size)
    while cursor < n_frames:
        if activity[cursor]:
            cursor += 1
            continue
        start = cursor
        while cursor < n_frames and not activity[cursor]:
            cursor += 1
        if cursor - start > best_end - best_start:
            best_start, best_end = start, cursor
    if best_end <= best_start:
        return None
    return best_start * frame_hop, best_end * frame_hop


def _min_activity_cut_offset(
    activity: np.ndarray, frame_hop: float
) -> float:
    """Pick the lowest local-activity frame; ties break toward the midpoint."""
    n_frames = int(activity.size)
    if n_frames == 0:
        return 0.0
    if n_frames == 1:
        return 0.0
    # Soft local density: count active frames in a ±2-frame neighborhood.
    radius = min(2, n_frames - 1)
    padded = np.pad(activity.astype(np.int32), radius, mode="edge")
    density = np.array(
        [
            int(np.sum(padded[index : index + 2 * radius + 1]))
            for index in range(n_frames)
        ],
        dtype=np.int32,
    )
    midpoint = (n_frames - 1) / 2.0
    best_index = min(
        range(n_frames),
        key=lambda index: (int(density[index]), abs(index - midpoint), index),
    )
    return best_index * frame_hop


def _ownership_boundary(
    overlap_start: float,
    overlap_end: float,
    activity_turns: list[Turn] | None,
    frame_hop: float,
) -> float:
    midpoint = (overlap_start + overlap_end) / 2.0
    if not activity_turns or overlap_end <= overlap_start:
        return midpoint
    activity = _paint(activity_turns, overlap_start, overlap_end, frame_hop)
    silence = _longest_silence_gap(activity, frame_hop)
    if silence is not None:
        gap_start, gap_end = silence
        return overlap_start + (gap_start + gap_end) / 2.0
    return overlap_start + _min_activity_cut_offset(activity, frame_hop)


def chunk_ownership_spans(
    chunks: list[ChunkWindow],
    activity_turns: list[Turn] | None = None,
    frame_hop: float = 0.02,
) -> list[dict[str, Any]]:
    spans = [
        {
            "chunk_index": index,
            "start": float(chunk.start),
            "end": float(chunk.end),
        }
        for index, chunk in enumerate(chunks)
    ]
    for index in range(len(chunks) - 1):
        if chunks[index].end <= chunks[index + 1].start:
            continue
        overlap_start = max(chunks[index].start, chunks[index + 1].start)
        overlap_end = min(chunks[index].end, chunks[index + 1].end)
        boundary = _ownership_boundary(
            overlap_start, overlap_end, activity_turns, frame_hop
        )
        spans[index]["end"] = float(boundary)
        spans[index + 1]["start"] = float(boundary)
    return spans


def reconcile_chunk_ownership(
    raw_turns: list[Turn],
    mapping: dict[str, str],
    ownership_spans: list[dict[str, Any]],
) -> list[Turn]:
    spans = {int(span["chunk_index"]): span for span in ownership_spans}
    clipped: list[Turn] = []
    for current in raw_turns:
        chunk_index = parse_chunk_index(current.speaker_id)
        if chunk_index not in spans or current.speaker_id not in mapping:
            continue
        span = spans[chunk_index]
        start = max(current.start, float(span["start"]))
        end = min(current.end, float(span["end"]))
        if end <= start:
            continue
        clipped.append(
            replace(
                current,
                start=start,
                end=end,
                speaker_id=mapping[current.speaker_id],
            )
        )
    clipped.sort(key=lambda item: (item.speaker_id, item.start, item.end))
    merged: list[Turn] = []
    for current in clipped:
        if (
            merged
            and merged[-1].speaker_id == current.speaker_id
            and current.start <= merged[-1].end + 1e-9
            and not merged[-1].text
            and not current.text
            and merged[-1].asr_status == current.asr_status
            and merged[-1].source == current.source
        ):
            previous = merged[-1]
            merged[-1] = replace(
                previous,
                end=max(previous.end, current.end),
                text="",
                asr_status=previous.asr_status,
                source=previous.source,
                confidence=min(previous.confidence, current.confidence),
            )
        else:
            merged.append(current)
    return sorted(merged, key=lambda item: (item.start, item.end, item.speaker_id))


def detect_exploded_chunks(
    moss_turns: list[Turn],
    diarizen_turns: list[Turn],
    chunks: list[ChunkWindow],
    ownership_spans: list[dict[str, Any]],
    config: StitchConfig = StitchConfig(),
) -> list[dict[str, Any]]:
    spans = {int(span["chunk_index"]): span for span in ownership_spans}
    exploded: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        moss_ids = {
            current.speaker_id
            for current in moss_turns
            if parse_chunk_index(current.speaker_id) == chunk_index
            and current.end > chunk.start
            and current.start < chunk.end
        }
        diarizen_ids = {
            current.speaker_id
            for current in diarizen_turns
            if current.end > chunk.start and current.start < chunk.end
        }
        n_moss, n_diarizen = len(moss_ids), len(diarizen_ids)
        if (
            n_moss > config.explosion_abs_cap
            and n_moss > config.explosion_ratio * max(n_diarizen, 1)
        ):
            span = spans.get(
                chunk_index,
                {
                    "start": float(chunk.start),
                    "end": float(chunk.end),
                },
            )
            exploded.append(
                {
                    "chunk_index": chunk_index,
                    "chunk": {"start": float(chunk.start), "end": float(chunk.end)},
                    "owned_span": {
                        "start": float(span["start"]),
                        "end": float(span["end"]),
                    },
                    "n_moss": n_moss,
                    "n_diarizen": n_diarizen,
                    "moss_local_ids": sorted(moss_ids),
                    "diarizen_speakers": sorted(diarizen_ids),
                }
            )
    return exploded


def stitch_identities(
    moss_turns: list[Turn],
    chunks: list[ChunkWindow],
    diarizen_turns: list[Turn],
    moss_embeddings: dict[str, list[float] | np.ndarray],
    diarizen_centroids: dict[str, list[float] | np.ndarray],
    config: StitchConfig = StitchConfig(),
) -> StitchResult:
    nodes = build_nodes(moss_turns, moss_embeddings)
    compute_diarizen_anchors(nodes, diarizen_turns, diarizen_centroids, config)
    adjacent_edges, adjacent_rejected = match_adjacent_chunks(nodes, chunks, config)
    embedding_edges, embedding_rejected = build_embedding_edges(nodes, config)
    mapping, cluster_metadata = cluster_nodes(
        nodes, adjacent_edges + embedding_edges, config
    )
    ownership_spans = chunk_ownership_spans(
        chunks, moss_turns, config.frame_hop
    )
    turns = reconcile_chunk_ownership(moss_turns, mapping, ownership_spans)
    exploded = detect_exploded_chunks(
        moss_turns, diarizen_turns, chunks, ownership_spans, config
    )
    rejected_adjacent_candidates = adjacent_rejected + [
        item
        for item in cluster_metadata["rejected_edges"]
        if item.get("method") == "moss_overlap"
    ]
    rejected_adjacent_switches = {
        tuple(sorted((item["left"], item["right"])))
        for item in rejected_adjacent_candidates
        if item.get("score", item.get("strength", -math.inf)) + 1e-9
        >= config.overlap_link_score
        and item.get("left") in mapping
        and item.get("right") in mapping
        and mapping[item["left"]] != mapping[item["right"]]
    }
    accepted_edges = cluster_metadata["accepted_edges"]
    rejected_edges = (
        adjacent_rejected
        + embedding_rejected
        + cluster_metadata["rejected_edges"]
    )
    assignments: dict[str, dict[str, Any]] = {}
    for node in nodes:
        incident_edges = [
            edge
            for edge in accepted_edges
            if node.local_id in (edge["left"], edge["right"])
        ]
        if incident_edges:
            evidence = max(
                incident_edges,
                key=lambda edge: (
                    float(edge["strength"]),
                    str(edge["method"]),
                    str(edge["left"]),
                    str(edge["right"]),
                ),
            )
            method = str(evidence["method"])
            confidence = float(evidence["strength"])
        elif node.anchor is not None:
            method = "diarizen_anchor"
            confidence = float(node.anchor_purity)
        else:
            method = "singleton"
            confidence = 0.0
        assignments[node.local_id] = {
            "global_id": mapping[node.local_id],
            "method": method,
            "confidence": confidence,
            "diarizen_anchor": node.anchor,
            "diarizen_purity": float(node.anchor_purity),
        }
    metadata: dict[str, Any] = {
        "version": "hierarchical_v2",
        "config": asdict(config),
        "assignments": assignments,
        "assignment_method": cluster_metadata["assignment_method"],
        "edges": {"accepted": accepted_edges, "rejected": rejected_edges},
        "cluster_diagnostics": cluster_metadata["clusters"],
        "ownership_spans": ownership_spans,
        "exploded_chunks": exploded,
        "cross_chunk_identity_switch_count": len(rejected_adjacent_switches),
        "anchors": {
            node.local_id: {
                "speaker": node.anchor,
                "purity": node.anchor_purity,
            }
            for node in nodes
        },
    }
    exploded_spans = [
        (
            float(item["owned_span"]["start"]),
            float(item["owned_span"]["end"]),
        )
        for item in exploded
    ]
    return StitchResult(turns, mapping, metadata, exploded_spans)
