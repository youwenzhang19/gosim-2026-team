"""Reversible, prequential calibration for observed telescope exposures.

OnlineOutcomeCalibrator predicts only the positive-hit fraction among
previously selected, completed exposures. ScienceGainCalibrator estimates the
realized positive best-score increment of an executed observation. Neither
class estimates unselected-candidate effects, aggregate science utility, or
REQUIRED/request completion probabilities.

The hit-fraction model is always shadow-only. The science-gain model may return
a bounded correction for a matching condition only after enough mature
examples and better prequential squared loss; unsupported conditions use the
uncorrected gain. Both models retain frozen inputs and replay after retraction.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone


MAX_RECORDS = 256
MIN_CONDITION_SUPPORT = 8
LOG_CLIP = 1e-6
PROGRAMS = ("DARK", "BRIGHT", "BACKUP")
DIRECTION_NAMES = {
    "N": 0, "NE": 1, "E": 2, "SE": 3,
    "S": 4, "SW": 5, "W": 6, "NW": 7,
}
GAIN_FACTOR_MIN = 0.8
GAIN_FACTOR_MAX = 1.2
GAIN_LOG_RATIO_MIN = math.log(GAIN_FACTOR_MIN)
GAIN_LOG_RATIO_MAX = math.log(GAIN_FACTOR_MAX)
GAIN_PRIOR_STRENGTH = 1.0
EXPOSURE_BUCKETS = ((600, "le_600s"), (900, "601_900s"), (float("inf"), "gt_900s"))
COUNT_BUCKETS = ((8, "1_8"), (32, "9_32"), (float("inf"), "33_plus"))


def _timestamp_epoch(value, field):
    if isinstance(value, bool):
        raise ValueError(field + " must be a timestamp")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        result = dt.timestamp()
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            raise ValueError(field + " must be an ISO timestamp") from None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        result = dt.timestamp()
    else:
        raise ValueError(field + " must be a timestamp")
    if not math.isfinite(result):
        raise ValueError(field + " must be finite")
    return result


def _timestamp_text(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _finite_number(value, field, *, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ValueError(field + " must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(field + " must be finite numeric data") from None
    if not math.isfinite(result):
        raise ValueError(field + " must be finite numeric data")
    if minimum is not None and result < minimum:
        raise ValueError(field + " is below its supported range")
    if maximum is not None and result > maximum:
        raise ValueError(field + " is above its supported range")
    return result


def _optional_hit_fraction(value):
    if value is None:
        return None
    return _finite_number(value, "base_probability", minimum=0.0, maximum=1.0)


def _action_index(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("action_index must be a nonnegative integer")
    return value


def _epoch_key(value, default):
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("epoch must be a nonempty string or integer")
    if isinstance(value, str) and not value.strip():
        raise ValueError("epoch must be a nonempty string or integer")
    return value


def _input_hash(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("input_hash must be a nonempty string")
    return value


def _direction_bucket(features):
    value = features.get("direction_bucket", features.get("sector", features.get("direction")))
    if isinstance(value, str):
        return DIRECTION_NAMES.get(value.upper())
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 8:
        return None
    return value


def _positive_features(features):
    if not isinstance(features, Mapping):
        return None
    direction = _direction_bucket(features)
    program = features.get("program")
    if direction is None or program not in PROGRAMS:
        return None
    return {"direction_bucket": direction, "program": program}


def _science_features(features):
    if not isinstance(features, Mapping):
        return None
    base = _positive_features(features)
    if base is None:
        return None
    try:
        seconds = _finite_number(features.get("exposure_seconds"), "exposure_seconds", minimum=1.0)
        count_value = features.get("assignments_count", features.get("assigned_count"))
        if isinstance(count_value, bool) or not isinstance(count_value, int) or count_value <= 0:
            return None
    except ValueError:
        return None
    exposure_bucket = next(name for upper, name in EXPOSURE_BUCKETS if seconds <= upper)
    count_bucket = next(name for upper, name in COUNT_BUCKETS if count_value <= upper)
    return {
        **base,
        "exposure_seconds": seconds,
        "assignments_count": count_value,
        "exposure_bucket": exposure_bucket,
        "count_bucket": count_bucket,
    }


def _half_open_index_window(start, end):
    return (
        isinstance(start, int) and not isinstance(start, bool)
        and isinstance(end, int) and not isinstance(end, bool)
        and max(abs(start), abs(end)) < 100_000_000
    )


def _brier(prediction, observed):
    return prediction * prediction - 2.0 * prediction * observed + observed


def _mean(total, count):
    return total / count if count else None


class OnlineOutcomeCalibrator:
    """Shadow-predict selected-exposure positive hit fractions.

    ``base_probability`` keeps the prototype API name, but its meaning is
    strictly a positive-hit-fraction baseline for completed selected
    exposures. When omitted, it is the same-epoch Beta(1, 1) estimate from
    labels received before the prediction time. ``predict`` returns only this
    shadow estimate; the class exposes no action-enabling method or gate.
    """

    def __init__(self, max_records=MAX_RECORDS, min_condition_support=MIN_CONDITION_SUPPORT):
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records <= 0:
            raise ValueError("max_records must be a positive integer")
        if isinstance(min_condition_support, bool) or not isinstance(min_condition_support, int) \
                or min_condition_support <= 0:
            raise ValueError("min_condition_support must be a positive integer")
        self.max_records = max_records
        self.min_condition_support = min_condition_support
        self._records = {}
        self._last_action_index = None
        self._epoch_generation = 0
        self._replay()

    @property
    def action_gate_enabled(self):
        return False

    def freeze(self, action_index, base_probability, features, issued_at, end_time,
               input_hash=None, *, epoch=None):
        """Freeze one selected exposure before its public result is known.

        No action object is accepted. ``action_index`` is the monotone observe
        index; ``input_hash`` and ``epoch`` identify the public input lineage.
        """
        index = _action_index(action_index)
        if index in self._records or (
            self._last_action_index is not None and index <= self._last_action_index
        ):
            raise ValueError("action_index must be unique and increasing")
        issue_epoch = _timestamp_epoch(issued_at, "issued_at")
        end_epoch = _timestamp_epoch(end_time, "end_time")
        if end_epoch < issue_epoch:
            raise ValueError("end_time must be at or after issued_at")
        record_epoch = _epoch_key(epoch, self._epoch_generation)
        record = {
            "action_index": index,
            "epoch": record_epoch,
            "input_hash": _input_hash(input_hash),
            "base_hit_fraction": None,
            "shadow_hit_fraction": None,
            "features": _positive_features(features),
            "issued_at": _timestamp_text(issue_epoch),
            "end_time": _timestamp_text(end_epoch),
            "label": None,
            "received_at": None,
            "active_at_issue": False,
            "_provided_base": _optional_hit_fraction(base_probability),
            "_issue_epoch": issue_epoch,
            "_end_epoch": end_epoch,
            "_received_epoch": None,
        }
        self._records[index] = record
        self._last_action_index = index
        while len(self._records) > self.max_records:
            del self._records[next(iter(self._records))]
        self._replay()
        return self._public_record(self._records[index])

    def label(self, action_index, y, received_at, public=True):
        """Attach a mature public hit fraction; incomplete labels do not update."""
        try:
            index = _action_index(action_index)
        except ValueError:
            return False
        if public is not True or isinstance(y, bool) or not isinstance(y, (int, float)):
            return False
        observed = float(y)
        if not math.isfinite(observed) or not 0.0 <= observed <= 1.0:
            return False
        record = self._records.get(index)
        if record is None or record["label"] is not None:
            return False
        received_epoch = _timestamp_epoch(received_at, "received_at")
        if received_epoch < record["_end_epoch"]:
            return False
        record["label"] = observed
        record["received_at"] = _timestamp_text(received_epoch)
        record["_received_epoch"] = received_epoch
        self._replay()
        return True

    def retract(self, action_index):
        """Remove one exposure and replay all retained predictions."""
        try:
            index = _action_index(action_index)
        except ValueError:
            return False
        if index not in self._records:
            return False
        del self._records[index]
        self._replay()
        return True

    def retract_window(self, start, end):
        """Retract a half-open observe-index window or an overlapping time window."""
        index_window = _half_open_index_window(start, end)
        if index_window:
            if end < start:
                raise ValueError("retraction window end must be at or after start")
            removed = [i for i in self._records if start <= i < end]
        else:
            start_epoch = _timestamp_epoch(start, "window.start")
            end_epoch = _timestamp_epoch(end, "window.end")
            if end_epoch < start_epoch:
                raise ValueError("retraction window end must be at or after start")
            removed = [
                i for i, row in self._records.items()
                if row["_issue_epoch"] < end_epoch and row["_end_epoch"] > start_epoch
            ]
        for index in removed:
            del self._records[index]
        if removed:
            self._replay()
        return len(removed)

    def reset(self):
        """Discard history and start a new implicit epoch."""
        self._records.clear()
        self._last_action_index = None
        self._epoch_generation += 1
        self._replay()

    def predict(self, base_probability, features, *, epoch=None, as_of=None):
        """Return the selected-exposure hit-fraction shadow estimate.

        Unsupported conditions return the supplied or same-epoch Beta(1, 1)
        base unchanged. ``as_of`` can bound replay to labels received strictly
        before a historical prediction time.
        """
        prediction_epoch = _epoch_key(epoch, self._epoch_generation)
        as_of_epoch = None if as_of is None else _timestamp_epoch(as_of, "as_of")
        eligible = [
            row for row in self._records.values()
            if row["epoch"] == prediction_epoch and row["label"] is not None
            and row["_received_epoch"] >= row["_end_epoch"]
            and (as_of_epoch is None or row["_received_epoch"] < as_of_epoch)
        ]
        base = _optional_hit_fraction(base_probability)
        if base is None:
            base = (1.0 + sum(row["label"] for row in eligible)) / (2.0 + len(eligible))
        clean = _positive_features(features)
        if clean is None:
            return base
        group = [row for row in eligible if row["features"] == clean]
        if len(group) < self.min_condition_support:
            return base
        return (2.0 * base + sum(row["label"] for row in group)) / (2.0 + len(group))

    def summary(self, epoch=None, *, include_records=False):
        selected_epoch = None if epoch is None else _epoch_key(epoch, self._epoch_generation)
        rows = [row for row in self._records.values()
                if selected_epoch is None or row["epoch"] == selected_epoch]
        labeled = [row for row in rows if row["label"] is not None]
        by_epoch = {}
        for row in labeled:
            key = str(row["epoch"])
            summary = by_epoch.setdefault(key, {"count": 0, "sum_labels": 0.0})
            summary["count"] += 1
            summary["sum_labels"] += row["label"]
        for values in by_epoch.values():
            values["base_hit_fraction"] = (1.0 + values["sum_labels"]) / (2.0 + values["count"])
        metric_rows = [value for key, value in self._metrics.items()
                       if selected_epoch is None or key == selected_epoch]
        count = sum(row["count"] for row in metric_rows)
        base_brier = _mean(sum(row["base_brier_sum"] for row in metric_rows), count)
        shadow_brier = _mean(sum(row["shadow_brier_sum"] for row in metric_rows), count)
        base_hit_fraction = None
        if selected_epoch is not None and str(selected_epoch) in by_epoch:
            base_hit_fraction = by_epoch[str(selected_epoch)]["base_hit_fraction"]
        elif selected_epoch is None and len(by_epoch) == 1:
            base_hit_fraction = next(iter(by_epoch.values()))["base_hit_fraction"]
        result = {
            "object_kind": "selected_exposure_positive_hit_fraction",
            "estimand": "positive hit fraction among completed selected exposures",
            "record_count": len(rows),
            "labeled_exposure_count": len(labeled),
            "sum_labels": sum(row["label"] for row in labeled),
            "base_hit_fraction": base_hit_fraction,
            "epochs": by_epoch,
            "max_records": self.max_records,
            "minimum_condition_support": self.min_condition_support,
            "base_brier_mean": base_brier,
            "shadow_brier_mean": shadow_brier,
            "active": any(row["active"] for row in metric_rows),
            "active_scope": "diagnostic_only_no_action_promotion",
            "action_gate_enabled": False,
        }
        if include_records:
            result["records"] = [self._public_record(row) for row in rows]
        return result

    def records(self, epoch=None):
        selected_epoch = None if epoch is None else _epoch_key(epoch, self._epoch_generation)
        return [
            self._public_record(row) for row in self._records.values()
            if selected_epoch is None or row["epoch"] == selected_epoch
        ]

    @staticmethod
    def _public_record(record):
        return {key: copy.deepcopy(value) for key, value in record.items()
                if not key.startswith("_")}

    @staticmethod
    def _diagnostic_active(metrics, minimum):
        return bool(metrics["count"] >= minimum
                    and metrics["shadow_brier_sum"] < metrics["base_brier_sum"])

    def _replay(self):
        totals = {}
        groups = {}
        metrics = {}
        events = []
        for index, row in self._records.items():
            events.append((row["_issue_epoch"], 0, index, "freeze", row))
            if row["label"] is not None:
                events.append((row["_received_epoch"], 1, index, "label", row))
        events.sort(key=lambda event: (event[0], event[2], event[1]))
        for _time, _priority, _index, kind, row in events:
            epoch = row["epoch"]
            if kind == "freeze":
                total = totals.get(epoch, [0, 0.0])
                base = row["_provided_base"]
                if base is None:
                    base = (1.0 + total[1]) / (2.0 + total[0])
                clean = row["features"]
                shadow = base
                if clean is not None:
                    group = groups.get((epoch, clean["direction_bucket"], clean["program"]), [0, 0.0])
                    if group[0] >= self.min_condition_support:
                        shadow = (2.0 * base + group[1]) / (2.0 + group[0])
                row["base_hit_fraction"] = base
                row["shadow_hit_fraction"] = shadow
                row["active_at_issue"] = self._diagnostic_active(
                    metrics.get(epoch, self._empty_metrics()), self.min_condition_support)
                continue
            score = metrics.setdefault(epoch, self._empty_metrics())
            y = row["label"]
            score["count"] += 1
            score["base_brier_sum"] += _brier(row["base_hit_fraction"], y)
            score["shadow_brier_sum"] += _brier(row["shadow_hit_fraction"], y)
            total = totals.setdefault(epoch, [0, 0.0])
            total[0] += 1
            total[1] += y
            clean = row["features"]
            if clean is not None:
                group = groups.setdefault((epoch, clean["direction_bucket"], clean["program"]), [0, 0.0])
                group[0] += 1
                group[1] += y
        for score in metrics.values():
            score["active"] = self._diagnostic_active(score, self.min_condition_support)
        self._metrics = metrics

    @staticmethod
    def _empty_metrics():
        return {"count": 0, "base_brier_sum": 0.0, "shadow_brier_sum": 0.0, "active": False}


class ScienceGainCalibrator:
    """Calibrate realized score increments for executed observe actions only.

    The label is the sum over frozen assigned targets of
    ``max(0, hit_score - old_best_score)``; assigned targets absent from the
    public hit list contribute zero. A condition uses the same program,
    direction, exposure-time bucket, and assignment-count bucket. Its bounded
    ratio is exposed only after at least eight mature positive-base examples
    and better prequential squared loss than the uncorrected base.

    This is not a forecast for an unselected candidate or a counterfactual.
    """

    def __init__(self, max_records=MAX_RECORDS, min_condition_support=MIN_CONDITION_SUPPORT):
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records <= 0:
            raise ValueError("max_records must be a positive integer")
        if isinstance(min_condition_support, bool) or not isinstance(min_condition_support, int) \
                or min_condition_support <= 0:
            raise ValueError("min_condition_support must be a positive integer")
        self.max_records = max_records
        self.min_condition_support = min_condition_support
        self._records = {}
        self._last_action_index = None
        self._epoch_generation = 0
        self._last_gate_enabled = False
        self._replay()

    @property
    def action_gate_enabled(self):
        return self._last_gate_enabled

    def freeze(self, action_index, base_gain, features, old_best_scores, assigned_ids,
               issued_at, end_time, input_hash=None):
        """Freeze one selected observe action and its pre-action score baselines."""
        index = _action_index(action_index)
        if index in self._records or (
            self._last_action_index is not None and index <= self._last_action_index
        ):
            raise ValueError("action_index must be unique and increasing")
        issue_epoch = _timestamp_epoch(issued_at, "issued_at")
        end_epoch = _timestamp_epoch(end_time, "end_time")
        if end_epoch < issue_epoch:
            raise ValueError("end_time must be at or after issued_at")
        base = _finite_number(base_gain, "base_gain", minimum=0.0)
        assigned = self._normalize_assigned_ids(assigned_ids)
        old_best = self._normalize_old_best(old_best_scores, assigned)
        clean = _science_features(features)
        record = {
            "action_index": index,
            "epoch": self._epoch_generation,
            "input_hash": _input_hash(input_hash),
            "base_gain": base,
            "features": clean,
            "old_best_scores": old_best,
            "assigned_ids": list(assigned),
            "issued_at": _timestamp_text(issue_epoch),
            "end_time": _timestamp_text(end_epoch),
            "observed_score_increment": None,
            "received_at": None,
            "ratio_factor_at_issue": 1.0,
            "shadow_gain_at_issue": base,
            "predicted_gain": base,
            "action_gate_enabled_at_issue": False,
            "_issue_epoch": issue_epoch,
            "_end_epoch": end_epoch,
            "_received_epoch": None,
        }
        self._records[index] = record
        self._last_action_index = index
        while len(self._records) > self.max_records:
            del self._records[next(iter(self._records))]
        self._replay()
        return self._public_record(self._records[index])

    def label(self, action_index, result, received_at):
        """Attach a complete public observe result for the frozen assignment."""
        try:
            index = _action_index(action_index)
        except ValueError:
            return False
        row = self._records.get(index)
        if row is None or row["observed_score_increment"] is not None:
            return False
        if not isinstance(result, Mapping) or result.get("action") != "observe":
            return False
        if result.get("observe_index") != index:
            return False
        count = result.get("assigned_count")
        hits_count = result.get("hit_count")
        hits = result.get("hits")
        if isinstance(count, bool) or not isinstance(count, int) or count != len(row["assigned_ids"]):
            return False
        if isinstance(hits_count, bool) or not isinstance(hits_count, int) \
                or not 0 <= hits_count <= count or not isinstance(hits, list) or hits_count != len(hits):
            return False
        hit_scores = {}
        assigned = set(row["assigned_ids"])
        for hit in hits:
            if not isinstance(hit, Mapping) or "target_id" not in hit or "score" not in hit:
                return False
            try:
                target = self._normalize_target_id(hit["target_id"])
            except ValueError:
                return False
            if target not in assigned or target in hit_scores:
                return False
            try:
                hit_scores[target] = _finite_number(hit["score"], "hit score", minimum=0.0)
            except ValueError:
                return False
        observed = sum(
            max(0.0, hit_scores.get(target, 0.0) - row["old_best_scores"][target])
            for target in row["assigned_ids"]
        )
        received_epoch = _timestamp_epoch(received_at, "received_at")
        if received_epoch < row["_end_epoch"]:
            return False
        row["observed_score_increment"] = observed
        row["received_at"] = _timestamp_text(received_epoch)
        row["_received_epoch"] = received_epoch
        self._replay()
        return True

    def retract(self, action_index):
        try:
            index = _action_index(action_index)
        except ValueError:
            return False
        if index not in self._records:
            return False
        del self._records[index]
        self._replay()
        return True

    def retract_window(self, start, end):
        """Retract a half-open observe-index window or an overlapping time window."""
        if _half_open_index_window(start, end):
            if end < start:
                raise ValueError("retraction window end must be at or after start")
            removed = [i for i in self._records if start <= i < end]
        else:
            start_epoch = _timestamp_epoch(start, "window.start")
            end_epoch = _timestamp_epoch(end, "window.end")
            if end_epoch < start_epoch:
                raise ValueError("retraction window end must be at or after start")
            removed = [
                i for i, row in self._records.items()
                if row["_issue_epoch"] < end_epoch and row["_end_epoch"] > start_epoch
            ]
        for index in removed:
            del self._records[index]
        if removed:
            self._replay()
        return len(removed)

    def reset(self):
        self._records.clear()
        self._last_action_index = None
        self._epoch_generation += 1
        self._last_gate_enabled = False
        self._replay()

    def predict_gain(self, base_gain, features, *, record_gate=True):
        """Return a bounded selected-action score-increment estimate."""
        base = _finite_number(base_gain, "base_gain", minimum=0.0)
        clean = _science_features(features)
        if clean is None or base <= 0.0:
            if record_gate:
                self._last_gate_enabled = False
            return base
        state = self._group_state(self._epoch_generation, clean)
        enabled = self._gate_open(state)
        factor = self._ratio_factor(state) if enabled else 1.0
        if record_gate:
            self._last_gate_enabled = enabled
        return base * factor

    def summary(self, *, include_records=False):
        labeled = [row for row in self._records.values()
                   if row["observed_score_increment"] is not None]
        groups = []
        gate_enabled = False
        for (epoch, direction, program, exposure_bucket, count_bucket), state in sorted(
            self._group_metrics.items(), key=lambda item: tuple(str(part) for part in item[0])
        ):
            if epoch != self._epoch_generation:
                continue
            open_gate = self._gate_open(state)
            gate_enabled = gate_enabled or open_gate
            groups.append({
                "direction_bucket": direction,
                "program": program,
                "exposure_bucket": exposure_bucket,
                "count_bucket": count_bucket,
                "support": state["count"],
                "ratio_factor": self._ratio_factor(state),
                "base_squared_loss_mean": _mean(state["base_squared_loss_sum"], state["count"]),
                "shadow_squared_loss_mean": _mean(state["shadow_squared_loss_sum"], state["count"]),
                "action_gate_enabled": open_gate,
            })
        all_base = sum(row["base_squared_loss_sum"] for row in self._group_metrics.values())
        all_shadow = sum(row["shadow_squared_loss_sum"] for row in self._group_metrics.values())
        support = sum(row["count"] for row in self._group_metrics.values())
        result = {
            "object_kind": "selected_action_observed_score_increment",
            "record_count": len(self._records),
            "labeled_exposure_count": len(labeled),
            "sum_labels": sum(row["observed_score_increment"] for row in labeled),
            "positive_base_labeled_count": support,
            "base_squared_loss_mean": _mean(all_base, support),
            "shadow_squared_loss_mean": _mean(all_shadow, support),
            "minimum_condition_support": self.min_condition_support,
            "ratio_factor_bounds": [GAIN_FACTOR_MIN, GAIN_FACTOR_MAX],
            "exposure_bucket_seconds": ["<=600", "601-900", ">900"],
            "assignment_count_buckets": ["1-8", "9-32", "33+"],
            "action_gate_enabled": gate_enabled,
            "last_prediction_gate_enabled": self._last_gate_enabled,
            "effect_unverified": True,
            "policy_support": "executed selected actions only; no unselected or counterfactual effects",
            "groups": groups,
        }
        if include_records:
            result["records"] = self.records()
        return result

    def records(self):
        """Return retained frozen and labeled records on explicit request."""
        return [self._public_record(row) for row in self._records.values()]

    @staticmethod
    def _normalize_target_id(value):
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("target id must be a string or integer")
        text = str(value)
        if not text:
            raise ValueError("target id must not be empty")
        return text

    @classmethod
    def _normalize_assigned_ids(cls, values):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
            raise ValueError("assigned_ids must be a nonempty sequence")
        normalized = [cls._normalize_target_id(value) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("assigned_ids must be unique")
        return tuple(normalized)

    @classmethod
    def _normalize_old_best(cls, values, assigned):
        if not isinstance(values, Mapping):
            raise ValueError("old_best_scores must be a mapping")
        normalized = {}
        for key, value in values.items():
            target = cls._normalize_target_id(key)
            if target in normalized:
                raise ValueError("old_best_scores has duplicate normalized target ids")
            normalized[target] = value
        result = {}
        for target in assigned:
            if target not in normalized:
                raise ValueError("old_best_scores must cover every assigned target")
            result[target] = _finite_number(normalized[target], "old_best_score", minimum=0.0)
        return result

    @staticmethod
    def _public_record(record):
        return {key: copy.deepcopy(value) for key, value in record.items()
                if not key.startswith("_")}

    @staticmethod
    def _group_key(epoch, features):
        if features is None:
            return None
        return (
            epoch,
            features["direction_bucket"],
            features["program"],
            features["exposure_bucket"],
            features["count_bucket"],
        )

    def _group_state(self, epoch, features):
        key = self._group_key(epoch, features)
        return self._group_metrics.get(key, self._empty_group_metrics()) if key is not None \
            else self._empty_group_metrics()

    @staticmethod
    def _empty_group_metrics():
        return {
            "count": 0,
            "log_ratio_sum": 0.0,
            "base_squared_loss_sum": 0.0,
            "shadow_squared_loss_sum": 0.0,
        }

    def _ratio_factor(self, state):
        if state["count"] <= 0:
            return 1.0
        mean_log_ratio = state["log_ratio_sum"] / (state["count"] + GAIN_PRIOR_STRENGTH)
        return min(GAIN_FACTOR_MAX, max(GAIN_FACTOR_MIN, math.exp(mean_log_ratio)))

    def _gate_open(self, state):
        return bool(
            state["count"] >= self.min_condition_support
            and state["shadow_squared_loss_sum"] < state["base_squared_loss_sum"]
        )

    def _replay(self):
        self._group_metrics = {}
        events = []
        for index, row in self._records.items():
            events.append((row["_issue_epoch"], 0, index, "freeze", row))
            if row["observed_score_increment"] is not None:
                events.append((row["_received_epoch"], 1, index, "label", row))
        events.sort(key=lambda event: (event[0], event[2], event[1]))
        for _time, _priority, _index, kind, row in events:
            key = self._group_key(row["epoch"], row["features"])
            state = self._group_metrics.setdefault(key, self._empty_group_metrics()) if key else None
            if kind == "freeze":
                factor = self._ratio_factor(state) if state else 1.0
                enabled = self._gate_open(state) if state else False
                row["ratio_factor_at_issue"] = factor if enabled else 1.0
                row["shadow_gain_at_issue"] = row["base_gain"] * factor
                row["predicted_gain"] = row["base_gain"] * (factor if enabled else 1.0)
                row["action_gate_enabled_at_issue"] = enabled
                continue
            base = row["base_gain"]
            observed = row["observed_score_increment"]
            if state is None or base <= 0.0:
                continue
            diagnostic_factor = self._ratio_factor(state)
            state["base_squared_loss_sum"] += (observed - base) ** 2
            state["shadow_squared_loss_sum"] += (observed - base * diagnostic_factor) ** 2
            ratio = observed / base
            ratio = min(GAIN_FACTOR_MAX, max(GAIN_FACTOR_MIN, ratio))
            state["log_ratio_sum"] += math.log(ratio)
            state["count"] += 1
        self._last_gate_enabled = False


GainCalibrator = ScienceGainCalibrator
