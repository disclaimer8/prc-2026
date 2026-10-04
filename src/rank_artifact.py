"""Fail-closed checkpointing and atomic publication for E130 ranking models."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_output_path(mode, features_path, override=None):
    legacy = Path(f"submissions/e130_{mode}_rank.npz")
    output = Path(override) if override else legacy
    features = Path(features_path)
    if any("fin1b" in name.lower() for name in (features.name, features.resolve().name)):
        if not override or output.resolve() == legacy.resolve():
            raise ValueError("FIN1b ranking requires explicit E130_OUT with a distinct path "
                             f"(for example submissions/e130_{mode}_fin1b_rank.npz); "
                             "E130_OVERWRITE does not bypass this protection")
    if output.suffix != ".npz":
        raise ValueError("E130_OUT must end in .npz")
    return output


def validate_rank_ids(mvt_id, expected_rows):
    ids = np.asarray(mvt_id)
    if ids.ndim != 1 or len(ids) != expected_rows:
        raise ValueError(f"ranking requires exactly {expected_rows} one-dimensional IDs")
    if ids.dtype.kind not in "iuf" or not np.all(np.isfinite(ids)):
        raise ValueError("ranking IDs must be finite integers")
    if ids.dtype.kind == "f" and (np.any(np.abs(ids) > 2**53) or not np.all(ids == np.floor(ids))):
        raise ValueError("ranking IDs must be exact integers in the safe float range")
    if len(np.unique(ids)) != expected_rows:
        raise ValueError("ranking IDs must be unique")
    return ids


class RankArtifactWriter:
    """One writer per output; final filenames only ever contain complete runs.

    A lock surviving SIGKILL is intentionally not reclaimed automatically.
    Checkpoints are diagnostics, never candidates for final publication.
    """
    def __init__(self, output, expected_seeds, rounds, features_path, mode,
                 provenance=None, expected_rows=344841, overwrite=False):
        if overwrite:
            raise ValueError("E130_OVERWRITE=1 is disabled because replacing an existing final "
                             "can race legacy writers; unset it and choose a fresh E130_OUT")
        for name, value in (("expected_seeds", expected_seeds), ("rounds", rounds),
                            ("expected_rows", expected_rows)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.output = Path(output).resolve()
        if self.output.suffix != ".npz":
            raise ValueError("ranking output must end in .npz")
        self.checkpoint_path = self.output.with_name(self.output.stem + ".checkpoint.npz")
        self.lock_path = self.output.with_name(self.output.name + ".lock")
        self.expected_seeds = expected_seeds
        self.rounds = rounds
        self.expected_rows = expected_rows
        self.features_path = Path(features_path).resolve()
        self.mode = mode
        self.provenance = dict(provenance or {})
        self._lock_fd = None
        self._published = False

    def __enter__(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(self._lock_fd, f"pid={os.getpid()}\n".encode())
            if self.output.exists():
                raise FileExistsError(f"final output already exists: {self.output}; "
                                      "choose a fresh E130_OUT; in-place overwrite is disabled")
            self.features_sha256 = file_sha256(self.features_path)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
            self.lock_path.unlink()

    def _payload(self, mvt_id, per_seed, complete):
        if self._lock_fd is None or self._published:
            raise RuntimeError("ranking writer must be active and not already published")
        ids = validate_rank_ids(mvt_id, self.expected_rows)
        predictions = np.asarray(per_seed)
        if predictions.ndim != 2 or predictions.shape[1] != self.expected_rows:
            raise ValueError("per_seed predictions must have shape (completed_seeds, ranking_rows)")
        completed = predictions.shape[0]
        if not 1 <= completed <= self.expected_seeds or (complete and completed != self.expected_seeds):
            raise ValueError(f"expected {self.expected_seeds} seeds, received {completed}")
        if predictions.dtype.kind not in "iuf" or not np.all(np.isfinite(predictions)):
            raise ValueError("ranking predictions must be finite numbers")
        # Preserve the original seed-mean calculation, including its dtype.
        reg = np.mean(predictions, axis=0)
        if not np.all(np.isfinite(reg)):
            raise ValueError("ranking mean predictions must be finite")
        return dict(mvt_id=ids, reg=reg, per_seed=predictions,
                    complete=complete, status="complete" if complete else "checkpoint",
                    expected_seeds=self.expected_seeds, completed_seeds=completed,
                    expected_rows=self.expected_rows, features_path=str(self.features_path),
                    features_sha256=self.features_sha256, mode=self.mode, rounds=self.rounds,
                    provenance_json=json.dumps(self.provenance, sort_keys=True, allow_nan=False))

    def _save(self, path, payload, replace):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                             dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())
            if replace:
                os.replace(temporary, path)
            else:
                # Atomic no-clobber: unlike exists()+rename(), link cannot race an arrival.
                os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def checkpoint(self, mvt_id, per_seed):
        self._save(self.checkpoint_path, self._payload(mvt_id, per_seed, complete=False), replace=True)

    def publish(self, mvt_id, per_seed):
        payload = self._payload(mvt_id, per_seed, complete=True)
        if file_sha256(self.features_path) != self.features_sha256:
            raise ValueError("features source changed during ranking; choose an immutable source and fresh E130_OUT")
        self._save(self.output, payload, replace=False)
        self._published = True
