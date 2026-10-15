"""The tier counts on the box-list rows (A2; A3 decision
13 shares the module with Step 4).

Each Session and Curation row shows how many labels of the patch sit in
each triage tier -- ``good / review / bad / garbage / curated`` -- as the
features CSV
ON DISK says, refreshed when the disk changes (an extraction finishes,
Commit Triage or Save Curation writes, a folder load, a run switch, the step
entered, a Forget or a stale-QC delete), never live while tiers move in the
plot.

Two parts:

* the pure counter :func:`tier_counts_from_csv` and the record helper
  :func:`tier_items` (no Qt) -- Step 3's five tiers by default; a caller
  gives ``order`` (the words, in drawing order) and ``bucket_of`` (the rule
  that maps one ``qc_tier`` value to one of them) to count something else:
  Step 4's good / bad / unmarked through ``wb_tier_bucket`` (A3);
* the worker, on the ``_step3_thumbs.py`` pattern: ONE batch per trigger
  (:func:`start_tier_counts` stops the older batch first), the thread never
  parented, ``finished`` AND ``error`` connected to ``quit``, a thread still
  running after :data:`STOP_WAIT_MS` parked in :data:`_LIVE_COUNT_THREADS`,
  stopped by :func:`stop_tier_counts` on step exit and in ``shutdown()``.
  Qt is imported only when a worker is built.

The batch carries a caller's *stamp* (the workspace, the run, its epoch);
the controller discards a result whose stamp is no longer current.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: the tiers counted, in the order EVERY tier line draws them (A3 fold,
#: owner 2026-10-10: curated first, then good, review, bad, garbage) -- the
#: one constant the rows, the counts and the tooltips follow
TIERS = ("curated", "good", "review", "bad", "garbage")
#: label statuses that are not active (the ghosts)
INACTIVE_STATUSES = ("deleted", "superseded")
#: the tier column (the old name ``tier`` is read when ``qc_tier`` is absent)
TIER_COLUMNS = ("qc_tier", "tier")
#: how long a step exit / shutdown waits for a running batch (Step 4's rule)
STOP_WAIT_MS = 3000

#: threads still running after their stop timed out (never destroyed while
#: running); a thread leaves on its own ``finished``
_LIVE_COUNT_THREADS: set = set()


def _diag(msg: str) -> None:
    try:
        print(f"[STEP3-COUNTS] {msg}", flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pure
# ---------------------------------------------------------------------------

def tier_counts_from_csv(path, *, order: Sequence[str] = TIERS,
                         bucket_of=None) -> Optional[Dict[str, int]]:
    """``{tier: n}`` of the ACTIVE rows of a features CSV (``label_status``
    not deleted / superseded; a CSV without the column counts every row),
    for the words of *order* (default the five :data:`TIERS`; zeros
    included); ``None`` when there is no file, no tier column, or the file
    cannot be read.  Reads two columns only.

    With *bucket_of* (a function of one ``qc_tier`` value, NaN included)
    each distinct value is mapped to a word of *order* and the rows are
    counted per word; without it the tier values are counted as they are."""
    import pandas as pd

    try:
        p = Path(path)
        if not p.is_file():
            return None
        wanted = set(TIER_COLUMNS) | {"label_status"}
        df = pd.read_csv(p, usecols=lambda c: c in wanted)
    except Exception as exc:      # noqa: BLE001 -- a row's decoration
        _diag(f"could not read {path}: {type(exc).__name__}: {exc}")
        return None
    col = next((c for c in TIER_COLUMNS if c in df.columns), None)
    if col is None:
        if bucket_of is None:
            return None
        # (audit L3) a bucket rule counts a table with no tier column as
        # all unmarked -- a fresh Measure may write none
        try:
            head = pd.read_csv(p, nrows=0).columns
            keep = {head[0], "label_status"} if len(head) else set()
            df = pd.read_csv(p, usecols=lambda c: c in keep)
        except Exception as exc:      # noqa: BLE001 -- a row's decoration
            _diag(f"could not count {path}: {type(exc).__name__}: {exc}")
            return None
        tiers = pd.Series([None] * len(df), dtype=object, index=df.index)
    else:
        tiers = df[col]
    if "label_status" in df.columns:
        tiers = tiers[~df["label_status"].isin(INACTIVE_STATUSES)]
    if bucket_of is None:
        vc = tiers.value_counts()
        return {t: int(vc.get(t, 0)) for t in order}
    out = {t: 0 for t in order}
    vc = tiers.value_counts(dropna=False)
    for value, n in vc.items():
        word = bucket_of(None if value != value else value)
        if word in out:
            out[word] += int(n)
    return out


def tier_items(counts: Optional[Mapping[str, int]],
               colours: Mapping[str, Sequence[int]], *,
               order: Sequence[str] = TIERS
               ) -> Optional[Tuple[Tuple[str, int, Tuple[int, int, int]], ...]]:
    """The row record's ``tier_counts``: ``((word, n, (r, g, b)), ...)`` in
    :data:`TIERS` order, from *counts* and the plot's tier colours
    (``qc_plot_widget._TIER_COLORS``, passed in -- this module never copies
    them).  ``None`` for no counts, or when every tier is 0 (a patch whose
    CSV has no tiered label yet shows nothing)."""
    if not counts:
        return None
    out = []
    for t in order:
        n = int(counts.get(t, 0) or 0)
        rgb = tuple(int(v) for v in colours.get(t, (153, 153, 153)))[:3]
        out.append((t, n, rgb))
    if not any(n for _t, n, _c in out):
        return None
    return tuple(out)


def count_requests(csv_for, names: Iterable[str]) -> List[Tuple[str, str]]:
    """``[(patch, csv path), ...]`` -- built on the main thread from
    ``csv_for(name) -> path`` (path joins only; no file is touched)."""
    out = []
    for n in names:
        try:
            out.append((str(n), str(csv_for(n))))
        except Exception as exc:      # noqa: BLE001
            _diag(f"no CSV path for {n}: {exc!r}")
    return out


def count_batch(requests: Iterable[Tuple[str, str]], cancel=None, *,
                order: Sequence[str] = TIERS, bucket_of=None
                ) -> Dict[str, Optional[Dict[str, int]]]:
    """The worker's body: ``{patch: counts or None}``; stops early (with
    what it has) when *cancel* (a ``threading.Event``) is set."""
    out: Dict[str, Optional[Dict[str, int]]] = {}
    for name, path in requests:
        if cancel is not None and cancel.is_set():
            break
        out[name] = tier_counts_from_csv(path, order=order,
                                         bucket_of=bucket_of)
    return out


# ---------------------------------------------------------------------------
# The worker (Qt imported on first use)
# ---------------------------------------------------------------------------

_CLASSES = None


def _classes():
    """``(_TierCountWorker, _TierCountThread)`` -- built once, on first use,
    so importing this module needs no Qt."""
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    from qtpy.QtCore import QObject, QThread, Signal, Slot

    class _TierCountWorker(QObject):
        """Counts one batch, then ends.  Every exit of :meth:`run` emits
        ``finished`` (``try/finally``); a bug also emits ``error`` first.
        ``counted(stamp, {patch: counts or None})`` carries the result."""

        counted = Signal(object, object)
        error = Signal(str)
        finished = Signal()

        def __init__(self, requests, stamp, parent=None, *,
                     order=TIERS, bucket_of=None):
            super().__init__(parent)
            self._requests = list(requests)
            self._stamp = stamp
            self._order = tuple(order)
            self._bucket_of = bucket_of
            self._cancel = threading.Event()

        def cancel(self) -> None:
            self._cancel.set()

        @Slot()
        def run(self) -> None:
            try:
                if self._order == TIERS and self._bucket_of is None:
                    # Step 3's own call, exactly as it was
                    result = count_batch(self._requests, self._cancel)
                else:
                    result = count_batch(self._requests, self._cancel,
                                         order=self._order,
                                         bucket_of=self._bucket_of)
                if not self._cancel.is_set():
                    self.counted.emit(self._stamp, result)
            except Exception as exc:      # a bug, not a bad CSV
                _diag(f"run error: {type(exc).__name__}: {exc}")
                self.error.emit(f"{type(exc).__name__}: {exc}")
            finally:
                self.finished.emit()

    class _TierCountThread(QThread):
        """A never-parented thread that holds its worker, records that
        ``quit`` was reached, and leaves :data:`_LIVE_COUNT_THREADS` on its
        own ``finished``."""

        quit_called = False
        _count_worker = None

        @Slot()
        def quit(self) -> None:  # noqa: A003 - Qt's name
            self.quit_called = True
            super().quit()

        @Slot()
        def _forget(self) -> None:
            self.wait(2000)
            _LIVE_COUNT_THREADS.discard(self)

    _CLASSES = (_TierCountWorker, _TierCountThread)
    return _CLASSES


class TierCountState:
    """The running batch, if any (``thread`` / ``worker``).  The controller
    keeps ONE and hands it to :func:`start_tier_counts` and
    :func:`stop_tier_counts`."""

    __slots__ = ("thread", "worker")

    def __init__(self):
        self.thread = None
        self.worker = None

    def running(self) -> bool:
        th = self.thread
        if th is None:
            return False
        try:
            return bool(th.isRunning())
        except RuntimeError:
            return False


_SLOTS = ("_on_counted", "_on_finished", "_on_error")


def _on_count_error(message: str) -> None:
    _diag(f"worker error: {message}")


def start_tier_counts(requests, stamp, on_counted, *,
                      state: Optional[TierCountState] = None,
                      on_finished=None, on_error=None,
                      order: Sequence[str] = TIERS,
                      bucket_of=None) -> TierCountState:
    """Start ONE batch -> the state.  The state's older batch is stopped
    first (cancelled, its slots disconnected, not waited on: a thread still
    reading its last CSV stays parked).  No requests: the older batch is
    still stopped and nothing starts.

    ``on_counted(stamp, {patch: counts or None})`` arrives on the main
    thread; pass bound methods, never lambdas (a slot held only by PyQt can
    lose its closure when napari garbage-collects during a layer removal).
    They are
    connected BEFORE ``start()``.  ``order`` / ``bucket_of`` are
    :func:`tier_counts_from_csv`'s (Step 4: its three buckets)."""
    if state is None:
        state = TierCountState()
    elif state.thread is not None or state.worker is not None:
        stop_tier_counts(state, wait_ms=0)
    requests = list(requests or ())
    if not requests:
        return state
    from qtpy.QtCore import Qt

    Worker, Thread = _classes()
    thread = Thread()
    worker = Worker(requests, stamp, order=order, bucket_of=bucket_of)
    worker.moveToThread(thread)
    thread._count_worker = worker
    for attr, sig, fn in (
            ("_on_counted", worker.counted, on_counted),
            ("_on_finished", worker.finished, on_finished),
            ("_on_error", worker.error,
             on_error if on_error is not None else _on_count_error)):
        setattr(worker, attr, fn)
        if fn is not None:
            sig.connect(fn)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit, Qt.DirectConnection)
    worker.error.connect(thread.quit, Qt.DirectConnection)
    thread.finished.connect(thread._forget)
    _LIVE_COUNT_THREADS.add(thread)
    state.thread, state.worker = thread, worker
    thread.start()
    _diag(f"{len(requests)} patch(es) to count")
    return state


def stop_tier_counts(state: Optional[TierCountState],
                     wait_ms: int = STOP_WAIT_MS) -> bool:
    """Stop the state's batch -> True when no thread is left running.
    Cancels, DISCONNECTS the caller's slots (a result already queued can no
    longer land), quits and waits *wait_ms*; a thread still running then
    stays parked in :data:`_LIVE_COUNT_THREADS`.  The state is cleared.
    Idempotent."""
    if state is None:
        return True
    thread, worker = state.thread, state.worker
    state.thread = state.worker = None
    if worker is not None:
        try:
            worker.cancel()
        except RuntimeError:
            pass
        for attr, sig_name in zip(_SLOTS, ("counted", "finished", "error")):
            fn = getattr(worker, attr, None)
            if fn is None:
                continue
            try:
                getattr(worker, sig_name).disconnect(fn)
            except (TypeError, RuntimeError):
                pass
    try:
        if thread is None or not thread.isRunning():
            return True
        thread.quit()
        ok = bool(thread.wait(int(wait_ms))) if wait_ms else False
        if not ok:
            _diag("stop: a batch still running -- parked")
        return ok
    except RuntimeError:
        return True
