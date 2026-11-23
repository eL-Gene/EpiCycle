"""A finish handler must know WHICH worker finished without Qt's ``sender()``.

Root cause (traced 2026-11-14, a live Add whole-patch Preview that died with
exit 139 inside ``_on_add_finished -> _result_token -> _signal_sender ->
self.sender()``).  For a slot that is a Python callable, PyQt answers
``sender()`` from ONE pointer shared by the whole process: each slot call
saves it, sets its own sender and writes the saved value back on its way
out.  A worker's ``run`` is such a slot call too, in the WORKER thread (the
thread's ``started -> worker.run``): it saves whatever the main thread had
when the thread started and writes it back when ``run`` returns -- just
after ``finished`` is emitted, while the main thread may be inside the
finish handler (a ``print`` lets the worker thread in).  ``sender()`` then
names the object that was the sender when the thread started: a stranger
(no token, so the stale gate abstains) or a deleted one (an access violation
inside ``sender()``).

The fix tags each worker / thread connection with its sender
(``curation_3d_widget._from_sender``); ``_signal_sender()`` reads the tag
and never calls Qt's ``sender()``.  The end-to-end tests below drive the
REAL ``_launch_add_worker`` / ``_on_add_finished`` against a real QThread
and force the race deterministically; the access-violation variant runs in
a subprocess so the old code fails the test instead of killing pytest.
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from napari_epicycle.annotator import curation_3d_widget as CW
from napari_epicycle.annotator import reshape_worker as RW

from .add_harness import FakeThread, Harness, install_fakes

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------- #
# the race, through the real Add launch and finish handler
# --------------------------------------------------------------------------- #
def race(dead_decoy: bool, qapp=None):
    """Run one Add Preview whose finish lands while the worker thread's
    ``run`` slot writes back the sender it saved at thread start (a decoy).

    Returns ``(tokens the stale gate was handed, the launch token)``.
    """
    from qtpy import sip
    from qtpy.QtCore import QObject, Signal
    from qtpy.QtWidgets import QApplication

    from packages.epicycle_annotator import add_preview as AP

    app = qapp or QApplication.instance() or QApplication([])
    run_started = threading.Event()
    main_in_handler = threading.Event()

    class RacingWorker(RW.RefineWorker):
        def run(self):          # entered through PyQt's slot proxy
            run_started.set()
            self.finished.emit(RW.RefineOutcome(
                result=AP.AddPreviewResult(status="cancelled"), kind="add"))
            main_in_handler.wait(5)
            # returning: the proxy writes back the sender it saved at start

    class QHarness(Harness, QObject):
        def __init__(self, *a, **k):
            QObject.__init__(self)
            Harness.__init__(self, *a, **k)
            del self.sender            # Qt's own sender(), as the widget has
            self.gate_tokens = []
            self._launch_t_value = None

        # the real seam and token reader, not the harness's stand-ins
        _signal_sender = CW.Curation3DAnnotator._signal_sender
        _result_token = CW.Curation3DAnnotator._result_token

        # ``_on_add_finished`` reads this FIRST: hold the main thread here,
        # GIL released (as the handler's print does), while the worker
        # thread's slot returns and writes its saved sender back
        @property
        def _add_launch_t(self):
            if self._launch_t_value is not None and run_started.is_set():
                main_in_handler.set()
                time.sleep(0.4)
            return self._launch_t_value

        @_add_launch_t.setter
        def _add_launch_t(self, v):
            self._launch_t_value = v

        def _add_is_stale(self, result_token=None):
            self.gate_tokens.append(result_token)
            return True                 # discard: nothing else to draw

    old_worker = RW.RefineWorker
    RW.RefineWorker = RacingWorker
    try:
        vol = np.zeros((6, 12, 12), np.uint32)
        h = QHarness(vol)
        h.place_box((slice(0, 6), slice(0, 12), slice(0, 12)))

        class Decoy(QObject):
            poke = Signal()

        decoy = Decoy()

        def launch():                   # a slot whose sender is the decoy
            h._launch_add_worker({"preset": "nuclei"})
            assert run_started.wait(5), "the worker never started"

        decoy.poke.connect(launch)
        decoy.poke.emit()
        if dead_decoy:
            sip.delete(decoy)
            del decoy
            gc.collect()
        th = h._add_thread
        t0 = time.time()
        while time.time() - t0 < 10 and (
                not h.gate_tokens or h._add_thread is not None):
            app.processEvents()
            time.sleep(0.01)
        if th is not None:
            th.wait(3000)
        return list(h.gate_tokens), int(h._add_token)
    finally:
        RW.RefineWorker = old_worker


def test_the_finish_handler_reads_the_finishing_workers_token(qapp):
    tokens, launch_token = race(dead_decoy=False, qapp=qapp)
    assert tokens == [launch_token], (
        "the stale gate was handed {!r}, not the finishing worker's token "
        "{} -- the handler read a stranger as its sender".format(
            tokens, launch_token))


def test_a_deleted_sender_at_thread_start_does_not_crash_the_finish():
    """The crash itself, in a child process: the decoy is deleted before
    the finish lands.  The old code faulted inside ``sender()``."""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONFAULTHANDLER"] = "1"
    code = textwrap.dedent("""
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from tests.annotator.test_worker_sender_race import race
        tokens, launch = race(dead_decoy=True)
        print("TOKENS", tokens, "LAUNCH", launch, flush=True)
    """)
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                          env=env, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, (
        "child exited {} (an access violation is 3221225477 / -1073741819)"
        "\n--- stdout ---\n{}\n--- stderr ---\n{}".format(
            proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert "TOKENS [1] LAUNCH 1" in proc.stdout, proc.stdout[-2000:]


# --------------------------------------------------------------------------- #
# the stale gate still works on the tag (Qt-free harness)
# --------------------------------------------------------------------------- #
@pytest.fixture
def h(monkeypatch):
    from .add_harness import nuclei_volume
    from .test_add_nuclei_golden import box_of

    install_fakes(monkeypatch)
    img, labels = nuclei_volume()
    hh = Harness(labels, images={"nuc": img})
    hh.place_box(box_of(labels.shape))
    # the REAL seam: the tag, not the harness's hand-set ``sender``
    monkeypatch.setattr(Harness, "_signal_sender",
                        CW.Curation3DAnnotator._signal_sender)
    return hh


def test_a_replaced_workers_late_finish_is_still_discarded(h):
    h._preview_add()
    th1, w1 = FakeThread.instances[-1], h._add_worker
    h._add_abandon_workers()            # Close mid-run ...
    h._preview_add()                    # ... and a new Preview on the box
    th2, w2 = FakeThread.instances[-1], h._add_worker
    assert w2 is not w1 and h._add_token == 2
    # the first worker's late answer, as if its cancel came too late (a
    # cancelled run takes the stopped path and never reaches the gate)
    w1.cancel_event.clear()
    seen = []
    real_gate = h._add_is_stale

    def gate(result_token=None):
        seen.append(result_token)
        return real_gate(result_token)

    h._add_is_stale = gate
    th1.run()
    assert seen == [1], "the gate must be handed the FIRST worker's token"
    assert h._add_preview_result is None, (
        "a replaced worker's late result was drawn")
    assert h.rendered == []
    th2.run()                           # the live one still lands
    assert h._add_preview_result is not None


def test_the_thread_finished_check_names_the_thread(h):
    h._preview_add()
    th, _w = FakeThread.instances[-1], h._add_worker
    th.run()
    seen = []
    orig = h._add_thread_cleanup

    def spy(which="preview"):
        seen.append(h._signal_sender())
        return orig(which)

    h._add_thread_cleanup = spy
    th.end()
    assert seen == [th]
    assert not h._add_busy


def test_a_gone_sender_still_carries_its_launch_token():
    """If the sender's wrapper is gone by the time its slot runs, the token
    it was launched with still reaches the gate (never an abstention)."""

    class Owner:
        _signal_sender = CW.Curation3DAnnotator._signal_sender
        _result_token = CW.Curation3DAnnotator._result_token

        def handler(self, *args):
            self.seen = (self._signal_sender(), self._result_token(), args)

    class Sender:
        _epicycle_token = 7

    o, s = Owner(), Sender()
    slot = CW._from_sender(o, s, o.handler)
    slot("x")
    assert o.seen == (s, 7, ("x",))
    o.seen = None
    del s
    gc.collect()
    slot("y")
    assert o.seen == (None, 7, ("y",))
    assert o._signal_sender() is None, "the tag outlives its handler"


def test_the_seam_never_asks_qt(qapp):
    """Outside a tagged slot the seam answers None -- even on a QObject,
    whose ``sender()`` would read PyQt's process-wide pointer."""
    from qtpy.QtCore import QObject

    class Owner(QObject):
        _signal_sender = CW.Curation3DAnnotator._signal_sender

    o = Owner()
    o.sender = lambda: pytest.fail("Qt's sender() was called")
    assert o._signal_sender() is None


# --------------------------------------------------------------------------- #
# review of the seam (2026-11-14)
# --------------------------------------------------------------------------- #
def test_an_owner_wired_to_a_moved_worker_is_collectable(qapp):
    """PyQt never frees the slot of a signal whose sender was moved to a
    thread: a slot holding its owner strongly kept every closed annotator
    alive.  Weak owner + handler: collectable, as with a bound method."""
    import weakref

    from qtpy import sip
    from qtpy.QtCore import QCoreApplication, QEvent, QObject, QThread, Signal

    class Owner:
        def handler(self, *a):
            self.got = a

    class W(QObject):
        sig = Signal(object)

    def wire(bound):
        o, th, w = Owner(), QThread(), W()
        w.moveToThread(th)
        if bound:
            w.sig.connect(o.handler)
        else:
            w.sig.connect(CW._from_sender(o, w, o.handler))
        ref = weakref.ref(o)
        sip.delete(w)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        del o, w, th
        gc.collect()
        return ref

    assert wire(bound=True)() is None, "the bound-method control"
    assert wire(bound=False)() is None, (
        "an owner wired through _from_sender outlived its worker")


def test_the_slot_does_nothing_once_its_owner_is_gone():
    class Owner:
        def handler(self, *a):
            raise AssertionError("ran on a gone owner")

    class Sender:
        _epicycle_token = 3

    o, s = Owner(), Sender()
    slot = CW._from_sender(o, s, o.handler)
    del o
    gc.collect()
    assert slot("late") is None


class _Bare:
    """The real readers on a plain object (no Qt)."""
    _signal_sender = CW.Curation3DAnnotator._signal_sender
    _result_token = CW.Curation3DAnnotator._result_token
    _add_guide_is_stale = CW.Curation3DAnnotator._add_guide_is_stale
    _add_clear_current = CW.Curation3DAnnotator._add_clear_current
    _on_add_info_finished = CW.Curation3DAnnotator._on_add_info_finished
    _on_add_info_error = CW.Curation3DAnnotator._on_add_info_error
    _on_add_thread_finished = CW.Curation3DAnnotator._on_add_thread_finished

    def probe(self, *a):
        self.seen = {
            "guide_stale": self._add_guide_is_stale(),
            "clear_current": self._add_clear_current(),
            "token": self._result_token(),
        }


class _Tok:
    def __init__(self, t):
        self._epicycle_token = t


def _late_slot(owner, handler_name, token=1):
    """A slot whose sender (token ``token``) is gone before it runs."""
    old = _Tok(token)
    slot = CW._from_sender(owner, old, getattr(owner, handler_name))
    del old
    gc.collect()
    return slot


def test_a_gone_workers_late_answer_is_never_taken_for_the_current_one():
    b = _Bare()
    b._add_guide_worker = _Tok(2)
    b._add_run_guide_token = 2
    b._add_clear_worker = _Tok(2)
    b._add_clear_token = 2
    b._add_clear_phase = "counting"
    _late_slot(b, "probe")()
    assert b.seen == {"guide_stale": True, "clear_current": False,
                      "token": 1}
    # a direct call (no slot) still reads the current worker
    b.probe()
    assert b.seen["guide_stale"] is False
    assert b.seen["clear_current"] is True


def test_a_gone_info_workers_late_answer_is_ignored(capsys):
    b = _Bare()
    b._add_info_worker = _Tok(2)          # the live one: must not lend keys
    _late_slot(b, "_on_add_info_finished")(object())
    _late_slot(b, "_on_add_info_error")("boom")
    assert "late answer of a dropped worker" in capsys.readouterr().out


def test_a_gone_threads_finish_never_cleans_the_current_thread():
    b = _Bare()
    b._add_thread = object()
    cleaned = []
    b._add_thread_cleanup = lambda which="preview": cleaned.append(which)
    _late_slot(b, "_on_add_thread_finished")()
    assert cleaned == []


def test_every_wired_handler_takes_its_signals_arguments():
    """The slot passes every signal argument on (PyQt's retry with fewer
    does not reach through it): each handler wired by ``_from_sender``
    must accept its signal's full argument list."""
    import inspect
    import re

    from qtpy.QtCore import QThread

    from napari_epicycle.annotator import batch_refine_worker as BW

    src = inspect.getsource(CW)
    sites = re.findall(
        r"([\w.]+)\.(\w+)\.connect\(_from_sender\(\s*self,\s*([\w.]+),\s*"
        r"self\.(\w+)\)\)", src)
    assert len(sites) == 23, len(sites)
    by_sender = {
        "self._add_worker": RW.RefineWorker,
        "self._add_thread": QThread, "th": QThread,
        "self._batch_worker": BW.BatchRefineWorker,
        "self._seed_find_worker": RW.SeedFindWorker,
        "self._reshape_worker": RW.ReshapeWorker,
        "self._refine_worker": RW.RefineWorker,
    }
    for _tx, sig_name, sender, handler in sites:
        cls = by_sender.get(sender)
        if cls is None:                    # `w`: the guide or a RefineWorker
            cls = (RW.GuideLoadWorker if handler.startswith("_on_add_guide")
                   else RW.RefineWorker)
        sig = getattr(cls, sig_name)
        sigs = getattr(sig, "signatures", None) or (str(sig),)
        # "finished()" for a Qt signal, "PyQt_PyObject,int,int)" for ours
        inner = sigs[0].rsplit("(", 1)[-1].rstrip(")").strip()
        n = 0 if not inner else inner.count(",") + 1
        params = list(inspect.signature(
            getattr(CW.Curation3DAnnotator, handler)).parameters.values())[1:]
        pos = [p for p in params
               if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        required = [p for p in pos if p.default is p.empty]
        assert len(required) <= n <= len(pos) or any(
            p.kind == p.VAR_POSITIONAL for p in params), (
            f"{handler} takes {len(required)}..{len(pos)} positional "
            f"arguments; {cls.__name__}.{sig_name} sends {n}")
