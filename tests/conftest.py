"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _keep_session_log_out_of_the_home_folder(tmp_path_factory):
    """Send any crash log to a temp folder and keep the console mirror off.

    Building the Step 1 widget installs the crash logger; without this it
    would write into, and prune, the developer's real log folder.
    """
    mp = pytest.MonkeyPatch()
    mp.setenv("EPICYCLE_CRASH_LOG_DIR",
              str(tmp_path_factory.mktemp("crash_logs")))
    mp.setenv("EPICYCLE_SESSION_LOG", "0")
    yield
    mp.undo()
