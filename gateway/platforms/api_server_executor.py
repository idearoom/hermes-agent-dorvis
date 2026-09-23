"""Size the event loop's default executor to the API server's admission cap.

The API server runs every agent turn with ``loop.run_in_executor(None, ...)``
and its auth checks and session-DB calls with ``asyncio.to_thread``. Both use
the loop's default executor, which asyncio creates lazily with
``min(32, os.process_cpu_count() + 4)`` workers: 8 on a 4-vCPU task. A turn
holds its thread for minutes, so without this module at most 8 turns ran at
once however high ``gateway.api_server.max_concurrent_runs`` was set, and once
8 were running every ``to_thread`` call for a new request queued behind them.

``install_api_server_default_executor`` replaces that pool, once per loop,
with one that fits the cap plus headroom for the short ``to_thread`` work.
The loop owns the installed executor: ``asyncio.run`` shuts it down through
``loop.shutdown_default_executor()`` when the gateway exits, after drain and
blue/green shutdown have finished the in-flight turns. The adapter never shuts
it down on ``disconnect()``, because a reconnect builds a new adapter on the
same loop and other components keep using the default executor.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import weakref

logger = logging.getLogger(__name__)

# Threads beyond the run cap, for asyncio.to_thread callers (auth verification,
# session-DB opens and reads, payload builds) that must not wait on turns.
EXECUTOR_HEADROOM = 16
# Never smaller than the stock asyncio ceiling.
EXECUTOR_FLOOR = 32
# max_concurrent_runs <= 0 disables the cap; the pool still needs a bound.
UNCAPPED_EXECUTOR_WORKERS = 64

THREAD_NAME_PREFIX = "hermes-api-server"
# asyncio's lazily created default executor uses this prefix.
_ASYNCIO_DEFAULT_PREFIX = "asyncio"

_installed: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, concurrent.futures.ThreadPoolExecutor]" = (
    weakref.WeakKeyDictionary()
)
_installed_lock = threading.Lock()


def api_server_executor_workers(max_concurrent_runs: int) -> int:
    """Return the default-executor size for a resolved run cap."""
    if max_concurrent_runs <= 0:
        return UNCAPPED_EXECUTOR_WORKERS
    return max(EXECUTOR_FLOOR, max_concurrent_runs + EXECUTOR_HEADROOM)


def _executor_is_live(executor: concurrent.futures.ThreadPoolExecutor) -> bool:
    return not getattr(executor, "_shutdown", False)


def install_api_server_default_executor(
    loop: asyncio.AbstractEventLoop,
    max_concurrent_runs: int,
) -> concurrent.futures.ThreadPoolExecutor:
    """Install a cap-sized default executor on *loop*, at most once per size.

    Keeps the executor already installed by this function when it is still
    the loop's default, still live, and at least as large as needed, so
    adapter reconnects are no-ops. Otherwise installs a new one. The replaced
    executor is shut down without waiting (work already submitted still runs
    to completion) when this module or asyncio created it; an executor that
    someone else installed is left for its owner to shut down.
    """
    workers = api_server_executor_workers(max_concurrent_runs)
    with _installed_lock:
        previous = getattr(loop, "_default_executor", None)
        ours = _installed.get(loop)
        if (
            ours is not None
            and previous is ours
            and _executor_is_live(ours)
            and ours._max_workers >= workers
        ):
            return ours

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=THREAD_NAME_PREFIX,
        )
        loop.set_default_executor(executor)
        _installed[loop] = executor

    if previous is not None and previous is not executor:
        prefix = getattr(previous, "_thread_name_prefix", "")
        if previous is ours or prefix.startswith(_ASYNCIO_DEFAULT_PREFIX):
            previous.shutdown(wait=False)

    logger.info(
        "API server default executor: %d workers (max_concurrent_runs=%d, "
        "headroom=%d, replaced=%s)",
        workers,
        max_concurrent_runs,
        EXECUTOR_HEADROOM,
        "none" if previous is None else getattr(previous, "_max_workers", "?"),
    )
    return executor
