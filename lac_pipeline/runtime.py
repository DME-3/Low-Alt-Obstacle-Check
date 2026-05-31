import argparse
import atexit
import json
import logging
import logging.handlers
import os
import signal
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, TypeVar


T = TypeVar("T")


class LockError(RuntimeError):
    """Raised when another pipeline run appears to be active."""


class MaxRuntimeExceeded(TimeoutError):
    """Raised when the process-level runtime guard expires."""


@dataclass(frozen=True)
class RuntimeSettings:
    run_id: str
    target_date: Optional[str]
    publish: bool
    target: str
    confirm_production: bool
    skip_reload: bool
    lock_file: Path
    stale_lock_seconds: int
    max_runtime_seconds: int
    log_file: Optional[Path]
    log_level: str
    http_timeout_seconds: int
    query_attempts: int
    query_retry_delay_seconds: int
    ssh_timeout_seconds: float
    tunnel_timeout_seconds: float

    @property
    def dry_run(self) -> bool:
        return not self.publish


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the nightly Low Alt Cologne ADS-B pipeline."
    )
    parser.add_argument(
        "--date",
        dest="target_date",
        help="Process this UTC/local server date instead of the default two days ago.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Upload results. Without this flag the run is a dry run and writes no DB rows.",
    )
    parser.add_argument(
        "--target",
        choices=("test", "prod"),
        default="test",
        help="Database/table target used only when --publish is set.",
    )
    parser.add_argument(
        "--confirm-production",
        action="store_true",
        help="Required together with --publish --target prod.",
    )
    parser.add_argument(
        "--skip-reload",
        action="store_true",
        help="Do not call the PythonAnywhere web app reload API after a successful publish.",
    )
    parser.add_argument(
        "--lock-file",
        default="/tmp/obstaclecheck-nightly.lock",
        help="Path to the single-run lock file.",
    )
    parser.add_argument(
        "--stale-lock-seconds",
        type=int,
        default=8 * 60 * 60,
        help="Recover locks older than this only when their PID is no longer running.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=6 * 60 * 60,
        help="Hard process runtime limit enforced with SIGALRM on Unix.",
    )
    parser.add_argument(
        "--log-file",
        help="Optional rotating log file. Cron stdout/stderr logging still works without it.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    parser.add_argument(
        "--http-timeout-seconds",
        type=int,
        default=30,
        help="Timeout for PythonAnywhere reload API calls.",
    )
    parser.add_argument(
        "--query-attempts",
        type=int,
        default=2,
        help="Bounded retry attempts for OpenSky/Trino query exceptions.",
    )
    parser.add_argument(
        "--query-retry-delay-seconds",
        type=int,
        default=60,
        help="Delay between OpenSky/Trino query retry attempts.",
    )
    parser.add_argument(
        "--ssh-timeout-seconds",
        type=float,
        default=15.0,
        help="SSH connection timeout for PythonAnywhere database tunnels.",
    )
    parser.add_argument(
        "--tunnel-timeout-seconds",
        type=float,
        default=15.0,
        help="Tunnel establishment timeout for PythonAnywhere database tunnels.",
    )
    return parser


def parse_runtime_settings(argv: Optional[list[str]] = None) -> RuntimeSettings:
    args = build_arg_parser().parse_args(argv)
    run_id = uuid.uuid4().hex[:12]
    return RuntimeSettings(
        run_id=run_id,
        target_date=args.target_date,
        publish=args.publish,
        target=args.target,
        confirm_production=args.confirm_production,
        skip_reload=args.skip_reload,
        lock_file=Path(args.lock_file),
        stale_lock_seconds=args.stale_lock_seconds,
        max_runtime_seconds=args.max_runtime_seconds,
        log_file=Path(args.log_file) if args.log_file else None,
        log_level=args.log_level,
        http_timeout_seconds=args.http_timeout_seconds,
        query_attempts=max(1, args.query_attempts),
        query_retry_delay_seconds=max(0, args.query_retry_delay_seconds),
        ssh_timeout_seconds=args.ssh_timeout_seconds,
        tunnel_timeout_seconds=args.tunnel_timeout_seconds,
    )


def configure_logging(settings: RuntimeSettings) -> None:
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s run_id=%(run_id)s %(name)s %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(_RunIdFilterHandler(stream_handler, settings.run_id))

    if settings.log_file:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            settings.log_file, maxBytes=10_000_000, backupCount=7
        )
        file_handler.setFormatter(formatter)
        root.addHandler(_RunIdFilterHandler(file_handler, settings.run_id))


class _RunIdFilterHandler(logging.Handler):
    def __init__(self, wrapped: logging.Handler, run_id: str) -> None:
        super().__init__(wrapped.level)
        self.wrapped = wrapped
        self.run_id = run_id

    def emit(self, record: logging.LogRecord) -> None:
        if not hasattr(record, "run_id"):
            record.run_id = self.run_id
        self.wrapped.emit(record)

    def setFormatter(self, fmt: logging.Formatter) -> None:  # noqa: N802
        self.wrapped.setFormatter(fmt)


class PipelineLock:
    def __init__(
        self,
        path: Path,
        run_id: str,
        stale_lock_seconds: int,
        logger: logging.Logger,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.stale_lock_seconds = stale_lock_seconds
        self.logger = logger
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recover_stale_lock()

        payload = {
            "pid": os.getpid(),
            "run_id": self.run_id,
            "created_at": int(time.time()),
            "cwd": os.getcwd(),
            "argv": sys.argv,
        }
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(str(self.path), flags, 0o644)
        except FileExistsError as exc:
            holder = self._read_lock()
            raise LockError(
                f"lock already exists at {self.path}; holder={holder}"
            ) from exc

        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        self.acquired = True
        atexit.register(self.release)
        self.logger.info("lock_acquired path=%s", self.path)

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            holder = self._read_lock()
            if holder.get("run_id") == self.run_id:
                self.path.unlink(missing_ok=True)
                self.logger.info("lock_released path=%s", self.path)
        finally:
            self.acquired = False

    def _recover_stale_lock(self) -> None:
        if not self.path.exists():
            return

        holder = self._read_lock()
        pid = holder.get("pid")
        age_seconds = time.time() - self.path.stat().st_mtime
        pid_running = isinstance(pid, int) and _pid_is_running(pid)

        if pid_running:
            return

        if age_seconds >= self.stale_lock_seconds or not pid:
            self.logger.warning(
                "recovering_stale_lock path=%s age_seconds=%.0f holder=%s",
                self.path,
                age_seconds,
                holder,
            )
            self.path.unlink(missing_ok=True)

    def _read_lock(self) -> dict:
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def install_max_runtime_guard(seconds: int, logger: logging.Logger) -> None:
    if seconds <= 0:
        return
    if not hasattr(signal, "SIGALRM"):
        logger.warning("max_runtime_guard_unavailable platform_has_no_SIGALRM")
        return

    def _handle_timeout(signum: int, frame: object) -> None:
        raise MaxRuntimeExceeded(f"max runtime exceeded after {seconds} seconds")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    logger.info("max_runtime_guard_installed seconds=%s", seconds)


@contextmanager
def stage(logger: logging.Logger, name: str) -> Iterator[None]:
    started = time.monotonic()
    logger.info("stage_start name=%s", name)
    try:
        yield
    except Exception:
        logger.exception("stage_failed name=%s", name)
        raise
    finally:
        duration = time.monotonic() - started
        logger.info("stage_end name=%s duration_seconds=%.2f", name, duration)


def retry(
    label: str,
    attempts: int,
    delay_seconds: int,
    logger: logging.Logger,
    operation: Callable[[], T],
) -> T:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "operation_failed label=%s attempt=%s attempts=%s error=%s",
                label,
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts and delay_seconds:
                time.sleep(delay_seconds)
    assert last_exc is not None
    raise last_exc

