"""
Sorbot AI Engine v3.0 — Continuous Retraining Scheduler
==========================================================
Automatically retrains the XGBoost model on fresh market data
at configurable intervals. Includes:

  - Periodic retraining (default: every 6 hours)
  - Model validation gate: new model must beat or match old metrics
  - Backup of previous model before swap
  - Hot-reload of predictor without server restart
  - Full retraining history log (JSON)
  - Graceful error handling (never crashes the server)

Usage:
  scheduler = RetrainingScheduler(predictor)
  scheduler.start()   # called automatically in FastAPI lifespan
  scheduler.stop()    # called on shutdown
"""

import json
import logging
import shutil
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_DIR, DATA_DIR,
    RETRAIN_INTERVAL_HOURS, RETRAIN_MIN_IMPROVEMENT,
    RETRAIN_FORCE_AFTER_HOURS, RETRAIN_ENABLED,
)
from ml_core.data_loader import fetch_all_timeframes
from ml_core.feature_eng import build_dataset
from ml_core.trainer import train_model

logger = logging.getLogger("sorbot.retrainer")

HISTORY_FILE = MODEL_DIR / "retrain_history.json"
BACKUP_MODEL = MODEL_DIR / "btc_model_backup.json"
BACKUP_META = MODEL_DIR / "btc_meta_backup.json"
CURRENT_MODEL = MODEL_DIR / "btc_model.json"
CURRENT_META = MODEL_DIR / "btc_meta.json"


class RetrainingScheduler:
    """
    Background scheduler that continuously retrains the model
    on fresh market data fetched from yfinance/Binance.
    """

    def __init__(self, predictor):
        """
        Args:
            predictor: The Predictor instance to hot-reload after retraining.
        """
        self._predictor = predictor
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_retrain: Optional[datetime] = None
        self._retrain_count = 0
        self._consecutive_failures = 0
        self._history: list = []
        self._lock = threading.Lock()

        # Load existing history
        self._load_history()

    # ── Public API ───────────────────────────────

    def start(self):
        """Start the background retraining loop."""
        if not RETRAIN_ENABLED:
            logger.info("Continuous retraining is DISABLED (RETRAIN_ENABLED=false)")
            return

        if self._running:
            logger.warning("Retraining scheduler already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="retrain-scheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Retraining scheduler STARTED (interval=%dh, force_after=%dh)",
            RETRAIN_INTERVAL_HOURS,
            RETRAIN_FORCE_AFTER_HOURS,
        )

    def stop(self):
        """Stop the background retraining loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("Retraining scheduler STOPPED")

    def get_status(self) -> dict:
        """Return current scheduler status for the /retrain-status endpoint."""
        with self._lock:
            old_metrics = self._get_current_model_metrics()
            return {
                "enabled": RETRAIN_ENABLED,
                "running": self._running,
                "interval_hours": RETRAIN_INTERVAL_HOURS,
                "force_after_hours": RETRAIN_FORCE_AFTER_HOURS,
                "min_improvement": RETRAIN_MIN_IMPROVEMENT,
                "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
                "total_retrains": self._retrain_count,
                "consecutive_failures": self._consecutive_failures,
                "current_model_metrics": old_metrics,
                "history_count": len(self._history),
                "recent_history": self._history[-5:] if self._history else [],
            }

    def force_retrain(self) -> dict:
        """Manually trigger a retrain cycle. Returns the result."""
        logger.info("Manual retrain triggered")
        return self._retrain_cycle(force=True)

    # ── Background Loop ──────────────────────────

    def _run_loop(self):
        """Main loop: sleep, then retrain. Runs in a daemon thread."""
        interval_seconds = RETRAIN_INTERVAL_HOURS * 3600

        # Initial delay: wait 60s after startup to let things settle
        logger.info("Retraining loop: waiting 60s for startup to complete...")
        self._interruptible_sleep(60)

        while self._running:
            try:
                result = self._retrain_cycle(force=False)
                status = result.get("status", "unknown")
                if status == "success":
                    self._consecutive_failures = 0
                    logger.info("Retrain cycle succeeded. Next in %dh.", RETRAIN_INTERVAL_HOURS)
                elif status == "skipped":
                    logger.info("Retrain skipped (model still fresh). Next check in %dh.", RETRAIN_INTERVAL_HOURS)
                elif status == "rejected":
                    logger.warning("New model rejected (worse metrics). Keeping old model.")
                else:
                    self._consecutive_failures += 1
                    logger.error("Retrain failed (%d consecutive). Retrying in %dh.",
                                 self._consecutive_failures, RETRAIN_INTERVAL_HOURS)
            except Exception as e:
                self._consecutive_failures += 1
                logger.error("Retrain loop exception: %s (failure #%d)", e, self._consecutive_failures)

            # Backoff on repeated failures: double interval up to 24h
            sleep_time = interval_seconds
            if self._consecutive_failures >= 3:
                sleep_time = min(interval_seconds * (2 ** (self._consecutive_failures - 2)), 24 * 3600)
                logger.warning("Backing off: next retrain in %.1fh", sleep_time / 3600)

            self._interruptible_sleep(sleep_time)

    def _interruptible_sleep(self, seconds: float):
        """Sleep in small increments so we can stop quickly."""
        step = 5.0
        elapsed = 0.0
        while elapsed < seconds and self._running:
            time.sleep(min(step, seconds - elapsed))
            elapsed += step

    # ── Retrain Cycle ────────────────────────────

    def _retrain_cycle(self, force: bool = False) -> dict:
        """
        Execute one full retrain cycle:
          1. Check if retrain is needed
          2. Fetch fresh data
          3. Train new model
          4. Validate against old model
          5. Swap if better (or force)
          6. Hot-reload predictor
        """
        cycle_start = datetime.now(timezone.utc)
        record = {
            "timestamp": cycle_start.isoformat(),
            "forced": force,
        }

        try:
            # 1. Check freshness
            if not force and not self._needs_retrain():
                record["status"] = "skipped"
                record["reason"] = "Model is still fresh"
                self._append_history(record)
                return record

            # 2. Get old model metrics for comparison
            old_metrics = self._get_current_model_metrics()
            record["old_metrics"] = old_metrics

            # 3. Backup current model
            self._backup_model()

            # 4. Fetch fresh data
            logger.info("[Retrain] Fetching fresh market data...")
            data = fetch_all_timeframes(force_refresh=True)
            if "1h" not in data or data["1h"] is None or data["1h"].empty:
                raise RuntimeError("Failed to fetch 1h data")

            record["data_rows"] = {tf: len(df) for tf, df in data.items()}

            # 5. Build features & train
            logger.info("[Retrain] Building features...")
            htf_data = {"4h": data.get("4h"), "1d": data.get("1d")}
            dataset = build_dataset(data["1h"], include_target=True, htf_data=htf_data)
            record["dataset_size"] = len(dataset)

            logger.info("[Retrain] Training new model on %d samples...", len(dataset))
            meta = train_model(dataset)

            new_metrics = meta.get("final_metrics", {})
            cv_metrics = meta.get("cv_metrics", {})
            record["new_metrics"] = new_metrics
            record["cv_metrics"] = cv_metrics

            # 6. Validation gate: compare new vs old
            if not force and old_metrics and not self._is_improvement(old_metrics, new_metrics):
                # New model is worse — restore backup
                logger.warning("[Retrain] New model is WORSE. Restoring backup.")
                self._restore_backup()
                record["status"] = "rejected"
                record["reason"] = "New model metrics did not meet threshold"
                self._append_history(record)
                return record

            # 7. Model already saved by train_model(). Hot-reload predictor.
            logger.info("[Retrain] New model accepted! Hot-reloading predictor...")
            self._predictor.load()

            self._last_retrain = datetime.now(timezone.utc)
            self._retrain_count += 1

            elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            record["status"] = "success"
            record["elapsed_seconds"] = round(elapsed, 1)
            record["retrain_number"] = self._retrain_count
            self._append_history(record)

            logger.info(
                "[Retrain] SUCCESS #%d — CV: acc=%.3f auc=%.3f | Final: acc=%.3f auc=%.3f | %.1fs",
                self._retrain_count,
                cv_metrics.get("accuracy", 0), cv_metrics.get("auc_roc", 0),
                new_metrics.get("accuracy", 0), new_metrics.get("auc_roc", 0),
                elapsed,
            )
            return record

        except Exception as e:
            logger.error("[Retrain] FAILED: %s", e, exc_info=True)
            # Restore backup if we have one
            self._restore_backup()
            record["status"] = "error"
            record["error"] = str(e)
            self._append_history(record)
            return record

    # ── Helpers ──────────────────────────────────

    def _needs_retrain(self) -> bool:
        """Check if model is stale and needs retraining."""
        if not CURRENT_META.exists():
            logger.info("[Retrain] No model found — training needed")
            return True

        try:
            with open(CURRENT_META) as f:
                meta = json.load(f)
            trained_at = datetime.fromisoformat(meta["trained_at"]).replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - trained_at).total_seconds() / 3600

            if age_hours >= RETRAIN_FORCE_AFTER_HOURS:
                logger.info("[Retrain] Model is %.1fh old (force threshold: %dh) — retraining",
                            age_hours, RETRAIN_FORCE_AFTER_HOURS)
                return True

            if age_hours >= RETRAIN_INTERVAL_HOURS:
                logger.info("[Retrain] Model is %.1fh old — due for retrain", age_hours)
                return True

            logger.debug("[Retrain] Model is only %.1fh old — still fresh", age_hours)
            return False

        except Exception as e:
            logger.warning("[Retrain] Could not read model age: %s — retraining", e)
            return True

    def _get_current_model_metrics(self) -> Optional[dict]:
        """Read current model's final metrics from meta.json."""
        try:
            if CURRENT_META.exists():
                with open(CURRENT_META) as f:
                    meta = json.load(f)
                return meta.get("final_metrics")
        except Exception:
            pass
        return None

    def _is_improvement(self, old: dict, new: dict) -> bool:
        """
        Check if new model metrics are acceptable compared to old.
        Uses AUC-ROC as primary metric, with a minimum improvement threshold.
        If old model had no metrics, always accept.
        """
        if not old:
            return True

        old_auc = old.get("auc_roc", 0)
        new_auc = new.get("auc_roc", 0)
        old_f1 = old.get("f1", 0)
        new_f1 = new.get("f1", 0)

        # New model must not be significantly worse
        # Allow slight degradation (within threshold) since market changes
        min_auc = old_auc - RETRAIN_MIN_IMPROVEMENT
        min_f1 = old_f1 - RETRAIN_MIN_IMPROVEMENT

        accepted = new_auc >= min_auc and new_f1 >= min_f1

        logger.info(
            "[Retrain] Validation: old(auc=%.4f, f1=%.4f) vs new(auc=%.4f, f1=%.4f) -> %s",
            old_auc, old_f1, new_auc, new_f1,
            "ACCEPTED" if accepted else "REJECTED",
        )
        return accepted

    def _backup_model(self):
        """Backup current model files before retraining."""
        try:
            if CURRENT_MODEL.exists():
                shutil.copy2(CURRENT_MODEL, BACKUP_MODEL)
            if CURRENT_META.exists():
                shutil.copy2(CURRENT_META, BACKUP_META)
            logger.debug("[Retrain] Model backed up")
        except Exception as e:
            logger.warning("[Retrain] Backup failed: %s", e)

    def _restore_backup(self):
        """Restore backup model files if retrain failed."""
        try:
            if BACKUP_MODEL.exists():
                shutil.copy2(BACKUP_MODEL, CURRENT_MODEL)
            if BACKUP_META.exists():
                shutil.copy2(BACKUP_META, CURRENT_META)
            # Reload predictor with restored model
            try:
                self._predictor.load()
            except Exception:
                pass
            logger.info("[Retrain] Backup restored")
        except Exception as e:
            logger.warning("[Retrain] Restore failed: %s", e)

    def _load_history(self):
        """Load retrain history from disk."""
        try:
            if HISTORY_FILE.exists():
                with open(HISTORY_FILE) as f:
                    self._history = json.load(f)
                # Track last retrain time
                successes = [h for h in self._history if h.get("status") == "success"]
                if successes:
                    self._last_retrain = datetime.fromisoformat(successes[-1]["timestamp"])
                    self._retrain_count = len(successes)
                logger.info("Loaded %d retrain history records", len(self._history))
        except Exception as e:
            logger.warning("Could not load retrain history: %s", e)
            self._history = []

    def _append_history(self, record: dict):
        """Append a record to history and persist."""
        with self._lock:
            self._history.append(record)
            # Keep last 100 entries
            if len(self._history) > 100:
                self._history = self._history[-100:]
            try:
                with open(HISTORY_FILE, "w") as f:
                    json.dump(self._history, f, indent=2, default=str)
            except Exception as e:
                logger.warning("Could not save retrain history: %s", e)
