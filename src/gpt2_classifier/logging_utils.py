"""Thin wandb wrapper that degrades to a safe no-op when disabled or unavailable.

Training must never hard-fail because wandb isn't installed, the user has no
API key configured, or the network is unreachable -- ``--use-wandb`` is
opt-in, and any failure to actually start a run just disables logging with a
printed warning.
"""

from dotenv import load_dotenv
from typing import Any


class RunLogger:
    """Wraps ``wandb.init``/``log``/``finish``, no-op when disabled."""

    def __init__(
        self,
        enabled: bool,
        project: str = "gpt2-classifier",
        config: dict[str, Any] | None = None,
    ) -> None:
        """Start a wandb run, or configure this logger as a no-op.

        Args:
            enabled: Whether wandb logging was requested (e.g. via
                ``--use-wandb``). If ``False``, this logger is a no-op and
                wandb is never imported.
            project: wandb project name.
            config: Run config to record (e.g. hyperparameters).
        """
        self.enabled = enabled
        self._wandb = None
        self._run = None

        if self.enabled:
            try:
                import wandb

                load_dotenv()
                wandb.login()
                self._wandb = wandb
                self._run = wandb.init(project=project, config=config)
            except Exception as exc:
                print(f"[wandb] disabled due to init failure: {exc}")
                self.enabled = False

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dict of metrics, if enabled.

        Args:
            metrics: Metric name/value pairs to log.
            step: Optional explicit step index.
        """
        if self.enabled and self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Finish the wandb run, if one was started."""
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
