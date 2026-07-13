from gpt2_classifier.logging_utils import RunLogger


def test_disabled_logger_is_safe_noop():
    logger = RunLogger(enabled=False)
    logger.log({"loss": 1.0})
    logger.finish()
    assert logger.enabled is False


def test_enabled_logger_degrades_to_disabled_on_init_failure(monkeypatch):
    import gpt2_classifier.logging_utils as logging_utils_module

    class _FakeWandbModule:
        @staticmethod
        def init(**kwargs):
            raise RuntimeError("no network / no API key")

    monkeypatch.setitem(__import__("sys").modules, "wandb", _FakeWandbModule())

    logger = RunLogger(enabled=True)

    assert logger.enabled is False
    # Logging/finishing after a failed init must still be safe no-ops.
    logger.log({"loss": 1.0})
    logger.finish()


def test_enabled_logger_calls_wandb_log_and_finish(monkeypatch):
    calls = {"log": [], "finished": False}

    class _FakeWandbModule:
        @staticmethod
        def init(**kwargs):
            return object()

        @staticmethod
        def log(metrics, step=None):
            calls["log"].append((metrics, step))

        @staticmethod
        def finish():
            calls["finished"] = True

    monkeypatch.setitem(__import__("sys").modules, "wandb", _FakeWandbModule())

    logger = RunLogger(enabled=True)
    assert logger.enabled is True

    logger.log({"loss": 0.5}, step=3)
    logger.finish()

    assert calls["log"] == [({"loss": 0.5}, 3)]
    assert calls["finished"] is True
