import sys
from gpt2_classifier.logging_utils import RunLogger


def test_disabled_logger_is_safe_noop():
    logger = RunLogger(enabled=False)
    logger.log({"loss": 1.0})
    logger.finish()
    assert logger.enabled is False


def test_enabled_logger_degrades_to_disabled_on_init_failure(monkeypatch):
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

    calls = {
        "load_dotenv_called": False,
        "login_called": False,
        "log": [], 
        "finished": False
    }

    def fake_load_dotenv(*args, **kwargs):
        calls["load_dotenv_called"] = True
        return True

    import gpt2_classifier.logging_utils as logger_module
   
    monkeypatch.setattr(logger_module, "load_dotenv", fake_load_dotenv)

    class _FakeWandbModule:
        @staticmethod
        def login(**kwargs):
            calls["login_called"] = True
            return True

        @staticmethod
        def init(**kwargs):
            return object()

        @staticmethod
        def log(metrics, step=None):
            calls["log"].append((metrics, step))

        @staticmethod
        def finish():
            calls["finished"] = True

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandbModule())

    logger = logger_module.RunLogger(enabled=True)
    assert logger.enabled is True

    logger.log({"loss": 0.5}, step=3)
    logger.finish()

    assert (
        calls["load_dotenv_called"] is True
    ), "load_dotenv was not called during RunLogger init"
    assert (
        calls["login_called"] is True
    ), "wandb.login was not called during RunLogger init"
    assert calls["log"] == [
        ({"loss": 0.5}, 3)
    ], "Logged metrics or steps do not match"
    assert calls["finished"] is True, "wandb.finish was not triggered"