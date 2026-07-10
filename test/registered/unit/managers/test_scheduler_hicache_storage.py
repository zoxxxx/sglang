import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.scheduler import Scheduler


class TestSchedulerHiCacheStorage(unittest.TestCase):
    def test_prefetch_is_disabled_without_hierarchical_cache(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_hicache_storage = True
        scheduler.tree_cache = object()
        req = MagicMock()

        scheduler._prefetch_kvcache(req)

        req.init_next_round_input.assert_not_called()

    def test_host_kv_release_is_delegated_to_runtime(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.host_kv_pool_runtime = MagicMock()
        req = MagicMock()

        scheduler.maybe_release_host_kv_object(req, "decode_finished")

        scheduler.host_kv_pool_runtime.schedule_release.assert_called_once_with(
            req, "decode_finished"
        )


if __name__ == "__main__":
    unittest.main()
