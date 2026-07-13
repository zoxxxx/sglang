import ctypes
import re
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.host_kv_pool import (
    HostKVObjectInfo,
    HostKVPoolRuntime,
    HostKVReceiver,
    HostKVSender,
    _HostKVFetchResult,
    _HostKVPushResult,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class FakeMetadataBuffers:
    def __init__(self, item_lens, rows):
        self.item_lens = item_lens
        self.buffers = [
            (ctypes.c_ubyte * (item_len * rows))() for item_len in item_lens
        ]

    def get_buf_infos(self):
        ptrs = [ctypes.addressof(buf) for buf in self.buffers]
        return ptrs, None, self.item_lens

    def write_row(self, row_idx, values):
        for buf, item_len, value in zip(self.buffers, self.item_lens, values):
            assert len(value) == item_len
            ctypes.memmove(ctypes.addressof(buf) + item_len * row_idx, value, item_len)

    def read_row(self, row_idx):
        values = []
        for buf, item_len in zip(self.buffers, self.item_lens):
            start = item_len * row_idx
            values.append(bytes(buf[start : start + item_len]))
        return values


class FakeStore:
    def __init__(self):
        self.values = {}
        self.remove_patterns = []
        self.remove_forces = []

    def put(self, key, payload):
        self.values[key] = payload
        return 0

    def get(self, key):
        return self.values.get(key)

    def remove_by_regex(self, pattern, force=False):
        self.remove_patterns.append(pattern)
        self.remove_forces.append(force)
        matcher = re.compile(pattern)
        removed = [key for key in self.values if matcher.search(key)]
        for key in removed:
            del self.values[key]
        return len(removed)


class FakeStorage:
    def __init__(self, extra_backend_tag=None):
        self.store = FakeStore()
        self.extra_backend_tag = extra_backend_tag

    def _tag_keys(self, keys):
        if self.extra_backend_tag is None:
            return keys
        return [f"{self.extra_backend_tag}_{key}" for key in keys]


class DeferredPushRuntime:
    def __init__(self):
        self.tasks = []
        self.events = []

    def stage_write(self, host_kv_id, page_indices, start_page, state_indices=None):
        self.events.append(("stage", start_page, len(page_indices)))
        return SimpleNamespace(
            transfer_total_bytes=len(page_indices) * 1024,
            state_page_counts={"indexer": 1} if state_indices else {},
            start_page=start_page,
        )

    def commit_staged_write(self, host_kv_id, staged):
        self.events.append(("commit", staged.start_page))
        return _HostKVPushResult(
            submit_latency_s=0.001,
            ready_wait_s=0.002,
            kv_put_s=0.003,
            state_put_s=0.004,
            metadata_put_s=0.0,
            task_latency_s=0.009,
            transfer_total_bytes=1024,
        )

    def put_metadata(
        self,
        req,
        metadata_buffers,
        metadata_index,
        num_pages,
        state_page_counts=None,
    ):
        self.events.append(("metadata", num_pages, state_page_counts))

    def submit_push(self, fn):
        future = Future()
        self.tasks.append((fn, future))
        return future

    def run_all(self):
        while self.tasks:
            fn, future = self.tasks.pop(0)
            try:
                future.set_result(fn())
            except BaseException as exc:
                future.set_exception(exc)


def make_host_kv_runtime(tp_rank=1, pp_rank=2, storage=None):
    runtime = object.__new__(HostKVPoolRuntime)
    runtime.storage = storage or FakeStorage()
    runtime.page_size = 16
    runtime.tp_rank = tp_rank
    runtime.pp_rank = pp_rank
    runtime.controller = SimpleNamespace(
        mem_pool_host=SimpleNamespace(layout="page_first"),
        mem_pool_device=SimpleNamespace(store_dtype="float16", head_num=8),
    )
    return runtime


class TestHostKVPoolRequestNormalization(unittest.TestCase):
    def test_single_host_kv_id_stays_scalar(self):
        req = GenerateReqInput(text="hello", host_kv_id="kv-single")

        req.normalize_batch_and_arguments()

        self.assertEqual(req.host_kv_id, "kv-single")

    def test_batch_parallel_list_host_kv_id_is_sample_scoped(self):
        req = GenerateReqInput(
            text=["hello", "world"],
            sampling_params={"n": 2},
            host_kv_id=["kv-a", "kv-b"],
        )

        req.normalize_batch_and_arguments()

        self.assertEqual(
            req.host_kv_id,
            ["kv-a-0", "kv-b-0", "kv-a-1", "kv-b-1"],
        )
        self.assertEqual(req[2].host_kv_id, "kv-a-1")

    def test_parallel_scalar_host_kv_id_is_unique_per_sample(self):
        req = GenerateReqInput(
            text="hello",
            sampling_params={"n": 2},
            host_kv_id="kv-single",
        )

        req.normalize_batch_and_arguments()

        self.assertEqual(req.host_kv_id, ["kv-single-0", "kv-single-1"])


class TestHostKVPoolServerArgs(unittest.TestCase):
    def _make_args(self, **overrides):
        args = object.__new__(ServerArgs)
        defaults = {
            "disaggregation_host_kv_pool": True,
            "disaggregation_mode": "decode",
            "hicache_storage_backend": "mooncake",
            "enable_hisparse": False,
            "speculative_algorithm": None,
            "disaggregation_decode_enable_radix_cache": False,
            "disaggregation_decode_enable_offload_kvcache": False,
        }
        defaults.update(overrides)
        for key, value in defaults.items():
            setattr(args, key, value)
        return args

    def test_requires_pd_role(self):
        with self.assertRaisesRegex(ValueError, "requires --disaggregation-mode"):
            self._make_args(disaggregation_mode=None)._handle_pd_disaggregation()

    def test_requires_mooncake_hicache_backend(self):
        with self.assertRaisesRegex(ValueError, "requires --hicache-storage-backend"):
            self._make_args(hicache_storage_backend="file")._handle_pd_disaggregation()

    def test_rejects_decode_radix_cache(self):
        with self.assertRaisesRegex(ValueError, "decode-enable-radix-cache"):
            self._make_args(
                disaggregation_decode_enable_radix_cache=True
            )._handle_pd_disaggregation()

    def test_accepts_minimal_host_kv_pool_config(self):
        self._make_args()._handle_pd_disaggregation()


class TestHostKVPoolRuntimeController(unittest.TestCase):
    def _make_scheduler(self, controller):
        return SimpleNamespace(
            server_args=SimpleNamespace(hicache_storage_backend="mooncake"),
            page_size=16,
            tp_rank=0,
            pp_rank=0,
            tree_cache=SimpleNamespace(cache_controller=controller),
        )

    def test_reuses_existing_hicache_controller(self):
        controller = SimpleNamespace(
            storage_backend_type="mooncake",
            storage_backend=FakeStorage(),
        )
        scheduler = self._make_scheduler(controller)

        runtime = HostKVPoolRuntime.get_or_create(scheduler)

        self.assertIs(runtime.controller, controller)
        self.assertIs(runtime.storage, controller.storage_backend)
        self.assertIs(HostKVPoolRuntime.get_or_create(scheduler), runtime)

    def test_builds_private_controller_without_existing_hicache_controller(self):
        scheduler = self._make_scheduler(controller=None)
        controller = SimpleNamespace(
            storage_backend_type="mooncake",
            storage_backend=FakeStorage(),
        )

        with patch.object(
            HostKVPoolRuntime, "_build_private_controller", return_value=controller
        ) as build_private_controller:
            runtime = HostKVPoolRuntime.get_or_create(scheduler)

        build_private_controller.assert_called_once_with()
        self.assertIs(runtime.controller, controller)
        self.assertIs(runtime.storage, controller.storage_backend)

    def test_requires_existing_controller_to_use_mooncake(self):
        controller = SimpleNamespace(
            storage_backend_type="file",
            storage_backend=FakeStorage(),
        )
        scheduler = self._make_scheduler(controller)

        with self.assertRaisesRegex(RuntimeError, "Mooncake storage"):
            HostKVPoolRuntime.get_or_create(scheduler)


class TestHostKVPoolMetadata(unittest.TestCase):
    def test_metadata_roundtrip_copies_object_info_and_rows(self):
        runtime = make_host_kv_runtime()
        req = SimpleNamespace(host_kv_id="host-kv-1", origin_input_ids=[1, 2, 3])
        src = FakeMetadataBuffers(item_lens=[3, 4], rows=2)
        dst = FakeMetadataBuffers(item_lens=[3, 4], rows=2)
        src.write_row(1, [b"abc", b"defg"])

        runtime.put_metadata(req, src, metadata_index=1, num_pages=7)
        info = runtime.get_metadata(req, dst, metadata_index=0)

        self.assertEqual(info.host_kv_id, "host-kv-1")
        self.assertEqual(info.token_len, 3)
        self.assertEqual(info.page_size, 16)
        self.assertEqual(info.num_pages, 7)
        self.assertEqual(info.tp_rank, 1)
        self.assertEqual(info.pp_rank, 2)
        self.assertEqual(dst.read_row(0), [b"abc", b"defg"])

    def test_metadata_rejects_token_len_mismatch(self):
        runtime = make_host_kv_runtime()
        prefill_req = SimpleNamespace(
            host_kv_id="host-kv-1", origin_input_ids=[1, 2, 3]
        )
        decode_req = SimpleNamespace(host_kv_id="host-kv-1", origin_input_ids=[1, 2])
        metadata = FakeMetadataBuffers(item_lens=[3], rows=1)
        metadata.write_row(0, [b"abc"])

        runtime.put_metadata(prefill_req, metadata, metadata_index=0, num_pages=1)

        with self.assertRaisesRegex(RuntimeError, "token length mismatch"):
            runtime.get_metadata(decode_req, metadata, metadata_index=0)

    def test_decode_reads_metadata_from_prefill_source_rank(self):
        storage = FakeStorage()
        prefill_runtime = make_host_kv_runtime(tp_rank=1, storage=storage)
        decode_runtime = make_host_kv_runtime(tp_rank=9, storage=storage)
        prefill_req = SimpleNamespace(
            host_kv_id="host-kv-cross-rank", origin_input_ids=[1, 2, 3]
        )
        decode_req = SimpleNamespace(
            host_kv_id="host-kv-cross-rank",
            origin_input_ids=[1, 2, 3],
            disagg_prefill_dp_rank=1,
        )
        src = FakeMetadataBuffers(item_lens=[3], rows=1)
        dst = FakeMetadataBuffers(item_lens=[3], rows=1)
        src.write_row(0, [b"abc"])

        prefill_runtime.put_metadata(prefill_req, src, metadata_index=0, num_pages=1)
        info = decode_runtime.get_metadata(decode_req, dst, metadata_index=0)

        self.assertEqual(info.tp_rank, 1)
        self.assertEqual(decode_runtime.tp_rank, 9)
        self.assertEqual(dst.read_row(0), [b"abc"])


class TestHostKVPoolLifecycle(unittest.TestCase):
    def test_release_removes_only_the_source_rank_object_and_is_idempotent(self):
        storage = FakeStorage(extra_backend_tag="model-a")
        runtime = make_host_kv_runtime(tp_rank=9, pp_rank=2, storage=storage)

        def submit_immediately(fn):
            future = Future()
            try:
                future.set_result(fn())
            except BaseException as exc:
                future.set_exception(exc)
            return future

        runtime.submit_push = submit_immediately
        storage.store.values.update(
            {
                "model-a_hostkv:kv-1:tp1:pp2:page0_mla_k": b"kv",
                "model-a_hostkv:kv-1:tp1:pp2:page0_mla_indexer": b"state",
                "model-a_hostkv:kv-1:tp1:pp2:meta": b"meta",
                "model-a_hostkv:kv-1:tp9:pp2:meta": b"other-rank",
                "model-a_hostkv:kv-2:tp1:pp2:meta": b"other-object",
            }
        )
        req = SimpleNamespace(
            rid="req-1",
            host_kv_id="kv-1",
            disagg_prefill_dp_rank=1,
            host_kv_object_info=None,
            host_kv_release_scheduled=False,
        )

        future = runtime.schedule_release(req, "decode_finished")
        duplicate = runtime.schedule_release(req, "decode_finished")

        self.assertIsNone(duplicate)
        self.assertEqual(future.result()[0], 3)
        self.assertEqual(
            storage.store.remove_patterns,
            [r"model\-a_hostkv:kv\-1:tp1:pp2:.*"],
        )
        self.assertEqual(storage.store.remove_forces, [True])
        self.assertEqual(
            set(storage.store.values),
            {
                "model-a_hostkv:kv-1:tp9:pp2:meta",
                "model-a_hostkv:kv-2:tp1:pp2:meta",
            },
        )


class FakePullRuntime:
    def __init__(self):
        self.calls = []
        self.release_reasons = []
        self.tasks = []
        self.attach_ready = False
        self.info = HostKVObjectInfo(
            host_kv_id="host-kv-1",
            token_len=3,
            page_size=16,
            num_pages=2,
            tp_rank=1,
            pp_rank=0,
            layout="page_first",
            dtype="float16",
            is_mla_backend=True,
            state_page_counts={"indexer": 1},
        )

    def stage_read(self, req, page_indices, state_indices=None):
        self.calls.append(("stage", len(page_indices), state_indices))
        return SimpleNamespace(
            attach_failure=None,
            attach_start_s=None,
            released=False,
        )

    def submit_pull(self, fn):
        future = Future()
        future.set_running_or_notify_cancel()
        self.tasks.append((fn, future))
        return future

    def run_all(self):
        while self.tasks:
            fn, future = self.tasks.pop(0)
            try:
                future.set_result(fn())
            except BaseException as exc:
                future.set_exception(exc)

    def fetch_staged_read(
        self, req, metadata_buffers, metadata_index, staged, submitted_at
    ):
        info = self.get_metadata(req, metadata_buffers, metadata_index)
        attached_tokens = self.attach_pages(
            req.host_kv_id, np.asarray([4, 5]), info.num_pages, info.tp_rank
        )
        attached_state_tokens = self.attach_state_pages(
            req.host_kv_id, [[6]], info, info.tp_rank
        )
        return _HostKVFetchResult(
            object_info=info,
            queue_wait_s=0.001,
            metadata_get_s=0.002,
            kv_get_s=0.003,
            state_get_s=0.004,
            attached_tokens=attached_tokens,
            attached_state_tokens=attached_state_tokens,
        )

    def start_staged_read_attach(self, staged):
        self.calls.append(("attach",))
        staged.attach_start_s = 0.0

    def staged_read_ready(self, staged):
        return self.attach_ready

    def release_staged_read(self, staged):
        staged.released = True
        self.calls.append(("release",))

    def get_metadata(self, req, metadata_buffers, metadata_index):
        self.calls.append(("metadata", metadata_index))
        return self.info

    def attach_pages(
        self, host_kv_id, page_indices, expected_num_pages, source_tp_rank=None
    ):
        self.calls.append(
            ("pages", host_kv_id, len(page_indices), expected_num_pages, source_tp_rank)
        )
        return 32

    def attach_state_pages(
        self, host_kv_id, state_indices, object_info, source_tp_rank=None
    ):
        self.calls.append(("state", host_kv_id, source_tp_rank))
        return 16

    def schedule_release(self, req, reason):
        self.release_reasons.append(reason)


class TestHostKVReceiver(unittest.TestCase):
    def test_pull_and_attach_are_polled_asynchronously(self):
        runtime = FakePullRuntime()
        req = SimpleNamespace(
            rid="req-1",
            host_kv_id="host-kv-1",
            host_kv_object_info=None,
        )
        receiver = HostKVReceiver(runtime, req, metadata_buffers=object())

        receiver.send_metadata(
            np.asarray([4, 5], dtype=np.int32),
            aux_index=3,
            state_indices=[[6]],
        )

        self.assertEqual(receiver.poll(), KVPoll.Transferring)
        self.assertEqual(runtime.calls, [("stage", 2, [[6]])])

        runtime.run_all()

        self.assertEqual(receiver.poll(), KVPoll.Transferring)
        self.assertEqual(runtime.calls[-1], ("attach",))

        runtime.attach_ready = True

        self.assertEqual(receiver.poll(), KVPoll.Success)
        self.assertIs(req.host_kv_object_info, runtime.info)
        self.assertEqual(
            runtime.calls,
            [
                ("stage", 2, [[6]]),
                ("metadata", 3),
                ("pages", "host-kv-1", 2, 2, 1),
                ("state", "host-kv-1", 1),
                ("attach",),
                ("release",),
            ],
        )

    def test_abort_waits_for_running_fetch_before_releasing_buffers(self):
        runtime = FakePullRuntime()
        req = SimpleNamespace(
            rid="req-1",
            host_kv_id="host-kv-1",
            host_kv_object_info=None,
        )
        receiver = HostKVReceiver(runtime, req, metadata_buffers=object())
        receiver.send_metadata(np.asarray([4, 5], dtype=np.int32), aux_index=3)

        receiver.abort()

        self.assertEqual(receiver.poll(), KVPoll.Transferring)
        self.assertNotIn(("release",), runtime.calls)

        runtime.run_all()

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertIn(("release",), runtime.calls)
        self.assertNotIn(("attach",), runtime.calls)
        self.assertEqual(runtime.release_reasons, ["receiver_aborted"])

    def test_abort_schedules_object_release(self):
        runtime = FakePullRuntime()
        req = SimpleNamespace(rid="req-1", host_kv_id="host-kv-1")
        receiver = HostKVReceiver(runtime, req, metadata_buffers=object())

        receiver.abort()

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertEqual(runtime.release_reasons, ["receiver_aborted"])


class TestHostKVSenderAsyncPush(unittest.TestCase):
    def _make_sender(self):
        runtime = DeferredPushRuntime()
        req = SimpleNamespace(rid="req-1", host_kv_id="host-kv-1")
        sender = HostKVSender(runtime, req, metadata_buffers=object())
        sender.init(num_kv_indices=2, aux_index=0)
        return runtime, sender

    def test_send_returns_before_background_push_finishes(self):
        runtime, sender = self._make_sender()
        sender.should_send_kv_chunk(num_pages=2, last_chunk=True)

        sender.send(np.asarray([3, 4], dtype=np.int32), state_indices=[[5]])

        self.assertEqual(sender.poll(), KVPoll.Transferring)
        self.assertNotIn("metadata", [event[0] for event in runtime.events])

        runtime.run_all()

        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertEqual(
            runtime.events,
            [
                ("stage", 0, 2),
                ("commit", 0),
                ("metadata", 2, {"indexer": 1}),
            ],
        )

    def test_metadata_is_published_after_all_chunks(self):
        runtime, sender = self._make_sender()
        sender.should_send_kv_chunk(num_pages=1, last_chunk=False)
        sender.send(np.asarray([3], dtype=np.int32))
        sender.should_send_kv_chunk(num_pages=1, last_chunk=True)
        sender.send(np.asarray([4], dtype=np.int32))

        self.assertEqual(sender.poll(), KVPoll.Transferring)
        runtime.run_all()

        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertEqual(
            runtime.events,
            [
                ("stage", 0, 1),
                ("stage", 1, 1),
                ("commit", 0),
                ("commit", 1),
                ("metadata", 2, {}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
