from __future__ import annotations

import ctypes
import json
import logging
import re
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.disaggregation.base.conn import KVTransferMetric, KVPoll, StateType
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_anchor_sidecar_stack,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSAIndexerPoolHost,
)
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import MetadataBuffers
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

_METADATA_MAGIC = b"SGL_HOSTKV_META_V1\0"


@dataclass
class HostKVObjectInfo:
    host_kv_id: str
    token_len: int
    page_size: int
    num_pages: int
    tp_rank: int
    pp_rank: int
    layout: str
    dtype: str
    is_mla_backend: bool
    state_page_counts: Optional[dict[str, int]] = None


@dataclass
class _HostKVStagedWrite:
    host_indices: Optional[torch.Tensor]
    moved_host_indices: Optional[torch.Tensor]
    moved_device_indices: Optional[torch.Tensor]
    keys: List[str]
    state_transfers: List[PoolTransfer]
    moved_state_transfers: List[PoolTransfer]
    start_event: Optional[object]
    ready_event: Optional[object]
    transfer_total_bytes: int
    state_page_counts: dict[str, int]
    submit_latency_s: float


@dataclass
class _HostKVPushResult:
    submit_latency_s: float
    ready_wait_s: float
    kv_put_s: float
    state_put_s: float
    metadata_put_s: float
    task_latency_s: float
    transfer_total_bytes: int


@dataclass
class _HostKVStagedRead:
    host_kv_id: str
    source_tp_rank: int
    num_pages: int
    host_indices: Optional[torch.Tensor]
    moved_host_indices: Optional[torch.Tensor]
    moved_device_indices: Optional[torch.Tensor]
    state_transfers: List[PoolTransfer]
    moved_state_transfers: List[PoolTransfer]
    dependency_events: List[object]
    transfer_total_bytes: int
    state_page_counts: dict[str, int]
    ready_event: Optional[object] = None
    attach_failure: Optional[BaseException] = None
    attach_start_s: Optional[float] = None
    released: bool = False


@dataclass
class _HostKVFetchResult:
    object_info: HostKVObjectInfo
    queue_wait_s: float
    metadata_get_s: float
    kv_get_s: float
    state_get_s: float
    attached_tokens: int
    attached_state_tokens: int


def _expand_page_indices(
    page_indices: npt.NDArray[np.int32], page_size: int, device: str
) -> torch.Tensor:
    if len(page_indices) == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    pages = torch.as_tensor(page_indices, dtype=torch.int64, device=device)
    if page_size == 1:
        return pages
    offsets = torch.arange(page_size, dtype=torch.int64, device=device)
    return (pages[:, None] * page_size + offsets[None, :]).reshape(-1)


class HostKVPoolRuntime:
    """Mooncake-backed Host KV Object runtime for the host-centric PD prototype."""

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.server_args = scheduler.server_args
        self.page_size = scheduler.page_size
        self.tp_rank = scheduler.tp_rank
        self.pp_rank = scheduler.pp_rank
        self.controller = self._build_controller()
        self.storage = self.controller.storage_backend
        if self.storage is None:
            raise RuntimeError("Host KV Pool requires a Mooncake storage backend")
        self._register_existing_host_pools_v2()
        self._push_stream = None
        self._push_executor: Optional[ThreadPoolExecutor] = None
        self._push_executor_lock = threading.Lock()
        self._pull_stream = None
        self._pull_executor: Optional[ThreadPoolExecutor] = None
        self._pull_executor_lock = threading.Lock()

    @classmethod
    def get_or_create(cls, scheduler: Scheduler) -> "HostKVPoolRuntime":
        runtime = getattr(scheduler, "host_kv_pool_runtime", None)
        if runtime is None:
            runtime = cls(scheduler)
            scheduler.host_kv_pool_runtime = runtime
        return runtime

    def _build_controller(self) -> HiCacheController:
        if self.server_args.hicache_storage_backend != "mooncake":
            raise ValueError(
                "--disaggregation-host-kv-pool requires "
                "--hicache-storage-backend mooncake"
            )

        controller = self._get_existing_controller()
        if controller is None:
            return self._build_private_controller()

        self._validate_controller(controller)
        return controller

    def _get_existing_controller(self) -> Optional[HiCacheController]:
        return getattr(
            getattr(self.scheduler, "tree_cache", None), "cache_controller", None
        )

    def _validate_controller(self, controller: HiCacheController) -> None:
        if getattr(controller, "storage_backend_type", None) != "mooncake":
            raise RuntimeError(
                "Host KV Pool requires the existing HiCache controller to use "
                "Mooncake storage."
            )
        if getattr(controller, "storage_backend", None) is None:
            raise RuntimeError(
                "Host KV Pool requires an attached Mooncake storage backend"
            )

    def _build_private_controller(self) -> HiCacheController:
        kv_cache = self.scheduler.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, HybridLinearKVPool):
            kv_cache = kv_cache.full_kv_pool

        extra_config = {}
        if self.server_args.hicache_storage_backend_extra_config:
            extra_config = json.loads(
                self.server_args.hicache_storage_backend_extra_config
            )

        if isinstance(kv_cache, NSATokenToKVPool):
            logger.info(
                "Host KV Pool is using a private Mooncake staging controller with "
                "an NSA indexer sidecar pool because HiCache is not enabled on this "
                "worker."
            )
            params = CacheInitParams(
                disable=False,
                req_to_token_pool=getattr(self.scheduler, "req_to_token_pool", None),
                token_to_kv_pool_allocator=self.scheduler.token_to_kv_pool_allocator,
                page_size=self.page_size,
                tp_cache_group=self.scheduler.tp_group,
                attn_cp_cache_group=getattr(self.scheduler, "attn_cp_cpu_group", None),
                attn_tp_cache_group=getattr(self.scheduler, "attn_tp_cpu_group", None),
                pp_rank=self.pp_rank,
                pp_size=self.scheduler.pp_size,
                enable_metrics=getattr(self.scheduler, "enable_metrics", False),
            )
            _, controller = build_anchor_sidecar_stack(
                params=params,
                server_args=self.server_args,
                kv_pool=kv_cache,
                sidecar_pool_name=PoolName.INDEXER,
                full_layer_mapping={
                    layer_id: layer_id for layer_id in range(kv_cache.layer_num)
                },
                page_size=self.page_size,
                tp_group=self.scheduler.tp_group,
                load_cache_event=None,
                attn_cp_group=getattr(self.scheduler, "attn_cp_cpu_group", None),
                attn_tp_group=getattr(self.scheduler, "attn_tp_cpu_group", None),
                storage_backend=self.server_args.hicache_storage_backend,
                use_mla=True,
                override_kv_cache_dim=getattr(kv_cache, "kv_cache_dim", None),
                sidecar_host_pool_factory=lambda kv_host_pool: NSAIndexerPoolHost(
                    kv_cache,
                    kv_host_pool,
                    self.server_args.hicache_mem_layout,
                    allocator_type=self.server_args.hicache_storage_backend,
                ),
                model_name=self.server_args.served_model_name,
                storage_backend_extra_config=extra_config,
                pp_rank=self.pp_rank,
                pp_size=self.scheduler.pp_size,
                enable_storage_metrics=getattr(self.scheduler, "enable_metrics", False),
            )
            return controller

        common_kwargs = dict(
            host_to_device_ratio=self.server_args.hicache_ratio,
            host_size=self.server_args.hicache_size,
            page_size=self.page_size,
            layout=self.server_args.hicache_mem_layout,
        )
        if isinstance(kv_cache, MHATokenToKVPool):
            mem_pool_host = MHATokenToKVPoolHost(kv_cache, **common_kwargs)
        elif isinstance(kv_cache, MLATokenToKVPool):
            mem_pool_host = MLATokenToKVPoolHost(kv_cache, **common_kwargs)
        else:
            raise ValueError(
                "Host KV Pool MVP supports MHA/MLA KV pools only, got "
                f"{type(kv_cache).__name__}"
            )

        logger.info(
            "Host KV Pool is using a private Mooncake staging controller because "
            "HiCache is not enabled on this worker."
        )
        return HiCacheController(
            token_to_kv_pool_allocator=self.scheduler.token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=self.page_size,
            tp_group=self.scheduler.tp_group,
            attn_cp_group=getattr(self.scheduler, "attn_cp_cpu_group", None),
            attn_tp_group=getattr(self.scheduler, "attn_tp_cpu_group", None),
            io_backend=self.server_args.hicache_io_backend,
            load_cache_event=None,
            storage_backend=self.server_args.hicache_storage_backend,
            model_name=self.server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=self.pp_rank,
            pp_size=self.scheduler.pp_size,
            enable_storage_metrics=getattr(self.scheduler, "enable_metrics", False),
        )

    def _register_existing_host_pools_v2(self) -> None:
        register_fn = getattr(self.storage, "register_mem_host_pool_v2", None)
        if register_fn is None:
            return

        registered_pools = getattr(self.storage, "registered_pools", {})
        for entry in getattr(self.controller.mem_pool_host, "entries", ()) or ():
            if entry.name in registered_pools:
                continue
            register_fn(entry.host_pool, entry.name)

    def _get_push_stream(self):
        if self._push_stream is None:
            self._push_stream = get_device_module().Stream()
        return self._push_stream

    def submit_push(self, fn: Callable[[], object]) -> Future:
        if self._push_executor is None:
            with self._push_executor_lock:
                if self._push_executor is None:
                    self._push_executor = ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix=f"host-kv-push-tp{self.tp_rank}",
                    )
        return self._push_executor.submit(fn)

    def _get_pull_stream(self):
        if self._pull_stream is None:
            self._pull_stream = get_device_module().Stream()
        return self._pull_stream

    def submit_pull(self, fn: Callable[[], object]) -> Future:
        """Run blocking Mooncake reads outside the decode scheduler thread."""
        if self._pull_executor is None:
            with self._pull_executor_lock:
                if self._pull_executor is None:
                    self._pull_executor = ThreadPoolExecutor(
                        max_workers=4,
                        thread_name_prefix=f"host-kv-pull-tp{self.tp_rank}",
                    )
        return self._pull_executor.submit(fn)

    def schedule_release(self, req: Req, reason: str) -> Optional[Future]:
        """Remove one logical Host KV Object after its request lifetime ends."""
        host_kv_id = getattr(req, "host_kv_id", None)
        if host_kv_id is None or getattr(req, "host_kv_release_scheduled", False):
            return None

        object_info = getattr(req, "host_kv_object_info", None)
        source_tp_rank = (
            object_info.tp_rank
            if object_info is not None
            else self._source_tp_rank(req)
        )
        source_pp_rank = (
            object_info.pp_rank if object_info is not None else self.pp_rank
        )
        prefix = f"{self._object_prefix(host_kv_id, source_tp_rank, source_pp_rank)}:"
        tag_keys = getattr(self.storage, "_tag_keys", None)
        if tag_keys is not None:
            prefix = tag_keys([prefix])[0]
        pattern = f"{re.escape(prefix)}.*"
        rid = getattr(req, "rid", "unknown")
        req.host_kv_release_scheduled = True

        def remove_object():
            remove_by_regex = getattr(self.storage.store, "remove_by_regex", None)
            if remove_by_regex is None:
                raise RuntimeError(
                    "Mooncake Host KV lifecycle cleanup requires remove_by_regex"
                )
            start = time.perf_counter()
            last_error = None
            for attempt in range(3):
                try:
                    result = remove_by_regex(pattern, True)
                    if result >= 0:
                        return result, time.perf_counter() - start
                    last_error = RuntimeError(
                        f"Mooncake Host KV object removal failed with code {result}"
                    )
                except BaseException as exc:
                    last_error = exc
                if attempt < 2:
                    time.sleep(0.05 * (2**attempt))
            assert last_error is not None
            raise last_error

        try:
            future = self.submit_push(remove_object)
        except BaseException:
            req.host_kv_release_scheduled = False
            logger.exception(
                "Failed to schedule Host KV object release for rid=%s host_kv_id=%s",
                rid,
                host_kv_id,
            )
            return None

        def log_result(done: Future) -> None:
            try:
                result, latency_s = done.result()
                logger.info(
                    "HostKVReleaseStats(rid=%s, host_kv_id=%s, reason=%s): "
                    "result=%s, latency=%.2fms",
                    rid,
                    host_kv_id,
                    reason,
                    result,
                    latency_s * 1000,
                )
            except BaseException:
                logger.exception(
                    "Host KV object release failed for rid=%s host_kv_id=%s "
                    "reason=%s pattern=%s",
                    rid,
                    host_kv_id,
                    reason,
                    pattern,
                )

        future.add_done_callback(log_result)
        return future

    def _object_prefix(
        self,
        host_kv_id: str,
        tp_rank: Optional[int] = None,
        pp_rank: Optional[int] = None,
    ) -> str:
        tp_rank = self.tp_rank if tp_rank is None else tp_rank
        pp_rank = self.pp_rank if pp_rank is None else pp_rank
        return f"hostkv:{host_kv_id}:tp{tp_rank}:pp{pp_rank}"

    def _page_keys(
        self,
        host_kv_id: str,
        start_page: int,
        num_pages: int,
        tp_rank: Optional[int] = None,
    ) -> List[str]:
        prefix = self._object_prefix(host_kv_id, tp_rank=tp_rank)
        return [f"{prefix}:page{i}" for i in range(start_page, start_page + num_pages)]

    def _meta_key(self, host_kv_id: str, tp_rank: Optional[int] = None) -> str:
        key = f"{self._object_prefix(host_kv_id, tp_rank=tp_rank)}:meta"
        tag_keys = getattr(self.storage, "_tag_keys", None)
        if tag_keys is None:
            return key
        return tag_keys([key])[0]

    def _source_tp_rank(self, req: Req) -> int:
        source_rank = getattr(req, "disagg_prefill_dp_rank", None)
        return self.tp_rank if source_rank is None else int(source_rank)

    def _sync_device(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _record_tensor_on_stream(tensor, stream) -> None:
        if tensor is not None and getattr(tensor, "is_cuda", False):
            tensor.record_stream(stream)

    def _release_staged_write(self, staged: _HostKVStagedWrite) -> None:
        if staged.host_indices is not None:
            self.controller.mem_pool_host.free(staged.host_indices)
        self._free_state_pool_transfers(staged.state_transfers)

    def _record_pull_dependencies(self) -> List[object]:
        """Fence index setup and HBM reuse against already queued GPU work."""
        device_module = get_device_module()
        schedule_event = device_module.Event()
        schedule_event.record()
        events = [schedule_event]
        forward_stream = (
            getattr(self.scheduler, "forward_stream", None)
            if getattr(self.scheduler, "enable_overlap", False)
            else None
        )
        if forward_stream is not None:
            forward_event = device_module.Event()
            forward_event.record(forward_stream)
            events.append(forward_event)
        return events

    def stage_read(
        self,
        req: Req,
        page_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ) -> _HostKVStagedRead:
        """Reserve staging buffers and destination indices without doing I/O."""
        host_indices = None
        state_transfers: List[PoolTransfer] = []
        source_tp_rank = self._source_tp_rank(req)

        try:
            num_pages = len(page_indices)
            moved_host_indices = None
            moved_device_indices = None
            transfer_total_bytes = 0
            if num_pages > 0:
                device_indices = _expand_page_indices(
                    page_indices, self.page_size, self.controller.device
                )
                host_indices = self.controller.mem_pool_host.alloc(
                    num_pages * self.page_size
                )
                if host_indices is None:
                    raise RuntimeError(
                        "Host KV Pool staging allocation failed while fetching "
                        "decode KV"
                    )
                moved_host_indices, moved_device_indices = self.controller.move_indices(
                    host_indices, device_indices
                )
                transfer_total_bytes += int(
                    device_indices.numel()
                    * self.controller.mem_pool_host.size_per_token
                )

            state_transfers = self._build_state_pool_transfers(
                req.host_kv_id, state_indices, tp_rank=source_tp_rank
            )
            moved_state_transfers = []
            for transfer in state_transfers:
                moved_host, moved_device = self.controller.move_indices(
                    transfer.host_indices, transfer.device_indices
                )
                moved_state_transfers.append(
                    PoolTransfer(
                        name=transfer.name,
                        host_indices=moved_host,
                        device_indices=moved_device,
                        keys=transfer.keys,
                    )
                )
                entry = self.controller.mem_pool_host.entry_map[transfer.name]
                transfer_total_bytes += int(
                    transfer.device_indices.numel() * entry.host_pool.size_per_token
                )

            state_page_counts = {
                transfer.name.value: len(transfer.keys or [])
                for transfer in state_transfers
            }
            dependency_events = []
            if host_indices is not None or moved_state_transfers:
                dependency_events = self._record_pull_dependencies()

            return _HostKVStagedRead(
                host_kv_id=req.host_kv_id,
                source_tp_rank=source_tp_rank,
                num_pages=num_pages,
                host_indices=host_indices,
                moved_host_indices=moved_host_indices,
                moved_device_indices=moved_device_indices,
                state_transfers=state_transfers,
                moved_state_transfers=moved_state_transfers,
                dependency_events=dependency_events,
                transfer_total_bytes=transfer_total_bytes,
                state_page_counts=state_page_counts,
            )
        except BaseException:
            if host_indices is not None:
                self.controller.mem_pool_host.free(host_indices)
            self._free_state_pool_transfers(state_transfers)
            raise

    def fetch_staged_read(
        self,
        req: Req,
        metadata_buffers: MetadataBuffers,
        metadata_index: int,
        staged: _HostKVStagedRead,
        submitted_at: float,
    ) -> _HostKVFetchResult:
        """Fetch metadata and KV into pinned host buffers on a worker thread."""
        fetch_start = time.perf_counter()
        metadata_start = time.perf_counter()
        object_info = self.get_metadata(req, metadata_buffers, metadata_index)
        metadata_get_s = time.perf_counter() - metadata_start

        if staged.num_pages != object_info.num_pages:
            raise RuntimeError(
                f"Host KV Object page count mismatch for {req.host_kv_id}: "
                f"decode={staged.num_pages}, object={object_info.num_pages}"
            )
        if staged.state_page_counts != (object_info.state_page_counts or {}):
            raise RuntimeError(
                f"Host KV Object state page count mismatch for {req.host_kv_id}: "
                f"decode={staged.state_page_counts}, "
                f"object={object_info.state_page_counts or {}}"
            )

        kv_get_s = 0.0
        if staged.host_indices is not None:
            kv_start = time.perf_counter()
            keys = self._page_keys(
                staged.host_kv_id,
                0,
                staged.num_pages,
                tp_rank=staged.source_tp_rank,
            )
            results = self.storage.batch_get_v1(keys, staged.host_indices)
            kv_get_s = time.perf_counter() - kv_start
            if not all(results):
                raise RuntimeError(
                    f"Mooncake Host KV page get failed for {staged.host_kv_id}: "
                    f"{results}"
                )

        state_get_s = 0.0
        if staged.state_transfers:
            state_start = time.perf_counter()
            results = self.storage.batch_get_v2(staged.state_transfers)
            state_get_s = time.perf_counter() - state_start
            for transfer in staged.state_transfers:
                pool_results = results.get(transfer.name, [])
                if not all(pool_results):
                    raise RuntimeError(
                        f"Mooncake Host KV state get failed for {staged.host_kv_id} "
                        f"pool={transfer.name}: {pool_results}"
                    )

        return _HostKVFetchResult(
            object_info=object_info,
            queue_wait_s=fetch_start - submitted_at,
            metadata_get_s=metadata_get_s,
            kv_get_s=kv_get_s,
            state_get_s=state_get_s,
            attached_tokens=staged.num_pages * self.page_size,
            attached_state_tokens=sum(
                int(transfer.device_indices.numel())
                for transfer in staged.state_transfers
            ),
        )

    def start_staged_read_attach(self, staged: _HostKVStagedRead) -> None:
        """Enqueue host-to-device copies and return without synchronizing."""
        if staged.host_indices is None and not staged.moved_state_transfers:
            return

        device_module = get_device_module()
        pull_stream = self._get_pull_stream()
        ready_event = device_module.Event()
        staged.ready_event = ready_event
        staged.attach_start_s = time.perf_counter()

        try:
            with device_module.stream(pull_stream):
                for dependency_event in staged.dependency_events:
                    dependency_event.wait(pull_stream)
                if staged.host_indices is not None:
                    for layer_id in range(self.controller.layer_num):
                        self.controller.mem_pool_host.load_to_device_per_layer(
                            self.controller.mem_pool_device,
                            staged.moved_host_indices,
                            staged.moved_device_indices,
                            layer_id,
                            self.controller.io_backend,
                        )
                for transfer in staged.moved_state_transfers:
                    entry = self.controller.mem_pool_host.entry_map.get(transfer.name)
                    if entry is None:
                        raise RuntimeError(
                            f"Host KV Pool missing host pool for {transfer.name}"
                        )
                    for layer_id in range(self.controller.layer_num):
                        local_layer_id = entry.layer_mapper(layer_id)
                        if local_layer_id is None:
                            continue
                        entry.host_pool.load_to_device_per_layer(
                            entry.device_pool,
                            transfer.host_indices,
                            transfer.device_indices,
                            local_layer_id,
                            self.controller.io_backend,
                        )
                ready_event.record()
                self._record_tensor_on_stream(staged.moved_host_indices, pull_stream)
                self._record_tensor_on_stream(staged.moved_device_indices, pull_stream)
                for transfer in staged.moved_state_transfers:
                    self._record_tensor_on_stream(transfer.host_indices, pull_stream)
                    self._record_tensor_on_stream(transfer.device_indices, pull_stream)
        except BaseException as exc:
            staged.attach_failure = exc
            # Kernels may already have been queued before a later layer failed.
            # Put the completion marker after them so buffers are not recycled early.
            with device_module.stream(pull_stream):
                ready_event.record()

    @staticmethod
    def staged_read_ready(staged: _HostKVStagedRead) -> bool:
        return staged.ready_event is None or staged.ready_event.query()

    def release_staged_read(self, staged: _HostKVStagedRead) -> None:
        if staged.released:
            return
        staged.released = True
        if staged.host_indices is not None:
            self.controller.mem_pool_host.free(staged.host_indices)
        self._free_state_pool_transfers(staged.state_transfers)

    def stage_write(
        self,
        host_kv_id: str,
        page_indices: npt.NDArray[np.int32],
        start_page: int,
        state_indices: Optional[List] = None,
    ) -> _HostKVStagedWrite:
        """Submit GPU-to-host copies without blocking the scheduler thread."""
        submit_start = time.perf_counter()
        host_indices = None
        moved_host_indices = None
        moved_device_indices = None
        keys: List[str] = []
        state_transfers: List[PoolTransfer] = []
        moved_state_transfers: List[PoolTransfer] = []
        transfer_total_bytes = 0

        try:
            num_pages = len(page_indices)
            if num_pages > 0:
                device_indices = _expand_page_indices(
                    page_indices, self.page_size, self.controller.device
                )
                host_indices = self.controller.mem_pool_host.alloc(len(device_indices))
                if host_indices is None:
                    raise RuntimeError(
                        "Host KV Pool staging allocation failed while backing up "
                        "prefill KV"
                    )
                moved_host_indices, moved_device_indices = self.controller.move_indices(
                    host_indices, device_indices
                )
                keys = self._page_keys(host_kv_id, start_page, num_pages)
                transfer_total_bytes += int(
                    device_indices.numel()
                    * self.controller.mem_pool_host.size_per_token
                )

            state_transfers = self._build_state_pool_transfers(
                host_kv_id, state_indices
            )
            for transfer in state_transfers:
                moved_host, moved_device = self.controller.move_indices(
                    transfer.host_indices, transfer.device_indices
                )
                moved_state_transfers.append(
                    PoolTransfer(
                        name=transfer.name,
                        host_indices=moved_host,
                        device_indices=moved_device,
                        keys=transfer.keys,
                    )
                )
                entry = self.controller.mem_pool_host.entry_map[transfer.name]
                transfer_total_bytes += int(
                    transfer.device_indices.numel() * entry.host_pool.size_per_token
                )

            state_page_counts = {
                transfer.name.value: len(transfer.keys or [])
                for transfer in state_transfers
            }
            start_event = None
            ready_event = None
            if host_indices is not None or moved_state_transfers:
                device_module = get_device_module()
                push_stream = self._get_push_stream()
                start_event = device_module.Event()
                ready_event = device_module.Event()
                start_event.record()
                with device_module.stream(push_stream):
                    start_event.wait(push_stream)
                    if host_indices is not None:
                        self.controller.mem_pool_host.backup_from_device_all_layer(
                            self.controller.mem_pool_device,
                            moved_host_indices,
                            moved_device_indices,
                            self.controller.io_backend,
                        )
                    for transfer in moved_state_transfers:
                        entry = self.controller.mem_pool_host.entry_map.get(
                            transfer.name
                        )
                        if entry is None:
                            raise RuntimeError(
                                f"Host KV Pool missing host pool for {transfer.name}"
                            )
                        entry.host_pool.backup_from_device_all_layer(
                            entry.device_pool,
                            transfer.host_indices,
                            transfer.device_indices,
                            self.controller.io_backend,
                        )
                    ready_event.record()
                    self._record_tensor_on_stream(moved_host_indices, push_stream)
                    self._record_tensor_on_stream(moved_device_indices, push_stream)
                    for transfer in moved_state_transfers:
                        self._record_tensor_on_stream(
                            transfer.host_indices, push_stream
                        )
                        self._record_tensor_on_stream(
                            transfer.device_indices, push_stream
                        )

            return _HostKVStagedWrite(
                host_indices=host_indices,
                moved_host_indices=moved_host_indices,
                moved_device_indices=moved_device_indices,
                keys=keys,
                state_transfers=state_transfers,
                moved_state_transfers=moved_state_transfers,
                start_event=start_event,
                ready_event=ready_event,
                transfer_total_bytes=transfer_total_bytes,
                state_page_counts=state_page_counts,
                submit_latency_s=time.perf_counter() - submit_start,
            )
        except BaseException:
            if host_indices is not None:
                self.controller.mem_pool_host.free(host_indices)
            self._free_state_pool_transfers(state_transfers)
            raise

    def commit_staged_write(
        self, host_kv_id: str, staged: _HostKVStagedWrite
    ) -> _HostKVPushResult:
        ready_wait_s = 0.0
        kv_put_s = 0.0
        state_put_s = 0.0
        try:
            if staged.ready_event is not None:
                wait_start = time.perf_counter()
                staged.ready_event.synchronize()
                ready_wait_s = time.perf_counter() - wait_start

            if staged.host_indices is not None:
                put_start = time.perf_counter()
                results = self.storage.batch_set_v1(staged.keys, staged.host_indices)
                kv_put_s = time.perf_counter() - put_start
                if not all(results):
                    raise RuntimeError(
                        f"Mooncake Host KV page put failed for {host_kv_id}: {results}"
                    )

            if staged.state_transfers:
                state_start = time.perf_counter()
                results = self.storage.batch_set_v2(staged.state_transfers)
                state_put_s = time.perf_counter() - state_start
                for transfer in staged.state_transfers:
                    pool_results = results.get(transfer.name, [])
                    if not all(pool_results):
                        raise RuntimeError(
                            f"Mooncake Host KV state put failed for {host_kv_id} "
                            f"pool={transfer.name}: {pool_results}"
                        )

            return _HostKVPushResult(
                submit_latency_s=staged.submit_latency_s,
                ready_wait_s=ready_wait_s,
                kv_put_s=kv_put_s,
                state_put_s=state_put_s,
                metadata_put_s=0.0,
                task_latency_s=ready_wait_s + kv_put_s + state_put_s,
                transfer_total_bytes=staged.transfer_total_bytes,
            )
        finally:
            self._release_staged_write(staged)

    def _backup_device_pages_to_host(
        self, device_indices: torch.Tensor
    ) -> torch.Tensor:
        host_indices = self.controller.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            raise RuntimeError(
                "Host KV Pool staging allocation failed while backing up prefill KV"
            )
        moved_host, moved_device = self.controller.move_indices(
            host_indices, device_indices
        )
        self.controller.mem_pool_host.backup_from_device_all_layer(
            self.controller.mem_pool_device,
            moved_host,
            moved_device,
            self.controller.io_backend,
        )
        self._sync_device()
        return host_indices

    def _backup_state_pages_to_host(self, transfer: PoolTransfer) -> None:
        entry = self.controller.mem_pool_host.entry_map.get(transfer.name)
        if entry is None:
            raise RuntimeError(f"Host KV Pool missing host pool for {transfer.name}")
        moved_host, moved_device = self.controller.move_indices(
            transfer.host_indices, transfer.device_indices
        )
        entry.host_pool.backup_from_device_all_layer(
            entry.device_pool,
            moved_host,
            moved_device,
            self.controller.io_backend,
        )

    def _load_host_pages_to_device(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> None:
        moved_host, moved_device = self.controller.move_indices(
            host_indices, device_indices
        )
        for layer_id in range(self.controller.layer_num):
            self.controller.mem_pool_host.load_to_device_per_layer(
                self.controller.mem_pool_device,
                moved_host,
                moved_device,
                layer_id,
                self.controller.io_backend,
            )
        self._sync_device()

    def _load_state_pages_to_device(self, transfer: PoolTransfer) -> None:
        entry = self.controller.mem_pool_host.entry_map.get(transfer.name)
        if entry is None:
            raise RuntimeError(f"Host KV Pool missing host pool for {transfer.name}")
        moved_host, moved_device = self.controller.move_indices(
            transfer.host_indices, transfer.device_indices
        )
        for layer_id in range(self.controller.layer_num):
            local_layer_id = entry.layer_mapper(layer_id)
            if local_layer_id is None:
                continue
            entry.host_pool.load_to_device_per_layer(
                entry.device_pool,
                moved_host,
                moved_device,
                local_layer_id,
                self.controller.io_backend,
            )

    def _state_types(self) -> List[StateType]:
        for queue_name in (
            "disagg_prefill_bootstrap_queue",
            "disagg_decode_prealloc_queue",
        ):
            queue = getattr(self.scheduler, queue_name, None)
            kv_manager = getattr(queue, "kv_manager", None)
            kv_args = getattr(kv_manager, "kv_args", None)
            state_types = getattr(kv_args, "state_types", None)
            if state_types is not None:
                return list(state_types)
        return []

    def _pool_name_for_state_type(self, state_type) -> PoolName:
        value = (
            state_type.value if isinstance(state_type, StateType) else str(state_type)
        )
        if value == StateType.NSA.value:
            return PoolName.INDEXER
        raise RuntimeError(
            "Host KV Pool currently supports auxiliary state transfer only for "
            f"NSA/indexer, got {value}"
        )

    def _normalize_state_indices(self, indices) -> npt.NDArray[np.int32]:
        if indices is None:
            return np.empty((0,), dtype=np.int32)
        if isinstance(indices, (list, tuple)):
            parts = [
                np.asarray(item, dtype=np.int32).reshape(-1)
                for item in indices
                if item is not None
            ]
            if not parts:
                return np.empty((0,), dtype=np.int32)
            return np.concatenate(parts)
        return np.asarray(indices, dtype=np.int32).reshape(-1)

    def _build_state_pool_transfers(
        self,
        host_kv_id: str,
        state_indices: Optional[List],
        tp_rank: Optional[int] = None,
    ) -> List[PoolTransfer]:
        if not state_indices:
            return []

        state_types = self._state_types()
        if not state_types:
            raise RuntimeError(
                "Host KV Pool received auxiliary state indices but could not "
                "resolve state_types from the local KV manager"
            )

        transfers = []
        for idx, indices in enumerate(state_indices):
            if indices is None:
                continue
            if idx >= len(state_types):
                raise RuntimeError(
                    "Host KV Pool state_indices length exceeds state_types length: "
                    f"{len(state_indices)} > {len(state_types)}"
                )

            page_indices = self._normalize_state_indices(indices)
            if len(page_indices) == 0:
                continue

            pool_name = self._pool_name_for_state_type(state_types[idx])
            entry = getattr(self.controller.mem_pool_host, "entry_map", {}).get(
                pool_name
            )
            if entry is None:
                raise RuntimeError(f"Host KV Pool missing host pool for {pool_name}")

            registered_pools = getattr(self.storage, "registered_pools", {})
            if pool_name not in registered_pools:
                raise RuntimeError(
                    f"Mooncake Host KV storage has not registered pool {pool_name}"
                )

            device_indices = _expand_page_indices(
                page_indices, self.page_size, self.controller.device
            )
            host_indices = entry.host_pool.alloc(len(device_indices))
            if host_indices is None:
                raise RuntimeError(
                    "Host KV Pool staging allocation failed for auxiliary "
                    f"{pool_name} pages"
                )

            transfers.append(
                PoolTransfer(
                    name=pool_name,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    keys=self._page_keys(
                        host_kv_id, 0, len(page_indices), tp_rank=tp_rank
                    ),
                )
            )

        return transfers

    def _free_state_pool_transfers(self, transfers: List[PoolTransfer]) -> None:
        for transfer in transfers:
            entry = getattr(self.controller.mem_pool_host, "entry_map", {}).get(
                transfer.name
            )
            if entry is not None and transfer.host_indices is not None:
                entry.host_pool.free(transfer.host_indices)

    def put_state_pages(
        self, host_kv_id: str, state_indices: Optional[List]
    ) -> tuple[dict[str, int], int]:
        transfers = self._build_state_pool_transfers(host_kv_id, state_indices)
        if not transfers:
            return {}, 0

        try:
            for transfer in transfers:
                self._backup_state_pages_to_host(transfer)
            self._sync_device()
            results = self.storage.batch_set_v2(transfers)
            for transfer in transfers:
                pool_results = results.get(transfer.name, [])
                if not all(pool_results):
                    raise RuntimeError(
                        f"Mooncake Host KV state put failed for {host_kv_id} "
                        f"pool={transfer.name}: {pool_results}"
                    )
            page_counts = {
                transfer.name.value: len(transfer.keys or []) for transfer in transfers
            }
            bytes_written = sum(
                int(
                    transfer.device_indices.numel()
                    * self.controller.mem_pool_host.entry_map[
                        transfer.name
                    ].host_pool.size_per_token
                )
                for transfer in transfers
            )
            return page_counts, bytes_written
        finally:
            self._free_state_pool_transfers(transfers)

    def attach_state_pages(
        self,
        host_kv_id: str,
        state_indices: Optional[List],
        object_info: HostKVObjectInfo,
        source_tp_rank: Optional[int] = None,
    ) -> int:
        expected_counts = object_info.state_page_counts or {}
        transfers = self._build_state_pool_transfers(
            host_kv_id, state_indices, tp_rank=source_tp_rank
        )
        actual_counts = {
            transfer.name.value: len(transfer.keys or []) for transfer in transfers
        }
        if actual_counts != expected_counts:
            if expected_counts or actual_counts:
                raise RuntimeError(
                    f"Host KV Object state page count mismatch for {host_kv_id}: "
                    f"decode={actual_counts}, object={expected_counts}"
                )
            return 0
        if not transfers:
            return 0

        try:
            results = self.storage.batch_get_v2(transfers)
            for transfer in transfers:
                pool_results = results.get(transfer.name, [])
                if not all(pool_results):
                    raise RuntimeError(
                        f"Mooncake Host KV state get failed for {host_kv_id} "
                        f"pool={transfer.name}: {pool_results}"
                    )
            for transfer in transfers:
                self._load_state_pages_to_device(transfer)
            self._sync_device()
            return sum(int(transfer.device_indices.numel()) for transfer in transfers)
        finally:
            self._free_state_pool_transfers(transfers)

    def put_pages(
        self,
        host_kv_id: str,
        page_indices: npt.NDArray[np.int32],
        start_page: int,
    ) -> int:
        num_pages = len(page_indices)
        if num_pages == 0:
            return 0

        device_indices = _expand_page_indices(
            page_indices, self.page_size, self.controller.device
        )
        host_indices = self._backup_device_pages_to_host(device_indices)
        try:
            keys = self._page_keys(host_kv_id, start_page, num_pages)
            results = self.storage.batch_set_v1(keys, host_indices)
            if not all(results):
                raise RuntimeError(
                    f"Mooncake Host KV page put failed for {host_kv_id}: {results}"
                )
        finally:
            self.controller.mem_pool_host.free(host_indices)
        return int(
            device_indices.numel() * self.controller.mem_pool_host.size_per_token
        )

    def attach_pages(
        self,
        host_kv_id: str,
        page_indices: npt.NDArray[np.int32],
        expected_num_pages: int,
        source_tp_rank: Optional[int] = None,
    ) -> int:
        num_pages = len(page_indices)
        if num_pages != expected_num_pages:
            raise RuntimeError(
                f"Host KV Object page count mismatch for {host_kv_id}: "
                f"decode={num_pages}, object={expected_num_pages}"
            )
        if num_pages == 0:
            return 0

        host_indices = self.controller.mem_pool_host.alloc(num_pages * self.page_size)
        if host_indices is None:
            raise RuntimeError(
                "Host KV Pool staging allocation failed while attaching decode KV"
            )
        try:
            keys = self._page_keys(host_kv_id, 0, num_pages, tp_rank=source_tp_rank)
            results = self.storage.batch_get_v1(keys, host_indices)
            if not all(results):
                raise RuntimeError(
                    f"Mooncake Host KV page get failed for {host_kv_id}: {results}"
                )
            device_indices = _expand_page_indices(
                page_indices, self.page_size, self.controller.device
            )
            self._load_host_pages_to_device(host_indices, device_indices)
            return int(device_indices.numel())
        finally:
            self.controller.mem_pool_host.free(host_indices)

    def _object_info(
        self,
        req: Req,
        num_pages: int,
        state_page_counts: Optional[dict[str, int]] = None,
    ) -> HostKVObjectInfo:
        kv_cache = self.controller.mem_pool_device
        return HostKVObjectInfo(
            host_kv_id=req.host_kv_id,
            token_len=len(req.origin_input_ids),
            page_size=self.page_size,
            num_pages=num_pages,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            layout=self.controller.mem_pool_host.layout,
            dtype=str(getattr(kv_cache, "store_dtype", "")),
            is_mla_backend=not hasattr(kv_cache, "head_num"),
            state_page_counts=state_page_counts or {},
        )

    def put_metadata(
        self,
        req: Req,
        metadata_buffers: MetadataBuffers,
        metadata_index: int,
        num_pages: int,
        state_page_counts: Optional[dict[str, int]] = None,
    ) -> None:
        info = self._object_info(req, num_pages, state_page_counts)
        ptrs, _, item_lens = metadata_buffers.get_buf_infos()
        rows = [
            ctypes.string_at(ptr + item_len * metadata_index, item_len)
            for ptr, item_len in zip(ptrs, item_lens)
        ]
        header = {
            "object": info.__dict__,
            "metadata_item_lens": item_lens,
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        payload = (
            _METADATA_MAGIC
            + struct.pack("<I", len(header_bytes))
            + header_bytes
            + b"".join(rows)
        )
        ret = self.storage.store.put(self._meta_key(req.host_kv_id), payload)
        if ret != 0:
            raise RuntimeError(f"Mooncake Host KV metadata put failed: ret={ret}")

    def get_metadata(
        self,
        req: Req,
        metadata_buffers: MetadataBuffers,
        metadata_index: int,
    ) -> HostKVObjectInfo:
        source_tp_rank = self._source_tp_rank(req)
        meta_key = self._meta_key(req.host_kv_id, tp_rank=source_tp_rank)
        payload = self.storage.store.get(meta_key)
        if payload is None:
            raise RuntimeError(
                f"Host KV Object metadata not found: {req.host_kv_id} "
                f"source_tp_rank={source_tp_rank}"
            )
        if not isinstance(payload, (bytes, bytearray)):
            raise RuntimeError(
                f"Unexpected Host KV metadata payload type: {type(payload).__name__}"
            )
        if not payload.startswith(_METADATA_MAGIC):
            raise RuntimeError(
                f"Invalid Host KV metadata payload for {req.host_kv_id} "
                f"source_tp_rank={source_tp_rank} key={meta_key}"
            )

        header_offset = len(_METADATA_MAGIC)
        header_len = struct.unpack("<I", payload[header_offset : header_offset + 4])[0]
        body_offset = header_offset + 4 + header_len
        header = json.loads(payload[header_offset + 4 : body_offset].decode("utf-8"))
        object_info = HostKVObjectInfo(**header["object"])
        self._validate_object_info(req, object_info, source_tp_rank)

        ptrs, _, item_lens = metadata_buffers.get_buf_infos()
        expected_lens = header["metadata_item_lens"]
        if item_lens != expected_lens:
            raise RuntimeError(
                "Host KV metadata layout mismatch: "
                f"decode={item_lens}, object={expected_lens}"
            )

        offset = body_offset
        for ptr, item_len in zip(ptrs, item_lens):
            row = payload[offset : offset + item_len]
            if len(row) != item_len:
                raise RuntimeError("Truncated Host KV metadata payload")
            ctypes.memmove(ptr + item_len * metadata_index, row, item_len)
            offset += item_len
        return object_info

    def _validate_object_info(
        self, req: Req, info: HostKVObjectInfo, source_tp_rank: int
    ) -> None:
        if info.host_kv_id != req.host_kv_id:
            raise RuntimeError(
                f"Host KV Object id mismatch: request={req.host_kv_id}, "
                f"object={info.host_kv_id}"
            )
        if info.token_len != len(req.origin_input_ids):
            raise RuntimeError(
                f"Host KV Object token length mismatch: request={len(req.origin_input_ids)}, "
                f"object={info.token_len}"
            )
        if info.page_size != self.page_size:
            raise RuntimeError(
                f"Host KV Object page size mismatch: runtime={self.page_size}, "
                f"object={info.page_size}"
            )
        if info.tp_rank != source_tp_rank or info.pp_rank != self.pp_rank:
            raise RuntimeError(
                "Host KV Object rank mismatch: "
                f"source=(tp{source_tp_rank},pp{self.pp_rank}), "
                f"runtime=(tp{self.tp_rank},pp{self.pp_rank}), "
                f"object=(tp{info.tp_rank},pp{info.pp_rank})"
            )
        if info.layout != self.controller.mem_pool_host.layout:
            raise RuntimeError(
                f"Host KV Object layout mismatch: runtime={self.controller.mem_pool_host.layout}, "
                f"object={info.layout}"
            )


class HostKVSender:
    """Sender-like adapter used by the prefill scheduler in Host KV Pool mode."""

    def __init__(
        self, runtime: HostKVPoolRuntime, req: Req, metadata_buffers: MetadataBuffers
    ):
        self.runtime = runtime
        self.req = req
        self.metadata_buffers = metadata_buffers
        self.status = KVPoll.WaitingForInput
        self.failure: Optional[BaseException] = None
        self.metadata_index: Optional[int] = None
        self.expected_pages = 0
        self.sent_pages = 0
        self.transfer_latency_s = 0.0
        self.transfer_total_bytes = 0
        self._next_is_last = False
        self._push_start_s: Optional[float] = None
        self._push_futures: List[Future] = []
        self._final_future: Optional[Future] = None
        self._closed = False
        self._stats_logged = False
        self._debug_stats: Optional[str] = None

    @property
    def kv_mgr(self):
        return None

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.expected_pages = num_kv_indices
        self.metadata_index = aux_index
        self.status = KVPoll.WaitingForInput

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        self._next_is_last = last_chunk
        return num_pages > 0 or last_chunk

    def pop_decode_prefix_len(self) -> int:
        return 0

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ):
        if self._closed:
            return
        if self.metadata_index is None:
            self.status = KVPoll.Failed
            self.failure = RuntimeError(
                "Host KV sender metadata index is not initialized"
            )
            return

        if self._push_start_s is None:
            self._push_start_s = time.perf_counter()

        staged = None
        try:
            is_last = self._next_is_last
            staged = self.runtime.stage_write(
                self.req.host_kv_id,
                kv_indices,
                self.sent_pages,
                state_indices=state_indices if is_last else None,
            )
            self.sent_pages += len(kv_indices)
            self.transfer_total_bytes += staged.transfer_total_bytes

            prior_futures = tuple(self._push_futures)
            sent_pages = self.sent_pages
            expected_pages = self.expected_pages
            metadata_index = self.metadata_index

            def commit() -> _HostKVPushResult:
                task_start = time.perf_counter()
                result = self.runtime.commit_staged_write(self.req.host_kv_id, staged)
                if is_last:
                    for prior in prior_futures:
                        prior.result()
                    if sent_pages != expected_pages:
                        raise RuntimeError(
                            f"Host KV sender wrote {sent_pages} pages, "
                            f"expected {expected_pages}"
                        )
                    metadata_start = time.perf_counter()
                    self.runtime.put_metadata(
                        self.req,
                        self.metadata_buffers,
                        metadata_index,
                        expected_pages,
                        state_page_counts=staged.state_page_counts,
                    )
                    result.metadata_put_s = time.perf_counter() - metadata_start
                result.task_latency_s = time.perf_counter() - task_start
                return result

            future = self.runtime.submit_push(commit)
            self._push_futures.append(future)
            if is_last:
                self._final_future = future
                self._closed = True
            self.status = KVPoll.Transferring
        except BaseException as exc:
            if staged is not None:
                self.runtime._release_staged_write(staged)
            self._defer_failure(exc)

    def _defer_failure(self, exc: BaseException) -> None:
        self.failure = exc
        self._closed = True
        prior_futures = tuple(self._push_futures)
        if not prior_futures:
            self.status = KVPoll.Failed
            logger.error(
                "Host KV sender submission failed for %s: %s", self.req.rid, exc
            )
            return

        def drain_then_fail() -> _HostKVPushResult:
            for prior in prior_futures:
                try:
                    prior.result()
                except BaseException:
                    pass
            raise exc

        self._final_future = self.runtime.submit_push(drain_then_fail)
        self._push_futures.append(self._final_future)
        self.status = KVPoll.Transferring

    def _refresh_status(self) -> None:
        if self.status in (KVPoll.Success, KVPoll.Failed):
            return
        if self._final_future is None or not self._final_future.done():
            return

        try:
            self._final_future.result()
            results = [future.result() for future in self._push_futures]
        except BaseException as exc:
            self.failure = self.failure or exc
            self.status = KVPoll.Failed
            if self._push_start_s is not None:
                self.transfer_latency_s = time.perf_counter() - self._push_start_s
            logger.error("Host KV sender failed for %s: %s", self.req.rid, self.failure)
            self.runtime.schedule_release(self.req, "sender_failed")
            return

        self.status = KVPoll.Success
        if not self._stats_logged:
            self._stats_logged = True
            submit_ms = sum(result.submit_latency_s for result in results) * 1000
            ready_wait_ms = sum(result.ready_wait_s for result in results) * 1000
            kv_put_ms = sum(result.kv_put_s for result in results) * 1000
            state_put_ms = sum(result.state_put_s for result in results) * 1000
            metadata_put_ms = sum(result.metadata_put_s for result in results) * 1000
            task_ms = sum(result.task_latency_s for result in results) * 1000
            component_ms = (
                submit_ms + ready_wait_ms + kv_put_ms + state_put_ms + metadata_put_ms
            )
            self.transfer_latency_s = component_ms / 1000
            transfer_total_mb = self.transfer_total_bytes / (1 << 20)
            self._debug_stats = (
                f"HostKVPushStats(host_kv_id={self.req.host_kv_id}, "
                f"pages={self.sent_pages}, total={component_ms:.2f}ms, "
                f"task={task_ms:.2f}ms, "
                f"submit={submit_ms:.2f}ms, ready_wait={ready_wait_ms:.2f}ms, "
                f"kv_put={kv_put_ms:.2f}ms, state_put={state_put_ms:.2f}ms, "
                f"metadata_put={metadata_put_ms:.2f}ms, "
                f"transfer_total={transfer_total_mb:.2f}MB)"
            )
            logger.info(
                "HostKVPushStats(rid=%s, host_kv_id=%s, pages=%d): "
                "total=%.2fms, task=%.2fms, submit=%.2fms, ready_wait=%.2fms, "
                "kv_put=%.2fms, state_put=%.2fms, metadata_put=%.2fms, "
                "transfer_total=%.2fMB",
                self.req.rid,
                self.req.host_kv_id,
                self.sent_pages,
                component_ms,
                task_ms,
                submit_ms,
                ready_wait_ms,
                kv_put_ms,
                state_put_ms,
                metadata_put_ms,
                transfer_total_mb,
            )

    def get_debug_stats(self) -> Optional[str]:
        self._refresh_status()
        return self._debug_stats

    def get_transfer_metric(self) -> KVTransferMetric:
        return KVTransferMetric(
            transfer_latency_s=self.transfer_latency_s,
            transfer_total_bytes=self.transfer_total_bytes,
        )

    def poll(self) -> KVPoll:
        self._refresh_status()
        return self.status

    def failure_exception(self):
        self._refresh_status()
        if self.failure is not None:
            raise self.failure

    def clear(self):
        self._push_futures.clear()
        self._final_future = None

    def abort(self):
        self._closed = True
        self.status = KVPoll.Failed
        self.failure = RuntimeError("Host KV sender aborted")
        self.runtime.schedule_release(self.req, "sender_aborted")


class HostKVReceiver:
    """Receiver-like adapter used by decode preallocation in Host KV Pool mode."""

    def __init__(
        self,
        runtime: HostKVPoolRuntime,
        req: Req,
        metadata_buffers: MetadataBuffers,
    ):
        self.runtime = runtime
        self.req = req
        self.metadata_buffers = metadata_buffers
        self.status = KVPoll.WaitingForInput
        self.failure: Optional[BaseException] = None
        self.object_info: Optional[HostKVObjectInfo] = None
        self.require_staging = False
        self._staged_read: Optional[_HostKVStagedRead] = None
        self._fetch_future: Optional[Future] = None
        self._fetch_result: Optional[_HostKVFetchResult] = None
        self._attach_started = False
        self._aborted = False
        self._release_requested = False
        self._pull_start_s: Optional[float] = None
        self._stats_logged = False

    def init(self, prefill_dp_rank: int):
        self.status = KVPoll.WaitingForInput

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        if decode_prefix_len not in (None, 0):
            self.status = KVPoll.Failed
            self.failure = RuntimeError(
                "Host KV Pool MVP does not support decode radix-cache prefix attach"
            )
            return
        if aux_index is None:
            self.status = KVPoll.Failed
            self.failure = RuntimeError("Host KV receiver metadata index is missing")
            return
        if self._fetch_future is not None or self._attach_started:
            self.status = KVPoll.Failed
            self.failure = RuntimeError("Host KV receiver pull was submitted twice")
            return

        try:
            self._pull_start_s = time.perf_counter()
            self.status = KVPoll.Transferring
            self._staged_read = self.runtime.stage_read(
                self.req,
                np.asarray(kv_indices, dtype=np.int32).copy(),
                state_indices=state_indices,
            )
            submitted_at = time.perf_counter()

            def fetch() -> _HostKVFetchResult:
                return self.runtime.fetch_staged_read(
                    self.req,
                    self.metadata_buffers,
                    aux_index,
                    self._staged_read,
                    submitted_at,
                )

            self._fetch_future = self.runtime.submit_pull(fetch)
        except BaseException as exc:
            if self._staged_read is not None:
                self.runtime.release_staged_read(self._staged_read)
            self._finish_failure(exc, "receiver_submit_failed")

    def _schedule_release_once(self, reason: str) -> None:
        if self._release_requested:
            return
        self._release_requested = True
        self.runtime.schedule_release(self.req, reason)

    def _finish_failure(self, exc: BaseException, reason: str) -> None:
        self.failure = self.failure or exc
        if self._staged_read is not None:
            self.runtime.release_staged_read(self._staged_read)
        self.status = KVPoll.Failed
        self._schedule_release_once(reason)
        logger.error("Host KV receiver failed for %s: %s", self.req.rid, self.failure)

    def _log_success(self) -> None:
        if self._stats_logged or self._fetch_result is None:
            return
        self._stats_logged = True
        now = time.perf_counter()
        total_s = now - self._pull_start_s if self._pull_start_s is not None else 0.0
        h2d_s = (
            now - self._staged_read.attach_start_s
            if self._staged_read is not None
            and self._staged_read.attach_start_s is not None
            else 0.0
        )
        result = self._fetch_result
        logger.info(
            "HostKVPullStats(rid=%s, host_kv_id=%s, source_tp_rank=%d, "
            "pages=%d): total=%.2fms, queue_wait=%.2fms, "
            "metadata_get=%.2fms, kv_get=%.2fms, state_get=%.2fms, "
            "h2d=%.2fms, kv_get_h2d=%.2fms, state_get_h2d=%.2fms, "
            "attached_tokens=%d, attached_state_tokens=%d",
            self.req.rid,
            self.req.host_kv_id,
            result.object_info.tp_rank,
            result.object_info.num_pages,
            total_s * 1000,
            result.queue_wait_s * 1000,
            result.metadata_get_s * 1000,
            result.kv_get_s * 1000,
            result.state_get_s * 1000,
            h2d_s * 1000,
            (result.kv_get_s + h2d_s) * 1000,
            result.state_get_s * 1000,
            result.attached_tokens,
            result.attached_state_tokens,
        )

    def _refresh_status(self) -> None:
        if self.status in (KVPoll.Success, KVPoll.Failed):
            return
        if self._fetch_future is None or not self._fetch_future.done():
            return

        if self._fetch_result is None:
            try:
                self._fetch_result = self._fetch_future.result()
            except BaseException as exc:
                if self._aborted and self.failure is not None:
                    exc = self.failure
                self._finish_failure(
                    exc, "receiver_aborted" if self._aborted else "receiver_failed"
                )
                return

            if self._aborted:
                self._finish_failure(
                    self.failure or RuntimeError("Host KV receiver aborted"),
                    "receiver_aborted",
                )
                return

        if not self._attach_started:
            try:
                self.runtime.start_staged_read_attach(self._staged_read)
                self._attach_started = True
            except BaseException as exc:
                self._finish_failure(exc, "receiver_attach_failed")
                return

        if not self.runtime.staged_read_ready(self._staged_read):
            return
        if self._staged_read.attach_failure is not None:
            self._finish_failure(
                self._staged_read.attach_failure, "receiver_attach_failed"
            )
            return
        if self._aborted:
            self._finish_failure(
                self.failure or RuntimeError("Host KV receiver aborted"),
                "receiver_aborted",
            )
            return

        self.object_info = self._fetch_result.object_info
        self.req.host_kv_object_info = self.object_info
        self._log_success()
        self.runtime.release_staged_read(self._staged_read)
        self.status = KVPoll.Success

    def poll(self) -> KVPoll:
        self._refresh_status()
        return self.status

    def failure_exception(self):
        self._refresh_status()
        if self.failure is not None:
            raise self.failure

    def clear(self):
        self._fetch_future = None
        self._fetch_result = None
        self._staged_read = None

    def abort(self):
        if self.status in (KVPoll.Success, KVPoll.Failed):
            return
        self._aborted = True
        self.failure = RuntimeError("Host KV receiver aborted")
        if self._fetch_future is None:
            self._finish_failure(self.failure, "receiver_aborted")
            return
        if self._fetch_future.cancel():
            self._finish_failure(self.failure, "receiver_aborted")
            return
        # A running fetch or H2D must retain its buffers and destination pages.
        # poll() finalizes the abort only after that in-flight work is safe.
        self.status = KVPoll.Transferring
