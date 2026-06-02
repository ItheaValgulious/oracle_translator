"""Phase 6 tests: GPU owner thread command/result channels."""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from engine.gpu_owner import (
    CameraFollowCommand,
    CameraTeleportCommand,
    ChunkPrefetchCommand,
    CreateWorldCommand,
    EntityGpuSyncCommand,
    GpuFramePayload,
    GpuCommand,
    GpuCommandType,
    GpuOptionsCommand,
    GpuOwnedWorldRuntime,
    GpuOwnerThread,
    GpuWorldCommandHandlers,
    GpuWorldMutationBuffer,
    InjectPressureRingWorldCommand,
    InjectPressureWorldCommand,
    PaintWorldCommand,
    RenderFrameCommand,
    SnapshotPollCommand,
    SnapshotRequestCommand,
    SnapshotServiceCommand,
    WorldCallableCommand,
    WorldStatusCommand,
)
from engine.snapshot_mailbox import SnapshotMailboxRegistry


class GpuOwnerLifecycleTests(unittest.TestCase):
    def test_start_and_shutdown(self) -> None:
        owner = GpuOwnerThread()
        owner.start()
        self.assertTrue(owner.is_running)
        owner.shutdown()
        self.assertFalse(owner.is_running)

    def test_double_start_is_noop(self) -> None:
        owner = GpuOwnerThread()
        owner.start()
        first_tid = owner.owner_thread_id
        owner.start()
        self.assertEqual(owner.owner_thread_id, first_tid)
        owner.shutdown()


class GpuOwnerCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.owner = GpuOwnerThread()
        self.owner.start()
        self.addCleanup(self.owner.shutdown)

    def test_run_callable_returns_value(self) -> None:
        result = self.owner.run_on_owner(lambda: 42, timeout=2.0)
        self.assertTrue(result.ok)
        self.assertEqual(result.value, 42)

    def test_result_if_ready_is_nonblocking(self) -> None:
        release = threading.Event()
        handle = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.RUN_CALLABLE,
                fn=lambda: (release.wait(timeout=2.0), 99)[1],
            )
        )
        self.assertIsNone(handle.result_if_ready())
        release.set()
        result = handle.wait(timeout=2.0)
        self.assertTrue(result.ok)
        self.assertEqual(result.value, 99)
        self.assertEqual(handle.result_if_ready(), result)

    def test_callable_runs_on_owner_thread(self) -> None:
        # The callable must run on the dedicated owner thread, not the
        # caller. We assert by reading the active thread id inside.
        main_tid = threading.get_ident()
        observed: dict[str, int] = {}

        def _on_owner() -> None:
            observed["tid"] = threading.get_ident()

        result = self.owner.run_on_owner(_on_owner, timeout=2.0)
        self.assertTrue(result.ok)
        self.assertNotEqual(observed["tid"], main_tid)
        self.assertEqual(observed["tid"], self.owner.owner_thread_id)

    def test_handler_exception_surfaces_as_failed_result(self) -> None:
        def _boom() -> None:
            raise RuntimeError("kaboom")

        result = self.owner.run_on_owner(_boom, timeout=2.0)
        self.assertFalse(result.ok)
        self.assertIn("kaboom", result.error)
        self.assertEqual(self.owner.stats.commands_failed, 1)

    def test_commands_execute_in_fifo_order(self) -> None:
        order: list[int] = []

        def make_appender(i: int):
            def _fn() -> None:
                order.append(i)
            return _fn

        handles = [self.owner.submit(GpuCommand(op=GpuCommandType.RUN_CALLABLE, fn=make_appender(i)))
                   for i in range(10)]
        for h in handles:
            h.wait(timeout=2.0)
        self.assertEqual(order, list(range(10)))

    def test_high_priority_command_runs_before_queued_normal_work(self) -> None:
        started = threading.Event()
        release = threading.Event()
        order: list[str] = []

        def _slow() -> None:
            started.set()
            release.wait(timeout=2.0)
            order.append("slow")

        self.owner.submit(
            GpuCommand(
                op=GpuCommandType.RUN_CALLABLE,
                fn=_slow,
            )
        )
        self.assertTrue(started.wait(timeout=2.0))
        normal = self.owner.submit(
            GpuCommand(op=GpuCommandType.RUN_CALLABLE, fn=lambda: order.append("normal"))
        )
        urgent = self.owner.submit_high_priority(
            GpuCommand(op=GpuCommandType.RUN_CALLABLE, fn=lambda: order.append("urgent"))
        )

        release.set()
        urgent.wait(timeout=2.0)
        normal.wait(timeout=2.0)

        self.assertEqual(order, ["slow", "urgent", "normal"])

    def test_registered_handler_is_called(self) -> None:
        received: list = []

        def handler(payload):
            received.append(payload)
            return f"got-{payload}"

        self.owner.register_handler(GpuCommandType.STEP_WORLD, handler)
        h = self.owner.submit(GpuCommand(op=GpuCommandType.STEP_WORLD, payload={"dt": 0.016}))
        result = h.wait(timeout=2.0)
        self.assertTrue(result.ok)
        self.assertEqual(result.value, "got-{'dt': 0.016}")
        self.assertEqual(received, [{"dt": 0.016}])

    def test_fire_and_forget_does_not_block(self) -> None:
        flag = threading.Event()

        def _slow():
            time.sleep(0.2)
            flag.set()

        start = time.perf_counter()
        self.owner.fire_and_forget(GpuCommand(op=GpuCommandType.RUN_CALLABLE, fn=_slow))
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.1)  # didn't wait for the 0.2s job
        flag.wait(timeout=2.0)
        self.assertTrue(flag.is_set())


class GpuWorldMutationBufferTests(unittest.TestCase):
    def test_world_mutations_are_buffered_and_flushed_in_order(self) -> None:
        buffer = GpuWorldMutationBuffer()
        overrides = {"vel_x": 2.0}
        buffer.paint_world(10, 11, 3, "stone", "stone_powder", overrides=overrides)
        overrides["vel_x"] = 99.0
        buffer.inject_pressure_world(12, 13, 4, 120.0)
        buffer.inject_pressure_ring_world(14, 15, 2, 5, 80.0)

        self.assertEqual(buffer.pending_count, 3)
        commands = buffer.drain()
        self.assertEqual(buffer.pending_count, 0)
        self.assertIsInstance(commands[0], PaintWorldCommand)
        self.assertEqual(commands[0].overrides["vel_x"], 2.0)
        self.assertIsInstance(commands[1], InjectPressureWorldCommand)
        self.assertIsInstance(commands[2], InjectPressureRingWorldCommand)

    def test_flush_to_world_applies_commands_and_clears_buffer(self) -> None:
        class FakeWorld:
            def __init__(self) -> None:
                self.calls: list[tuple] = []

            def paint_world(self, *args, **kwargs) -> None:
                self.calls.append(("paint", args, kwargs))

            def inject_pressure_world(self, *args) -> None:
                self.calls.append(("pressure", args))

            def inject_pressure_ring_world(self, *args) -> None:
                self.calls.append(("ring", args))

        buffer = GpuWorldMutationBuffer()
        buffer.paint_world(1, 2, 3, "fire", "fire", overrides={"vel_y": -2.0})
        buffer.inject_pressure_world(4, 5, 6, 7.0)
        buffer.inject_pressure_ring_world(8, 9, 10, 11, 12.0)

        world = FakeWorld()
        count = buffer.flush_to_world(world)

        self.assertEqual(count, 3)
        self.assertEqual(buffer.pending_count, 0)
        self.assertEqual([call[0] for call in world.calls], ["paint", "pressure", "ring"])
        self.assertEqual(world.calls[0][1], (1, 2, 3, "fire", "fire"))
        self.assertEqual(world.calls[0][2], {"overrides": {"vel_y": -2.0}})


class GpuWorldCommandHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.owner = GpuOwnerThread()
        self.owner.start()
        self.addCleanup(self.owner.shutdown)

    def test_world_handlers_run_on_owner_thread(self) -> None:
        class FakeGpu:
            def __init__(self) -> None:
                self.skip_liquid_physics = False
                self.calls: list[tuple] = []

            def register_entity_shape(self, *args) -> None:
                self.calls.append(("shape", threading.get_ident(), args))

            def upload_entity_states(self, states, *, origin_x: int, origin_y: int) -> None:
                self.calls.append(("states", threading.get_ident(), tuple(states), origin_x, origin_y))

            def update_entity_mask(self, rects) -> None:
                self.calls.append(("mask", threading.get_ident(), tuple(rects)))

            def set_skip_liquid_physics(self, value: bool) -> None:
                self.skip_liquid_physics = bool(value)
                self.calls.append(("skip_liquid", threading.get_ident(), bool(value)))

        class FakeWorld:
            def __init__(self) -> None:
                self.gpu_simulator = FakeGpu()
                self.active_width = 32
                self.active_height = 32
                self.viewport_width = 16
                self.viewport_height = 16
                self.world_width = 128
                self.world_height = 96
                self.active_origin_x = 0
                self.active_origin_y = 0
                self.camera_x = 0
                self.camera_y = 0
                self.chunk_cache = self
                self.paging_stats = SimpleNamespace(
                    shift_count=2,
                    last_shift_cache_hits=3,
                    last_shift_empty_hits=4,
                    last_shift_inflight_wait_hits=5,
                    last_shift_disk_loads=6,
                    last_shift_generates=7,
                    last_shift_saves=8,
                    last_shift_disk_load_seconds=0.011,
                    last_shift_generate_seconds=0.022,
                    last_shift_save_seconds=0.033,
                )
                self.gpu_writeback_queue_depth = 9
                self.active_chunk_patch_queue_depth = 10
                self.pending_writeback_count = 11
                self.calls: list[tuple] = []

            def snapshot_stats(self):
                return SimpleNamespace(
                    cached_chunks=12,
                    clean_resident_chunks=13,
                    dirty_resident_chunks=14,
                    prefetch_queued=15,
                    prefetch_inflight=16,
                    queued_read_count=17,
                    queued_write_count=18,
                    queued_generate_count=19,
                    inflight_io_count=20,
                    inflight_generation_count=21,
                    worker_fallback_count=22,
                    sync_blocking_fetch_count=23,
                    disk_load_count=24,
                    generate_count=25,
                    save_count=26,
                    disk_load_last_seconds=0.001,
                    generate_last_seconds=0.002,
                    save_last_seconds=0.003,
                    disk_load_total_seconds=0.101,
                    generate_total_seconds=0.202,
                    save_total_seconds=0.303,
                )

            def chunk_residency_state(self, chunk_x: int, chunk_y: int):
                del chunk_x, chunk_y
                from engine.world import ChunkResidency
                return ChunkResidency.RESIDENT_CLEAN

            def shift_time_last_ms(self) -> float:
                return 1.0

            def incoming_load_time_last_ms(self) -> float:
                return 2.0

            def stage_time_last_ms(self) -> float:
                return 3.0

            def overlap_copy_time_last_ms(self) -> float:
                return 4.0

            def overlap_transient_copy_time_last_ms(self) -> float:
                return 5.0

            def incoming_transient_clear_time_last_ms(self) -> float:
                return 6.0

            def anchor_build_time_last_ms(self) -> float:
                return 7.0

            def anchor_upload_time_last_ms(self) -> float:
                return 8.0

            def paint_world(self, *args, **kwargs) -> None:
                self.calls.append(("paint", threading.get_ident(), args, kwargs))

            def inject_pressure_world(self, *args) -> None:
                self.calls.append(("pressure", threading.get_ident(), args))

            def inject_pressure_ring_world(self, *args) -> None:
                self.calls.append(("ring", threading.get_ident(), args))

            def step(self, dt: float) -> None:
                self.calls.append(("step", threading.get_ident(), dt))

            def request_snapshot_cells_region_world(self, **kwargs):
                self.calls.append(("snapshot_request", threading.get_ident(), kwargs))
                return {"token": kwargs["entity_id"]}

            def poll_snapshot_cells_region_world(self, token, *, force_ready: bool = False):
                self.calls.append(("snapshot_poll", threading.get_ident(), token, force_ready))
                return {"snapshot": token}

            class FakeTexture:
                size = (4, 3)

                def read(self, *, alignment: int = 1) -> bytes:
                    return bytes([alignment]) * (4 * 3 * 4)

            def render(self, view_mode):
                self.calls.append(("render", threading.get_ident(), view_mode))
                if view_mode == "readback":
                    return self.FakeTexture()
                return f"frame:{view_mode}"

            def visible_uv_rect(self) -> tuple[float, float, float, float]:
                return (0.25, 0.5, 0.125, 0.25)

            def schedule_prefetch_for_rect(self, rect, *, margin_x: int, margin_y: int, prioritize: bool) -> None:
                self.calls.append(("prefetch", threading.get_ident(), rect, margin_x, margin_y, prioritize))

            def pan_camera(self, dx: int, dy: int) -> None:
                self.camera_x += dx
                self.camera_y += dy
                self.calls.append(("pan", threading.get_ident(), dx, dy))

            def mark_camera_activity(self, moved: bool, *, dt: float) -> None:
                self.calls.append(("activity", threading.get_ident(), moved, dt))

            def prefetch_camera_region(self, camera_x: int, camera_y: int, *, submit_chunks: int) -> None:
                self.calls.append(("camera_prefetch", threading.get_ident(), camera_x, camera_y, submit_chunks))

            def set_camera(self, camera_x: int, camera_y: int) -> None:
                self.camera_x = camera_x
                self.camera_y = camera_y
                self.calls.append(("set_camera", threading.get_ident(), camera_x, camera_y))

            def service_background_io(self) -> None:
                self.calls.append(("background_io", threading.get_ident()))

            def close(self) -> None:
                self.calls.append(("close", threading.get_ident()))

        world = FakeWorld()
        GpuWorldCommandHandlers(world).register(self.owner)

        mutations = [
            PaintWorldCommand(1, 2, 3, "fire", "fire", {}),
            InjectPressureWorldCommand(4, 5, 6, 7.0),
            InjectPressureRingWorldCommand(8, 9, 10, 11, 12.0),
        ]
        mutation_result = self.owner.submit(
            GpuCommand(op=GpuCommandType.APPLY_WORLD_MUTATIONS, payload=mutations)
        ).wait(timeout=2.0)
        self.assertTrue(mutation_result.ok)
        self.assertEqual(mutation_result.value, 3)

        step_result = self.owner.submit(
            GpuCommand(op=GpuCommandType.STEP_WORLD, payload={"dt": 0.25})
        ).wait(timeout=2.0)
        self.assertTrue(step_result.ok)

        snapshot_request = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.SUBMIT_SNAPSHOT,
                payload=SnapshotRequestCommand("hero", 1, 2, 3, 4, submitted_at=10.0),
            )
        ).wait(timeout=2.0)
        self.assertTrue(snapshot_request.ok)
        self.assertEqual(snapshot_request.value, {"token": "hero"})

        snapshot_poll = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.POLL_SNAPSHOT,
                payload=SnapshotPollCommand(snapshot_request.value, force_ready=False),
            )
        ).wait(timeout=2.0)
        self.assertTrue(snapshot_poll.ok)
        self.assertEqual(snapshot_poll.value, {"snapshot": {"token": "hero"}})

        registry = SnapshotMailboxRegistry()
        runtime_owner = GpuOwnerThread()
        runtime = GpuOwnedWorldRuntime()
        runtime.register(runtime_owner)
        runtime_owner.start()
        self.addCleanup(runtime_owner.shutdown)
        runtime.world = world
        runtime._snapshot_registry = registry
        service_result = runtime_owner.submit(
            GpuCommand(
                op=GpuCommandType.SERVICE_SNAPSHOTS,
                payload=SnapshotServiceCommand(
                    requests=(SnapshotRequestCommand("hero", 1, 2, 3, 4, submitted_at=10.0),)
                ),
            )
        ).wait(timeout=2.0)
        self.assertTrue(service_result.ok)
        self.assertEqual(service_result.value["submitted"], 1)
        self.assertEqual(service_result.value["completed"], 1)
        consumed = registry.consume("hero", current_tick=0)
        self.assertIsNotNone(consumed)

        sync_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.SYNC_ENTITY_STATE,
                payload=EntityGpuSyncCommand(
                    shapes=(("hero", 4, 8),),
                    states=(("hero", 10, 20, True, 1.0),),
                    mask_rects=((0, 0, 4, 8, 123),),
                    origin_x=5,
                    origin_y=6,
                ),
            )
        ).wait(timeout=2.0)
        self.assertTrue(sync_result.ok)
        self.assertEqual(sync_result.value, {"shapes": 1, "states": 1, "mask_rects": 1})

        render_result = self.owner.submit(
            GpuCommand(op=GpuCommandType.RENDER_FRAME, payload="material")
        ).wait(timeout=2.0)
        self.assertTrue(render_result.ok)
        self.assertEqual(render_result.value, "frame:material")

        readback_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.RENDER_FRAME,
                payload=RenderFrameCommand(view_mode="readback", readback_rgba=True),
            )
        ).wait(timeout=2.0)
        self.assertTrue(readback_result.ok)
        self.assertIsInstance(readback_result.value, GpuFramePayload)
        self.assertEqual(readback_result.value.width, 4)
        self.assertEqual(readback_result.value.height, 3)
        self.assertEqual(readback_result.value.rgba, bytes([1]) * (4 * 3 * 4))
        self.assertEqual(readback_result.value.uv_rect, (0.25, 0.5, 0.125, 0.25))
        self.assertEqual(readback_result.value.status["active_size"], (32, 32))

        prefetch_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.SCHEDULE_PREFETCH,
                payload=ChunkPrefetchCommand(rect=("rect",), margin_x=1, margin_y=2, prioritize=True),
            )
        ).wait(timeout=2.0)
        self.assertTrue(prefetch_result.ok)

        follow_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.CAMERA_FOLLOW,
                payload=CameraFollowCommand(target_x=10, target_y=12, dt=0.016),
            )
        ).wait(timeout=2.0)
        self.assertTrue(follow_result.ok)
        self.assertEqual(follow_result.value, {"camera_x": 10, "camera_y": 12})

        teleport_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.CAMERA_TELEPORT,
                payload=CameraTeleportCommand(target_x=80, target_y=90, submit_chunks=3),
            )
        ).wait(timeout=2.0)
        self.assertTrue(teleport_result.ok)
        self.assertEqual(teleport_result.value, {"camera_x": 80, "camera_y": 90})

        background_result = self.owner.submit(
            GpuCommand(op=GpuCommandType.SERVICE_BACKGROUND_IO)
        ).wait(timeout=2.0)
        self.assertTrue(background_result.ok)

        status_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.WORLD_STATUS,
                payload=WorldStatusCommand(include_gpu_debug=True),
            )
        ).wait(timeout=2.0)
        self.assertTrue(status_result.ok)
        self.assertEqual(status_result.value["camera"], (80, 90))
        self.assertEqual(status_result.value["active_size"], (32, 32))
        self.assertEqual(status_result.value["paging"]["last_shift_generates"], 7)
        self.assertEqual(status_result.value["chunk"]["worker_fallback"], 22)
        self.assertIn("0,0", status_result.value["chunk_cache_states"])

        callable_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.RUN_WORLD_CALLABLE,
                payload=WorldCallableCommand(
                    lambda world: ("callable", threading.get_ident(), world.camera_x, world.camera_y)
                ),
            )
        ).wait(timeout=2.0)
        self.assertTrue(callable_result.ok)
        self.assertEqual(callable_result.value, ("callable", self.owner.owner_thread_id, 80, 90))

        options_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.SET_GPU_OPTIONS,
                payload=GpuOptionsCommand(skip_liquid_physics=True),
            )
        ).wait(timeout=2.0)
        self.assertTrue(options_result.ok)
        self.assertEqual(options_result.value, {"skip_liquid_physics": True})

        close_result = self.owner.submit(
            GpuCommand(op=GpuCommandType.CLOSE_WORLD)
        ).wait(timeout=2.0)
        self.assertTrue(close_result.ok)

        observed_world_thread_ids = [call[1] for call in world.calls]
        observed_gpu_thread_ids = [call[1] for call in world.gpu_simulator.calls]
        self.assertTrue(observed_world_thread_ids or observed_gpu_thread_ids)
        allowed_world_tids = {self.owner.owner_thread_id, runtime_owner.owner_thread_id}
        self.assertTrue(all(tid in allowed_world_tids for tid in observed_world_thread_ids))
        self.assertTrue(all(tid == self.owner.owner_thread_id for tid in observed_gpu_thread_ids))
        self.assertIn("prefetch", [call[0] for call in world.calls])
        self.assertIn("pan", [call[0] for call in world.calls])
        self.assertIn("camera_prefetch", [call[0] for call in world.calls])
        self.assertIn("background_io", [call[0] for call in world.calls])
        self.assertIn("close", [call[0] for call in world.calls])
        self.assertIn("skip_liquid", [call[0] for call in world.gpu_simulator.calls])


class GpuOwnedWorldRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.owner = GpuOwnerThread()
        self.runtime = GpuOwnedWorldRuntime()
        self.runtime.register(self.owner)
        self.owner.start()
        self.addCleanup(self.owner.shutdown)

    def test_create_and_close_world_run_on_owner_thread(self) -> None:
        class FakeContext:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            def release(self) -> None:
                self.calls.append(("release", threading.get_ident()))

        class FakeWorld:
            def __init__(self, ctx: FakeContext) -> None:
                self.ctx = ctx
                self.calls: list[tuple] = [("create", threading.get_ident())]
                self.gpu_simulator = None
                self.camera_x = 1
                self.camera_y = 2
                self.active_origin_x = 3
                self.active_origin_y = 4
                self.active_width = 5
                self.active_height = 6
                self.viewport_width = 7
                self.viewport_height = 8
                self.world_width = 9
                self.world_height = 10

            def step(self, dt: float) -> None:
                self.calls.append(("step", threading.get_ident(), dt))

            def close(self) -> None:
                self.calls.append(("close", threading.get_ident()))

        created: dict[str, object] = {}

        def make_context() -> FakeContext:
            ctx = FakeContext()
            created["ctx"] = ctx
            return ctx

        def make_world(ctx: FakeContext) -> FakeWorld:
            world = FakeWorld(ctx)
            created["world"] = world
            return world

        create_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.CREATE_WORLD,
                payload=CreateWorldCommand(
                    context_factory=make_context,
                    world_factory=make_world,
                ),
            )
        ).wait(timeout=2.0)
        self.assertTrue(create_result.ok)
        self.assertEqual(create_result.value, {"world_created": True, "context_created": True})

        step_result = self.owner.submit(
            GpuCommand(op=GpuCommandType.STEP_WORLD, payload={"dt": 0.125})
        ).wait(timeout=2.0)
        self.assertTrue(step_result.ok)

        status_result = self.owner.submit(
            GpuCommand(op=GpuCommandType.WORLD_STATUS, payload=WorldStatusCommand())
        ).wait(timeout=2.0)
        self.assertTrue(status_result.ok)
        self.assertEqual(status_result.value["camera"], (1, 2))
        self.assertEqual(status_result.value["active_origin"], (3, 4))

        callable_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.RUN_WORLD_CALLABLE,
                payload=WorldCallableCommand(
                    lambda world: ("owned", threading.get_ident(), world.world_width)
                ),
            )
        ).wait(timeout=2.0)
        self.assertTrue(callable_result.ok)
        self.assertEqual(callable_result.value, ("owned", self.owner.owner_thread_id, 9))

        close_result = self.owner.submit(
            GpuCommand(op=GpuCommandType.CLOSE_WORLD)
        ).wait(timeout=2.0)
        self.assertTrue(close_result.ok)
        self.assertEqual(close_result.value, {"world_closed": True, "context_released": True})

        owner_tid = self.owner.owner_thread_id
        world = created["world"]
        ctx = created["ctx"]
        self.assertTrue(all(call[1] == owner_tid for call in world.calls))
        self.assertEqual(ctx.calls, [("release", owner_tid)])

    def test_owner_created_standalone_world_can_step_snapshot_and_close(self) -> None:
        try:
            import moderngl
        except Exception:
            self.skipTest("moderngl not available")
        from engine.materials import build_material_registry
        from engine.gpu_owner import SnapshotRequestCommand
        from engine.world import ActiveWorldWindow, WorldChunkStore
        import tempfile

        registry = build_material_registry()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)

        def make_context():
            return moderngl.create_standalone_context()

        def make_world(ctx):
            store = WorldChunkStore(32, 32, chunk_size=8, seed=7)
            return ActiveWorldWindow(
                store,
                registry,
                viewport_width=8,
                viewport_height=8,
                halo_cells=0,
                ctx=ctx,
                chunk_save_dir=tmp.name,
                chunk_cache_prefetch_x=0,
                chunk_cache_prefetch_y=0,
            )

        create_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.CREATE_WORLD,
                payload=CreateWorldCommand(
                    context_factory=make_context,
                    world_factory=make_world,
                ),
            )
        ).wait(timeout=10.0)
        if not create_result.ok:
            self.skipTest(f"standalone GL owner world unavailable: {create_result.error}")

        step_result = self.owner.submit(
            GpuCommand(op=GpuCommandType.STEP_WORLD, payload={"dt": 0.0})
        ).wait(timeout=5.0)
        self.assertTrue(step_result.ok, step_result.error)

        frame_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.RENDER_FRAME,
                payload=RenderFrameCommand(view_mode=None, readback_rgba=True),
            )
        ).wait(timeout=5.0)
        self.assertTrue(frame_result.ok, frame_result.error)
        self.assertIsInstance(frame_result.value, GpuFramePayload)
        self.assertEqual(frame_result.value.width, 8)
        self.assertEqual(frame_result.value.height, 8)
        self.assertEqual(len(frame_result.value.rgba), 8 * 8 * 4)

        token_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.SUBMIT_SNAPSHOT,
                payload=SnapshotRequestCommand(
                    "hero",
                    0,
                    0,
                    4,
                    4,
                    submitted_at=time.perf_counter(),
                ),
            )
        ).wait(timeout=5.0)
        self.assertTrue(token_result.ok, token_result.error)

        snapshot_result = self.owner.submit(
            GpuCommand(
                op=GpuCommandType.POLL_SNAPSHOT,
                payload=SnapshotPollCommand(token_result.value, force_ready=True),
            )
        ).wait(timeout=5.0)
        self.assertTrue(snapshot_result.ok, snapshot_result.error)
        self.assertEqual(snapshot_result.value.entity_id, "hero")

        close_result = self.owner.submit(
            GpuCommand(op=GpuCommandType.CLOSE_WORLD)
        ).wait(timeout=5.0)
        self.assertTrue(close_result.ok, close_result.error)
        self.assertEqual(close_result.value, {"world_closed": True, "context_released": True})


if __name__ == "__main__":
    unittest.main()
