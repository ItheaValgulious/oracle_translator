"""Phase 7 tests: async snapshot mailbox semantics."""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from engine.snapshot_mailbox import (
    SnapshotEnvelope,
    SnapshotMailbox,
    SnapshotMailboxRegistry,
)


def _envelope(entity_id: str, *, tick: int = 0, payload: bytes = b"x") -> SnapshotEnvelope:
    return SnapshotEnvelope(
        entity_id=entity_id,
        origin_x=10,
        origin_y=20,
        width=8,
        height=8,
        tick_id=tick,
        packed=payload,
    )


class SnapshotMailboxBasicsTests(unittest.TestCase):
    def test_empty_mailbox_returns_none(self) -> None:
        box = SnapshotMailbox("hero")
        self.assertIsNone(box.consume(current_tick=0))

    def test_submit_then_consume(self) -> None:
        box = SnapshotMailbox("hero")
        box.submit(_envelope("hero", tick=5, payload=b"alpha"))
        got = box.consume(current_tick=7)
        self.assertIsNotNone(got)
        self.assertEqual(got.entity_id, "hero")
        self.assertEqual(got.tick_id, 5)
        self.assertEqual(got.age_frames, 2)
        self.assertEqual(got.packed, b"alpha")

    def test_newer_submit_overwrites_older(self) -> None:
        box = SnapshotMailbox("hero")
        box.submit(_envelope("hero", tick=1, payload=b"v1"))
        box.submit(_envelope("hero", tick=2, payload=b"v2"))
        got = box.consume(current_tick=2)
        self.assertEqual(got.tick_id, 2)
        self.assertEqual(got.packed, b"v2")

    def test_consume_does_not_remove_envelope(self) -> None:
        box = SnapshotMailbox("hero")
        box.submit(_envelope("hero", tick=1, payload=b"v1"))
        got1 = box.consume(current_tick=1)
        got2 = box.consume(current_tick=2)
        self.assertEqual(got1.tick_id, 1)
        self.assertEqual(got2.tick_id, 1)
        self.assertEqual(got2.age_frames, 1)  # reused, age grew

    def test_clear_drops_envelope(self) -> None:
        box = SnapshotMailbox("hero")
        box.submit(_envelope("hero", tick=1))
        box.clear()
        self.assertIsNone(box.consume(current_tick=2))


class SnapshotMailboxNonBlockingTests(unittest.TestCase):
    def test_consume_returns_immediately_when_no_fresh_snapshot(self) -> None:
        box = SnapshotMailbox("hero")
        # Without any submit, consume must return None immediately.
        start = time.perf_counter()
        result = box.consume(current_tick=10)
        elapsed = time.perf_counter() - start
        self.assertIsNone(result)
        self.assertLess(elapsed, 0.05)

    def test_concurrent_submit_and_consume(self) -> None:
        box = SnapshotMailbox("hero")
        stop = threading.Event()
        errors: list[str] = []

        def submitter() -> None:
            tick = 0
            while not stop.is_set():
                tick += 1
                try:
                    box.submit(_envelope("hero", tick=tick, payload=bytes([tick % 256])))
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))

        def consumer() -> None:
            for _ in range(200):
                try:
                    box.consume(current_tick=0)
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))

        t_sub = threading.Thread(target=submitter, daemon=True)
        t_con = threading.Thread(target=consumer)
        t_sub.start()
        t_con.start()
        t_con.join(timeout=5.0)
        stop.set()
        t_sub.join(timeout=1.0)
        self.assertEqual(errors, [])
        # And the mailbox holds a coherent latest envelope.
        latest = box.consume(current_tick=0)
        self.assertIsNotNone(latest)


class SnapshotMailboxRegistryTests(unittest.TestCase):
    def test_per_entity_isolation(self) -> None:
        reg = SnapshotMailboxRegistry()
        reg.submit(_envelope("hero", tick=10))
        reg.submit(_envelope("enemyA", tick=20))
        self.assertEqual(reg.consume("hero", current_tick=12).tick_id, 10)
        self.assertEqual(reg.consume("enemyA", current_tick=22).tick_id, 20)
        self.assertIsNone(reg.consume("missing", current_tick=0))

    def test_freshness_report(self) -> None:
        reg = SnapshotMailboxRegistry()
        reg.submit(_envelope("hero", tick=5))
        reg.submit(_envelope("enemyA", tick=8))
        # Touch a third entity so it appears in the report but has never
        # received a snapshot.
        reg.get_or_create("enemyB")
        report = reg.freshness_report(current_tick=10)
        self.assertEqual(report["hero"], 5)
        self.assertEqual(report["enemyA"], 2)
        self.assertEqual(report["enemyB"], -1)

    def test_drop_removes_mailbox(self) -> None:
        reg = SnapshotMailboxRegistry()
        reg.submit(_envelope("ghost", tick=1))
        reg.drop("ghost")
        self.assertIsNone(reg.consume("ghost", current_tick=0))
        report = reg.freshness_report(current_tick=0)
        self.assertNotIn("ghost", report)


if __name__ == "__main__":
    unittest.main()
