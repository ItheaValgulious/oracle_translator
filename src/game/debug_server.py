"""Debug HTTP server for inspecting game state at runtime."""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.game.app import GameApp


class DebugHandler(BaseHTTPRequestHandler):
    """HTTP handler that reads game state."""

    app: GameApp  # set by factory

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        try:
            if path == "/status":
                data = self._get_status()
            elif path == "/hero":
                data = self._get_hero()
            elif path.startswith("/cells"):
                data = self._get_cells()
            elif path.startswith("/gpu_cells"):
                data = self._get_gpu_cells()
            elif path.startswith("/press"):
                data = self._press_key()
            else:
                data = {"error": "unknown endpoint", "endpoints": ["/status", "/hero", "/cells?x=..&y=..&w=..&h=..", "/gpu_cells?x=..&y=..&w=..&h=..", "/press?key=ENTER"]}
        except Exception as exc:
            data = {"error": str(exc)}
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _get_status(self) -> dict[str, Any]:
        app = self.app
        return {
            "screen": app.current_screen,
            "world_loaded": app.world is not None,
            "hero_state": app.hero.state if app.hero else None,
            "camera": (app.world.camera_x, app.world.camera_y) if app.world else None,
            "active_origin": (app.world.active_origin_x, app.world.active_origin_y) if app.world else None,
        }

    def _get_hero(self) -> dict[str, Any]:
        h = self.app.hero
        return {
            "x": h.x, "y": h.y,
            "top": h.top, "bottom": h.bottom,
            "left": h.left, "right": h.right,
            "vel_x": h.vel_x, "vel_y": h.vel_y,
            "on_ground": h.on_ground,
            "state": h.state,
            "hp": h.hp, "mp": h.mp,
            "facing_right": h.facing_right,
        }

    def _get_cells(self) -> dict[str, Any]:
        return self._get_gpu_cells()

    def _get_gpu_cells(self) -> dict[str, Any]:
        """Read cells directly from GPU textures (bypasses CPU grid)."""
        app = self.app
        if app.world is None or app.world.gpu_simulator is None:
            return {"error": "no world or no GPU"}
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = int(v)
        ax = app.world.active_origin_x
        ay = app.world.active_origin_y
        x = params.get("x", int(app.hero.x))
        y = params.get("y", int(app.hero.y + app.hero.height))
        w = params.get("w", 5)
        h = params.get("h", 10)
        # Convert world coords to local grid coords
        lx = x - ax
        ly = y - ay
        cells_2d = app.world.gpu_simulator.readback_cells_region(lx, ly, w, h)
        cells = []
        for row_idx, row_cells in enumerate(cells_2d):
            for col_idx, cell in enumerate(row_cells):
                wx = x + col_idx
                wy = y + row_idx
                cells.append({
                    "wx": wx, "wy": wy,
                    "family": cell.family_id,
                    "variant": cell.variant_id,
                    "integrity": round(cell.integrity, 4),
                    "temperature": round(cell.temperature, 2),
                })
        return {"x": x, "y": y, "w": w, "h": h, "source": "gpu", "cells": cells}

    def _press_key(self) -> dict[str, Any]:
        import pyglet
        from pyglet.window import key as pyglet_key
        params = {}
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v
        key_name = params.get("key", "ENTER").upper()
        key_map = {
            "ENTER": pyglet_key.ENTER,
            "SPACE": pyglet_key.SPACE,
            "W": pyglet_key.W, "A": pyglet_key.A,
            "S": pyglet_key.S, "D": pyglet_key.D,
            "UP": pyglet_key.UP, "DOWN": pyglet_key.DOWN,
            "LEFT": pyglet_key.LEFT, "RIGHT": pyglet_key.RIGHT,
        }
        symbol = key_map.get(key_name, pyglet_key.ENTER)
        app = self.app
        # Dispatch on main thread — OpenGL operations (e.g. _init_game_world)
        # must run on the thread that owns the GL context.
        event = threading.Event()
        def _do_press(dt):
            app.on_key_press(symbol, 0)
            event.set()
        pyglet.clock.schedule_once(_do_press, 0)
        event.wait(timeout=5)
        # Schedule key release after a short delay
        def _do_release(dt):
            app.on_key_release(symbol, 0)
        pyglet.clock.schedule_once(_do_release, 0.1)
        return {"pressed": key_name, "symbol": symbol, "screen": app.current_screen}

    def log_message(self, format: str, *args: Any) -> None:
        pass  # suppress request logging


def start_debug_server(app: GameApp, port: int = 9123) -> HTTPServer:
    """Start a debug HTTP server in a background thread."""
    handler = type("H", (DebugHandler,), {"app": app})
    server = HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
