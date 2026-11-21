"""Thread-owned, offscreen CGL for the macOS tabletop scene."""
from __future__ import annotations

import ctypes
import os
import sys
import threading
from contextlib import contextmanager

TABLETOP_REPO = "lerobot/so101-tabletop-mujoco"


@contextmanager
def render_for_queries(robot):
    """Let non-query steps advance the scene without refreshing unobserved pixels."""
    from .sim_cartesian import tabletop_world

    world = tabletop_world(robot, "--trace-only")
    original = world.render
    cache = {}
    refresh = True

    def capture(enabled):
        nonlocal refresh
        refresh = enabled

    def render(camera):
        if refresh or camera not in cache:
            cache[camera] = original(camera)
        return cache[camera]

    world.render = render
    try:
        yield capture
    finally:
        del world.render


def use_tabletop_cgl(env):
    return (sys.platform == "darwin"
            and (env or {}).get("hub_path", "").split("@", 1)[0] == TABLETOP_REPO
            and os.environ.get("MUJOCO_GL", "").lower().strip() in {"", "cgl", "enable", "enabled", "on", "true", "1"})


@contextmanager
def tabletop_renderer(enabled):
    """The pinned EnvHub scene constructs its renderer synchronously on reset."""
    if not enabled:
        yield
        return
    from mujoco import renderer

    original = renderer.gl_context.GLContext
    renderer.gl_context.GLContext = CGLContext
    try:
        yield
    finally:
        renderer.gl_context.GLContext = original


class CGLContext:
    """MuJoCo calls make_current on the same control thread for each camera."""

    def __init__(self, width, height):
        from mujoco.cgl import cgl

        self.cgl = cgl
        self.owner = threading.get_ident()
        self.ctx = cgl.CGLContextObj()
        a = cgl.CGLPixelFormatAttribute
        base = [a.CGLPFAOpenGLProfile, cgl.CGLOpenGLProfile.CGLOGLPVersion_Legacy,
                a.CGLPFAColorSize, 24, a.CGLPFAAlphaSize, 8, a.CGLPFADepthSize, 24, a.CGLPFAStencilSize, 8]
        try:
            self._open([*base, a.CGLPFAMultisample, a.CGLPFASampleBuffers, 1, a.CGLPFASample, 4, a.CGLPFAAccelerated])
        except cgl.CGLError as exc:
            self.free()
            print(f"Warning: hardware MuJoCo rendering unavailable ({exc}); trying slower software rendering.",
                  file=sys.stderr, flush=True)
            self._open([*base, a.CGLPFARendererID, 0x00020400])  # kCGLRendererGenericFloatID
        from OpenGL.GL import GL_RENDERER, glGetString

        renderer = glGetString(GL_RENDERER)
        print(f"MuJoCo renderer: {renderer.decode() if renderer else 'unknown CGL renderer'}", flush=True)

    def _open(self, values):
        cgl = self.cgl
        args = (ctypes.c_int * (len(values) + 1))(*values, 0)
        pix = cgl.CGLPixelFormatObj()
        count = cgl.GLint()
        try:
            cgl.CGLChoosePixelFormat(args, ctypes.byref(pix), ctypes.byref(count))
            if not count.value or not pix:
                raise cgl.CGLError("no matching CGL pixel format")
            cgl.CGLCreateContext(pix, 0, ctypes.byref(self.ctx))
            self.make_current()
        finally:
            if pix:
                cgl.CGLReleasePixelFormat(pix)

    def make_current(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("MuJoCo rendering must run on the thread that created the context")
        self.cgl.CGLSetCurrentContext(self.ctx)

    def free(self):
        if self.ctx:
            self.cgl.CGLSetCurrentContext(0)
            self.cgl.CGLReleaseContext(self.ctx)
            self.ctx = self.cgl.CGLContextObj()
