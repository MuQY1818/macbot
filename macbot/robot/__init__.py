"""Robot drivers. Only the reBot B601-RS + Arm 102 pair is implemented; add others next to it."""
from .fake import FakeFollower, FakeLeader  # noqa: F401
__all__ = ["FakeFollower", "FakeLeader"]
