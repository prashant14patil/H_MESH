"""
Placeholder for future BCM device integration. NOTHING HERE TALKS TO HARDWARE.

No Tanita device, SDK, protocol or data format is implemented, emulated or assumed. This module only fixes the
contract a future adapter must satisfy, so that adding one does not change the assessment structure:

    1. Implement ``BCMDeviceAdapter`` from the manufacturer's documented interface (USB/serial/Bluetooth/file
       export - whichever the real device offers).
    2. ``read_latest()`` returns a ``DeviceReading`` whose ``values`` use the parameter keys of
       ``backend.services.bcm.schema`` and whose ``raw`` field keeps the device's original output unchanged.
    3. Add a ``schema.Profile`` with ``verified_device_spec=True`` listing exactly what that model reports.
    4. ``register_adapter(...)``. Readings are then stored through the same ``records.build_record`` path as
       manual and imported data, with ``source.type == "device"``.

Until an adapter is registered the integration status is ``not_connected``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DeviceReading:
    measured_at: str                          # ISO-8601 time the device took the measurement
    profile: str                              # schema profile key for this device model
    values: dict                              # {parameter_key: {"value": float, "unit": str}}
    device_model: str | None = None
    device_serial: str | None = None
    raw: dict = field(default_factory=dict)   # the device's original output, preserved verbatim


class BCMDeviceAdapter(ABC):
    """Contract for a future device adapter (not implemented for any device)."""

    key: str = ""
    display_name: str = ""
    profile: str = ""

    @abstractmethod
    def status(self) -> dict:
        """Connection state, e.g. {"state": "connected" | "not_connected" | "error", "message": str}."""

    @abstractmethod
    def read_latest(self) -> DeviceReading:
        """Most recent completed measurement from the device."""


_ADAPTERS: dict[str, BCMDeviceAdapter] = {}


def register_adapter(adapter: BCMDeviceAdapter) -> None:
    if not adapter.key:
        raise ValueError("adapter.key is required")
    _ADAPTERS[adapter.key] = adapter


def get_adapter(key: str) -> BCMDeviceAdapter | None:
    return _ADAPTERS.get(key)


def integration_status() -> dict:
    adapters = [{"key": a.key, "display_name": a.display_name, "profile": a.profile, **a.status()}
                for a in _ADAPTERS.values()]
    connected = any(a.get("state") == "connected" for a in adapters)
    return {
        "state": "connected" if connected else "not_connected",
        "planned": True,
        "adapters": adapters,
        "message": ("A device adapter is registered." if adapters else
                    "Tanita device integration is planned but not connected. No device adapter is installed; "
                    "enter BCM values manually or import a bcm-json-v1 file."),
        "supported_inputs": ["manual", "import:bcm-json-v1", "demo"],
        "adapter_contract": {
            "interface": "backend.services.bcm.devices.BCMDeviceAdapter",
            "methods": ["status() -> dict", "read_latest() -> DeviceReading"],
            "reading_fields": ["measured_at", "profile", "values", "device_model", "device_serial", "raw"],
        },
    }
