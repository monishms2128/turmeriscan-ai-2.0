"""Hardware actuation controller for TurmeriScan AI physical sorting station.

Manages serial communication with Arduino Uno to trigger physical LEDs,
LCD display updates, and servo-motor mechanical sorting gates.
"""

from __future__ import annotations

import time
from typing import List

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


def list_available_ports() -> List[str]:
    """Return a list of available serial COM ports."""
    if not SERIAL_AVAILABLE:
        return []
    ports = []
    for p in serial.tools.list_ports.comports():
        ports.append(p.device)
    return ports


def auto_detect_arduino_port() -> str | None:
    """Attempt to auto-detect Arduino Uno or USB-Serial adapter port."""
    if not SERIAL_AVAILABLE:
        return None
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        dev = (p.device or "").lower()
        if "arduino" in desc or "ch340" in desc or "usb-serial" in desc:
            return p.device
    # Fallback to first non-Bluetooth COM port if found
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        if "bluetooth" not in desc:
            return p.device
    return None


def send_verdict_to_arduino(
    port: str,
    status: str,
    confidence: float,
    adulterant: str = "",
    baud_rate: int = 9600,
) -> bool:
    """Send purity verdict and confidence to Arduino Uno over Serial.

    Format sent:
      - Pure: "PURE,99.2\n"
      - Adulterated: "ADULTERATED,98.5,Metanil Yellow\n"
    """
    if not SERIAL_AVAILABLE or not port:
        return False

    try:
        with serial.Serial(port, baud_rate, timeout=1) as ser:
            # Arduino resets on DTR toggle upon opening serial; wait brief moment
            time.sleep(1.8)

            conf_pct = f"{confidence * 100:.1f}"
            if status.lower() == "pure":
                payload = f"PURE,{conf_pct}\n"
            else:
                clean_adult = adulterant.replace(",", " ").strip() if adulterant else "Synthetic Dye"
                payload = f"ADULTERATED,{conf_pct},{clean_adult}\n"

            ser.write(payload.encode("utf-8"))
            ser.flush()
            return True
    except Exception:
        return False
