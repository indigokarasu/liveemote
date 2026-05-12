from __future__ import annotations

class ProsodyTracker:
    def estimate(self, energy_values: list[float]) -> dict:
        if not energy_values:
            return {"speech_energy": 0.0, "speech_rate": 0.0}
        avg = sum(energy_values) / len(energy_values)
        changes = sum(1 for a, b in zip(energy_values, energy_values[1:]) if abs(a-b) > 0.05)
        return {"speech_energy": avg, "speech_rate": min(1.0, changes / max(1, len(energy_values)-1))}
