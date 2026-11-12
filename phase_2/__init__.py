"""Phase 2 SAR curriculum package.

Subpackages:
    phase_2A -- shape recognition, survivor always visible
    phase_2B -- partial occlusion, policy must re-acquire the survivor
    phase_2C -- full occlusion, recurrent policy must search systematically

The shared ``sensors`` and ``utils`` packages live at the repository root so
every phase can import the same raycast sensor and fog-of-war map.
"""
