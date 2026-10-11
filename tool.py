from langchain_core.tools import tool

NOTE_BEATS = {"whole" : 4, "half" : 2, "quarter": 1, "eighth": 0.5, "sixteenth": 0.25}


@tool
def note_duration_ms(bpm: float, note: str) -> float | str:
    """Calculate how long a musical note lasts in milliseconds at a given tempo.

    bpm: the tempo in beats per minute (must be greater than 0).
    note: the note type, one of: whole, half, quarter, eighth, sixteenth.
    """
    beats = NOTE_BEATS.get(note.lower())
    if beats is None:
        return f"Unknown note '{note}'. Valid notes: {', '.join(NOTE_BEATS)}"
    if bpm <= 0:
        return f"BPM must be greater than 0"
    return round(60000 / bpm * beats, 2)

if __name__ == "__main__":
    print(note_duration_ms.invoke({"bpm": 120, "note": "quarter"}))
    print(note_duration_ms.invoke({"bpm": 90, "note": "eighth"}))
    print(note_duration_ms.invoke({"bpm": 120, "note": "banana"}))
    print(note_duration_ms.invoke({"bpm": -5, "note": "quarter"}))