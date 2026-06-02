def _extract_interrupt_payload(value) -> dict:
    """
    Normalize interrupt payload regardless of shape.

    Interrupt value can be:
      - dict   : {"message": "...", "action": "...", "options": [...], "values": [...]}
      - str    : raw message string (legacy or simple interrupts)
      - object : has .value attribute (LangGraph Interrupt wrapper)
      - list   : list of interrupt objects (take first)
    """
    # ── Unwrap LangGraph Interrupt wrapper ────────────────────────────────────
    if hasattr(value, "value"):
        return _extract_interrupt_payload(value.value)

    # ── List — take the first item ────────────────────────────────────────────
    if isinstance(value, list):
        if value:
            return _extract_interrupt_payload(value[0])
        return {"message": "Awaiting your input."}

    # ── Dict — already the right shape ───────────────────────────────────────
    if isinstance(value, dict):
        return {
            "message": value.get("message") or value.get("content", ""),
            "action":  value.get("action"),
            "options": value.get("options"),
            "values":  value.get("values"),
        }

    # ── Plain string ──────────────────────────────────────────────────────────
    if isinstance(value, str):
        return {"message": value}

    # ── Fallback ──────────────────────────────────────────────────────────────
    return {"message": str(value)}