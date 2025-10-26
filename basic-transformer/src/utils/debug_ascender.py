# src/utils/debug_ascender.py
def verify_ascender_wiring(model):
    ok = True; msgs = []
    for li in range(min(2, len(model.decoder.layers))):
        layer = model.decoder.layers[li]
        has_biaser = getattr(layer, "biaser_self", None) is not None
        has_mha_ptr = getattr(layer.self_attn, "biaser", None) is not None
        if not (has_biaser and has_mha_ptr):
            ok = False
            msgs.append(f"L{li}: biaser_self={has_biaser}, self_attn.biaser={has_mha_ptr}")
    if ok:
        print("[ASC VERIFY][OK] Decoder L0/L1 self-attn biasers attached.")
    else:
        print("[ASC VERIFY][FAIL] Missing attachments →", "; ".join(msgs))
    return ok
