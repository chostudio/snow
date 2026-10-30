from tricks import decode, spin_degrees


def test_spin_hundreds_round_to_multiples_of_180():
    assert [spin_degrees(n) for n in (3, 5, 7, 9, 10, 12, 14, 16, 18, 19, 21, 23)] == [
        360, 540, 720, 900, 1080, 1260, 1440, 1620, 1800, 1980, 2160, 2340]


def test_backside_double_cork():
    t = decode("b-DC-10-Wed")
    assert (t["spin_deg"], t["direction"], t["switch"], t["inverts"], t["corked"], t["grab"]) == (
        1080, "backside", False, 2, True, "Wed")


def test_cab_is_switch_frontside_and_grabs_chain():
    t = decode("Cab-D-9-Wed-to-Tg")
    assert (t["direction"], t["switch"], t["inverts"], t["grab"], t["n_grabs"]) == (
        "frontside", True, 2, "Wed", 2)


def test_switch_prefix_and_triple_cork():
    t = decode("x-b-T-C-16-Me")
    assert (t["direction"], t["switch"], t["inverts"], t["corked"], t["spin_deg"]) == (
        "backside", True, 3, True, 1620)


def test_straight_air_has_no_spin_or_grab():
    t = decode("Str-A")
    assert (t["spin_deg"], t["direction"], t["inverts"], t["grab"]) == (0, "none", 0, "none")


def test_codes_typed_without_dashes_or_in_the_wrong_case():
    assert decode("bT14Wed")["tokens"] == ["b", "T", "14", "Wed"]
    assert decode("xbs12Tg")["tokens"] == ["x", "bs", "12", "Tg"]
    t = decode("X-b-d-14-Me")
    assert (t["switch"], t["inverts"], t["axis"], t["spin_deg"]) == (True, 2, "D", 1440)
    # Grabs are not un-glued: "Tdr" stays one token.
    assert decode("f-DC-10-Tdr")["grab"] == "Tdr"
