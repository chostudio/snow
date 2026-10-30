from parse import official_score, parse_run_tail, parse_text

SHEET = """FIS SNOWBOARD WORLD CUP 2025
Overall Results
BA
Women's Snowboard Big Air
Results from Final
1 5 9305481 FUKADA Mari JPN R 2007 Run 1 f-10-Wed 87 88 88 87 88 86 87.50A 181.501000.00
1 5 9305481 JPN 2007 Run 2 x-b-12-Jp 92 94 95 95 93 94 94.00B
7 32 9325080 YU Seung Eun KOR R 2008 Run 3 DNS
Results from Qualification
24 10 9105409 BAIRD Jasmine CAN R 1999 Run 1 b-DC-10-Wed 26 22 20 19 18 20 20.25 20.25 54.00
24 10 9105409 CAN 1999 Run 2 b-DC-10-Wed 27 23 19 20 18 21 DNI
30 24 9405053 van VREDEN Romy NED R 2005 Run 2 5 2 5 4 DNI
19 OCT 2024 / Chur (SUI) / 6167 Report created SAT 19 OCT 2024 21:13
"""


def test_official_score_drops_high_and_low_and_truncates():
    assert official_score([90, 90, 92, 92, 93, 92]) == 91.50
    assert official_score([27, 23, 19, 20, 18, 21]) == 20.75
    assert official_score([70, 71, 73]) == 71.33


def test_run_tail_with_jump_id_glued_to_score():
    r = parse_run_tail("f-10-Wed 87 88 88 87 88 86 87.50A 181.501000.00")
    assert (r["trick"], r["score"], r["jump_id"], len(r["judges"])) == ("f-10-Wed", 87.5, "A", 6)


def test_sheet_rows_rounds_and_rider_carry_forward():
    jumps = parse_text(SHEET)
    assert [(j["round"], j["name"], j["run"], j["trick"], j["score"]) for j in jumps] == [
        ("Final", "FUKADA Mari", 1, "f-10-Wed", 87.5),
        ("Final", "FUKADA Mari", 2, "x-b-12-Jp", 94.0),
        ("Qualification", "BAIRD Jasmine", 1, "b-DC-10-Wed", 20.25),
        # DNI with marks: score recomputed from the judges.
        ("Qualification", "BAIRD Jasmine", 2, "b-DC-10-Wed", 20.75),
    ]
    assert {(j["gender"], j["codex"], j["stance"]) for j in jumps} == {("W", 6167, "R")}
