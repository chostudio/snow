# The coach: Nemotron on the NVIDIA API

Covers `src/snow/coach.py` and `src/snow/serve.py`.

```bash
snow coach run1 --event air-2             # print one report
snow serve                                # http://127.0.0.1:8000
VITE_API_BASE=http://127.0.0.1:8000 npm run dev    # in web/
../.venv/bin/python tools/export_session.py run1 --coach nemotron   # bake one into the bundle
```

The dashboard's coach panel is a **Nemotron** reasoning model served by
NVIDIA's API (`integrate.api.nvidia.com/v1`, OpenAI-compatible chat
completions), called with the same `NVIDIA_API_KEY` as the Body Pose NIM.
Nothing else is involved: no other provider, no local model.

| | |
|---|---|
| model | `nvidia/nemotron-3-ultra-550b-a55b` (default) |
| override | `SNOW_COACH_MODEL`, `SNOW_COACH_BASE_URL` |
| latency | about 25 s per report (it reasons before answering) |

The default was chosen by listing `GET /v1/models` and then making a real
completion: a listed model can still 404 for a given account
(`nemotron-3-super-120b-a12b` and `llama-3.1-nemotron-ultra-253b-v1` did).
Check with a real call before switching.

## What the model sees

**Only the measurements**, from the session bundle, so it reads exactly what
the dashboard shows. It never sees the video or the pose. Each field is
renamed so its name carries its caveat:

| bundle | the model sees |
|---|---|
| `rotation_deg` | `spin_count_deg` (the signed angular path when `snow features` ran) |
| `nearest_spin_deg` | `nearest_180_deg`, described as a rounding, not a call |
| `peak_rpm`, `air_time_s`, `rotation_decay_dps2` | `..._clip_timebase`, or `..._real_time` only if `slowmo_factor` was set |
| `pop_height_m`, `min_tuck_radius_m` | `..._uncalibrated`, no unit |

Per-shot summaries from `data/poses/<name>.features.json` go in too, with
`spin_path_deg` as the count. **`axis_spin_deg` is left out on purpose**:
synthetic tricks showed it undercounts corks even on exact poses
([validation.md](validation.md)), and offering two spin counts invites the
model to quote the wrong one. Vertical yaw (`net_spin_deg`) stays out for the
same reason; it degenerates when the rider is inverted.

The system prompt carries [validation.md](validation.md)'s *Claims you may
not make*, applied to the clip: spin counts are unvalidated in absolute
terms, lengths are not metres, rates are not real-time unless the factor was
set, rotation across a cut was not measured, and direction is not
frontside/backside.

## What it may send back

A JSON object matching `CoachReport` in `web/src/types.ts`; `source` and
`model` are filled in by us, not the model. `validate_report` checks:

- exactly the keys `verdict`, `headline_metric`, `findings`, `cues`
- severities from the UI's four, 1 to 5 findings, 1 to 4 cues, no empty text
- `headline_metric.event_id` is a real event
- `headline_metric.value` is a number that was actually measured (the
  nearest-180 rounding does not count), and its unit is not a length unit

A reply that fails is sent back **once** with the reason. A second failure
raises `CoachSchemaError`: `snow serve` answers 502 with the reason, the
panel shows it, and `export_session.py --coach nemotron` exits without
writing the bundle. Nothing falls back to the template under a live badge.

The dashboard's header chip follows the report on screen: *Coach (template)*
for the rule-based one, *Nemotron coach* only while a validated Nemotron
report is shown.

## What this does not prove

Validation is about **shape and provenance**, not coaching quality. The
prompt tells the model not to make the forbidden claims, and the headline is
checked mechanically, but the free text in findings and cues is not: a
model can still phrase something more confidently than the numbers allow.
Read it as commentary on measurements whose own limits are in
[validation.md](validation.md).

## The server

`snow serve` is the standard library's `ThreadingHTTPServer`, no new
dependencies. It serves `web/public/sessions/` on the routes `web/src/api.ts`
already expects (byte ranges on the clip, so the video seeks) plus
`POST /<name>/coach`. It binds to localhost: it has no auth of its own, and
every coach request spends the key's quota.
