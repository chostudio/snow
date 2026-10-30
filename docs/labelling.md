# How to tell tricks apart

A guide for labelling real snowboard airs in `snow label`, written for
someone who has never watched snowboarding. The same text is the panel beside
the player. What the labels are for, and the CSV they go into, is in
[segmenter.md](segmenter.md#labelling-real-airs) and
[training.md](training.md).

```bash
snow label              # opens http://127.0.0.1:8765
snow label --fetch      # once: copy the segmenter and classifier weights from S3
```

Labels autosave to `data/labels/real_airs.csv` on every change. Machine
guesses are cached in `data/labels/prefill/` and recomputed when a clip's
poses, features or the weights change (`--refresh` forces it). A new clip is any
`data/clips/<name>.mp4` (synthetic `synth*` and `awstest*` copies are
skipped); reload the page to see it. Clips without pose output can be
labelled too, just without a skeleton or guesses.

## The two rules that matter most

- **Never guess.** Every field has `unsure`. A wrong label teaches the model
  something false, and the gold set is what both models are finally judged
  on. `unsure` costs nothing.
- **The machine guesses are not answers.** The rate detector, the segmenter,
  the classifier and the geometry are all unvalidated on real footage; that
  is why you are labelling. Use them to find the air, then check every frame
  yourself.

## Keys

| key | does |
|---|---|
| `←` / `→` | one frame back / forward |
| `Shift+←` / `Shift+→` (or `↓` / `↑`) | ten frames |
| `Space` | play / pause |
| `1` `2` `3` `4` | speed 1/8, 1/4, 1/2, 1x |
| `t` | set the selected air's **takeoff** to this frame |
| `l` | set its **landing** to this frame |
| `n` | new air |
| `s` | skeleton overlay on / off |
| `[` / `]` | jump to the selected air's takeoff / landing |
| `d` | delete the selected air |

## Takeoff and landing

The jump is a ramp of snow (the **kicker**); its top edge is the **lip**.
The rider rides up it, flies, and lands on the steep slope below (the
**landing**; the rounded edge at its top is the **knuckle**).

- **Takeoff** is the **first frame the whole board is off the snow**. The
  nose usually leaves first and the tail last: step frame by frame and stop
  on the first frame where the tail has left the lip too.
- **Landing** is the **first frame the board touches the snow again**, any
  part of it. Usually the tail touches first. Riders often bounce or slide
  out after that; it does not matter, the first touch is the landing.
- The board decides, not the body. A hand dragging on the snow mid-air is
  still air. A **butter** (pressing the nose or tail into the snow and
  pivoting) is not air.
- If you cannot see the moment (a camera cut, the rider behind a flag or
  out of frame, motion blur), put your best frame and turn on **not sure**.
- If the air is cut off by a camera cut, label the part on this clip and
  mark it not sure.
- One air per jump. Small hops on the run-in are not what we are after, but
  if the board clearly leaves the snow, it is an air: label it, with spin `0`.

Tip: get close with `Shift+→`, then step single frames. Slow-motion replays
make this much easier, since there are more frames per air.

## Stance: regular or goofy

The rider's feet are strapped across the board. Whichever foot is nearer
the **nose** (the front, the end pointing where they travel) is their front
foot.

- **Regular**: left foot forward.
- **Goofy**: right foot forward.
- **Switch**: riding with the other foot forward from usual (a regular rider
  leading with the right foot). Pros spin into and out of switch constantly,
  so stance is about the rider, not one moment.

How to tell: watch the run-in before the jump, when the rider is going
straight. Which foot leads downhill? If most of the clip shows them leading
with the left foot, they are regular. If you cannot tell which foot is
which (back to the camera, far away), stance is `unsure`. With the skeleton
on, the left foot has a yellow dot and the right a blue one; the pose model
can swap them, especially from behind, so confirm by eye. Commentators
often say "switch" or "cab" (see below), which helps with the **switch**
field.

## Frontside or backside

Every spin goes one of two ways. The name says which side of the body turns
toward the landing **first**:

- **Frontside**: the rider opens up with the **chest** (and toes) turning
  toward the landing first. In the first quarter turn after takeoff they can
  see where they are going.
- **Backside**: the rider turns the **back** toward the landing first. For
  the first quarter turn they cannot see the landing, which is why backside
  spins are often called "blind".

How to read it from the camera:

1. Find the moment just after takeoff. Ignore the wind-up before it (riders
   twist the other way on the lip to load the spin).
2. Watch the chest. Does it swing to face down the hill, toward the landing
   (frontside), or does the back swing that way (backside)?
3. Check with the stance if you know it. Seen **from above**, a regular
   rider's frontside spin turns **counter-clockwise** and a backside spin
   **clockwise**; for a goofy rider it is the other way round. The machine's
   "sense" (clockwise or counter-clockwise) is about world up, seen from
   above, and only becomes frontside/backside once the stance is known.

From a side-on camera both directions can look alike, because you see the
rider edge-on at the start. A front-on or follow camera is easier. When in
doubt, `unsure`.

Naming differs by source here, so do not over-think it:

- A spin taken off riding **switch** is called "switch frontside" or "switch
  backside". A switch frontside spin is usually called a **cab** (after the
  skateboarder Steve Caballero), so "cab 1080" means switch frontside 1080.
  For a cab, set direction `frontside` and switch `y`.
- In a halfpipe, "frontside" and "backside" also name the wall (see
  [Halfpipe clips](#halfpipe-clips)).
- For straight airs (no spin) direction is meaningless: leave it `unsure`.

## Counting the spin

Spin is named in degrees, in steps of 180: 180, 360, 540, 720, 900, 1080,
1260, 1440, 1620, 1800. A 1080 is three full turns. Count from takeoff to
landing only.

Three ways to count, use at least two:

- **Chest and back.** Every time the rider goes from showing the camera
  their chest to showing their back (or back to chest), that is about 180.
  Count the changes and multiply.
- **Nose and tail.** Pick the board's nose (the end that led on the
  takeoff). Each time it comes back to pointing the same way is 360.
- **How they land (the best check).** If the rider lands riding the same
  way they took off (same foot forward), the spin is a multiple of 360: 360,
  720, 1080, 1440, 1800. If they land the other way round (they took off
  regular and ride away switch, or the reverse), it is an odd multiple of
  180: 180, 540, 900, 1260, 1620. This holds for corks too. It cannot tell
  a 720 from a 1080, but it rules out half the answers.

Slow-motion replays are the easiest place to count. Commentators often name
the trick; if they do and it matches what you see, use it. If it does not
match, trust neither and mark `unsure`, or write what you heard in notes.

## Flat spin or cork

- **Flat** (just "a 720"): the rider stays upright the whole way, head
  above the board, turning like a figure on a music box. The axis of the
  turn is roughly vertical.
- **Cork** (short for corkscrew): the rider tips over sideways while
  spinning, so the turn is off-axis. At some point the rider is upside down
  or close to it: **the head comes level with the board or below it**, and
  the board passes over or beside the head. It is not a straight flip
  (head over heels in one plane); it is a tilted spin.
- **Double cork**: two of those off-axis flips in one air. Count the
  upside-down moments: **twice** the board goes above the head.
- **Triple cork**: three. Rare, but seen in top big air contests.

Sources differ on how far over a cork must go: some require the rider to be
fully inverted, others accept any clearly off-axis spin. This project's
geometry counts an inversion when the body tilts past 90 degrees (head at
or below the board), so use that: if the head never gets to the level of
the board, call it flat. Other off-axis families (rodeo, misty, underflip)
have no column of their own: pick `cork` for one off-axis flip, `double_cork`
for two, and write the name you heard in notes.

## Grabs

A grab is a hand holding the board in the air. The name depends on **which
hand**, **which edge** and **where** on the board. "Front hand" is the hand
on the nose side (left for a regular rider, right for goofy); the "rear
hand" is the other one. The **toe edge** is the long edge under the toes
(the side the rider faces), the **heel edge** is the one behind the heels.

| grab | hand | where |
|---|---|---|
| **indy** | rear | toe edge, between the feet |
| **mute** | front | toe edge, between the feet (or in front of the front foot) |
| **melon** | front | heel edge, between the feet |
| **stalefish** | rear | heel edge, between the feet, reaching behind the back leg |
| **nose** | front | the nose |
| **tail** | rear | the tail |
| **method** | front | heel edge, like a melon, with the knees bent and the board pulled up behind the rider, often at head height |
| **japan** | front | toe edge, like a mute, with the front knee tucked and the board pulled up behind the back |

How to spot one: pause at the middle of the air and look for a hand on the
board. Then ask which hand (front or rear) and which edge (the side with the
toes or the heels). In a fast cork the grab may last only a few frames;
step through the whole air. A grab you cannot name (crail, roast beef,
truck driver and many more): `other`, and write it in notes. No hand on the
board at all: `none`. Hand near the board but you cannot see it touch:
`unsure`.

## Replays: the same jump twice

Broadcasts show the live run and then replay the best tricks, often in slow
motion and from a different camera. A replay is the **same jump**, so it gets
the **same event id** as the live shot; otherwise the model is tested on a
jump it was trained on.

Cues that you are watching a replay: slow motion, a wipe or logo transition
before it, a camera angle that is too close or too low for live coverage, no
score or clock graphic, and the same rider in the same clothes doing the same
trick you just saw.

Event ids: anything consistent, for example `contest/round/rider/run/hit`,
like `xgames26/bigair-final/rider-name/run2/hit1`. The page suggests ids
you have already used; pick the same one for a replay.

The slow-motion factor shown under the player comes from the clip's
`meta.json`. Frame numbers are always the clip's frames, slow motion or not,
so label in frames and never in seconds. A clip with no `meta.json` shows the
factor as unknown; that does not affect the labels.

## Halfpipe clips

Some clips (`run1` is one) are halfpipe runs, not a single jump: a U-shaped
channel of snow, and the rider goes up one wall, flies above its top edge
(the **lip** or coping), comes back down the same wall, crosses and does
the next wall. **Every wall is a separate air** (a "hit"), so one run has
four to six: give each its own row, and event ids that share the run but
not the hit (`.../run1/hit1`, `.../run1/hit2`).

- Takeoff is the first frame the whole board is above the lip; landing is
  the first frame it touches the wall again.
- The **frontside wall** is the one the rider's toes face as they ride up
  it, the **backside wall** the one their heels face.
- Spin names in the pipe follow their own conventions (a spin's degrees
  count differently from a jump because the rider comes back down the wall
  they went up), and the landing rule in [Counting the spin](#counting-the-spin)
  is for jumps. If you are not sure how a pipe spin is named, take the
  commentator's name when they give one, and otherwise `unsure`. The
  takeoff and landing frames are what matter most.

## Naming a trick

Build the name from the fields, in this order:

**[switch] [frontside / backside] [axis] [spin] [grab]**

- frontside, double cork, 1080, mute: **frontside double cork 1080 mute**
- backside, flat, 360, indy: **backside 360 indy**
- switch frontside (a cab), double cork, 1440, no grab: **cab double cork 1440**
- no spin, method grab: **method** (a straight air)

The CSV stores the fields, not the name; the page composes the name so you
can check it reads like something a commentator would say.

## Common mistakes

- **Counting the wind-up.** Riders twist on the lip before they leave it.
  Rotation starts at takeoff.
- **Counting camera motion.** The camera pans and zooms to follow; the
  background swinging round is not the rider spinning. Watch the body
  against the board, not against the background.
- **Mirror confusion.** A rider seen from the front looks like a mirror
  image of the same rider from behind. Decide frontside/backside by the
  chest-first rule, not by left or right on the screen.
- **Landing on the first bounce.** The landing frame is the first touch,
  even if they bounce up again.
- **Labelling a fall as no air.** A crash is still an air: label takeoff and
  landing (first contact of the board, or of the body if the board never
  lands; mark not sure) and say so in notes.
- **Skipping boring clips.** A clip with no air is a label too: press
  "no air in this clip". The model needs negatives.
- **Forcing a field.** If it took a guess, it is `unsure`.

## How the pre-filled guesses work

Each clip with pose output shows what the machines think, labelled as such:

- **Rate detector**: the dashboard's air finder. It looks for fast body
  rotation, so it finds spins, not airs: it misses straight airs, and its
  start and end follow the turning, not the board leaving and touching the
  snow.
- **Segmenter**: a small model trained to mark airborne frames. Trained on
  synthetic renders only; on real footage it has looked wrong.
- **Geometry**: the measured spin (`spin_path_deg`, rounded to 180), its
  turn sense, and the number of inversions for the axis. "Margin" says how
  close the measurement is to a multiple of 180 (1 is exact, 0 is halfway);
  it is not a probability.
- **Classifier**: a model that names the trick, with a probability per field.
  On synthetic tests it beats the geometry only on turn sense, stance and
  grab, never on spin count or axis ([classifier.md](classifier.md)), and it
  has never been checked on real footage.

"Use as draft" copies a machine air into your labels so you can correct it;
nothing a machine said is saved until you do that.
