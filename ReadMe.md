# ResQ-AI (Stampede Manager) — How It Works
### A speaking script for the demo / viva

---

## 0. The one-line pitch

> "ResQ-AI watches a CCTV feed and answers one question continuously: *how many
> people are packed into how much floor space, and are they moving in a way that
> precedes a crush?* It converts a camera view into real-world square metres, so
> the danger figure it reports is a physical measurement, not a guess."

---

## 1. Opening — frame the problem correctly

**Say this:**

> "Stampedes are not caused by panic. That's a myth. Post-incident analyses of
> Hillsborough, the Love Parade, and Meron all point to the same physical cause:
> crowd density crossing roughly five people per square metre. Above that point
> a crowd starts behaving like a fluid — shockwaves travel through it, and
> individuals lose the ability to control their own movement. Nobody is running.
> People are being compressed.
>
> So the useful early-warning signal is not 'is anyone running' — it's
> **people per square metre**. That single number is what our system measures."

**Why this opening works:** it justifies your entire architecture in thirty
seconds. Everything after this is just "how do we measure that number from a
camera."

---

## 2. The pipeline in five stages

Draw this on the board:

```
   CCTV frame
       │
       ▼
 [1] DETECT      Faster R-CNN ResNet-50 FPN  →  bounding boxes for each person
       │
       ▼
 [2] PROJECT     Homography  →  each person's position in metres on the ground
       │
       ▼
 [3] DENSIFY     Bird's-eye grid  →  a density map in people / m²
       │
       ▼
 [4] MOTION      Optical flow  →  crowd speed + directional disorder
       │
       ▼
 [5] FUSE        Risk engine  →  0–100 % stampede risk + alarm
```

**Say this:**

> "Five stages. Detect, project, densify, measure motion, fuse. I'll walk through
> each one, and the important one is stage two — that's where a camera image
> becomes real-world measurements."

---

## 3. Stage 1 — Detection and counting

**Say this:**

> "We use Faster R-CNN with a ResNet-50 FPN backbone, pre-trained on COCO. We
> keep only class 1, which is 'person', and only detections above 50 % confidence.
>
> But the count alone is not the output we care about. What we actually extract
> from each bounding box is a single point: the **bottom-centre of the box**. That
> is the person's foot position — the point where they contact the floor.
>
> That choice matters, and here's why."

**Then explain the key insight:**

> "A person's head, torso and shoulders are all floating in 3D space at unknown
> heights. Only their feet are guaranteed to lie on the ground plane. And the
> ground plane is flat. That single geometric fact is what makes the next stage
> possible."

*Code reference: `PersonDetector.foot_points()` — `x = (x1+x2)/2, y = y2`.*

---

## 4. Stage 2 — Turning pixels into metres (the core of the project)

This is the part examiners will probe. Take your time here.

### The problem

**Say this:**

> "A camera cannot measure distance. A person one metre from the lens and a
> person twenty metres away produce completely different pixel sizes. So there is
> no fixed conversion between pixels and metres — the scale changes across the
> image. Near the bottom of the frame one metre might be eighty pixels; near the
> top it might be fifteen."

### The solution: planar homography

**Say this:**

> "But we don't need to solve full 3D reconstruction. Every person we care about
> is standing on one flat surface. And there is an exact mathematical relationship
> between a plane in the world and its image under a pinhole camera — a
> **homography**, a 3×3 projective transformation:"

$$
\begin{bmatrix} X' \\ Y' \\ W \end{bmatrix} =
\mathbf{H}
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix},
\qquad
X = \frac{X'}{W},\quad Y = \frac{Y'}{W}
$$

> "Here (u, v) is a pixel in the image, and (X, Y) is a position in metres on the
> ground. H has 8 degrees of freedom — it's a 3×3 matrix defined up to scale —
> so **four point correspondences are enough to solve for it exactly**. Each
> correspondence gives two equations, four points give eight, eight unknowns.
>
> The division by W is what makes this work. That's the perspective term. It's
> why a linear scale factor fails and a projective transform succeeds."

### How we obtain those four points — calibration

**Say this:**

> "Calibration is a one-time, thirty-second step per camera. We show the operator
> the first frame and they click four corners of any rectangle on the ground they
> can physically measure — a set of floor tiles, road markings, the width of a
> gate, a badminton court. Then they type in the real dimensions, say twelve
> metres by eight.
>
> We now have four image points and their four true world coordinates:
> (0,0), (12,0), (12,8), (0,8). OpenCV's `getPerspectiveTransform` solves for H
> in closed form. We store it in `calibration.json` and never repeat the step
> unless the camera moves."

**If asked "what if you can't calibrate?":**

> "We have a fallback. We take the median bounding-box height across all detected
> people and assume the average human is 1.65 metres tall. That gives us a
> pixels-per-metre figure. But I want to be honest about its limitation: it
> produces a *uniform* scale, so it ignores perspective entirely. It's acceptable
> for a demo or for a top-down mounted camera. For a real deployment, you
> calibrate."

### Computing the area

**Say this:**

> "Once we have H, the area is straightforward. We take the four corner points of
> our monitored region, push them through the homography to get their positions in
> metres, and apply the **shoelace formula**:"

$$
A = \frac{1}{2}\left| \sum_{i=1}^{n} \left( x_i y_{i+1} - x_{i+1} y_i \right) \right|
$$

> "That gives the true ground area of the region we're monitoring, in square
> metres — even though on screen it looks like a trapezoid, because perspective
> squashes the far end."

*Code reference: `GroundPlane.from_calibration()`, `polygon_area()`.*

---

## 5. Stage 3 — The density heatmap

### Why the obvious approach is wrong

**Say this:**

> "The naive way to build a crowd heatmap is to blur the detection boxes directly
> on the camera image. We deliberately did **not** do that, and I think this is
> the most interesting design decision in the project.
>
> Here's the failure: people far from the camera occupy fewer pixels. So an
> image-space blur makes the back of the crowd look sparse — even when it's
> physically the most tightly packed region in the frame. The heatmap would lie
> to you precisely where crushes actually begin, which is at the far end of a
> corridor or against a barrier."

### What we do instead

**Say this:**

> "We build the heatmap in **world space**, not image space.
>
> First we lay down a bird's-eye grid over the monitored area at twenty
> centimetres per cell. Then, for every detected person, we take their foot
> point, project it through the homography into metres, and deposit a value of
> 1.0 into the corresponding cell.
>
> Then we convolve that grid with a Gaussian of standard deviation 0.6 metres —
> roughly the radius of a person's personal space. The Gaussian kernel is
> normalised, so **it preserves total mass**: if forty-one people went in, the
> map still sums to forty-one after blurring. We verified this numerically.
>
> Finally we divide by the cell area, 0.2 × 0.2 = 0.04 m². That converts
> 'people per cell' into **people per square metre** — a real physical unit that
> can be compared against published crowd-safety thresholds.
>
> For display, we warp that metric grid back onto the video using the inverse
> homography. So what you see overlaid on the footage is perspective-correct."

### A number worth quoting

**Say this:**

> "A useful sanity check: a single isolated person produces a peak density of
> 1/(2πσ²) = 1/(2π × 0.36) ≈ **0.44 people per square metre**. That's exactly
> what it should be — one person standing alone is nowhere near dangerous. The
> scale is physically calibrated, not arbitrary."

### Average vs peak — say this, it's important

> "We report two density figures, and the distinction matters.
>
> **Average density** is just count divided by area. It is almost useless for
> safety, because crowds are never uniform. A stadium concourse can average 1.5
> people per square metre while one corner sits at 6.
>
> **Peak density** is the maximum of the smoothed map — the worst 0.6-metre
> neighbourhood anywhere in the scene. That's the number our risk engine uses.
> At Hillsborough the *stadium* was not overcrowded. Two pens were. Averages
> hide exactly the thing you're trying to detect."

*Code reference: `DensityGrid.compute()`, `DensityGrid.overlay()`.*

---

## 6. Stage 4 — Motion and turbulence

**Say this:**

> "Density alone isn't sufficient, because a packed but calmly moving queue is
> safe. What distinguishes a dangerous crowd is that its motion becomes
> **incoherent** — people start being pushed in conflicting directions rather
> than flowing together. Researchers call this crowd turbulence, and it's
> observable in the seconds before a crush.
>
> We measure it with Farnebäck dense optical flow between consecutive frames.
> That gives us a motion vector at every pixel. We extract two quantities:
>
> **Speed** — mean vector magnitude, converted from pixels-per-frame into metres
> per second using the local scale from our homography and the frame rate.
>
> **Directional disorder** — we normalise every flow vector to unit length,
> take the magnitude-weighted mean, and get a coherence value R between 0 and 1.
> If everyone moves the same way, the unit vectors add constructively and R
> approaches 1. If motion is conflicting, they cancel and R approaches 0. We
> report disorder as 1 − R."

*Code reference: `MotionAnalyzer.update()`.*

---

## 7. Stage 5 — The risk verdict

**Say this:**

> "We fuse three signals into a single risk score between 0 and 1:"

$$
\text{risk} = 0.62 \cdot d + 0.26 \cdot m + 0.12 \cdot s
$$

> "**d — the density term (62 %).** This is the dominant factor, and it's
> normalised against physical thresholds:
> d = (peak_density − 2.0) / (5.5 − 2.0), clipped to [0, 1].
> Below 2 people per square metre a crush is not physically possible, so d is
> zero. At 5.5 it saturates, because that's where crowd-turbulence literature
> places the loss of individual movement control.
>
> **m — the motion term (26 %).** Combines disorder and speed.
>
> **s — the surge term (12 %).** Rate of increase in headcount over a three-second
> window. A rapid influx is an independent early warning.
>
> Critically, both m and s are **gated** by density — they're multiplied by a
> factor that is zero when the space is empty. This prevents the obvious false
> positive: three people running across an empty plaza is high speed and high
> disorder, but it is not a stampede risk. Physically, you need bodies in contact
> before crowd forces can build."

**Then the smoothing:**

> "The output passes through an exponential moving average with α = 0.85, so a
> single bad frame — a detection glitch, someone walking close to the lens —
> can't trigger an alarm. And the alarm itself requires the score to stay above
> 80 % for a sustained two seconds. That's a deliberate trade: we accept a small
> detection delay in exchange for operator trust. An alarm that cries wolf gets
> ignored, and an ignored alarm is worse than no alarm."

### The threshold table — have this on a slide

| People / m² | Condition | System status |
|---|---|---|
| < 2.0 | Free flow, unrestricted walking | NORMAL |
| 2.0 – 3.5 | Busy, speed reduced but controllable | WARNING |
| 3.5 – 4.5 | Restricted, involuntary contact begins | DANGER |
| 4.5 – 5.5 | Dangerous, individual movement constrained | DANGER |
| > 5.5 | Crush conditions, turbulent shockwaves | CRITICAL |

> "These are not numbers we invented. They come from Fruin's Level of Service
> work on pedestrian planning and Keith Still's crowd-safety research."

---

## 8. The demo — run it in this order

1. **Show the raw footage first.** Let them look at it for five seconds. "Can you
   tell me if this crowd is dangerous? Neither can a human operator watching
   forty screens."
2. **Start the app.** Point out the count updating live.
3. **Point at the heatmap** — "note that the red region is *not* where the most
   pixels of people are, it's where they're most tightly packed in real metres."
4. **Point at the bird's-eye panel** — "this is the same crowd viewed from above.
   This is the view the system actually reasons about."
5. **Point at the risk timeline** — "this is what an operator monitors. One line
   per camera instead of one video feed per camera."
6. **If you have dense footage, let it hit an alarm.** The banner pulses red and
   the frame gets a red border.

---

## 9. Anticipated questions — prepare these answers

**"How accurate is your people count?"**
> "In moderate crowds, good. In dense crowds, it undercounts — that's an honest
> limitation of any detection-based counter, because at 5 people per square metre
> most bodies are 80 % occluded and there's no box to draw. Our roadmap fix is to
> add CSRNet, which regresses a density map directly from image texture rather
> than detecting individuals, and is specifically designed for dense crowds. We'd
> keep Faster R-CNN for the visible-count display and use CSRNet for the density
> figure that drives the risk score."

**"Why not just use YOLO?"**
> "We could, and it would be faster. Faster R-CNN's two-stage design gives better
> recall on small and partially occluded objects, which is the regime we operate
> in. For a real deployment on edge hardware, YOLOv8 with a tuned confidence
> threshold would be the practical choice — it's a speed-accuracy trade, not a
> correctness issue."

**"What if the ground isn't flat?"**
> "Then the single-homography assumption breaks. For stairs or terraced seating
> you'd fit a separate homography per planar section and stitch the density maps.
> The architecture supports it — `GroundPlane` is a swappable component."

**"What happens if the camera gets moved?"**
> "The calibration becomes invalid and the area is wrong. In production you'd
> detect this automatically by tracking static feature points between frames and
> raising a recalibration flag if they drift."

**"Does it track individuals? Is that a privacy concern?"**
> "It does not. We never store identities, faces, or trajectories — we extract a
> foot coordinate, add it to a density map, and discard the frame. The system's
> entire memory is the metrics CSV: counts and densities. That's a deliberate
> design choice, and it makes deployment in public spaces far easier to justify."

**"What's the latency?"**
> "Detection dominates — roughly 40–60 ms per frame on GPU. The homography,
> density map and optical flow together are under 10 ms. We also support running
> detection every N frames while the heatmap and flow update continuously, which
> makes it viable on CPU."

**"What's the false positive rate?"**
> "We haven't formally benchmarked it, and I won't claim a number I can't defend.
> What I can say is the design specifically targets the two most likely false
> triggers: the density gate suppresses fast motion in empty spaces, and the
> two-second sustain requirement suppresses transient detection glitches.
> Quantifying this against annotated crowd datasets is our next step."

---

## 10. Closing line

> "To summarise: we take a CCTV feed with no depth information, use a one-time
> four-point calibration to recover the ground plane geometry, and from that
> produce a physically meaningful measurement — people per square metre — that
> maps directly onto established crowd-safety thresholds. The system doesn't
> detect stampedes. It detects the conditions that cause them, which is the only
> version of this problem that's actually useful, because by the time a stampede
> is visible it is far too late to intervene."

---

## Quick reference — where each stage lives in the code

| Stage | Class / function | File |
|---|---|---|
| Detection, foot points | `PersonDetector` | `resq_ai_analyzer.py` |
| Homography, area | `GroundPlane`, `polygon_area` | `resq_ai_analyzer.py` |
| Calibration tool | `run_calibration` | `resq_ai_analyzer.py` |
| Density map | `DensityGrid.compute` | `resq_ai_analyzer.py` |
| Heatmap overlay | `DensityGrid.overlay` | `resq_ai_analyzer.py` |
| Optical flow | `MotionAnalyzer.update` | `resq_ai_analyzer.py` |
| Risk fusion | `RiskEngine.update` | `resq_ai_analyzer.py` |
| Orchestration | `CrowdAnalyzer.process` | `resq_ai_analyzer.py` |
| Live dashboard | Streamlit main loop | `app.py` |
