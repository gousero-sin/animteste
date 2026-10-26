# VFX edit — production spec

## What the client asked for (spoken in the video, Portuguese)
A 75.7 s vertical selfie video (360x480, locked-off phone camera) of a shirtless guy in a small
bathroom. He narrates the edit he wants, live, and every edit must happen **exactly when he says
it**. Goal: "o melhor de Motion e CGI que conseguir" — the best motion design and CGI possible.
Think high-end Hollywood/YouTube VFX: physically plausible light, interactive lighting on his
body, HDR glows, particles with depth/motion blur, no cheap flat clip-art.

Spoken lines (word timings are in `vfx/words.json`) and what must happen:

| time (s) | he says | edit |
|---|---|---|
| 0.0–4.8 | "Esse é um vídeo para testar o Claude" | original bathroom; captions start |
| 5.2–7.1 | "limpa o fundo aqui" (points right) | sink, towel, bottles, mirror, holder, outlet, toilet bits vanish |
| 9.1–11.3 | "limpa esse negócio de descarga" | beige flush plate on the wall vanishes |
| 11.5–12.1 | "e esse chuveiro" | black shower panel + head + hose + valve vanish |
| 12.5–15.1 | "eu coloco o meu estúdio todo preto" | bathroom becomes an all-black studio |
| 15.5–18.3 | "coloco uma roupa em mim, interna, tá tudo bem" | a suit/outfit materialises on him |
| 19.6–21.3 | "e vamos fazer algumas coisas" | anticipation |
| 22.4–25.3 | "bola de fogo nessa mão direita" | fireball in his RIGHT hand (frame-LEFT) |
| 25.3–28.5 | "bola de gelo nessa mão" | ice orb in the other hand (frame-RIGHT) |
| 28.8–30.6 | "a fusão das duas" | the two fuse (hands form a ring in front of his face) |
| 32.2–33.0 | "uma finalização" | a finishing blast |
| 34.4–42.2 | "todas as minhas falas devem ser legendadas atrás da minha cabeça, de forma que não apareça…" | ALL speech captioned BEHIND his head (head occludes the text) — whole video |
| 43.4–48.4 | "nesse momento um portal surge acima no teto" (points up 46–47.7) | portal opens on the ceiling above him |
| 49.4–50.3 | "e me suga" | it sucks him up |
| 51.1–53.4 | "e me leva para outra dimensão" | travel to another dimension |
| 53.4–61.3 | "nessa outra dimensão, eu estou no espaço" | he is in outer space |
| 62.0–63.1 | "aqui tem um buraco negro" (points up-right) | a black hole appears on the frame-right side |
| 63–69.6 | "e agora acabou, é só isso" | — |
| 70.1–73.2 | "finaliza com o portal do Dr. Estranho" (sling-ring gesture 70–74.5) | ends with a Doctor Strange spark portal |

The exact event times every module must use are in `vfx/config.py` → `EV` (do not invent
different ones; if you think one is wrong, say so in your report).

## Footage facts
* Camera is locked off (sub-pixel drift) → clean background plates exist.
* He stands centre, framed chest-up, in the lower half of the frame; the ceiling fills the top
  third. Output frame is 1080x1440 (3x source). Face is around y≈930–1130, x≈420–700.
* Tracking (MediaPipe, smoothed): pose, both hands, 478-pt face mesh. `R` = his right hand =
  appears on FRAME-LEFT = FIRE hand. `L` = his left hand = FRAME-RIGHT = ICE hand.
* Key gestures: 21.6–25.1 R hand raised palm-up at ≈(435,1260); 25.6–28.1 L hand raised high at
  ≈(800,900); 28.7–31.0 both hands in front of his face, fingertips forming a ring around his
  eyes/nose; 31.0–31.3 hands spread to the sides of his head, then drop out of frame;
  33.2–34.0 he turns and points behind his head (frame-left); 46.0–47.7 R arm straight up
  pointing at the ceiling (fingertip ≈ (580,570)); 58.5–60 gestures right then left; 61.5–63.0
  turns toward the right wall and points up-right with his L hand (fingertip ≈ (680–720,
  960–1060), pointing direction up-right); 70.0–72.4 R hand raised with two fingers up
  (sling-ring gesture), 72.5–74.5 hand circling in front of his chest; 74.6+ leans to the camera
  to stop recording.

## Code layout (read these files first)
* `vfx/config.py` — paths, FPS (29.9135), N_FRAMES=2265, W,H=1080,1440, `EV` timeline, `WORDS`.
* `vfx/assets.py` — per-frame loaders (linear float RGB, output px coordinates): `frame(i)`
  (raw SR frame), `fgr(i)` + `alpha(i)` (decontaminated person + matte), `plate(k)` (clean
  bathroom plates 0..3), `plate_mask(name)`, `pose(i)`, `hand(i,side)`, `palm(i,side)`,
  `palm_normal_up`, `hand_size`, `face(i)`, `face_oval(i)`, `head_box(i)`, colour conversions.
* `vfx/core.py` — `Scene`, `Effect` base class (hooks: background, person, lights, behind,
  front, post), `Layer` blending, `Light`, `warp_person`, easing helpers, the `Compositor`
  (relighting, distortion, bloom, shake, tonemap, grain). READ IT — it defines the contract.
* `vfx/glfx.py` — headless GLSL (`gl.program`, `gl.render`) with a noise library prelude, and
  `gl.sprites` (instanced additive gaussian sprites / motion-blur streaks) for particles.
  llvmpipe (CPU) — keep shaders reasonable and render only the crop you need (`offset=`).
* `vfx/render.py` — CLI: `stills`, `range`, `preview`, `full`. Module order:
  `cleanup, ceiling_portal, space, suit, orbs, captions, strange`.
* Fonts in `config.FONTS`: Anton-Regular.ttf, BebasNeue-Regular.ttf, Montserrat[wght].ttf,
  ArchivoBlack-Regular.ttf, Bangers-Regular.ttf (+ system Inter/DejaVu).
* Run everything from `/home/user/animteste` with `python3 -m vfx.render ...`. An X server is
  running on DISPLAY=:99 (glfx sets it). If GL fails: `Xvfb :99 -screen 0 1280x1024x24 &`.

## Rules for every module
1. Write ONLY your own file(s): `vfx/fx/<name>.py` defining `class FX(core.Effect)` (plus
   optional helpers `vfx/fx/<name>_*.py` and caches under `<DATA>/cache/<name>/`). Never edit
   config/core/assets/glfx/render or another module. Need a core change? Work around it locally
   and describe it in your report.
2. Pure function of the frame index: no state across frames, deterministic randomness
   (seeded RNG created in `__init__`, particle trajectories evaluated analytically from `t`).
   Frames are rendered in parallel, out of order.
3. Linear light, HDR allowed (emissive cores 2–8 are fine; the compositor blooms + tonemaps).
   Premultiplied rgb for 'over' layers. Use `Layer(..., box=(x0,y0))` crops for speed.
4. Interactive light: any glowing thing must return `core.Light`s so his skin/suit and the
   environment react (rim on the facing side, coloured wash). This sells the CGI.
5. Things attached to the body: use tracking (`A.palm`, `A.face_oval`, ...) and pass points
   through `S.tx(pt)` (adds the zero-g float offset in space).
6. `active(t)` must return False outside your time window (cheap early-out).
7. Budget: ≤ ~1.5 s per frame for your module at full res on this 4-core CPU box (other jobs
   may be running — measure, don't panic). Precompute heavy static things once (cache to disk).
8. Ease everything (no linear pops). Respect motion: things that appear must ANTICIPATE and
   SETTLE; add secondary motion (particles, flicker, wobble). Motion blur fast things (streak
   sprites).
9. Verify visually and iterate: render stills at key times AND `range` contact sheets (every
   2–3 frames) across your windows, then LOOK at them with the Read tool (PNG/JPG). Check
   edges, occlusion, timing, flicker between frames, and that it reads at phone size. Keep
   iterating until it looks genuinely impressive, not merely "working". Put your test renders
   in `<DATA>/dev/<name>/`.
10. Final answer: what you built, the exact time windows, per-frame cost, the paths of 3–6
    representative renders, and known limitations.

## Module briefs

### cleanup — bathroom cleanup + black studio (active t < 51.2)
* `background(S)`: bathroom = `A.plate(0..3)` (person is composited by the compositor).
  Removals at `EV['clean_clutter']`, `EV['clean_flush']`, `EV['clean_shower']` using masks
  `plate_mask('clutter'|'flush'|'shower')` (`'shower_region'` = what changes between plate2 and
  plate3). Make each removal a "magic eraser": a bright scan/outline traces the object, then it
  dissolves (noise-driven burn front with a hot glowing edge) into drifting glowing particles /
  ash that float up and fade, revealing the clean wall. Small light bloom on the wall.
* Studio at `EV['studio']`: the whole bathroom is consumed and replaced by an all-black studio.
  Must be spectacular but tasteful — e.g. a darkness/ink wave radiating from behind him with an
  energy front, tiles/walls disintegrating into particles and sucked away, lights "power off".
* Black studio look (until `EV['tunnel'][0]`): "todo preto" but premium: near-black backdrop,
  very subtle soft overhead pool of light / volumetric haze behind him, sparse slow dust motes
  catching light, deep vignette. Grade him for the studio (S.person_grade) and give him rim
  lights (lights()) so he separates from black. Expose light-catching haze so other modules'
  lights (Light.bg) read on the environment.

### suit — the outfit (active t ≥ EV['suit'][0])
* `person(S)`: garment mask = matte ∩ body, excluding face (face oval, dilated a bit, incl.
  ears), hair, and both hands (hand landmark hulls with margin; fall back to pose hand points).
  High collar up to under the jaw. Must survive arms raised over the head (46–48 s, 70–72 s),
  hands in front of the face (28.7–31), turning (33–34, 58–63), hands out of frame.
* Look: black high-tech suit (nanotech/tactical), matte with subtle micro-pattern (hex or
  carbon weave), shading driven by the original skin luminance so anatomy/folds read, soft
  specular sheen, a few thin emissive seams (cool white/cyan, subtle). Pattern must NOT swim:
  anchor texture coordinates to the torso (shoulders/hips affine) per frame.
* Materialisation `EV['suit']`: nanotech spreading from the sternum outward over the garment
  mask, advancing front of glowing hex cells + micro sparks, finishes at the end of the window.
* It is relit by other modules' lights after your hook (dark albedo → mostly rim/spec), fine.

### orbs — fire, ice, fusion, finale (active EV['anticipation'][0] … EV['finale_end'])
* Anticipation: faint energy wisps/sparks gathering at both hands.
* Fire `EV['fire_on']` on R palm (frame-left): ignition burst → sustained fireball hovering just
  above the palm (≈0.5× hand length radius), volumetric turbulent flames licking upward
  (GLSL fbm/curl), white-hot core, rising embers, heat-haze distortion (`S.distort`), strong
  orange light (diffuse + rim + bg). Follows the hand with slight lag/overshoot.
* Ice `EV['ice_on']` on L palm (frame-right): crystals grow and lock into an ice orb —
  faceted refractive shards, inner glow, frost mist, falling snow/ice glints, cyan light.
* Fusion `EV['fusion']`: orbs leave the palms, spiral toward the centre of the finger ring and
  become intertwined fire+ice helix streams orbiting around the ring (keep his eyes/face
  readable through it), lightning arcs between the hands, violet-white energy building.
* `EV['release']` 31.0: hands spread → energy condenses into one plasma orb that rises above
  his head and charges (contracting, arcs) until the boom.
* `EV['finale_boom']`: detonation — 2–3 frame flash, expanding refractive shockwave ring
  (distortion), radial fire + ice shards/sparks, smoke, camera shake (S.shake, decaying ~0.8 s),
  embers + snow drifting down until `EV['finale_end']`.

### ceiling_portal — ceiling portal, suction, tunnel, arrival (active 43.5 … 54.6)
* `EV['ceil_portal']`: an elliptical (seen in perspective on the ceiling) swirling vortex opens
  above him at top-centre (he points at it): violet/blue/teal energy, spiral arms, crackling
  rim, a deep interior with stars (hint of the other dimension), god-rays down onto him, studio
  dust pulled up into it. Bluish light from above on him (lights()).
* `EV['suck']`: he is pulled up into it — warp `S.fg/S.alpha` with `core.warp_person`
  (stretch toward the portal, twist, shrink, streaks), particles, chromatic streak; gone by the
  end of the window. Then `EV['ceil_portal_close']` collapse + flash.
* `EV['tunnel']`: full-frame wormhole/hyperspace flight (S.bg, person hidden: S.alpha[:]=0),
  accelerating, bright exit flash into space at the end.
* `EV['arrive']` (space background is drawn by `space`): a small portal at the top spits him
  out — inverse suction, settles to normal by the end of the window, portal closes.
* Person must be hidden from the end of `suck` until `arrive` begins.

### space — outer space + black hole (active t ≥ EV['space'][0] - 0.1)
* `background(S)`: cinematic deep space for "outra dimensão": layered starfield (twinkle,
  colour temperature variety, a few bright stars with diffraction glints), procedural nebula
  (teal/magenta/indigo fbm clouds with depth), a distant galaxy, slow parallax drift.
* Black hole at `EV['blackhole']` up and to the right of where he points (centre ≈ x 800–880,
  y 520–680, event horizon radius ≈ 70–110 px; lensing region larger; must not hide behind his
  head): space visibly warps as it forms; Interstellar-style accretion disk (thin, hot, Doppler
  beamed, lensed far side arching over the shadow, photon ring), real gravitational lensing of
  the stars/nebula behind (ray-march Schwarzschild geodesics in GLSL; precompute/cached maps
  are fine). Disk keeps rotating until the end.
* Zero-g float: set `S.person_offset` (slow drift, ±10–15 px, in `background()`), the
  compositor applies it after all person hooks. Grade him for space (cool fill) and add a warm
  rim light from the black hole once it exists.

### strange — Doctor Strange portal ending (active t ≥ EV['strange'][0])
* Sling-ring sparks: golden-orange sparks emitted tangentially from a spinning circle that
  follows his gesturing R hand (70–72.4) then his circling hand (72.5–74.5); the circle wobbles,
  grows from small to a large portal (radius ≈ 350–450 px) beside/behind him. Sparks: hundreds
  of tiny hot streaks with gravity, embers, rim glow; warm light on him (lights()).
* Through the portal (once open): the original bathroom `A.plate(0)` (drawn in background()
  inside the portal disk, so he stays in front).
* `EV['strange_wipe']`: the portal expands past the frame edges and wipes everything back to
  the raw original footage `A.frame(i)` (do it in `post(S, img)`; sparks ride the expanding
  edge). After that until the end: raw footage (he leans in to stop the recording), last sparks
  fade.

### captions — captions behind the head (whole video)
* Every spoken word from `config.WORDS` (fix obvious ASR casing; keep wording). Group into
  short phrases (1–3 words, ≲ 16 chars), break at punctuation/pauses; trim Whisper's over-long
  word ends (e.g. cap holds; you may analyse the audio RMS of SRC_VIDEO to find speech ends).
* Draw in `behind(S)` so the compositor puts his head/body IN FRONT of the text. Place the text
  just behind the top of his head: horizontally centred on the head (smoothed, clamped to the
  frame), vertically so the hair/head overlaps only the lower part of the letters — readable,
  but unmistakably "behind his head". Large bold uppercase (Anton or similar), ~70–90% frame
  width, karaoke highlight of the active word, pop/slide-in animation with slight blur,
  subtle shadow/stroke so it reads on the white tiles AND on black/space.
* Keyword accents: FOGO (fire gradient), GELO (ice), FUSÃO (violet), PORTAL, DIMENSÃO, ESPAÇO,
  BURACO NEGRO, DR. ESTRANHO (gold), CLAUDE (warm accent).
* When he is not visible (tunnel, mid-suction) centre the captions in the frame.

### sound — sound design (separate script `vfx/sound.py`, writes `<DATA>/audio/mix.wav`)
* Keep his voice (SRC_VIDEO audio stream 0) clear and on top; light cleanup/EQ/compression ok.
* Synthesised SFX (numpy/scipy) for every event above: eraser shimmers, studio blackout
  (whoosh + sub drop), nanotech assembly (granular ticks + riser), fire ignition + crackle bed,
  ice formation (crystalline chimes + crackle), fusion (rising electric hum), finale (sub boom +
  blast + debris tail), portal (deep swirling rumble/wind), suction (riser + whoosh), tunnel
  (rushing wind, flanged tones), arrival thump, space ambience, black hole (deep pulsing
  rumble), sling-ring sizzle, wipe whoosh. Plus a subtle cinematic music bed. Duck everything
  under speech. 48 kHz stereo, exactly `config.DURATION` long. No clipping (true-peak ≤ -1 dB).
