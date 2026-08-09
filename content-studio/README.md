# GolfIQ Content Studio

A local VS Code golf-content director for TikTok, Instagram Reels and YouTube Shorts.

## What changed

Content Studio is no longer designed around one still image pretending to be a video. The main workflow is now:

`idea -> golf-specific director -> continuity bible -> scene plan -> scene generation -> edit -> 4K portrait export`

The director fills in connective moments users normally omit: walking into address, target selection, holding the finish, reactions, walking off the tee while talking, eyelines, phone checks and natural transition actions.

## What works now

- Simple video-idea input rather than requiring a full screenplay.
- Golf-specific director for tee shots, iron shots and putting stories.
- Automatic hook, dialogue, camera instructions and scene timing.
- Character continuity bible: same golfer, face description, clothing and manner across scenes.
- Location continuity bible: same course, lighting, weather and visual style.
- Per-scene live-action generation prompts with golf-physics constraints.
- Storyboard preview mode when no video model is connected.
- Local ComfyUI multi-scene generation adapter.
- ComfyUI placeholders: `__PROMPT__`, `__DURATION__`, `__FPS__`, `__FRAMES__`, `__SEED__`.
- Scene-by-scene normalization to 1080x1920 for practical editing.
- Final 2160x3840 4K portrait H.264 export.
- 24/30/48/60 fps final-export options.
- Popular meme-template search for inspiration/optional use.
- Downloadable JSON story plan.

## Important limitation

The director and editor can be built entirely in Python/FFmpeg, but realistic humans actually walking, swinging and talking require a genuine video-generation model. Storyboard Preview deliberately labels itself as a preview rather than presenting an animated still as AI video.

The current recommended architecture is to generate short coherent scenes at model-friendly resolution, then edit and upscale once. Native 2160x3840 diffusion generation is unnecessarily expensive for local use.

## Run from VS Code

Install Python 3.11+ and FFmpeg, then:

```bash
cd content-studio
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The browser normally opens at `http://localhost:8501`.

## After pulling an update

From the repository:

```bash
git checkout content-studio
git pull origin content-studio
cd content-studio
.venv\Scripts\activate
streamlit run app.py
```

On macOS/Linux use `source .venv/bin/activate`.

## Storyboard Preview

Choose **Storyboard preview** to test the AI-director decisions, timing, scene order and final editing without a local video model. It creates scene cards and stitches them into the planned timeline. This is a structural preview only.

## Local live-action AI

Install ComfyUI separately, add a compatible open video model/workflow, and export the workflow in API format. Put placeholders where Content Studio should inject scene-specific values:

- `__PROMPT__` - full directed scene prompt
- `__DURATION__` - scene length in seconds
- `__FPS__` - requested generation frame rate
- `__FRAMES__` - requested frame count
- `__SEED__` - deterministic scene seed

Select **Local ComfyUI**, give Content Studio the workflow JSON path and server URL, and press **Generate complete video**.

A Wan/Hunyuan-class ComfyUI workflow is the intended direction, but the correct model depends on GPU VRAM. Model weights are intentionally not stored in GitHub.

## Why thousands of independent images are not the main approach

A video is technically a sequence of frames, but independently generating thousands of still images usually causes face, clothing, club, grass and lighting flicker. A video model is better because it models temporal consistency. Content Studio still exports thousands of frames in the final MP4, but they should originate from temporally coherent generated clips rather than unrelated still-image generations.

Frame interpolation can later be added between generated frames for additional smoothness; it does not replace the need for temporally coherent source motion.

## Golf realism rules

Per-scene prompts explicitly ask for:

- anatomically believable hands and grip;
- one normal club with a stable shaft/head;
- realistic setup, backswing, impact and follow-through;
- believable ball scale and launch;
- continuous walking and weight transfer;
- no teleporting people, duplicated clubs or floating balls;
- consistent golfer, clothing, golf bag, tee markers, course and sun direction.

These constraints improve prompts but do not guarantee perfect generations; the underlying model determines the actual visual quality.

## Meme/copyright policy

Meme search is for discovery/inspiration. Public availability does not automatically grant commercial rights. For GolfIQ advertising, prefer original, licensed, public-domain or compatible Creative-Commons media.

## Trend research

Study patterns rather than copying videos. Useful GolfIQ variables include hook wording, first-shot timing, camera movement, shot type, humour/education/challenge format, average cut length, caption density, product-screen time and ending CTA. Public video metadata can be analysed without downloading/reusing copyrighted source footage.
