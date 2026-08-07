from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from composer import render_scenes, save_plan, stitch_and_export_4k
from core import OUTPUTS, search_memes
from director import plan_golf_video

st.set_page_config(page_title="GolfIQ Content Studio", page_icon="⛳", layout="wide")

st.title("GolfIQ Content Studio")
st.caption("Idea → AI director → complete golf story → live-action scene generation → 4K social export")

with st.sidebar:
    st.subheader("Director")
    provider = st.selectbox("Video engine", ["Storyboard preview", "Local ComfyUI"], index=0)
    duration = st.select_slider("Final duration", options=[10, 15, 20, 30], value=15)
    fps = st.select_slider("Final smoothness", options=[24, 30, 48, 60], value=30)
    st.caption("Final portrait export: 2160 × 3840")

    st.divider()
    st.subheader("Local AI")
    workflow = st.text_input(
        "ComfyUI video workflow JSON",
        value=os.getenv("GOLFIQ_COMFY_WORKFLOW", ""),
        help="Export an API-format video workflow from ComfyUI. The workflow can contain __PROMPT__ placeholders.",
    )
    comfy_server = st.text_input(
        "ComfyUI server",
        value=os.getenv("GOLFIQ_COMFY_SERVER", "http://127.0.0.1:8188"),
    )

idea = st.text_area(
    "Describe the video idea",
    height=150,
    placeholder=(
        "Example: Funny relatable GolfIQ TikTok. A golfer is convinced his slice is fixed, hits a tee shot, "
        "watches it peel right, then walks off the tee talking to camera before briefly checking GolfIQ."
    ),
)

preset_cols = st.columns(4)
presets = {
    "Relatable tee shot": "Relatable golf TikTok. Golfer says his slice is fixed, hits driver, ball moves right, then walks off the tee talking to camera. Mention GolfIQ naturally near the end.",
    "Putting pain": "Funny putting TikTok. Golfer reads a short putt confidently, lips it out, walks to the hole talking to camera, then holes the next one.",
    "GolfIQ build": "Build-in-public golf TikTok. Golfer hits a normal tee shot, walks down from the tee discussing why he is building GolfIQ, checks the phone briefly, then continues playing.",
    "Practice challenge": "Fast-paced range challenge. Golfer tries to hit three controlled shots at one target, reacts naturally between shots, and ends by asking viewers if they could beat it.",
}
for col, (label, text) in zip(preset_cols, presets.items()):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state["director_idea"] = text
            st.rerun()

if "director_idea" in st.session_state and not idea:
    idea = st.session_state["director_idea"]

if idea.strip():
    plan = plan_golf_video(idea, duration=duration)

    st.subheader("AI director plan")
    st.caption("The director fills in connective actions you did not explicitly request so the result behaves like a complete video rather than disconnected shots.")

    top_a, top_b, top_c = st.columns(3)
    top_a.metric("Scenes", len(plan.scenes))
    top_b.metric("Duration", f"{plan.duration}s")
    top_c.metric("Hook", plan.hook)

    with st.expander("Continuity lock", expanded=False):
        st.markdown(f"**Golfer:** {plan.character.golfer}")
        st.markdown(f"**Appearance:** {plan.character.appearance}")
        st.markdown(f"**Clothing:** {plan.character.clothing}")
        st.markdown(f"**Location:** {plan.location.course}")
        st.markdown(f"**Lighting:** {plan.location.lighting}")
        st.markdown(f"**Weather:** {plan.location.weather}")

    for scene in plan.scenes:
        with st.expander(f"Scene {scene.index} · {scene.purpose.title()} · {scene.duration:.1f}s", expanded=scene.index <= 2):
            st.markdown(f"**Action:** {scene.action}")
            st.markdown(f"**Camera:** {scene.camera}")
            if scene.dialogue:
                st.markdown(f"**Dialogue:** “{scene.dialogue}”")
            if scene.on_screen_text:
                st.markdown(f"**On-screen text:** {scene.on_screen_text}")
            st.caption(f"Transition: {scene.transition}")
            st.text_area("Generation prompt", scene.generation_prompt, height=170, key=f"scene_prompt_{scene.index}")

    if st.button("Generate complete video", type="primary", use_container_width=True):
        if provider == "Local ComfyUI" and not workflow.strip():
            st.error("Choose a ComfyUI video workflow JSON first. Storyboard preview does not require ComfyUI.")
            st.stop()

        progress = st.progress(0, text="Starting director...")
        try:
            workflow_path = Path(workflow) if workflow.strip() else None
            actual_provider = "Local ComfyUI" if provider == "Local ComfyUI" else "Storyboard preview"

            renders = []
            # Render scene-by-scene so the UI can explain what is happening.
            for idx, scene in enumerate(plan.scenes):
                progress.progress(idx / max(1, len(plan.scenes) + 1), text=f"Generating scene {scene.index}/{len(plan.scenes)}: {scene.purpose}")
                one_scene_plan = type(plan)(
                    title=plan.title,
                    hook=plan.hook,
                    duration=max(1, int(round(scene.duration))),
                    character=plan.character,
                    location=plan.location,
                    scenes=[scene],
                )
                renders.extend(
                    render_scenes(
                        one_scene_plan,
                        provider=actual_provider,
                        workflow_path=workflow_path,
                        comfy_server=comfy_server,
                        fps=min(fps, 30),
                    )
                )

            progress.progress(0.92, text="Editing scenes together and exporting 4K...")
            final = stitch_and_export_4k(renders, fps=fps)
            plan_path = save_plan(plan)
            progress.progress(1.0, text="Complete")

            if provider == "Storyboard preview":
                st.warning(
                    "This is a storyboard/editing proof, not live-action AI footage. The director, timing and complete-story structure are working. "
                    "Install/connect a local ComfyUI video model to replace every storyboard scene with real generated motion."
                )
            else:
                st.success("Multi-scene AI golf video rendered and exported to 4K portrait.")

            st.video(str(final))
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Download 4K MP4", final.read_bytes(), file_name=final.name, mime="video/mp4", use_container_width=True)
            with col2:
                st.download_button("Download story plan", plan_path.read_bytes(), file_name=plan_path.name, mime="application/json", use_container_width=True)

        except Exception as exc:
            progress.empty()
            st.exception(exc)
else:
    st.info("Enter a simple video idea above. You do not need to describe every transition or walking moment — the director adds them automatically.")

st.divider()
with st.expander("Meme library (optional)"):
    st.caption("Use meme search for inspiration or overlays. Check usage rights before using third-party meme imagery in commercial ads.")
    meme_query = st.text_input("Search popular meme templates")
    if meme_query:
        try:
            results = search_memes(meme_query, 12)
            if results:
                cols = st.columns(4)
                for i, item in enumerate(results[:8]):
                    with cols[i % 4]:
                        st.image(item["url"], caption=item["name"], use_container_width=True)
            else:
                st.info("No matching popular template found.")
        except Exception as exc:
            st.warning(f"Meme search unavailable: {exc}")

st.caption(
    "Important: realistic live-action people and golf motion require a genuine local video-generation model. "
    "The app now directs and edits multi-scene stories; it does not pretend a moving still is equivalent to generated live-action video."
)
