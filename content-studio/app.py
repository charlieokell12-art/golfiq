from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from core import (
    OUTPUTS,
    animate_image_to_4k_video,
    comfyui_generate,
    download_url,
    make_prompt_card,
    overlay_meme,
    search_memes,
    upscale_image_4k,
)

st.set_page_config(page_title="GolfIQ Content Studio", page_icon="⛳", layout="wide")
st.title("GolfIQ Content Studio")
st.caption("Prompt → image/video → meme layer → TikTok 4K export")

with st.sidebar:
    st.subheader("Generation")
    provider = st.selectbox("Provider", ["Free fallback", "Local ComfyUI"])
    mode = st.radio("Output", ["Image", "15s video"], horizontal=True)
    fps = st.select_slider("Video smoothness", options=[24, 30, 48, 60], value=60)
    st.caption("4K portrait export = 2160×3840")

    workflow = st.text_input("ComfyUI API workflow JSON", value=os.getenv("GOLFIQ_COMFY_WORKFLOW", ""))
    comfy_server = st.text_input("ComfyUI server", value=os.getenv("GOLFIQ_COMFY_SERVER", "http://127.0.0.1:8188"))

prompt = st.text_area(
    "AI prompt",
    height=170,
    placeholder="Example: Fast-paced viral golf TikTok. Golfer at a driving range, relatable hook, handheld phone feel, realistic motion, subtle GolfIQ mention, dark-green/gold brand accents...",
)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Meme search")
    meme_query = st.text_input("Search popular meme templates", placeholder="drake, distracted boyfriend, two buttons...")
    meme = None
    if meme_query:
        try:
            results = search_memes(meme_query, 12)
            names = [r["name"] for r in results]
            if names:
                selected = st.selectbox("Template", names)
                meme = next(r for r in results if r["name"] == selected)
                st.image(meme["url"], caption=meme["name"], width=320)
            else:
                st.info("No popular template matched. Paste/upload your own meme below.")
        except Exception as exc:
            st.warning(f"Meme search unavailable: {exc}")

with col_b:
    st.subheader("Your own meme / visual")
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
    custom_url = st.text_input("Or paste a direct image URL")
    add_meme = st.checkbox("Overlay selected meme on video", value=False)

if st.button("Generate", type="primary", use_container_width=True):
    if not prompt.strip():
        st.error("Enter a prompt first.")
        st.stop()

    try:
        if provider == "Local ComfyUI":
            if not workflow:
                raise RuntimeError("Choose/export a ComfyUI API workflow JSON first.")
            result = comfyui_generate(prompt, "video" if mode == "15s video" else "image", Path(workflow), comfy_server)
            generated = result.path
        else:
            generated = make_prompt_card(prompt)

        if mode == "Image":
            if generated.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                raise RuntimeError("The selected provider returned a non-image file for Image mode.")
            final = upscale_image_4k(generated)
            st.success("4K portrait image ready")
            st.image(str(final))
            st.download_button("Download 4K PNG", final.read_bytes(), file_name=final.name, mime="image/png")
        else:
            if generated.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                final = animate_image_to_4k_video(generated, duration=15, fps=fps)
            else:
                final = generated

            meme_path = None
            if add_meme:
                if uploaded is not None:
                    meme_path = OUTPUTS / f"upload-{uploaded.name}"
                    meme_path.write_bytes(uploaded.getbuffer())
                elif custom_url:
                    meme_path = download_url(custom_url)
                elif meme:
                    meme_path = download_url(meme["url"])
                if meme_path:
                    final = overlay_meme(final, meme_path)

            st.success("Video ready")
            st.video(str(final))
            st.download_button("Download MP4", final.read_bytes(), file_name=final.name, mime="video/mp4")

    except Exception as exc:
        st.exception(exc)

st.divider()
st.caption(
    "The free fallback creates a branded motion-video from a still so the editor can be tested without a GPU. "
    "True prompt-generated moving video requires a local ComfyUI video model or another generation backend."
)
