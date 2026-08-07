from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass
class CharacterBible:
    golfer: str
    clothing: str
    appearance: str
    voice: str
    manner: str


@dataclass
class LocationBible:
    course: str
    lighting: str
    weather: str
    visual_style: str


@dataclass
class Scene:
    index: int
    duration: float
    purpose: str
    action: str
    camera: str
    dialogue: str
    on_screen_text: str
    transition: str
    generation_prompt: str


@dataclass
class StoryPlan:
    title: str
    hook: str
    duration: int
    character: CharacterBible
    location: LocationBible
    scenes: list[Scene]

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "hook": self.hook,
            "duration": self.duration,
            "character": asdict(self.character),
            "location": asdict(self.location),
            "scenes": [asdict(s) for s in self.scenes],
        }


def _mentions(prompt: str, words: Iterable[str]) -> bool:
    p = prompt.lower()
    return any(w.lower() in p for w in words)


def _shot_type(prompt: str) -> str:
    p = prompt.lower()
    if any(x in p for x in ["putt", "putting", "green"]):
        return "putt"
    if any(x in p for x in ["iron", "7 iron", "approach"]):
        return "iron"
    return "tee shot"


def _default_hook(prompt: str, shot_type: str) -> str:
    if "slice" in prompt.lower():
        return "POV: you swear the slice is gone..."
    if shot_type == "putt":
        return "Golf is easy until this putt matters."
    if shot_type == "iron":
        return "Most golfers are guessing this number."
    return "Every golfer knows this feeling."


def _scene_prompt(
    *,
    character: CharacterBible,
    location: LocationBible,
    action: str,
    camera: str,
    dialogue: str,
    shot_type: str,
) -> str:
    continuity = (
        f"CONTINUITY LOCK: same golfer in every scene: {character.golfer}; {character.appearance}; "
        f"wearing exactly {character.clothing}. Same location throughout: {location.course}; "
        f"{location.lighting}; {location.weather}. Do not change face, hair, age, body, clothing, "
        "club style, golf bag, tee markers, grass colour, weather, sun direction or time of day. "
    )
    golf_physics = (
        "Golf realism: anatomically correct hands and grip, one normal golf club, believable address posture, "
        "realistic backswing and follow-through, realistic contact, realistic golf ball scale and flight. "
        "No duplicated clubs, no warped shaft, no floating ball, no impossible swing or teleporting person. "
    )
    return (
        continuity
        + golf_physics
        + f"SHOT TYPE: {shot_type}. ACTION: {action}. CAMERA: {camera}. "
        + (f"The person says naturally: {dialogue!r}. " if dialogue else "")
        + f"STYLE: {location.visual_style}. Real live-action footage, physically continuous motion, subtle natural micro-movements, "
        "realistic footsteps and body weight transfer. Vertical 9:16 composition. No title cards inside generated footage."
    )


def plan_golf_video(prompt: str, duration: int = 15) -> StoryPlan:
    """Turn a loose idea into a complete golf short with implied connective moments.

    This planner deliberately adds the moments users often omit: setup, reactions, walking,
    eyelines, continuity, phone checks and natural transitions. The model backend then has
    concrete per-scene instructions instead of one overloaded 15-second prompt.
    """
    prompt = re.sub(r"\s+", " ", prompt).strip()
    shot_type = _shot_type(prompt)
    funny = _mentions(prompt, ["funny", "meme", "laugh", "slice", "shank"])
    golfiq = "golfiq" in prompt.lower()

    character = CharacterBible(
        golfer="one relatable male amateur golfer in his mid-20s",
        appearance="natural face, short brown hair, athletic-average build, realistic skin texture",
        clothing="a dark forest-green golf polo, stone chinos, white golf shoes and a plain dark cap",
        voice="casual British male voice, conversational rather than advert-like",
        manner="relaxed, slightly self-deprecating, natural hand gestures and eye contact",
    )
    location = LocationBible(
        course="a real modern British parkland golf course, first tee and adjacent fairway",
        lighting="soft late-afternoon golden-hour sunlight",
        weather="dry, light breeze, partly cloudy sky",
        visual_style="authentic high-end smartphone golf TikTok, handheld but stable, shallow depth of field only when natural",
    )

    hook = _default_hook(prompt, shot_type)

    if shot_type == "putt":
        actions = [
            ("hook", "Golfer crouches behind an eight-foot putt, studies the line, then glances at camera with a nervous half-smile.", "handheld low angle from behind the ball, tiny push-in", "This looks straight... which means it definitely isn't.", hook, "hard cut"),
            ("attempt", "Golfer stands, takes two realistic practice strokes and rolls the putt. The ball tracks smoothly across the green and lips around the edge.", "ground-level tracking shot following the ball, then rack focus to golfer", "", "", "match cut"),
            ("connective", "Golfer walks toward the cup while talking to camera, putter resting naturally in one hand. He reacts to the miss without stopping walking.", "backward walking medium shot, natural operator footsteps", "That's the kind of putt I remember all day.", "", "whip cut"),
            ("payoff", "Golfer drops another ball, makes the putt, casually picks it out of the cup and walks past camera.", "side-on medium shot, then quick close-up of ball dropping", "", "Would you give yourself that one?", "cut on motion"),
        ]
    else:
        actions = [
            ("hook", "Golfer places a ball on a tee, steps behind it to pick a target, looks briefly at camera, then walks into address.", "handheld chest-height three-quarter rear view with a subtle push-in", "I told myself today's the day I stop doing this.", hook, "hard cut"),
            ("strike", f"Golfer hits a realistic {shot_type}. Show the full motion from takeaway through balanced finish. The ball launches from the clubface and flies across frame with believable speed.", "down-the-line phone camera, hold the frame through impact then smoothly tilt to follow initial ball flight", "", "", "cut on impact"),
            ("reaction", "Golfer holds the finish, follows the ball with his eyes, then gives a small knowing reaction and starts walking off the tee toward his bag.", "medium side profile moving with golfer, natural parallax and footsteps", "Yeah... that's exactly where I didn't aim.", "", "match on movement"),
            ("connective", "Golfer walks down from the tee box along the edge of the fairway while talking casually to camera. He carries the club naturally, occasionally looking toward where the ball finished.", "backward tracking shot at walking pace, realistic handheld stabilization", "The annoying part is I never know if that was me or just what I normally do.", "", "soft cut"),
            ("golfiq" if golfiq else "payoff", "Golfer pauses beside his golf bag and checks his phone briefly, then pockets it and continues walking. The phone is secondary to the real golf scene.", "over-shoulder phone glance for one second, then wider walking shot", "That's why I'm building Golf I.Q. — I want the practice to actually tell you something.", "GolfIQ — would you use it?" if golfiq else "Be honest — where is your miss?", "cut on gesture"),
        ]

    # Spread the requested duration across the natural story beats. Short connective
    # scenes are important because they stop the result feeling like disconnected AI clips.
    weights = [1.6, 2.8, 2.2, 2.7, 2.2] if len(actions) == 5 else [2.0, 3.5, 3.0, 3.0]
    scale = duration / sum(weights)
    scenes: list[Scene] = []
    for i, (purpose, action, camera, dialogue, text, transition) in enumerate(actions):
        d = round(weights[i] * scale, 2)
        scenes.append(
            Scene(
                index=i + 1,
                duration=d,
                purpose=purpose,
                action=action,
                camera=camera,
                dialogue=dialogue,
                on_screen_text=text,
                transition=transition,
                generation_prompt=_scene_prompt(
                    character=character,
                    location=location,
                    action=action,
                    camera=camera,
                    dialogue=dialogue,
                    shot_type=shot_type,
                ),
            )
        )

    title = "GolfIQ Director — " + ("Putting" if shot_type == "putt" else "Tee Story")
    if funny:
        title += " / relatable"
    return StoryPlan(title=title, hook=hook, duration=duration, character=character, location=location, scenes=scenes)
