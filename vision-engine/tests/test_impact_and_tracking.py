from golfiq_vision.impact import detect_impact
from golfiq_vision.tracker import TrackerConfig, build_track, track_quality
from golfiq_vision.types import Detection


def test_impact_detects_ball_departure_and_audio_transient():
    ball = [
        (0, 100.0, 200.0, 0.95),
        (1, 100.5, 199.5, 0.95),
        (2, 101.0, 199.0, 0.95),
        (3, 145.0, 170.0, 0.92),
        (4, 190.0, 145.0, 0.90),
        (5, 230.0, 125.0, 0.88),
    ]
    motion = [0.1, 0.1, 0.2, 4.0, 2.0, 1.0]
    audio = [0.1, 0.1, 0.2, 9.0, 0.5, 0.3]
    result = detect_impact(ball, motion, audio, fps=120.0)
    assert result is not None
    assert result.frame_index in {3, 4}
    assert result.combined_score >= 0.52


def test_tracker_rejects_far_false_positive_and_keeps_real_path():
    detections = []
    for i in range(8):
        detections.append(Detection(i, i / 120.0, 100 + 15 * i, 200 - 7 * i, 0.9, 3.0))
        detections.append(Detection(i, i / 120.0, 1000 - 20 * i, 800, 0.99, 4.0))
    cfg = TrackerConfig(
        gate_radius_px=70.0,
        launch_origin_x_px=100.0,
        launch_origin_y_px=200.0,
        max_seed_distance_px=80.0,
    )
    track = build_track(detections, cfg)
    assert len(track) >= 6
    assert track[-1].x_px < 300
    quality = track_quality(track)
    assert quality["continuity"] >= 0.75
