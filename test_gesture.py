import time
from collections import namedtuple
from unittest.mock import patch

from main import (
    OneEuroFilter,
    GestureStateMachine,
    extract_features,
    PINCH_PRESS,
    PINCH_RELEASE,
    FREEZE_THRESHOLD,
)

Landmark = namedtuple("Landmark", ["x", "y", "z"])


def create_fake_hand(
    index_up=True,
    middle_up=False,
    ring_up=False,
    pinky_up=False,
    thumb_index_pinch=False,
    thumb_middle_pinch=False,
    pinch_dist_override=None,
):
    """Generates 21 normalized landmarks with x, y, z for 3D gesture testing."""
    lm = [Landmark(x=0.5, y=0.8, z=0.0) for _ in range(21)]

    # Wrist (0) at (0.5, 0.8, 0.0)
    lm[0] = Landmark(x=0.5, y=0.8, z=0.0)
    # Middle MCP (9) at (0.5, 0.5, 0.0) -> gives hand_size_3d = 0.3
    lm[9] = Landmark(x=0.5, y=0.5, z=0.0)

    # Palm base landmarks (for centroid)
    lm[5] = Landmark(x=0.45, y=0.52, z=0.0)   # Index MCP
    lm[13] = Landmark(x=0.55, y=0.52, z=0.0)  # Ring MCP
    lm[17] = Landmark(x=0.60, y=0.55, z=0.0)  # Pinky MCP

    # Base thumb position
    lm[2] = Landmark(x=0.40, y=0.65, z=0.0)
    lm[4] = Landmark(x=0.35, y=0.55, z=0.0)  # Thumb Tip default

    def set_finger(tip_idx, pip_idx, mcp_idx, base_x, is_up):
        lm[mcp_idx] = Landmark(x=base_x, y=0.52, z=0.0)
        if is_up:
            lm[pip_idx] = Landmark(x=base_x, y=0.38, z=0.0)
            lm[tip_idx] = Landmark(x=base_x, y=0.22, z=0.0)
        else:
            lm[pip_idx] = Landmark(x=base_x, y=0.50, z=0.0)
            lm[tip_idx] = Landmark(x=base_x, y=0.65, z=0.0)

    set_finger(8, 6, 5, 0.45, index_up)
    set_finger(12, 10, 9, 0.50, middle_up)
    set_finger(16, 14, 13, 0.55, ring_up)
    set_finger(20, 18, 17, 0.60, pinky_up)

    if thumb_index_pinch:
        idx_tip = lm[8]
        # Place thumb tip within 3D press threshold (< 0.28 * 0.3 = 0.084)
        offset = 0.04 if pinch_dist_override is None else pinch_dist_override * 0.3
        lm[4] = Landmark(x=idx_tip.x + offset, y=idx_tip.y, z=idx_tip.z)

    if thumb_middle_pinch:
        mid_tip = lm[12]
        offset = 0.04 if pinch_dist_override is None else pinch_dist_override * 0.3
        lm[4] = Landmark(x=mid_tip.x + offset, y=mid_tip.y, z=mid_tip.z)

    return lm


def test_cursor_tracking_reference():
    lm = create_fake_hand(index_up=True)
    feat = extract_features(lm)
    # Cursor reference should be blended Index MCP and Palm Center
    rx, ry = feat["cursor_ref"]
    assert 0 < rx < 640 and 0 < ry < 480
    # Ensure it's not the index tip (tip is at y=0.22*480=105.6, palm/MCP is around y=0.55*480=264)
    assert ry > 200.0, f"Expected palm/MCP blended y > 200, got {ry}"
    print(" [PASS] Cursor tracking reference uses Index MCP blended with palm center.")


def test_one_euro_filter():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    t = 0.0
    val = f(100.0, t)
    assert val == 100.0, f"Expected 100.0, got {val}"

    # Smooth low-speed jitter around 100.0
    jittery_inputs = [100.0, 101.0, 99.5, 100.5, 99.8, 100.2]
    outputs = []
    for x in jittery_inputs:
        t += 0.016
        outputs.append(f(x, t))

    assert max(outputs) < 101.0 and min(outputs) > 99.5
    print(" [PASS] OneEuroFilter jitter reduction passed.")


def test_gestures():
    sm = GestureStateMachine()

    with patch("pyautogui.click") as mock_click, \
         patch("pyautogui.mouseDown") as mock_mdown, \
         patch("pyautogui.mouseUp") as mock_mup, \
         patch("pyautogui.moveTo") as mock_move, \
         patch("pyautogui.scroll") as mock_scroll:

        now = 1000.0

        # 1. Test Move Gesture & Freeze Threshold
        # When fingers are wide apart (min dist > 0.55), cursor moves
        lm_move = create_fake_hand(index_up=True, middle_up=False)
        feat_move = extract_features(lm_move)
        assert min(feat_move["d_thumb_index"], feat_move["d_thumb_middle"]) >= FREEZE_THRESHOLD
        label, _, freeze = sm.update(feat_move, (500, 300), now)
        assert label == "MOVE" and not freeze
        assert mock_move.called
        print(" [PASS] Move gesture tracks when distance >= 0.55.")

        # Cursor freezes when min distance < 0.55
        lm_close = create_fake_hand(index_up=True, middle_up=False, thumb_index_pinch=True, pinch_dist_override=0.48)
        feat_close = extract_features(lm_close)
        assert feat_close["d_thumb_index"] < FREEZE_THRESHOLD
        assert feat_close["d_thumb_index"] > PINCH_PRESS
        mock_move.reset_mock()
        label, _, freeze = sm.update(feat_close, (500, 300), now)
        assert freeze, "Expected cursor to be frozen"
        assert not mock_move.called, "Cursor should not move when frozen"
        print(" [PASS] Cursor freeze active when min distance < 0.55.")

        # 2. Test Left Click with 2-frame confirmation on press and release
        lm_pinch = create_fake_hand(index_up=True, middle_up=False, thumb_index_pinch=True, pinch_dist_override=0.15)
        feat_pinch = extract_features(lm_pinch)
        assert feat_pinch["d_thumb_index"] < PINCH_PRESS

        # Frame 1 of pinch -> not yet active (requires 2 frames)
        now += 0.016
        label, is_pinched, _ = sm.update(feat_pinch, (500, 300), now)
        assert sm.active_pinch is None, "Should not confirm on frame 1"

        # Frame 2 of pinch -> confirmed active!
        now += 0.016
        label, is_pinched, _ = sm.update(feat_pinch, (500, 300), now)
        assert sm.active_pinch == "LEFT" and is_pinched
        print(" [PASS] Left pinch 2-frame press confirmation passed.")

        # Release frame 1 -> dist > 0.40, not yet released (requires 2 frames)
        now += 0.05
        label, is_pinched, _ = sm.update(feat_move, (500, 300), now)
        assert sm.active_pinch == "LEFT", "Should not release on frame 1"

        # Release frame 2 -> confirmed release! Since held < 0.4s, fires left click
        now += 0.016
        label, is_pinched, _ = sm.update(feat_move, (500, 300), now)
        assert sm.active_pinch is None
        assert label == "LEFT CLICK"
        assert mock_click.call_count == 1
        print(" [PASS] Left click fired on release with 2-frame confirmation.")

        # 3. Test Drag Gesture (hold > 0.4s)
        now += 1.0
        # 2 frames to activate
        sm.update(feat_pinch, (500, 300), now)
        now += 0.016
        sm.update(feat_pinch, (500, 300), now)
        assert sm.active_pinch == "LEFT"

        # Advance past 0.40s
        now += 0.45
        label, is_pinched, freeze = sm.update(feat_pinch, (520, 310), now)
        assert label == "DRAG" and sm.is_dragging
        assert mock_mdown.called
        assert not freeze, "During drag, cursor is not frozen"
        print(" [PASS] Pinch held > 0.4s initiates drag.")

        # Release drag (2 frames)
        now += 0.10
        sm.update(feat_move, (520, 310), now)
        now += 0.016
        label, is_pinched, _ = sm.update(feat_move, (520, 310), now)
        assert label == "DROP" and not sm.is_dragging
        assert mock_mup.called
        print(" [PASS] Drag release drops with mouseUp.")

        # 4. Test Right Click (Thumb-Middle pinch with index extended, pick smaller)
        lm_mid = create_fake_hand(index_up=True, middle_up=True, thumb_middle_pinch=True, pinch_dist_override=0.15)
        feat_mid = extract_features(lm_mid)
        assert feat_mid["d_thumb_middle"] < feat_mid["d_thumb_index"]
        assert feat_mid["index_up"]
        assert feat_mid["d_thumb_middle"] < PINCH_PRESS

        now += 1.0
        # 2 frames to activate
        sm.update(feat_mid, (500, 300), now)
        now += 0.016
        label, is_pinched, _ = sm.update(feat_mid, (500, 300), now)
        assert sm.active_pinch == "RIGHT"
        assert label == "RIGHT CLICK"
        mock_click.assert_called_with(button="right")
        print(" [PASS] Right click (thumb-middle with index extended) detected.")

        # Right pinch is rejected if index is not extended
        sm.active_pinch = None
        lm_mid_no_idx = create_fake_hand(index_up=False, middle_up=True, thumb_middle_pinch=True, pinch_dist_override=0.15)
        feat_mid_no_idx = extract_features(lm_mid_no_idx)
        now += 1.0
        sm.update(feat_mid_no_idx, (500, 300), now)
        now += 0.016
        label, _, _ = sm.update(feat_mid_no_idx, (500, 300), now)
        assert sm.active_pinch != "RIGHT", "Right pinch must require index extended"
        print(" [PASS] Right click correctly requires index extended.")

        # 5. Test Scroll
        lm_scroll = create_fake_hand(index_up=True, middle_up=True)
        feat_scroll = extract_features(lm_scroll)
        now += 0.05
        label, _, _ = sm.update(feat_scroll, (500, 300), now)
        assert label == "SCROLL"

        # Hand movement triggers scroll
        lm_scroll2 = list(lm_scroll)
        lm_scroll2[8] = Landmark(x=lm_scroll2[8].x, y=lm_scroll2[8].y + 0.05, z=0.0)
        lm_scroll2[12] = Landmark(x=lm_scroll2[12].y, y=lm_scroll2[12].y + 0.05, z=0.0)
        feat_scroll2 = extract_features(lm_scroll2)
        now += 0.05
        sm.update(feat_scroll2, (500, 300), now)
        assert mock_scroll.called
        print(" [PASS] Scroll gesture verified.")

        # 6. Test Fist Pause/Resume (hold 1s)
        lm_fist = create_fake_hand(index_up=False, middle_up=False, ring_up=False, pinky_up=False)
        feat_fist = extract_features(lm_fist)
        now += 1.0
        sm.update(feat_fist, (500, 300), now)
        now += 1.05
        label, _, _ = sm.update(feat_fist, (500, 300), now)
        assert sm.paused and label == "PAUSED"
        print(" [PASS] Pause toggled by fist.")


if __name__ == "__main__":
    test_cursor_tracking_reference()
    test_one_euro_filter()
    test_gestures()
    print("\nALL ACCURACY & GESTURE TESTS PASSED SUCCESSFULLY!")
