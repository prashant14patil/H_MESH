"""Post-fit view exclusion (body_reconstruction.select_view_exclusions)."""
from backend.services import body_reconstruction as br


def _s(iou, prot):
    return {"iou": iou, "protrusion_frac": prot}


def test_clean_views_are_kept():
    scores = [_s(0.88 + 0.002 * i, 0.012 + 0.001 * (i % 5)) for i in range(24)]
    assert br.select_view_exclusions(scores, [1.0] * 24, "protrusion") == []


def test_occluded_masks_are_found_by_protrusion_not_iou():
    # Pattern from the occlusion study: after a shrunken first fit, clean views fall to IoU ~0.70 and occluded views
    # span 0.46-0.83, but occluded views show large protrusion (body outside the holed mask).
    scores = [_s(0.72 + 0.004 * i, 0.010 + 0.001 * (i % 4)) for i in range(18)] + \
             [_s(v, p) for v, p in ((0.83, 0.09), (0.78, 0.11), (0.70, 0.12), (0.65, 0.14), (0.59, 0.16), (0.46, 0.2))]
    got = {i for i, _ in br.select_view_exclusions(scores, [1.0] * 24, "protrusion")}
    assert got == set(range(18, 24))
    iou_rule = {i for i, _ in br.select_view_exclusions(scores, [1.0] * 24, "iou")}
    assert iou_rule & set(range(18)) and not set(range(18, 24)) <= iou_rule  # the old rule's failure mode


def test_unusable_mask_is_caught_by_iou_floor_and_excluded_views_are_ignored():
    scores = [_s(0.88, 0.015) for _ in range(10)] + [_s(0.40, 0.0)]
    got = br.select_view_exclusions(scores, [1.0] * 11, "protrusion")
    assert [i for i, _ in got] == [10] and "unusable" in got[0][1]
    assert br.select_view_exclusions(scores, [1.0] * 10 + [0.0], "protrusion") == []


def test_never_drops_below_minimum_silhouettes():
    scores = [_s(0.85, 0.01 + 0.001 * i) for i in range(5)] + [_s(0.7, 0.15), _s(0.7, 0.2)]
    got = br.select_view_exclusions(scores, [1.0] * 7, "protrusion")
    assert len(got) == 7 - br.MIN_SILHOUETTE_VIEWS
    assert [i for i, _ in got] == [6]  # only the worst protrusion
