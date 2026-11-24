import pytest
import media.audio_occlusion_absorb as mod

CASES = [
    ("multi_ray_occlusion", (1,)),
    ("absorb_by_rate", (1,)),
    ("smooth_coeff", (1,)),
    ("cancel_on_destroy", (1,)),
    ("per_listener", (1,)),
    ("update", (1,)),
    ("filter", (1,)),
    ("mix", (1,)),
    ("destroy", (1,)),
    ("hysteresis", (1,)),
    ("validate_audio", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
