# Runs after your code, in the same namespace. Peeking is allowed. ;)
import numpy as _np

_w_best, _b_best = _np.polyfit(rain_hours, cups, 1)
challenge_passed = False
if not (_np.isfinite(w) and _np.isfinite(b)):
    print("❌ w or b became NaN/inf. Is the learning rate sneaking up? Is there a + where a - should be?")
elif abs(w - _w_best) < 0.05 and abs(b - _b_best) < 0.05:
    challenge_passed = True
    print(f"✅ Nailed it! Best possible line: {_b_best:.2f} cups + {_w_best:.2f} cups per hour of rain.")
elif w == 0 and b == 0:
    print("❌ w and b never moved. Did you fill in both TODOs?")
else:
    print(f"❌ Close-ish: you got w={w:.2f}, b={b:.2f}, but the best is w={_w_best:.2f}, b={_b_best:.2f}.")
    print("   Check the update uses MINUS (downhill) and that grad_w multiplies by rain_hours.")
