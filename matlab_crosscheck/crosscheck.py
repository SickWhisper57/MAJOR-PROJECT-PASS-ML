"""Python side of the MATLAB cross-check (run from the project root)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pass_sim import P, Trial, psi, zf_power, dBm

tr = Trial([[6, 7], [12, 21], [19, 9], [26, 24]], fading=False)
X = np.array([[2 + 5 * m + 0.13 * n for n in range(P.N)] for m in range(P.M)])
Psi = psi(X, tr)
print(f"|Psi(1,1)|   = {abs(Psi[0, 0]):.6e}")
print(f"|Psi(3,2)|   = {abs(Psi[2, 1]):.6e}")
print(f"angle Psi(1,1) = {np.angle(Psi[0, 0]):.4f} rad")
print(f"ZF Tx power  = {float(dBm(zf_power(Psi))):.4f} dBm")
