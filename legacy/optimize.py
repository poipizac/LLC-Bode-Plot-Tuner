import numpy as np
import sys
sys.path.append('.')
from app import generate_analysis, Lr, Cr, Lm, Vin, Vout, Pout_max

best_fc = 0
best_pm = 0
best_params = None

for r68 in np.linspace(20000, 100000, 20):
    c71_min = 1.0 / (2 * np.pi * r68 * 100)
    for c71 in np.linspace(c71_min, 1000e-9, 10):
        # We want high freq pole at 40kHz, so C70 is fixed by R68
        # fp = 1 / (2 * pi * R68 * C70) => C70 = 1 / (2 * pi * R68 * 40000)
        c70 = 1.0 / (2 * np.pi * r68 * 40000)
        
        res = generate_analysis(Lm, Lr, Cr, 9.66, 0.5, 2.0, r68, c71, c70, 1690e-6, 20e-3, 50, 3500, 1.0)
        fc = res[4]
        pm = res[5]
        gm = res[6]
        if fc is not None and pm is not None and gm is not None:
            if 6000 <= fc <= 8000 and pm >= 60 and gm >= 10:
                res_light = generate_analysis(Lm, Lr, Cr, 9.66, 0.5, 2.0, r68, c71, c70, 1690e-6, 20e-3, 5, 3500, 1.0)
                fc_l = res_light[4]
                pm_l = res_light[5]
                gm_l = res_light[6]
                if pm_l is not None and pm_l >= 45 and gm_l is not None and gm_l >= 10:
                    if pm > best_pm:
                        best_pm = pm
                        best_fc = fc
                        best_params = (r68, c71, c70, pm_l, fc_l)

if best_params:
    r68, c71, c70, pml, fcl = best_params
    fz = 1 / (2 * np.pi * r68 * c71)
    fp = 1 / (2 * np.pi * r68 * c70)
    print(f"Optimal Parameters: R68={r68/1000:.1f}k, C71={c71*1e9:.1f}nF, C70={c70*1e12:.1f}pF")
    print(f"fc={best_fc:.1f} Hz, pm={best_pm:.1f} deg")
    print(f"fz={fz:.1f} Hz, fp={fp:.1f} Hz")
    print(f"Light Load: fc={fcl:.1f} Hz, pm={pml:.1f} deg")
else:
    print("No parameters found in the 20k-100k constraint that yields 6k-8k crossover.")
    
    # scan for best outside R68 constraint just to see
    print("Scanning outside constraint...")
    for r68_out in np.linspace(1000, 20000, 20):
        c70_out = 1.0 / (2 * np.pi * r68_out * 40000)
        c71_out = 100e-9
        res = generate_analysis(Lm, Lr, Cr, 9.66, 0.5, 2.0, r68_out, c71_out, c70_out, 1690e-6, 20e-3, 50, 3500, 1.0)
        if res[4] and 6000 <= res[4] <= 8000:
            print(f"To get {res[4]:.1f}Hz, R68 needs to be {r68_out:.1f} ohms")
            break
