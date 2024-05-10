import matplotlib.pyplot as plt
import scienceplots
import numpy as np
plt.style.use('science')
effic = np.array(np.arange(0.9,1.02,0.02))
print(effic)
eta = np.array(np.arange(0.001,1.001,0.001))
ref_effic = 0.9637

plt.figure()
for f in effic:
    k = f/ref_effic * (-0.0162*eta-0.0059/eta+0.9858)
    plt.plot(eta, k, label=f"$\eta = {f:.2f}$")
    plt.legend()
plt.ylim([0,1])
plt.ylabel("Inverter Efficiency [-]")
plt.xlabel("$ \zeta $ [-]")
plt.title("PVWatts inverter part-load efficiency")
plt.savefig("Figures/inv_efficiency.png", dpi=500)
plt.show()