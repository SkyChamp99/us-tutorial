import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# --- 1. PARAMETERS ---
c = 1520              
fs = 100e6            
depth_max = 0.06      
t = np.arange(0, (2 * depth_max) / c, 1/fs)

num_channels = 128    
x_array = np.linspace(-0.015, 0.015, num_channels) 

# --- 2. PHANTOM GENERATION (High Density) ---
num_speckle = 40000    # Aumentiamo molto per un fondo continuo
x_s = np.random.uniform(-0.02, 0.02, num_speckle)
z_s = np.random.uniform(0.01, depth_max, num_speckle)
amps = np.random.normal(0, 0.3, num_speckle)

# CREIAMO LA CISTI: Eliminiamo i punti dentro un cerchio
# Centro: (0, 0.03), Raggio: 5mm
cisti_mask = np.sqrt((x_s - 0.0)**2 + (z_s - 0.03)**2) > 0.005
x_s = x_s[cisti_mask]
z_s = z_s[cisti_mask]
amps = amps[cisti_mask]

cisti_mask2 = np.sqrt((x_s - (-0.01))**2 + (z_s - 0.05)**2) > 0.004
x_s = x_s[cisti_mask2]
z_s = z_s[cisti_mask2]
amps = amps[cisti_mask2]

# Aggiungiamo solo 2-3 target molto brillanti per riferimento
#x_s = np.append(x_s, [0.008, -0.008])
#z_s = np.append(z_s, [0.045, 0.045])
#amps = np.append(amps, [15.0, 15.0])

# --- 3. IMPULSE e(t) ---
f0 = 5e6
pulse_t = np.arange(-1e-6, 1e-6, 1/fs)
et = np.exp(-0.5 * (pulse_t / 0.15e-6)**2) * np.cos(2 * np.pi * f0 * pulse_t)

# --- 4. SIMULATION ---
b_mode_matrix = np.zeros((len(t), num_channels))

print("Simulating scan lines...")
for i, x_tx in enumerate(x_array):
    rf_line = np.zeros_like(t)
    for xs, zs, amp in zip(x_s, z_s, amps):
        dist = np.sqrt((x_tx - xs)**2 + zs**2)
        tau = (2 * dist) / c
        idx_tau = np.argmin(np.abs(t - tau))
        start_idx = idx_tau - len(et)//2
        end_idx = start_idx + len(et)
        
        if 0 <= start_idx and end_idx < len(rf_line):
            lateral_weight = np.exp(-((x_tx - xs)**2) / (0.0005**2))
            rf_line[start_idx:end_idx] += et * amp * lateral_weight
            
    b_mode_matrix[:, i] = np.abs(hilbert(rf_line))

# --- 5. VISUALIZATION ---
plt.figure(figsize=(7, 9))
image_data = 20 * np.log10(b_mode_matrix + 1e-6) 
image_data -= np.max(image_data)

plt.imshow(image_data, aspect='auto', cmap='gray', vmin=-40, vmax=0,
           extent=[x_array[0]*100, x_array[-1]*100, depth_max*100, 0])

plt.title("Ultrasound B-mode: Multiple Target Sizes")
plt.xlabel("X Position (cm)")
plt.ylabel("Depth Z (cm)")
plt.colorbar(label="Intensity (dB)")
plt.show()