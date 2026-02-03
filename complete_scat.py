import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# --- 1. PARAMETERS ---
c = 1520 # Speed of sound in m/s in unman soft tissue             
fs = 100e6  # freaquency of sampling 100 MHz          
depth_max = 0.06 # Maximum depth in meters (6 cm)     
t = np.arange(0, (2 * depth_max) / c, 1/fs) # Time vector for round trip it samples every 10ns , (start,stop,step)
num_channels = 128  # Number of transducer elements  
x_array = np.linspace(-0.015, 0.015, num_channels) # Transducer element positions from -1.5 cm to 1.5 cm

# --- 2. PHANTOM GENERATION (High Density) ---
num_speckle = 40000    # increased number of scatterers for a denser image
x_s = np.random.uniform(-0.02, 0.02, num_speckle) # Scatterer x-positions
z_s = np.random.uniform(0.01, depth_max, num_speckle) # Scatterer z-positions
amps = np.random.normal(0, 0.3, num_speckle) # Scatterer amplitudes

# Creation of Cyst : Remove scatterers in a circular area to create a cyst-like anechoic region
# Center: (0, 0.03), Radius: 5mm
cisti_mask = np.sqrt((x_s - 0.0)**2 + (z_s - 0.03)**2) > 0.005   # Keep points outside the cyst
x_s = x_s[cisti_mask] # Apply mask
z_s = z_s[cisti_mask]
amps = amps[cisti_mask]

cisti_mask2 = np.sqrt((x_s - (-0.01))**2 + (z_s - 0.05)**2) > 0.004
x_s = x_s[cisti_mask2]
z_s = z_s[cisti_mask2]
amps = amps[cisti_mask2]

# Horizontal line mask (Rettangolo molto sottile)
# we use logic: "Keep the points that are off the line"
line_x_start, line_x_end = -0.01, 0.01  # Line extends from -1 cm to 1 cm in x
line_z_coord, line_thick = 0.055, 0.0006  # Line at 5.5 cm depth, thickness 0.6 mm

# The mask says: True if the point is outside the line, False if inside
line_mask = ~((z_s > line_z_coord - line_thick/2) & (z_s < line_z_coord + line_thick/2) & \
           (x_s > line_x_start) & (x_s < line_x_end))  
# ~ = means, all within the line are False, outside are True.
x_s, z_s, amps = x_s[line_mask], z_s[line_mask], amps[line_mask]

# Aggiungiamo solo 2-3 target molto brillanti per riferimento
#x_s = np.append(x_s, [0.008, -0.008])
#z_s = np.append(z_s, [0.045, 0.045])
#amps = np.append(amps, [15.0, 15.0])

# --- 3. IMPULSE e(t) ---
f0 = 5e6 # Center frequency of the pulse (5 MHz)
pulse_t = np.arange(-1e-6, 1e-6, 1/fs) # Time vector for the pulse
et = np.exp(-0.5 * (pulse_t / 0.15e-6)**2) * np.cos(2 * np.pi * f0 * pulse_t) # Gaussian-modulated sinusoid (invilupse * cosine)

# --- 4. SIMULATION ---
b_mode_matrix = np.zeros((len(t), num_channels)) # Initialize B-mode image matrix. Our image will be stored here

print("Simulating scan lines...")
for i, x_tx in enumerate(x_array): # For each transducer element
    rf_line = np.zeros_like(t) # Initialize RF line for this transducer element 
    for xs, zs, amp in zip(x_s, z_s, amps): # For each scatterer
        dist = np.sqrt((x_tx - xs)**2 + zs**2) # Distance from transducer element to scatterer
        tau = (2 * dist) / c # Round-trip time delay
        idx_tau = np.argmin(np.abs(t - tau)) # Find the index in time vector closest to tau
        start_idx = idx_tau - len(et)//2 # Center the pulse around the idx_tau
        end_idx = start_idx + len(et) # End index for adding the pulse
        
        if 0 <= start_idx and end_idx < len(rf_line): # Ensure indices are within bounds
            lateral_weight = np.exp(-((x_tx - xs)**2) / (0.0005**2)) # Lateral weighting (Gaussian)
            rf_line[start_idx:end_idx] += et * amp * lateral_weight # Add weighted pulse to RF line
            
    b_mode_matrix[:, i] = np.abs(hilbert(rf_line)) # Envelope detection using Hilbert transform

# --- 5. VISUALIZATION ---
plt.figure(figsize=(7, 9))
image_data = 20 * np.log10(b_mode_matrix + 1e-6) # Convert to dB scale, we use +1e-6 to avoid log(0), and image_data is used to decrease the white intensity and improve weak scatterers visibility
image_data -= np.max(image_data) # Normalize to max value

plt.imshow(image_data, aspect='auto', cmap='gray', vmin=-40, vmax=0,
           extent=[x_array[0]*100, x_array[-1]*100, depth_max*100, 0])

plt.title("Ultrasound B-mode: Multiple Target Sizes")
plt.xlabel("X Position (cm)")
plt.ylabel("Depth Z (cm)")
plt.colorbar(label="Intensity (dB)")
plt.show()