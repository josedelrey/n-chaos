import os
import random
import math
import gc
import hashlib

import cupy as cp
import numpy as np
import pandas as pd
import datashader as ds
from datashader import transfer_functions as tf
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from PIL import ImageDraw, ImageFont
from scipy.interpolate import CubicSpline  # Import for spline interpolation

# -----------------------
# Parameters
# -----------------------

# General Parameters
search_steps = int(1e6)
output_dir = "session"  # Changed to 'session' as per your request
span_factor = 1.5  # Span factor for shading
num_iterations = int(1e9)  # Number of attractors to find

# Initial States Parameters
N_initial_states = int(1e5)  # Number of initial states (adjust based on GPU memory)

# Starting Rotation Angles
starting_angle_x = 0.0  # degrees
starting_angle_y = 0.0  # degrees
starting_angle_z = 0.0  # degrees

# Canvas Resolution Parameters
canvas_width = 1280  # Width of the canvas in pixels
canvas_height = 960  # Height of the canvas in pixels

# Font Parameters
font_path = os.path.join("fonts", "courbd.ttf")  # Path to font
font_size = 20  # Font size for the text overlay

# Background and Text Colors
background_color = "#000000"  # Black background
text_color = (255, 255, 255)  # Bright white text

# Camera Parameters
camera_distance = 5.0  # Distance of the camera from the origin along the z-axis

# Padding Parameters
padding_factor = 1.10  # 10% padding for plot limits
padding_x = 20  # Horizontal padding for text overlay
padding_y = 20  # Vertical padding for text overlay

# Noise Parameters for Initial States
noise_scale = 0.05  # Adjust this value to control the amount of noise

# Thresholds for standard deviation and range checks
std_threshold = 0.1  # Threshold for standard deviation of x, y, z
range_threshold = 1.0  # Threshold for the range (max - min) of x, y, z

# Maximum allowed absolute value for x, y, z during the attractor search
convergence_threshold = 10  # Set maximum range from -10 to 10

# Percentage of initial steps to discard before plotting to avoid initial sphere artifact
discard_pct = 5.0  # Adjust as needed (e.g., 5%)

# -----------------------
# Custom Colormap Modification
# -----------------------

# Get the 'viridis' colormap with 256 colors
original_cmap = plt.get_cmap('viridis', 256)
viridis_colors = original_cmap(np.linspace(0, 1, 256))

# Replace the first color with black (RGBA: [0, 0, 0, 1])
viridis_colors[:1, :] = [0, 0, 0, 1]

# Find the index where the color is green
green_index = 170  # You can adjust this index if needed
green_color = viridis_colors[green_index, :3]  # Get RGB values, ignore alpha

# Create a gradient from green to white over the remaining indices
num_steps = 256 - green_index
interp_factors = np.linspace(0, 1, num_steps)[:, np.newaxis]
white_color = np.array([1, 1, 1])  # RGB for white

# Interpolate between green_color and white_color
new_colors = green_color + interp_factors * (white_color - green_color)

# Add alpha channel
new_colors_with_alpha = np.hstack((new_colors, np.ones((num_steps, 1))))

# Update the colormap from green_index to the end
viridis_colors[green_index:, :] = new_colors_with_alpha

# Create a new ListedColormap with the modified colors
custom_cmap = ListedColormap(viridis_colors)

# Assign the custom colormap to matplotlib_cmap
matplotlib_cmap = custom_cmap

# -----------------------
# Load the Font
# -----------------------

# Load the font with desired size
try:
    font = ImageFont.truetype(font_path, size=font_size)
    print("Font loaded successfully.")
except IOError:
    print("Font not found. Falling back to default font.")
    font = ImageFont.load_default()

# -----------------------
# 3D Rotation Matrices (Use CuPy)
# -----------------------

def rotation_matrix_x(angle):
    rad = cp.deg2rad(angle)
    c = cp.cos(rad)
    s = cp.sin(rad)
    R = cp.eye(3, dtype=cp.float32)
    R[1, 1] = c
    R[1, 2] = -s
    R[2, 1] = s
    R[2, 2] = c
    return R

def rotation_matrix_y(angle):
    rad = cp.deg2rad(angle)
    c = cp.cos(rad)
    s = cp.sin(rad)
    R = cp.eye(3, dtype=cp.float32)
    R[0, 0] = c
    R[0, 2] = s
    R[2, 0] = -s
    R[2, 2] = c
    return R

def rotation_matrix_z(angle):
    rad = cp.deg2rad(angle)
    c = cp.cos(rad)
    s = cp.sin(rad)
    R = cp.eye(3, dtype=cp.float32)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R

# -----------------------
# 4D Attractor Functions
# -----------------------

def find_attractor_4d(search_steps=search_steps, convergence_threshold=convergence_threshold, std_threshold=std_threshold, range_threshold=range_threshold):
    """
    Finds parameters for a 4D attractor ensuring chaotic behavior.
    Returns the parameters and Lyapunov exponent.
    """
    found = False
    while not found:
        lyapunov = 0
        converging = False

        # Initialize state
        x = random.uniform(-0.5, 0.5)
        y = random.uniform(-0.5, 0.5)
        z = random.uniform(-0.5, 0.5)
        w = random.uniform(-2, 2)  # Random w value
        initial_state = [x, y, z, w]

        # Perturbation for Lyapunov calculation
        x_prime = x + random.uniform(-0.5, 0.5) / 1e3
        y_prime = y + random.uniform(-0.5, 0.5) / 1e3
        z_prime = z + random.uniform(-0.5, 0.5) / 1e3
        w_prime = w + random.uniform(-0.5, 0.5) / 1e3

        dx = x_prime - x
        dy = y_prime - y
        dz = z_prime - z
        dw = w_prime - w
        initial_distance = math.sqrt(dx**2 + dy**2 + dz**2 + dw**2)

        # Initialize 8 parameters: a, b, c, d, e, f, g, h
        a_params = [random.uniform(-2, 2) for _ in range(8)]

        # Lists to store points for spread analysis
        x_list = []
        y_list = []
        z_list = []

        for i in range(search_steps):
            a, b, c, d, e, f, g, h = a_params

            # System of equations for the attractor
            x_new = math.sin(a * y + b * w) - z * math.cos(c * x + d * w)
            y_new = math.sin(e * z + f * w) + x * math.cos(g * y + h * w)
            z_new = math.sin(a * x + b * w) + y * math.cos(c * z + d * w)
            w_new = w  # w remains constant for this attractor

            x_prime_new = math.sin(a * y_prime + b * w_prime) - z_prime * math.cos(c * x_prime + d * w_prime)
            y_prime_new = math.sin(e * z_prime + f * w_prime) + x_prime * math.cos(g * y_prime + h * w_prime)
            z_prime_new = math.sin(a * x_prime + b * w_prime) + y_prime * math.cos(c * z_prime + d * w_prime)
            w_prime_new = w_prime  # w_prime remains constant

            # Check for divergence
            if (abs(x_new) > convergence_threshold or
                abs(y_new) > convergence_threshold or
                abs(z_new) > convergence_threshold):
                converging = True
                break

            # Check for convergence
            if (abs(x_new - x) < 1/convergence_threshold and
                abs(y_new - y) < 1/convergence_threshold and
                abs(z_new - z) < 1/convergence_threshold):
                converging = True
                break

            if i > 1000:  # Start collecting data after transient period
                x_list.append(x_new)
                y_list.append(y_new)
                z_list.append(z_new)

            if i > 1e3:
                dx = x_prime_new - x_new
                dy = y_prime_new - y_new
                dz = z_prime_new - z_new
                dw = w_prime_new - w_new
                current_distance = math.sqrt(dx**2 + dy**2 + dz**2 + dw**2)

                if current_distance == 0:
                    lyapunov = -math.inf
                    converging = True
                    break

                lyapunov += math.log(abs(current_distance / initial_distance))
                x_prime = x_new + (initial_distance * dx / current_distance)
                y_prime = y_new + (initial_distance * dy / current_distance)
                z_prime = z_new + (initial_distance * dz / current_distance)
                w_prime = w_new  # w remains constant

            x, y, z, w = x_new, y_new, z_new, w_new
            x_prime, y_prime, z_prime, w_prime = x_prime_new, y_prime_new, z_prime_new, w_prime_new

        # Criteria for a valid attractor
        if not converging and (lyapunov / search_steps) >= 1e-3:
            # Compute standard deviations
            x_array = np.array(x_list)
            y_array = np.array(y_list)
            z_array = np.array(z_list)

            x_std = np.std(x_array)
            y_std = np.std(y_array)
            z_std = np.std(z_array)

            # Compute ranges
            x_range_value = np.max(x_array) - np.min(x_array)
            y_range_value = np.max(y_array) - np.min(y_array)
            z_range_value = np.max(z_array) - np.min(z_array)

            # Check if standard deviations and ranges are above thresholds
            if (x_std > std_threshold and y_std > std_threshold and z_std > std_threshold and
                x_range_value > range_threshold and y_range_value > range_threshold and z_range_value > range_threshold):
                print("Found a valid and interesting attractor with Lyapunov exponent:", lyapunov / search_steps)
                print("Parameters:", a_params)
                found = True
            else:
                print("Attractor discarded due to insufficient spread.")
        else:
            print("Attractor discarded due to convergence or low Lyapunov exponent.")

    return a_params, lyapunov / search_steps

def generate_unique_identifier(a_params):
    """
    Generates a unique identifier for the attractor based on its parameters.
    Format: ATR-{number}, where number is derived from the MD5 hash of the parameters.
    """
    # Convert parameters to a fixed precision string
    params_str = ",".join(f"{p:.15f}" for p in a_params)
    
    # Compute MD5 hash
    hash_obj = hashlib.md5(params_str.encode('utf-8'))
    hash_digest = hash_obj.hexdigest()
    
    # Take the first 6 characters of the hash and convert to an integer
    number = int(hash_digest[:6], 16)
    
    # Format the identifier
    identifier = f"ATR-{number}"
    return identifier

def generate_data_points_4d_parallel(a_params, initial_states, steps_per_initial_state, convergence_threshold=convergence_threshold):
    """
    Generates x, y, z points with fixed w based on the 4D attractor parameters,
    for multiple initial states in parallel using CuPy.
    """
    a, b, c, d, e, f, g, h = a_params

    # Convert initial_states to CuPy arrays
    initial_states = cp.asarray(initial_states)  # Shape: (N, 4)
    x = initial_states[:, 0]
    y = initial_states[:, 1]
    z = initial_states[:, 2]
    w = initial_states[:, 3]

    # Pre-allocate arrays to store the results
    x_values = cp.zeros((steps_per_initial_state + 1, x.size), dtype=cp.float32)
    y_values = cp.zeros_like(x_values)
    z_values = cp.zeros_like(x_values)

    # Store initial values
    x_values[0, :] = x
    y_values[0, :] = y
    z_values[0, :] = z

    for i in range(steps_per_initial_state):
        # System of equations for the attractor
        x_new = cp.sin(a * y + b * w) - z * cp.cos(c * x + d * w)
        y_new = cp.sin(e * z + f * w) + x * cp.cos(g * y + h * w)
        z_new = cp.sin(a * x + b * w) + y * cp.cos(c * z + d * w)

        # Check for divergence
        mask = (cp.abs(x_new) <= convergence_threshold) & \
               (cp.abs(y_new) <= convergence_threshold) & \
               (cp.abs(z_new) <= convergence_threshold)

        # Update states where trajectories have not diverged
        x = cp.where(mask, x_new, x)
        y = cp.where(mask, y_new, y)
        z = cp.where(mask, z_new, z)

        # Store updated states
        x_values[i + 1, :] = x
        y_values[i + 1, :] = y
        z_values[i + 1, :] = z

    return x_values, y_values, z_values  # Return CuPy arrays

# -----------------------
# Projection Functions (Use CuPy)
# -----------------------

def perspective_projection(points, camera_distance=camera_distance):
    """
    Performs perspective projection of 3D points onto a 2D plane.

    Parameters:
    - points: Nx3 CuPy array of 3D points.
    - camera_distance: Distance of the camera from the origin along the z-axis.

    Returns:
    - Nx2 CuPy array of projected 2D points.
    """
    # Avoid division by zero by ensuring z + camera_distance != 0
    z_adjusted = points[:, 2] + camera_distance

    # Handle points where z_adjusted is zero by setting a minimum value
    epsilon = 1e-6
    z_adjusted = cp.where(z_adjusted == 0, epsilon, z_adjusted)

    # Perform the projection
    x_proj = (points[:, 0] * camera_distance) / z_adjusted
    y_proj = (points[:, 1] * camera_distance) / z_adjusted

    return cp.vstack((x_proj, y_proj)).T

# -----------------------
# 3D Rotation Function (Use CuPy)
# -----------------------

def rotate_3d(points, angle_x, angle_y, angle_z):
    """
    Rotates 3D points around the X, Y, and Z axes.

    Parameters:
    - points: Nx3 CuPy array of 3D points.
    - angle_x: Rotation angle around the X-axis in degrees.
    - angle_y: Rotation angle around the Y-axis in degrees.
    - angle_z: Rotation angle around the Z-axis in degrees.

    Returns:
    - Nx3 CuPy array of rotated 3D points.
    """
    R_x = rotation_matrix_x(angle_x)
    R_y = rotation_matrix_y(angle_y)
    R_z = rotation_matrix_z(angle_z)
    rotated = points @ R_x.T @ R_y.T @ R_z.T
    return rotated

# -----------------------
# Text Overlay Function
# -----------------------

def add_text_overlay(image, text_lines, font, text_color=text_color, padding_x=padding_x, padding_y=padding_y):
    """
    Adds multiple lines of styled text to an image.

    Parameters:
    - image: PIL Image object.
    - text_lines: List of strings, each representing a line of text.
    - font: PIL ImageFont object.
    - text_color: Tuple representing RGB color of the text.
    - padding_x: Horizontal padding from the right edge.
    - padding_y: Vertical padding from the top edge and between lines.
    """
    draw = ImageDraw.Draw(image)

    # Use getmetrics to determine ascent and descent
    ascent, descent = font.getmetrics()
    line_height = ascent + descent + 4  # Increased line height

    image_width, image_height = image.size

    for i, line in enumerate(text_lines):
        # Calculate text width using textbbox
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]

        # Position on the right side with additional horizontal padding
        x_position = image_width - padding_x - text_width
        y_position = padding_y + i * line_height

        # Draw main text
        draw.text((x_position, y_position), line, font=font, fill=text_color)

# -----------------------
# Function to Generate Noisy Sphere Initial States
# -----------------------

def random_points_in_noisy_sphere(N, R=0.5, noise_scale=noise_scale):
    """
    Generates N random points within a sphere of radius R, with noise added to the positions.
    """
    # Generate random directions
    x = np.random.normal(size=(N, 3))
    x_norm = np.linalg.norm(x, axis=1)
    x_unit = x / x_norm[:, np.newaxis]

    # Generate random radii within the sphere
    r = np.random.rand(N) ** (1/3) * R

    # Compute points inside sphere
    x_in_sphere = x_unit * r[:, np.newaxis]

    # Add noise to the points
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=x_in_sphere.shape)
    noisy_points = x_in_sphere + noise
    return noisy_points

# -----------------------
# Preview Image Function (Modified)
# -----------------------

def generate_preview_image(a_params, lyapunov_exponent, identifier, output_dir=output_dir):
    """
    Generates a preview image of the attractor immediately after discovering it.
    """
    print("Generating preview image...")

    # Set up necessary parameters
    current_w = random.uniform(-2, 2)  # Random w value for preview
    steps_per_initial_state = 1000  # Adjust as needed

    # Generate N_initial_states random initial states within a noisy sphere
    noisy_points = random_points_in_noisy_sphere(N_initial_states, R=0.5, noise_scale=noise_scale)
    initial_states = np.zeros((N_initial_states, 4))
    initial_states[:, 0:3] = noisy_points
    initial_states[:, 3] = current_w  # Set w to current_w

    # Generate data points in parallel using CuPy
    x_values_cp, y_values_cp, z_values_cp = generate_data_points_4d_parallel(
        a_params, initial_states, steps_per_initial_state, convergence_threshold
    )

    # Calculate number of initial steps to discard
    num_discard_steps = int((steps_per_initial_state + 1) * discard_pct / 100)
    num_discard_steps = max(num_discard_steps, 1)

    # Discard initial steps
    x_values_cp = x_values_cp[num_discard_steps:, :]
    y_values_cp = y_values_cp[num_discard_steps:, :]
    z_values_cp = z_values_cp[num_discard_steps:, :]

    # Flatten the arrays (keep as CuPy arrays)
    x_values = x_values_cp.flatten()
    y_values = y_values_cp.flatten()
    z_values = z_values_cp.flatten()

    # Stack the points into a CuPy array
    points = cp.vstack((x_values, y_values, z_values)).T  # Shape: (N_total_points, 3)

    # -----------------------
    # Centering Using Frame Data (No Alpha Blending)
    # -----------------------
    x_min = cp.min(points[:, 0])
    x_max = cp.max(points[:, 0])
    y_min = cp.min(points[:, 1])
    y_max = cp.max(points[:, 1])
    z_min = cp.min(points[:, 2])
    z_max = cp.max(points[:, 2])

    # Compute midpoints for x, y, z
    frame_mid_x = (x_min + x_max) / 2
    frame_mid_y = (y_min + y_max) / 2
    frame_mid_z = (z_min + z_max) / 2

    frame_mid = cp.array([frame_mid_x, frame_mid_y, frame_mid_z])

    # Center the points using frame_mid
    points_centered = points - frame_mid

    # -----------------------
    # 3D Rotation (Set angles to desired values)
    # -----------------------
    angle_x = starting_angle_x
    angle_y = starting_angle_y
    angle_z = starting_angle_z
    rotated = rotate_3d(points_centered, angle_x, angle_y, angle_z)

    # -----------------------
    # Perspective Projection
    # -----------------------
    projected = perspective_projection(rotated, camera_distance=camera_distance)

    # -----------------------
    # Prepare a DataFrame for the projected points
    # -----------------------
    # Convert projected points to NumPy arrays for Pandas
    projected_np = cp.asnumpy(projected)

    df_projected = pd.DataFrame({
        'x': projected_np[:, 0],
        'y': projected_np[:, 1]
    })

    # -----------------------
    # Initialize the Datashader Canvas with ranges based on data
    # -----------------------
    x_min_proj = projected[:, 0].min().item()
    x_max_proj = projected[:, 0].max().item()
    y_min_proj = projected[:, 1].min().item()
    y_max_proj = projected[:, 1].max().item()

    # Add padding to the ranges
    x_range_span = x_max_proj - x_min_proj
    y_range_span = y_max_proj - y_min_proj

    # Handle zero spans by setting to a default span
    if x_range_span == 0:
        x_range_span = 1.0  # Default span if all x are the same
    if y_range_span == 0:
        y_range_span = 1.0  # Default span if all y are the same

    x_padding = x_range_span * (padding_factor - 1)
    y_padding = y_range_span * (padding_factor - 1)
    x_range = (x_min_proj - x_padding, x_max_proj + x_padding)
    y_range = (y_min_proj - y_padding, y_max_proj + y_padding)

    canvas = ds.Canvas(plot_width=canvas_width, plot_height=canvas_height,
                       x_range=x_range,
                       y_range=y_range)

    # -----------------------
    # Aggregation and Shading
    # -----------------------
    # Aggregation Step: Count the number of points per pixel
    agg = canvas.points(df_projected, 'x', 'y')

    # Apply Logarithmic Scaling
    agg_log = np.log1p(agg)  # log(1 + agg) to handle zero counts

    # Dynamic Span Calculation Using Percentiles on log-scaled agg
    p_min, p_max = np.percentile(agg_log.values.flatten(), [1, 99])

    # Handle cases where p_max == p_min to avoid span issues
    if p_max == p_min:
        p_max = p_min + 1  # Arbitrary increment to prevent zero span

    # Shading Step: Apply the Matplotlib colormap with dynamic span
    shaded = tf.shade(
        agg_log,
        cmap=matplotlib_cmap,  # Use the custom Matplotlib colormap
        how='linear',  # Linear scaling on log-transformed data
        span=[0, p_max * span_factor]  # Dynamic span based on log-scaled data percentiles
    )

    # Set the Background Color
    shaded = tf.set_background(shaded, background_color)

    # Convert the Datashader image to a PIL Image
    img_pil = shaded.to_pil()

    # -----------------------
    # Add Text Overlay
    # -----------------------
    info_text = [
        identifier,  # Added unique identifier as the first line
        f"Lyapunov Exponent: {lyapunov_exponent:.6f}",
        f"a={a_params[0]:.6f}, b={a_params[1]:.6f}, c={a_params[2]:.6f}",
        f"d={a_params[3]:.6f}, e={a_params[4]:.6f}, f={a_params[5]:.6f}",
        f"g={a_params[6]:.6f}, h={a_params[7]:.6f}",
        f"w = {current_w:.4f}",
        "Preview Image"
    ]

    add_text_overlay(
        image=img_pil,
        text_lines=info_text,
        font=font,
        text_color=text_color,
        padding_x=padding_x,
        padding_y=padding_y
    )

    # Save the image in the 'session' directory
    filename = os.path.join(output_dir, f'{identifier}.png')
    img_pil.save(filename, "PNG")

    # Clean up
    del x_values, y_values, z_values, points, points_centered, rotated, projected, projected_np, df_projected, agg, agg_log, shaded, img_pil
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    print(f"Preview image saved as '{filename}'.")

# -----------------------
# Main Execution
# -----------------------

if __name__ == "__main__":
    # Create the session directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Path to the log file in the session directory
    log_file = os.path.join(output_dir, "log.csv")

    # Check if log.csv exists; if not, create it and write the header
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("Identifier,a,b,c,d,e,f,g,h,Lyapunov_Exponent\n")

    for i in range(num_iterations):
        print(f"\nFinding attractor {i+1} of {num_iterations}...")
        a_params, lyapunov_exponent = find_attractor_4d(
            search_steps, convergence_threshold, std_threshold, range_threshold
        )

        # Generate unique identifier
        identifier = generate_unique_identifier(a_params)
        print(f"Attractor Identifier: {identifier}")

        # Append the current run's information with full precision
        with open(log_file, 'a') as f:
            # Format: Identifier,a,b,c,d,e,f,g,h,Lyapunov_Exponent
            param_str = ",".join(f"{p:.15f}" for p in a_params)
            lyapunov_str = f"{lyapunov_exponent:.15f}"
            f.write(f"{identifier},{param_str},{lyapunov_str}\n")

        print(f"Attractor information appended to '{log_file}'.")

        # Generate Preview Image
        generate_preview_image(a_params, lyapunov_exponent, identifier, output_dir)

        print(f"Preview image for attractor {identifier} saved.")

    print("\nAll attractors have been processed. Preview images and log file are saved in the 'session' directory.")
