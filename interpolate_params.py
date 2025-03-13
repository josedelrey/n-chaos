import os
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
num_frames = 360 * 4
output_dir = "frames"
span_factor = 1.5  # Span factor for shading
# alpha is no longer used and has been removed

# Initial States Parameters
N_initial_states = int(1e5)  # Number of initial states (adjust based on GPU memory)

# Calculate steps per initial state separately for both passes
steps_per_initial_state_first_pass = int(1e3)
steps_per_initial_state_second_pass = steps_per_initial_state_first_pass

# Starting Rotation Angles
starting_angle_x = 0.0  # degrees
starting_angle_y = 0.0  # degrees
starting_angle_z = 0.0  # degrees

# Rotation Parameters
angle_x_per_frame = 0.0  # degrees per frame
angle_y_per_frame = 1.0  # degrees per frame
angle_z_per_frame = 0.0  # degrees per frame

# Attractor Parameters
w_min = 1.84
w_max = 1.9067
w_values = np.linspace(w_min, w_max, num_frames)

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
padding_factor = 1.15  # 15% padding for plot limits
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

# Number of samples for dynamic plot limits
num_samples = int(num_frames / 30)  # Number of sampled points in time for plot limits

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

def generate_preview_image(a_params, lyapunov_exponent, identifier):
    """
    Generates a preview image of the attractor immediately after discovering it.
    """
    print("Generating preview image...")

    # Set up necessary parameters
    current_w = w_values[0]  # Use the first w value
    steps_per_initial_state = steps_per_initial_state_second_pass  # Use the same as in second pass

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
        f"g={a_params[6]:.2f}, h={a_params[7]:.2f}",
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

    # Save the image
    filename = f'preview_{identifier}.png'
    img_pil.save(filename, "PNG")

    # Clean up
    del x_values, y_values, z_values, points, points_centered, rotated, projected, projected_np, df_projected, agg, agg_log, shaded, img_pil
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    print(f"Preview image saved as '{filename}'.")

# -----------------------
# Plotting Function with Two-Pass Approach and Dynamic Plot Limits
# -----------------------

def plot_frames_as_2d(a_params, lyapunov_exponent, identifier, num_frames=num_frames, output_dir=output_dir):
    """
    Generates and saves frames of the attractor with perspective projection and rotation.
    Adjusts plot limits dynamically by sampling 30 points in time and interpolating between them.
    Also interpolates midpoints using spline interpolation for smooth transitions.

    Parameters:
    - a_params: List of attractor parameters.
    - lyapunov_exponent: Calculated Lyapunov exponent.
    - identifier: Unique identifier for the attractor.
    - num_frames: Number of frames to generate.
    - output_dir: Directory to save the frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize variables for sampled plot limits and midpoints
    sampled_limits = []  # List to store (x_min, x_max, y_min, y_max) for sampled frames
    sampled_midpoints = []  # List to store (mid_x, mid_y, mid_z) for sampled frames
    sampled_frame_indices = [int(i * (num_frames - 1) / (num_samples - 1)) for i in range(num_samples)]

    # -----------------------
    # First Pass: Sample Plot Limits and Midpoints at 30 Points in Time with Exact Midpoints
    # -----------------------
    print("Starting first pass to sample plot limits and midpoints...")
    for frame in tqdm(range(num_frames), desc="First Pass - Sampling Plot Limits and Midpoints"):
        # Check if current frame is a sampled frame
        if frame in sampled_frame_indices:
            # Define the fixed w value for this frame
            current_w = w_values[frame]

            # Generate N_initial_states random initial states within a noisy sphere
            noisy_points = random_points_in_noisy_sphere(N_initial_states, R=0.5, noise_scale=noise_scale)
            initial_states = np.zeros((N_initial_states, 4))
            initial_states[:, 0:3] = noisy_points
            initial_states[:, 3] = current_w  # Set w to current_w

            # Use steps_per_initial_state_first_pass
            x_values_cp, y_values_cp, z_values_cp = generate_data_points_4d_parallel(
                a_params, initial_states, steps_per_initial_state_first_pass, convergence_threshold
            )

            # Calculate number of initial steps to discard
            num_discard_steps = int((steps_per_initial_state_first_pass + 1) * discard_pct / 100)
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
            # Calculate Midpoints Between Extremes of X, Y, Z
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

            # Center the points using frame_mid (No Alpha Blending)
            points_centered = points - frame_mid

            # -----------------------
            # 3D Rotation (Per-Frame Rotations)
            # -----------------------
            angle_x = starting_angle_x + angle_x_per_frame * frame
            angle_y = starting_angle_y + angle_y_per_frame * frame
            angle_z = starting_angle_z + angle_z_per_frame * frame
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
            # Calculate Plot Limits with Padding
            # -----------------------
            x_min_proj = projected[:, 0].min().item()
            x_max_proj = projected[:, 0].max().item()
            y_min_proj = projected[:, 1].min().item()
            y_max_proj = projected[:, 1].max().item()

            # Add padding to the ranges
            x_range_span = x_max_proj - x_min_proj
            y_range_span = y_max_proj - y_min_proj
            x_padding = x_range_span * (padding_factor - 1)
            y_padding = y_range_span * (padding_factor - 1)
            x_range = (x_min_proj - x_padding, x_max_proj + x_padding)
            y_range = (y_min_proj - y_padding, y_max_proj + y_padding)

            # Store the plot limits for this sampled frame
            sampled_limits.append((x_range[0], x_range[1], y_range[0], y_range[1]))

            # Store the exact midpoints for this sampled frame
            sampled_midpoints.append((frame_mid_x.get(), frame_mid_y.get(), frame_mid_z.get()))

            # -----------------------
            # Clean up
            # -----------------------
            del x_values, y_values, z_values, points, points_centered, rotated, projected, projected_np, df_projected, x_values_cp, y_values_cp, z_values_cp
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

    # -----------------------
    # Interpolate Plot Limits and Midpoints for All Frames
    # -----------------------
    print("Interpolating plot limits and midpoints for all frames...")

    # Convert sampled_limits and sampled_midpoints to NumPy arrays for easier manipulation
    sampled_limits_np = np.array(sampled_limits)  # Shape: (num_samples, 4)
    sampled_midpoints_np = np.array(sampled_midpoints)  # Shape: (num_samples, 3)
    sampled_frame_indices_np = np.array(sampled_frame_indices)

    # Create spline interpolators for plot limits
    spline_x_min = CubicSpline(sampled_frame_indices_np, sampled_limits_np[:, 0])
    spline_x_max = CubicSpline(sampled_frame_indices_np, sampled_limits_np[:, 1])
    spline_y_min = CubicSpline(sampled_frame_indices_np, sampled_limits_np[:, 2])
    spline_y_max = CubicSpline(sampled_frame_indices_np, sampled_limits_np[:, 3])

    # Create spline interpolators for midpoints
    spline_mid_x = CubicSpline(sampled_frame_indices_np, sampled_midpoints_np[:, 0])
    spline_mid_y = CubicSpline(sampled_frame_indices_np, sampled_midpoints_np[:, 1])
    spline_mid_z = CubicSpline(sampled_frame_indices_np, sampled_midpoints_np[:, 2])

    # -----------------------
    # Second Pass: Generate and Save Frames with Interpolated Plot Limits and Midpoints
    # -----------------------
    print("Starting second pass to generate and save frames...")
    for frame in tqdm(range(num_frames), desc="Second Pass - Generating Frames"):
        # Define the fixed w value for this frame
        current_w = w_values[frame]

        # Generate N_initial_states random initial states within a noisy sphere
        noisy_points = random_points_in_noisy_sphere(N_initial_states, R=0.5, noise_scale=noise_scale)
        initial_states = np.zeros((N_initial_states, 4))
        initial_states[:, 0:3] = noisy_points
        initial_states[:, 3] = current_w  # Set w to current_w

        # Use steps_per_initial_state_second_pass
        x_values_cp, y_values_cp, z_values_cp = generate_data_points_4d_parallel(
            a_params, initial_states, steps_per_initial_state_second_pass, convergence_threshold
        )

        # Calculate number of initial steps to discard
        num_discard_steps = int((steps_per_initial_state_second_pass + 1) * discard_pct / 100)
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
        # Calculate Midpoints Between Extremes of X, Y, Z
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

        # -----------------------
        # Interpolated Midpoints for Current Frame
        # -----------------------
        interpolated_mid_x = spline_mid_x(frame)
        interpolated_mid_y = spline_mid_y(frame)
        interpolated_mid_z = spline_mid_z(frame)

        interpolated_mid = cp.array([interpolated_mid_x, interpolated_mid_y, interpolated_mid_z])

        # Center the points using interpolated_mid
        points_centered = points - interpolated_mid

        # -----------------------
        # 3D Rotation (Per-Frame Rotations)
        # -----------------------
        angle_x = starting_angle_x + angle_x_per_frame * frame
        angle_y = starting_angle_y + angle_y_per_frame * frame
        angle_z = starting_angle_z + angle_z_per_frame * frame
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
        # Retrieve Interpolated Plot Limits for Current Frame
        # -----------------------
        x_min_plot = spline_x_min(frame)
        x_max_plot = spline_x_max(frame)
        y_min_plot = spline_y_min(frame)
        y_max_plot = spline_y_max(frame)

        # Add padding to the ranges
        x_range_span = x_max_plot - x_min_plot
        y_range_span = y_max_plot - y_min_plot
        x_padding = x_range_span * (padding_factor - 1)
        y_padding = y_range_span * (padding_factor - 1)

        # Define the new plot ranges with padding
        x_range = (x_min_plot - x_padding, x_max_plot + x_padding)
        y_range = (y_min_plot - y_padding, y_max_plot + y_padding)

        # Initialize the Datashader Canvas with interpolated and padded ranges
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

        # Angle values for the current frame
        angle_x_display = angle_x % 360
        angle_y_display = angle_y % 360
        angle_z_display = angle_z % 360

        # -----------------------
        # Add Text Overlay
        # -----------------------
        info_text = [
            identifier,  # Added unique identifier as the first line
            f"Lyapunov Exponent: {lyapunov_exponent:.6f}",
            f"a={a_params[0]:.6f}, b={a_params[1]:.6f}, c={a_params[2]:.6f}",
            f"d={a_params[3]:.6f}, e={a_params[4]:.6f}, f={a_params[5]:.6f}",
            f"g={a_params[6]:.2f}, h={a_params[7]:.2f}",
            f"w = {current_w:.4f}",
            f"Rotation: X={angle_x_display:.2f}°, Y={angle_y_display:.2f}°, Z={angle_z_display:.2f}°",
            f"Frame: {frame + 1}/{num_frames}"
        ]

        add_text_overlay(
            image=img_pil,
            text_lines=info_text,
            font=font,
            text_color=text_color,
            padding_x=padding_x,
            padding_y=padding_y
        )

        # Save the image
        filename = os.path.join(output_dir, f"frame_{frame:04d}.png")
        img_pil.save(filename, "PNG")

        # -----------------------
        # Clean up
        # -----------------------
        del x_values, y_values, z_values, points, points_centered, rotated, projected, projected_np, df_projected, agg, agg_log, shaded, img_pil
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    print(f"All {num_frames} frames have been saved to the '{output_dir}' directory.")

# -----------------------
# Main Execution
# -----------------------

if __name__ == "__main__":
    # -----------------------
    # Manual Definition of Attractor Parameters
    # -----------------------
    # Define your attractor parameters here. Ensure you have 8 parameters: a, b, c, d, e, f, g, h
    a_params = [1.341717991231581, -0.109130377750538, -1.222433639170113,
                0.183534217149607, -0.966406667444416, -1.476072726203629,
                -0.742202450340568, -1.506377099919801]

    # -----------------------
    # Manual Assignment of Lyapunov Exponent
    # -----------------------
    # Updated Lyapunov exponent from CSV data
    lyapunov_exponent = 8.243889624666691

    # -----------------------
    # Manually Specify Unique Identifier
    # -----------------------
    identifier = "ATR-123456"  # Manually specified identifier

    # -----------------------
    # Generate Preview Image
    # -----------------------
    generate_preview_image(a_params, lyapunov_exponent, identifier)

    # -----------------------
    # Plotting Frames
    # -----------------------
    print("Plotting 2D frames...")
    plot_frames_as_2d(a_params, lyapunov_exponent, identifier, num_frames, output_dir)

    print("Frames saved. You can now create a video using ffmpeg.")
    print("Example ffmpeg command:")
    print(f"ffmpeg -r 30 -f image2 -s {canvas_width}x{canvas_height} -i frames/frame_%04d.png -vcodec libx264 -crf 0 -preset veryslow -pix_fmt yuv420p {identifier}.mp4")
