 import os
import sys
import time
import json
import random
import math
import requests
import traceback
import io
import datetime
import uuid
import websocket
import numpy as np
from threading import Thread, Lock
from flask import Flask, render_template_string, request, jsonify, send_file
from PIL import Image
from random import randint
from flask import (
    Flask,
    render_template_string,
    request,
    redirect,
    send_file,
    flash,
    url_for
)
import requests
from moviepy.editor import (
    ImageClip,
    concatenate_videoclips,
    AudioFileClip,
    VideoFileClip,
    vfx,
    CompositeAudioClip
)
from pydub import AudioSegment
from werkzeug.utils import secure_filename
from icecream import ic
# NO COMFY COMFY_URL = "http://192.168.1.41:5001"
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, LCMScheduler
import logging
import warnings
import datetime
import inspect

'''
# Define the log file path
LOG_FILE_PATH = "static/app_log.txt"

# Ensure the log file exists or create it
if not os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, "w"):
        pass  # Create an empty log file if it doesn't exist


def logit(*args):
    """
    Lightweight file-based logging utility for debugging and runtime diagnostics.
    Accepts ANY number of positional arguments and safely joins them into a
    single log message.

    Examples:
        logit("Hello world")
        logit("x:", x, "y:", y)
        logit("Paths:", image_paths)
        logit(["list", "of", "values"])
    """

    try:
        # --- timestamp ---
        timestr = datetime.datetime.now().strftime("%A_%b-%d-%Y_%H-%M-%S")

        # --- caller info ---
        frame = inspect.stack()[1]
        filename = frame.filename
        lineno = frame.lineno

        # --- normalize all inputs ---
        parts = []
        for arg in args:
            if isinstance(arg, (list, tuple, set)):
                parts.append(" ".join(map(str, arg)))
            else:
                parts.append(str(arg))

        message_str = " ".join(parts)

        # --- final log line ---
        log_message = f"{timestr} - File: {filename}, Line: {lineno}: {message_str}\n"

        # --- write log ---
        with open(LOG_FILE_PATH, "a") as f:
            f.write(log_message)

        # Optional console echo (leave commented if you want quiet logs)
        # print(log_message, end="")

    except Exception as e:
        print(f"[LOGIT ERROR] {e}")

'''




logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================
# CONFIG & PATHS
# ============================================
OLLAMA_URL = "http://localhost:11434"

# Global progress state
comfy_progress = 0
comfy_max_steps = 0
DEFAULT_WIDTH = 340
DEFAULT_HEIGHT = 512

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure static/spaceexplorer exists
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "spaceexplorer2")
STATE_FILE = os.path.join(OUTPUT_DIR, "spaceexplorer2.json")
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "spaceexplorer2.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# GLOBAL STATE
# ============================================================
state_lock = Lock()
render_lock = Lock()

running = False
paused = False
current_frame = 0
frames_current = 300
current_seed = 12345

model_name = "dreamshaper_8LCM.safetensors"
vae_name = "vae-ft-mse-840000-ema-pruned.safetensors"
lora1_name = "more_details.safetensors"
lora2_name = "None"
lora3_name = "None"
lora1_strength = 0.5
lora2_strength = 0.5
lora3_strength = 0.5

use_visual_director = False
visual_director_interval = 25
visual_director_model = "LlaVa:latest"
latest_vision_desc = "No vision data yet."

use_prompt_interpolation = True
use_video_interpolation = False

denoise_current = 0.35
teleport_image = None
active_caption = ""
caption_remaining = 0

# Caption custom style variables
caption_font_size = 12
caption_x = 10
caption_y = 10
caption_bg_r = 61
caption_bg_g = 81
caption_bg_b = 92
caption_bg_a = 0.4

temp_caption_font_size = 20
temp_caption_x = 20
temp_caption_y = 20
temp_caption_bg_r = 0
temp_caption_bg_g = 0
temp_caption_bg_b = 0
temp_caption_bg_a = 0.5
active_caption_font = "Default"
active_caption_font_size = 20
logo_filename = "None"
logo_x = 0
logo_y = 0
logo_w = 100
logo_h = 100
logo_opacity = 1.0

# AI Seed Guide variables
use_seed_guide = False
seed_guide_filename = "None"
seed_guide_x = 0
seed_guide_y = 0
seed_guide_w = 100
seed_guide_h = 100
seed_guide_opacity = 0.1
feedback_color_boost = 1.03
feedback_contrast_boost = 1.01
feedback_sharpness_boost = 1.10
current_prompt = "Highly detailed Centered Science fiction image of a star-gate with semi transparent space creatures swimming in space similar to mythical sea monsters, surrounded with space, stars, planets, nebula, dust and space debris <lora:more_details:.8>"
original_starting_prompt = current_prompt
rendering_prompt = "No active prompt yet."

def logit(*args):
    try:
        msg = " ".join(map(str, args))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE_PATH, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except: pass
logit("SpaceShortExplorer2.py")    

# ============================================================
# LOCAL MODEL PATHS AND PIPELINE CACHING
# ============================================================
#CHECKPOINTS_DIR = "/home/jack/Desktop/Comfy-UI/models/checkpoints"
#LORAS_DIR = "/home/jack/Desktop/Comfy-UI/models/loras"
#VAES_DIR = "/home/jack/Desktop/Comfy-UI/models/vae"
CHECKPOINTS_DIR = "/media/jack/9930-14E11/ComfyUI/models/checkpoints"
LORAS_DIR = "/media/jack/9930-14E11/ComfyUI/models/loras"
VAES_DIR = "/media/jack/9930-14E11/ComfyUI/models/vae"
def get_available_checkpoints():
    if not os.path.exists(CHECKPOINTS_DIR):
        return []
    files = []
    for root, _, filenames in os.walk(CHECKPOINTS_DIR):
        for f in filenames:
            if f.lower().endswith((".safetensors", ".ckpt")):
                rel = os.path.relpath(os.path.join(root, f), CHECKPOINTS_DIR)
                files.append(rel)
    return sorted(files)

def get_available_loras():
    if not os.path.exists(LORAS_DIR):
        return ["None"]
    files = []
    for root, _, filenames in os.walk(LORAS_DIR):
        for f in filenames:
            if f.lower().endswith((".safetensors", ".ckpt")):
                rel = os.path.relpath(os.path.join(root, f), LORAS_DIR)
                files.append(rel)
    return ["None"] + sorted(files)

def get_available_vaes():
    if not os.path.exists(VAES_DIR):
        return ["None"]
    files = []
    for root, _, filenames in os.walk(VAES_DIR):
        for f in filenames:
            if f.lower().endswith((".safetensors", ".ckpt")):
                rel = os.path.relpath(os.path.join(root, f), VAES_DIR)
                files.append(rel)
    return ["None"] + sorted(files)

_cached_pipe = None
_cached_model = None
_cached_vae = None
_cached_loras = None # list of tuples: (lora_path, scale)

def load_or_update_pipeline(model_name, vae_name, loras):
    global _cached_pipe, _cached_model, _cached_vae, _cached_loras
    
    model_path = os.path.join(CHECKPOINTS_DIR, model_name) if not os.path.isabs(model_name) else model_name
    vae_path = os.path.join(VAES_DIR, vae_name) if (vae_name and vae_name != "None" and not os.path.isabs(vae_name)) else None
    
    needs_new_base = (_cached_pipe is None or _cached_model != model_path)
    
    if needs_new_base:
        logit(f"Loading new base model: {model_path}")
        _cached_pipe = None
        import gc
        gc.collect()
        
        _cached_pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        _cached_pipe = _cached_pipe.to("cpu")
        from diffusers import EulerAncestralDiscreteScheduler
        _cached_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(_cached_pipe.scheduler.config)
        _cached_pipe.enable_attention_slicing()
        _cached_pipe.enable_vae_slicing()
        _cached_model = model_path
        _cached_vae = None
        _cached_loras = None
    
    if vae_path and _cached_vae != vae_path:
        logit(f"Loading external VAE: {vae_path}")
        _cached_pipe.vae = _cached_pipe.vae.from_single_file(vae_path).to("cpu")
        _cached_vae = vae_path
        
    active_loras = []
    for l in loras:
        if l["path"] and l["path"] != "None" and l["strength"] > 0:
            l_path = os.path.join(LORAS_DIR, l["path"]) if not os.path.isabs(l["path"]) else l["path"]
            active_loras.append((l_path, l["strength"]))
            
    if _cached_loras != active_loras:
        logit("Updating LoRAs...")
        try:
            _cached_pipe.unload_lora_weights()
        except Exception as e:
            logit(f"Unloading LoRAs failed or not needed: {e}")
            
        if active_loras:
            try:
                for idx, (l_path, strength) in enumerate(active_loras):
                    adapter_name = f"lora_{idx}"
                    _cached_pipe.load_lora_weights(l_path, adapter_name=adapter_name)
                
                adapter_names = [f"lora_{idx}" for idx in range(len(active_loras))]
                adapter_weights = [strength for _, strength in active_loras]
                _cached_pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                logit(f"Loaded {len(active_loras)} LoRAs using set_adapters: {active_loras}")
            except Exception as e:
                logit(f"Failed loading multi-LoRAs with set_adapters: {e}. Falling back to fused loading.")
                try:
                    for l_path, strength in active_loras:
                        _cached_pipe.load_lora_weights(l_path)
                        _cached_pipe.fuse_lora(lora_scale=strength)
                except Exception as e2:
                    logit(f"Fallback LoRA load failed: {e2}")
        _cached_loras = active_loras

    return _cached_pipe

tk = len(current_prompt.split(" "))
logit(f"Len Prompt: {tk}")
logit(f"Prompt: {current_prompt}")
negative_prompt = "bad anatomy, extra limbs, blurry, low quality, nsfw, nudity, breasts, nipples, watermark, text, deformed, mutated hands, worst quality, pure black background"

injection_lines = []
MAX_LINES = 5
keyframes = {
    "0": {
        "prompt": "A majestic spacecraft 'Explorer1' from a futuristic launchpad on Earth, giant plumes of fire and smoke, cinematic lighting, photorealistic, 8k resolution, dramatic clouds in the atmosphere",
        "denoise": 1.0,
        "seed_offset": 0
    },
    "50": {
        "prompt": "A majestic spacecraft 'Explorer1' in space and orbiting the blue Earth, sun glinting off solar panels, deep black space, stars, realistic reflections, high detail",
        "denoise": 0.35,
        "seed_offset": 5
    },
    "100": {
        "prompt": "A majestic spacecraft 'Explorer1' cruising through a vast starfield, nebula gas clouds glowing blue and purple, distant galaxies, cosmic scale, cinematic sci-fi, the background is NASA style deep space",
        "denoise": 0.32,
        "seed_offset": 12
    },
    "135": {
    "prompt": "A majestic spacecraft 'Explorer1' cruising through a vast starfield, nebula gas clouds glowing blue and purple, distant galaxies, cosmic scale, cinematic sci-fi, cruising through a vast starfield, approaching an asteroid",
    "denoise": 0.32,
    "seed_offset": 14
    },
    "155": {
    "prompt": "A majestic spacecraft 'Explorer1' landing on an asteroid.",
    "denoise": 0.32,
    "seed_offset": 19
    },
    "200": {
    "prompt": "A majestic spacecraft 'Explorer1' launching from an asteroid, towards deep space, nebula gas clouds glowing blue and purple, distant galaxies, cosmic scale, cinematic sci-fi,",
    "denoise": 0.32,
    "seed_offset": 26
    },    
    "250": {
    "prompt": "A majestic spacecraft 'Explorer1' cruising through a vast starfield, approaching Earth",
    "denoise": 0.32,
    "seed_offset": 31
    }        
}

# Motion Zoom Params
use_motion_zoom = True
use_metadata_caption = False
zoom_start = 1.0
zoom_end = 1.01
pan_start_x = 0.5
pan_end_x = 0.5
pan_start_y = 0.5
pan_end_y = 0.5
zoom_mode = "in" # "in" or "out"
zoom_shrink = 10
zoom_blur = 8
zoom_strength = 0.28
zoom_guidance_scale = 1.4
zoom_num_inference_steps = 12

roll_mode = "none" # "none", "right", "left"

default_steps = 12
default_cfg = 1.4

# ============================================================
# NEW: SPACESHIP MOVEMENT FUNCTION
# ============================================================
def move_spaceship(img, frame_idx, w, h, spaceship_path="static/blank.png"):
    """
    Spaceship crosses right → left, loops, with subtle organic motion
    """
    if not os.path.exists(spaceship_path):
        try:
            os.makedirs(os.path.dirname(spaceship_path), exist_ok=True)
            ship_temp = Image.new("RGBA", (48, 24), (0, 0, 0, 0))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(ship_temp)
            '''
            draw.polygon([(48, 12), (10, 0), (0, 12), (10, 24)], fill=(255, 60, 60, 255))
            draw.ellipse([8, 8, 20, 16], fill=(100, 200, 255, 255))            
            '''
            draw.polygon([(48, 12), (10, 0), (0, 12), (10, 24)], fill=(255, 60, 60, 0))
            draw.ellipse([8, 8, 20, 16], fill=(100, 200, 255, 0))
            ship_temp.save(spaceship_path)
            logit(f"Generated placeholder spaceship at {spaceship_path}")
        except Exception as e:
            logit(f"Failed to create spaceship placeholder: {e}")
            return img

    try:
        ship = Image.open(spaceship_path).convert("RGBA")
        ship_w, ship_h = ship.size

        # === horizontal movement (same as yours, but smoother float)
        speed = 2.0
        cycle_len = w + ship_w
        offset = (frame_idx * speed) % cycle_len
        ship_x = w - offset

        # === vertical drift (this is the magic)
        drift = int(20 * np.sin(frame_idx * 0.05))
        ship_y = (h // 2 - ship_h // 2) + drift

        # === subtle scale change (depth illusion)
        scale = 1.0 + 0.05 * np.sin(frame_idx * 0.03)
        new_w = int(ship_w * scale)
        new_h = int(ship_h * scale)
        ship_resized = ship.resize((new_w, new_h), Image.LANCZOS)

        # adjust position after scaling
        ship_x_adj = int(ship_x)
        ship_y_adj = int(ship_y)

        # === composite
        ship_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ship_layer.paste(ship_resized, (ship_x_adj, ship_y_adj), ship_resized)

        return Image.alpha_composite(img, ship_layer)

    except Exception as e:
        logit(f"Spaceship overlay error: {e}")
        return img

# ============================================================
# NEW: MOVIE CREATION FUNCTION
# ============================================================
def create_movie_from_frames(output_filename="production_August1.mp4"):
    """
    Joins the generated images into a movie file using ffmpeg.
    """
    logit("Joining images to create movie...")
    try:
        import subprocess
        # Search for frames and compile
        if use_video_interpolation:
            cmd = [
                "ffmpeg", "-y", "-framerate", "5", 
                "-i", os.path.join(OUTPUT_DIR, "frame_%03d.png"),
                "-vf", "minterpolate=fps=24:mi_mode=mci:mc_me=epzs:me_mode=bidir",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", 
                os.path.join(OUTPUT_DIR, output_filename)
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-framerate", "5", 
                "-i", os.path.join(OUTPUT_DIR, "frame_%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", 
                os.path.join(OUTPUT_DIR, output_filename)
            ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logit(f"Movie successfully created: {output_filename}")
            return True
        else:
            logit(f"FFmpeg error: {result.stderr}")
            return False
    except Exception as e:
        logit(f"Movie creation failed: {e}")
        return False

# ============================================================
# ZOOM & FINAL RENDER FUNCTION (UPDATED)
# ============================================================
def apply_pil_zoom(img, frame_idx, total_frames, overlay_png_path=None, overlay_opacity=0.10, spaceship_path="static/spaceship.png"):
    w, h = img.size
    img = img.convert("RGBA")

    # 1. Apply fullscreen overlay if requested
    if overlay_png_path and os.path.exists(overlay_png_path):
        overlay = Image.open(overlay_png_path).convert("RGBA")
        if overlay.size != img.size:
            overlay = overlay.resize((w, h), Image.LANCZOS)
        alpha = overlay.getchannel("A")
        alpha = alpha.point(lambda p: int(p * overlay_opacity))
        overlay.putalpha(alpha)
        img = Image.alpha_composite(img, overlay)

    # 2. CALL THE NEW SPACESHIP FUNCTION
    img = move_spaceship(img, frame_idx, w, h, spaceship_path=spaceship_path)

    # 3. Motion Zoom Logic
    # Disabled digital crop-zooming to avoid conflicting with the recursive zoom loop
    zoom_data = (1.0, 0.5, 0.5)
    
    # 4. Apply Roll (Rotation)
    if roll_mode != "none":
        # Tiny increment per frame: 0.005 degrees (Slowed for cinematic weight)
        # Right roll = clockwise (negative), Left roll = counter-clockwise (positive)
        direction = -1 if roll_mode == "right" else 1
        angle = frame_idx * 0.005 * direction
        # resample=Image.BICUBIC for high quality, expand=False to keep same size
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

    return img.convert("RGB"), zoom_data

def apply_logo_to_image(img, filename, lx, ly, lw, lh, l_opacity):
    if filename and filename != "None":
        logo_path = os.path.join(BASE_DIR, "static", "overlays", filename)
        if os.path.exists(logo_path):
            try:
                logo_img = Image.open(logo_path).convert("RGBA")
                if lw > 0 and lh > 0:
                    logo_img = logo_img.resize((lw, lh), Image.LANCZOS)
                
                if l_opacity < 1.0:
                    alpha = logo_img.getchannel("A")
                    alpha = alpha.point(lambda p: int(p * l_opacity))
                    logo_img.putalpha(alpha)
                
                w, h = img.size
                logo_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                logo_layer.paste(logo_img, (lx, ly), logo_img)
                orig_mode = img.mode
                img = Image.alpha_composite(img.convert("RGBA"), logo_layer)
                if orig_mode != "RGBA":
                    img = img.convert(orig_mode)
            except Exception as le:
                logit(f"Custom logo composition error: {le}")
    return img

def apply_custom_logo(img):
    global logo_filename, logo_x, logo_y, logo_w, logo_h, logo_opacity
    return apply_logo_to_image(img, logo_filename, logo_x, logo_y, logo_w, logo_h, logo_opacity)

def apply_seed_guide(img):
    global seed_guide_filename, seed_guide_x, seed_guide_y, seed_guide_w, seed_guide_h, seed_guide_opacity, use_seed_guide
    if use_seed_guide and seed_guide_filename and seed_guide_filename != "None":
        return apply_logo_to_image(img, seed_guide_filename, seed_guide_x, seed_guide_y, seed_guide_w, seed_guide_h, seed_guide_opacity)
    return img

def draw_metadata_caption(img, frame_idx, total_frames, metadata, curr_zoom, curr_pan_x, curr_pan_y):
    try:
        from PIL import ImageDraw, ImageFont
        # Create an RGBA version of the image to support transparency in drawing
        img_rgba = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        lines = [
            f"Frame: {frame_idx} of {total_frames} - Seed: {metadata.get('seed')} - Step: {metadata.get('steps')} - CFG: {metadata.get('cfg')}",
            f"Denoise: {metadata.get('denoise'):.2f} - Zoom: {curr_zoom:.3f} - Yaw: {curr_pan_x:.2f} - Pitch: {curr_pan_y:.2f}"
        ]
        text = "\n".join(lines)
        
        try:
            font = ImageFont.load_default(size=caption_font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate bounding box of the multiline text
        text_x = caption_x + 10
        text_y = caption_y + 6
        left, top, right, bottom = draw.multiline_textbbox((text_x, text_y), text, font=font)
        
        # Create box with padding
        box_left = left - 10
        box_top = top - 6
        box_right = right + 10
        box_bottom = bottom + 6
        
        # Convert opacity from 0.0-1.0 to 0-255
        alpha_val = int(caption_bg_a * 255)
        bg_color = (caption_bg_r, caption_bg_g, caption_bg_b, alpha_val)
        
        draw.rectangle(
            [box_left, box_top, box_right, box_bottom],
            fill=bg_color
        )
        draw.multiline_text(
            (text_x, text_y),
            text,
            fill=(255, 255, 255, 255),
            font=font
        )
        img_rgba = Image.alpha_composite(img_rgba, overlay)
        return img_rgba.convert("RGB")
    except Exception as e:
        logit(f"Caption error: {e}")
        return img
def apply_border(img, border_path="Border.png"):
    if border_path is None:
        border_path = random.choice([
            "static/border_dirty.png",
            "static/border_dirty1.png",
            "static/border_dirty2.png",
            "static/border_dirty3.png"
        ])
    """
    Overlays a frame/border on the image. This is only for local storage.
    """
    if not os.path.exists(border_path):
        return img
    try:
        border = Image.open(border_path).convert("RGBA")
        # Resize border to match image if necessary
        if border.size != img.size:
            border = border.resize(img.size, Image.LANCZOS)
        
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, border)
        return img.convert("RGB")
    except Exception as e:
        logit(f"Border overlay error: {e}")
        return img

def get_available_fonts():
    """
    Scans the fonts subdirectory recursively for TTF/OTF fonts.
    Returns a list of relative paths from the fonts directory.
    """
    fonts_dir = os.path.join(BASE_DIR, "fonts")
    font_files = []
    if os.path.exists(fonts_dir):
        for root, dirs, files in os.walk(fonts_dir):
            for file in files:
                if file.lower().endswith((".ttf", ".otf")):
                    rel_path = os.path.relpath(os.path.join(root, file), fonts_dir)
                    font_files.append(rel_path)
    font_files.sort()
    return font_files

def draw_top_caption(img, text, font_name="Default", font_size=20):
    """
    Draws text with customized font name, size, position, and background color.
    """
    if not text:
        return img
    try:
        from PIL import ImageDraw, ImageFont
        # Create an RGBA version of the image to support transparency in drawing
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Load font if not Default
        font = None
        if font_name and font_name != "Default":
            font_path = os.path.join(BASE_DIR, "fonts", font_name)
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, temp_caption_font_size)
                except Exception as fe:
                    logit(f"Failed to load font {font_path}: {fe}")
        
        if font is None:
            try:
                font = ImageFont.load_default(size=temp_caption_font_size)
            except:
                font = ImageFont.load_default()
        
        text_x = temp_caption_x
        text_y = temp_caption_y
        
        # Calculate bounding box of the text to draw the background box
        left, top, right, bottom = draw.multiline_textbbox((text_x, text_y), text, font=font)
        
        # Add padding to background box
        box_left = left - 10
        box_top = top - 6
        box_right = right + 10
        box_bottom = bottom + 6
        
        alpha_val = int(temp_caption_bg_a * 255)
        bg_color = (temp_caption_bg_r, temp_caption_bg_g, temp_caption_bg_b, alpha_val)
        
        draw.rectangle([box_left, box_top, box_right, box_bottom], fill=bg_color)
        draw.multiline_text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
        
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay)
        return img.convert("RGB")
    except Exception as e:
        logit(f"Top caption error: {e}")
        return img

# ============================================================
# STATE
# ============================================================
def save_state():
    with state_lock:
        state = {
            "current_frame": current_frame,
            "frames_total": frames_current,
            "seed": current_seed,
            "prompt": current_prompt,
            "original_starting_prompt": original_starting_prompt,
            "negative_prompt": negative_prompt,
            "model": model_name,
            "lora1": lora1_name,
            "lora2": lora2_name,
            "lora3": lora3_name,
            "denoise": denoise_current,
            "keyframes": keyframes,
            "injection_lines": injection_lines,
            "use_motion_zoom": use_motion_zoom,
            "use_metadata_caption": use_metadata_caption,
            "z_s": zoom_start,
            "z_e": zoom_end,
            "px_s": pan_start_x,
            "px_e": pan_end_x,
            "py_s": pan_start_y,
            "py_e": pan_end_y,
            "roll_mode": roll_mode,
            "steps": default_steps,
            "cfg": default_cfg,
            "logo_filename": logo_filename,
            "logo_x": logo_x,
            "logo_y": logo_y,
            "logo_w": logo_w,
            "logo_h": logo_h,
            "logo_opacity": logo_opacity,
            "feedback_color_boost": feedback_color_boost,
            "feedback_contrast_boost": feedback_contrast_boost,
            "feedback_sharpness_boost": feedback_sharpness_boost,
            "lora1_strength": lora1_strength,
            "lora2_strength": lora2_strength,
            "lora3_strength": lora3_strength,
            "use_visual_director": use_visual_director,
            "visual_director_interval": visual_director_interval,
            "visual_director_model": visual_director_model,
            "use_prompt_interpolation": use_prompt_interpolation,
            "use_video_interpolation": use_video_interpolation,
            "caption_font_size": caption_font_size,
            "caption_x": caption_x,
            "caption_y": caption_y,
            "caption_bg_r": caption_bg_r,
            "caption_bg_g": caption_bg_g,
            "caption_bg_b": caption_bg_b,
            "caption_bg_a": caption_bg_a,
            "temp_caption_font_size": temp_caption_font_size,
            "temp_caption_x": temp_caption_x,
            "temp_caption_y": temp_caption_y,
            "temp_caption_bg_r": temp_caption_bg_r,
            "temp_caption_bg_g": temp_caption_bg_g,
            "temp_caption_bg_b": temp_caption_bg_b,
            "temp_caption_bg_a": temp_caption_bg_a,
            "zoom_mode": zoom_mode,
            "zoom_shrink": zoom_shrink,
            "zoom_blur": zoom_blur,
            "zoom_strength": zoom_strength,
            "zoom_guidance_scale": zoom_guidance_scale,
            "zoom_num_inference_steps": zoom_num_inference_steps,
            "use_seed_guide": use_seed_guide,
            "seed_guide_filename": seed_guide_filename,
            "seed_guide_x": seed_guide_x,
            "seed_guide_y": seed_guide_y,
            "seed_guide_w": seed_guide_w,
            "seed_guide_h": seed_guide_h,
            "seed_guide_opacity": seed_guide_opacity
        }
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except:
            pass

def load_state():
    global current_frame, current_seed, current_prompt, negative_prompt, model_name, denoise_current
    global keyframes, injection_lines, frames_current, lora1_name, lora2_name, lora3_name
    global use_motion_zoom, zoom_start, zoom_end, pan_start_x, pan_end_x, pan_start_y, pan_end_y
    global default_steps, default_cfg, use_metadata_caption
    global logo_filename, logo_x, logo_y, logo_w, logo_h, logo_opacity
    global caption_font_size, caption_x, caption_y, caption_bg_r, caption_bg_g, caption_bg_b, caption_bg_a
    global temp_caption_font_size, temp_caption_x, temp_caption_y, temp_caption_bg_r, temp_caption_bg_g, temp_caption_bg_b, temp_caption_bg_a
    global original_starting_prompt
    global feedback_color_boost, feedback_contrast_boost, feedback_sharpness_boost
    global lora1_strength, lora2_strength, lora3_strength
    global use_visual_director, visual_director_interval, visual_director_model
    global use_prompt_interpolation, use_video_interpolation, roll_mode
    global zoom_mode, zoom_shrink, zoom_blur, zoom_strength, zoom_guidance_scale, zoom_num_inference_steps
    global use_seed_guide, seed_guide_filename, seed_guide_x, seed_guide_y, seed_guide_w, seed_guide_h, seed_guide_opacity

    if not os.path.exists(STATE_FILE):
        return False

    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)

        with state_lock:
            current_frame = state.get("current_frame", 0)
            frames_current = state.get("frames_total", 500)
            current_seed = state.get("seed", 999)
            current_prompt = state.get("prompt", "")
            original_starting_prompt = state.get("original_starting_prompt", current_prompt)
            negative_prompt = state.get("negative_prompt", "")
            model_name = state.get("model", "")
            lora1_name = state.get("lora1", "None")
            lora2_name = state.get("lora2", "None")
            lora3_name = state.get("lora3", "None")
            denoise_current = state.get("denoise", 0.35)
            default_steps = int(state.get("steps", 12))
            default_cfg = float(state.get("cfg", 5.5))

            keyframes = state.get("keyframes", {})
            injection_lines = state.get("injection_lines", [])

            use_motion_zoom = state.get("use_motion_zoom", True)
            use_metadata_caption = state.get("use_metadata_caption", False)
            zoom_start = state.get("z_s", 1.0)
            logo_filename = state.get("logo_filename", "None")
            logo_x = int(state.get("logo_x", 0))
            logo_y = int(state.get("logo_y", 0))
            logo_w = int(state.get("logo_w", 100))
            logo_h = int(state.get("logo_h", 100))
            logo_opacity = float(state.get("logo_opacity", 1.0))
            feedback_color_boost = float(state.get("feedback_color_boost", 1.03))
            feedback_contrast_boost = float(state.get("feedback_contrast_boost", 1.01))
            feedback_sharpness_boost = float(state.get("feedback_sharpness_boost", 1.10))
            lora1_strength = float(state.get("lora1_strength", 0.8))
            lora2_strength = float(state.get("lora2_strength", 0.8))
            lora3_strength = float(state.get("lora3_strength", 0.8))
            use_visual_director = bool(state.get("use_visual_director", False))
            visual_director_interval = int(state.get("visual_director_interval", 25))
            visual_director_model = state.get("visual_director_model", "moondream")
            use_prompt_interpolation = bool(state.get("use_prompt_interpolation", False))
            use_video_interpolation = bool(state.get("use_video_interpolation", False))
            zoom_end = state.get("z_e", 1.1)
            pan_start_x = state.get("px_s", 0.5)
            pan_end_x = state.get("px_e", 0.5)
            pan_start_y = state.get("py_s", 0.5)
            pan_end_y = state.get("py_e", 0.5)
            roll_mode = state.get("roll_mode", "none")
            caption_font_size = int(state.get("caption_font_size", 12))
            caption_x = int(state.get("caption_x", 10))
            caption_y = int(state.get("caption_y", 10))
            caption_bg_r = int(state.get("caption_bg_r", 61))
            caption_bg_g = int(state.get("caption_bg_g", 81))
            caption_bg_b = int(state.get("caption_bg_b", 92))
            caption_bg_a = float(state.get("caption_bg_a", 0.4))
            temp_caption_font_size = int(state.get("temp_caption_font_size", 20))
            temp_caption_x = int(state.get("temp_caption_x", 20))
            temp_caption_y = int(state.get("temp_caption_y", 20))
            temp_caption_bg_r = int(state.get("temp_caption_bg_r", 0))
            temp_caption_bg_g = int(state.get("temp_caption_bg_g", 0))
            temp_caption_bg_b = int(state.get("temp_caption_bg_b", 0))
            temp_caption_bg_a = float(state.get("temp_caption_bg_a", 0.5))
            zoom_mode = state.get("zoom_mode", "in")
            zoom_shrink = int(state.get("zoom_shrink", 10))
            zoom_blur = int(state.get("zoom_blur", 8))
            zoom_strength = float(state.get("zoom_strength", 0.28))
            zoom_guidance_scale = float(state.get("zoom_guidance_scale", 7.0))
            zoom_num_inference_steps = default_steps
            use_seed_guide = bool(state.get("use_seed_guide", False))
            seed_guide_filename = state.get("seed_guide_filename", "None")
            seed_guide_x = int(state.get("seed_guide_x", 0))
            seed_guide_y = int(state.get("seed_guide_y", 0))
            seed_guide_w = int(state.get("seed_guide_w", 100))
            seed_guide_h = int(state.get("seed_guide_h", 100))
            seed_guide_opacity = float(state.get("seed_guide_opacity", 0.1))

        # CRITICAL: Detect actual last frame on disk
        files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("frame_") and f.endswith(".png")])
        if files:
            last_file = files[-1]
            try:
                # Extract number from frame_XXX.png
                last_disk_frame = int(last_file.split("_")[1].split(".")[0])
                current_frame = last_disk_frame + 1
                logit(f"Resume: Detected frame {last_disk_frame} on disk. Starting at {current_frame}")
            except:
                pass

        if current_frame >= frames_current:
            logit(f"Session reached limit ({current_frame}/{frames_current}). Increase 'Frames' to continue.")

        return True

    except Exception as e:
        logit("Load state error:", e)
        return False

# ============================================================
# RENDER LOOP
# ============================================================
# local diffusers pipeline is used instead of ComfyUI workflow JSON

def parse_prompt(prompt_text):
    import re
    loras = re.findall(r"<lora:[^>]+>", prompt_text)
    clean_prompt = re.sub(r"<lora:[^>]+>", "", prompt_text).strip()
    clean_prompt = re.sub(r"\s+", " ", clean_prompt)
    clean_prompt = clean_prompt.strip(", ")
    return clean_prompt, loras

def clean_desc(desc_text):
    if not desc_text:
        return ""
    desc_text = desc_text.strip()
    
    # Strip quotes
    while True:
        stripped = False
        for q in ['"', "'", '`']:
            if desc_text.startswith(q) and desc_text.endswith(q) and len(desc_text) > 1:
                desc_text = desc_text[1:-1].strip()
                stripped = True
        if not stripped:
            break
            
    # Strip trailing period
    if desc_text.endswith('.'):
        desc_text = desc_text[:-1].strip()
        
    # Strip common visual model prefixes
    lower_desc = desc_text.lower()
    prefixes = [
        "i see a ", "i see ", "this is an image of ", "this is a picture of ", 
        "this is a ", "this is ", "the image shows ", "the image depicts ", 
        "the picture shows ", "there is a ", "there are ", "shows a ", "depicts a "
    ]
    for prefix in prefixes:
        if lower_desc.startswith(prefix):
            desc_text = desc_text[len(prefix):].strip()
            if desc_text:
                desc_text = desc_text[0].upper() + desc_text[1:]
            break
            
    return desc_text

def query_llava(image_path, system_instruction):
    """
    Queries local Ollama LlaVa/Moondream model with base64 encoded image.
    """
    import gc
    gc.collect()  # Force Python to release unused memory
    import base64
    try:
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        
        payload = {
            "model": "LlaVa:latest",
            "prompt": system_instruction,
            "images": [img_data],
            "stream": False,
            "options": {
                "temperature": 0.5,
                "max_tokens": 120
            }
        }
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=1000)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        else:
            logit(f"Ollama query returned status code {r.status_code}: {r.text}")
    except Exception as e:
        logit(f"LlaVa query failed: {e}")
    return None

def get_interpolated_prompt(frame_idx, base_prompt):
    """
    Finds the surrounding keyframes and returns a weighted blend of their prompts.
    """
    if not keyframes:
        return base_prompt
        
    kf_nums = sorted([int(k) for k in keyframes.keys()])
    if not kf_nums:
        return base_prompt
        
    if frame_idx in kf_nums:
        return keyframes[str(frame_idx)].get("prompt", base_prompt)
        
    prev_kf = None
    next_kf = None
    for k in kf_nums:
        if k < frame_idx:
            prev_kf = k
        elif k > frame_idx and next_kf is None:
            next_kf = k
            break
            
    if prev_kf is None and next_kf is None:
        return base_prompt
    elif prev_kf is None:
        return keyframes[str(next_kf)].get("prompt", base_prompt)
    elif next_kf is None:
        return keyframes[str(prev_kf)].get("prompt", base_prompt)
        
    p_prev = keyframes[str(prev_kf)].get("prompt", base_prompt)
    p_next = keyframes[str(next_kf)].get("prompt", base_prompt)
    
    if p_prev == p_next:
        return p_prev
        
    total_dist = next_kf - prev_kf
    weight_next = (frame_idx - prev_kf) / total_dist
    weight_prev = 1.0 - weight_next
    
    p_prev_clean = p_prev.replace("(", "").replace(")", "")
    p_next_clean = p_next.replace("(", "").replace(")", "")
    
    return f"({p_prev_clean}:{weight_prev:.2f}), ({p_next_clean}:{weight_next:.2f})"

def get_interpolated_denoise(frame_idx, default_d):
    """
    Finds the surrounding keyframes and returns a weighted blend of their denoise values.
    """
    if not keyframes:
        return default_d
        
    kf_nums = sorted([int(k) for k in keyframes.keys()])
    if not kf_nums:
        return default_d
        
    if frame_idx in kf_nums:
        return float(keyframes[str(frame_idx)].get("denoise", default_d))
        
    prev_kf = None
    next_kf = None
    for k in kf_nums:
        if k < frame_idx:
            prev_kf = k
        elif k > frame_idx and next_kf is None:
            next_kf = k
            break
            
    if prev_kf is None and next_kf is None:
        return default_d
    elif prev_kf is None:
        return float(keyframes[str(next_kf)].get("denoise", default_d))
    elif next_kf is None:
        return float(keyframes[str(prev_kf)].get("denoise", default_d))
        
    d_prev = float(keyframes[str(prev_kf)].get("denoise", default_d))
    d_next = float(keyframes[str(next_kf)].get("denoise", default_d))
    
    total_dist = next_kf - prev_kf
    weight_next = (frame_idx - prev_kf) / total_dist
    weight_prev = 1.0 - weight_next
    
    return d_prev * weight_prev + d_next * weight_next


def encode_prompt_weighted(pipe, prompt, negative_prompt):
    import re
    # Check if there are matches for weighted prompts like (some text:0.60)
    pattern = r"\(([^:]+):([0-9.]+)\)"
    matches = re.findall(pattern, prompt)
    
    if not matches:
        # Standard encoding using the pipeline's built-in encode_prompt
        res = pipe.encode_prompt(
            prompt=prompt,
            device="cpu",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt
        )
        return res[0], res[1]
    
    # Interpolated prompts
    blended_embeds = None
    total_weight = 0.0
    
    # Encode negative prompt once to get negative_prompt_embeds
    res_neg = pipe.encode_prompt(
        prompt="",
        device="cpu",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt
    )
    negative_prompt_embeds = res_neg[1]
    
    for sub_prompt, weight_str in matches:
        weight = float(weight_str)
        sub_prompt = sub_prompt.strip()
        
        # Encode sub-prompt
        res_sub = pipe.encode_prompt(
            prompt=sub_prompt,
            device="cpu",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt
        )
        sub_embeds = res_sub[0]
        
        if blended_embeds is None:
            blended_embeds = sub_embeds * weight
        else:
            blended_embeds += sub_embeds * weight
        total_weight += weight
        
    if total_weight > 0:
        blended_embeds = blended_embeds / total_weight
        
    return blended_embeds, negative_prompt_embeds


def render_video(resume=False):
    global running, current_frame, paused, current_seed, teleport_image
    global caption_remaining, active_caption, roll_mode, current_prompt, original_starting_prompt, rendering_prompt
    global zoom_mode, zoom_shrink, zoom_blur, zoom_strength, zoom_guidance_scale, zoom_num_inference_steps, zoom_start, zoom_end, default_steps, default_cfg
    global latest_vision_desc
    global use_seed_guide, seed_guide_filename, seed_guide_x, seed_guide_y, seed_guide_w, seed_guide_h, seed_guide_opacity
    if running: return
    logit("ENGINE STARTED: Entering render loop.")
    
    if resume:
        if not load_state():
            logit("Failed to load state, starting fresh.")
            current_frame = 0
            injection_lines.clear()
    else:
        current_frame = 0
        injection_lines.clear()
        original_starting_prompt = current_prompt
        # Cleanup existing frames from previous runs to prevent frame bleeding
        for f in os.listdir(OUTPUT_DIR):
            if (f.startswith("frame_") and f.endswith(".png")) or (f.startswith("temp_clean_") and f.endswith(".png")) or (f.startswith("clean_") and f.endswith(".png")):
                try:
                    os.remove(os.path.join(OUTPUT_DIR, f))
                except Exception as e:
                    logit(f"Failed to remove old frame {f}: {e}")
    
    running = True

    try:
        while current_frame < frames_current:
            if not running: break
            if paused:
                time.sleep(1)
                continue
            
            # Determine starting/init image path
            init_image_path = None
            if current_frame > 0:
                prev_clean = os.path.join(OUTPUT_DIR, f"clean_{current_frame-1:03d}.png")
                prev_framed = os.path.join(OUTPUT_DIR, f"frame_{current_frame-1:03d}.png")
                init_image_path = prev_clean if os.path.exists(prev_clean) else prev_framed
                
            with state_lock:
                if teleport_image:
                    init_image_path = teleport_image
                    teleport_image = None
                    logit(f"Teleporting! Using external image for frame {current_frame}")

            seed = current_seed + current_frame
            
            if use_prompt_interpolation:
                prompt_base = get_interpolated_prompt(current_frame, current_prompt)
                active_d = get_interpolated_denoise(current_frame, denoise_current)
            else:
                prompt_base = current_prompt
                active_d = denoise_current
                kf_nums = sorted([int(k) for k in keyframes.keys()])
                last_kf = None
                for k in kf_nums:
                    if k <= current_frame:
                        last_kf = k
                if last_kf is not None:
                    active_d = float(keyframes[str(last_kf)].get("denoise", denoise_current))
                
            prompt = prompt_base + (", " + ", ".join(injection_lines[-MAX_LINES:]) if injection_lines else "")
            
            # Calculate active params (for keyframe support)
            active_p, active_s = prompt, seed
            kf = keyframes.get(str(current_frame))
            if kf:
                kf_prompt = kf.get("prompt", active_p)
                active_p = kf_prompt
                active_d = float(kf.get("denoise", active_d))
                active_s = seed + int(kf.get("seed_offset", 0))
                with state_lock:
                    current_prompt = kf_prompt
                    original_starting_prompt = kf_prompt
                logit(f"Keyframe {current_frame} applied. Redirecting Visual Director baseline to: '{kf_prompt}'")

            with state_lock:
                rendering_prompt = active_p

            # Load/update local pipeline
            loras = [
                {"path": lora1_name, "strength": lora1_strength},
                {"path": lora2_name, "strength": lora2_strength},
                {"path": lora3_name, "strength": lora3_strength}
            ]
            try:
                pipe = load_or_update_pipeline(model_name, vae_name, loras)
            except Exception as e:
                logit(f"Failed to load or update pipeline: {e}")
                time.sleep(5)
                continue

            # Calculate generation dimensions (must be multiples of 64 to avoid UNet/VAE shape mismatches)
            gen_width = (DEFAULT_WIDTH // 64) * 64
            gen_height = (DEFAULT_HEIGHT // 64) * 64

            # Load or create starting image
            if init_image_path and os.path.exists(init_image_path):
                try:
                    init_image = Image.open(init_image_path).convert("RGB")
                    init_image = init_image.resize((gen_width, gen_height), Image.LANCZOS)
                    
                    if use_motion_zoom and current_frame > 0:
                        w, h = init_image.size
                        if zoom_mode == "out":
                            from PIL import ImageFilter
                            blurred = init_image.filter(ImageFilter.GaussianBlur(zoom_blur))
                            sw = w - zoom_shrink * 2
                            sh = h - zoom_shrink * 2
                            if sw > 0 and sh > 0:
                                small = init_image.resize((sw, sh), Image.LANCZOS)
                                canvas = blurred.copy()
                                canvas.paste(small, (zoom_shrink, zoom_shrink))
                                pixels = canvas.load()
                                for y in range(h):
                                    for x in range(w):
                                        if zoom_shrink <= x < w - zoom_shrink and zoom_shrink <= y < h - zoom_shrink:
                                            continue
                                        nx = min(w - zoom_shrink - 1, max(zoom_shrink, x))
                                        ny = min(h - zoom_shrink - 1, max(zoom_shrink, y))
                                        pixels[x, y] = pixels[nx, ny]
                                init_image = canvas
                            else:
                                logit("Warning: zoom_shrink too large, skipping shrink preprocessing.")
                        elif zoom_mode == "in":
                            # Crop and expand (physical recursive zoom-in) based on zoom_start and zoom_end configurations
                            try:
                                total_frames_val = max(frames_current - 1, 1)
                                z_start = float(zoom_start) if zoom_start is not None else 1.0
                                z_end = float(zoom_end) if zoom_end is not None else 1.1
                                r = (z_end / z_start) ** (1.0 / total_frames_val) if z_start > 0 else 1.01
                            except Exception as ex:
                                logit(f"Error calculating zoom ratio: {ex}, fallback to default.")
                                r = 1.01
                            
                            cw = w / r
                            ch = h / r
                            if cw > 0 and ch > 0 and (cw < w or ch < h):
                                left = (w - cw) / 2
                                top = (h - ch) / 2
                                right = left + cw
                                bottom = top + ch
                                init_image = init_image.crop((left, top, right, bottom)).resize((w, h), Image.LANCZOS)
                            else:
                                logit("Warning: calculated crop size matches or exceeds image, skipping crop preprocessing.")

                    strength = active_d
                except Exception as e:
                    logit(f"Error loading init image {init_image_path}: {e}")
                    import numpy as np
                    noise_array = np.random.randint(0, 256, (gen_height, gen_width, 3), dtype=np.uint8)
                    init_image = Image.fromarray(noise_array)
                    strength = 1.0
            else:
                import numpy as np
                noise_array = np.random.randint(0, 256, (gen_height, gen_width, 3), dtype=np.uint8)
                init_image = Image.fromarray(noise_array)
                strength = 1.0

            # Determine steps, cfg and strength values to use
            if use_motion_zoom and current_frame > 0:
                kf = keyframes.get(str(current_frame))
                strength_val = float(kf.get("denoise", zoom_strength)) if kf and "denoise" in kf else zoom_strength
                cfg_val = zoom_guidance_scale
                steps_val = default_steps
            else:
                strength_val = strength
                cfg_val = default_cfg
                steps_val = default_steps

            # Define progress callback
            def progress_callback(step, timestep, latents):
                global comfy_progress, comfy_max_steps
                # Calculate actual steps executed by the diffusers img2img pipeline
                actual_steps = max(1, int(steps_val * strength_val)) if strength_val < 1.0 else steps_val
                with state_lock:
                    comfy_progress = step + 1
                    comfy_max_steps = actual_steps

            logit(f"Inference: Frame {current_frame} starting. Prompt: '{active_p}'")
            
            try:
                generator = torch.Generator("cpu").manual_seed(active_s)
                # Encode the prompt, blending embeddings if it is an interpolated keyframe prompt
                prompt_embeds, negative_prompt_embeds = encode_prompt_weighted(pipe, active_p, negative_prompt)
                
                with torch.inference_mode():
                    img = pipe(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        image=init_image,
                        strength=strength_val,
                        guidance_scale=cfg_val,
                        num_inference_steps=steps_val,
                        generator=generator,
                        callback=progress_callback,
                        callback_steps=1
                    ).images[0]
                
                # Resize the generated image back to the project dimensions (340x512)
                if img.size != (DEFAULT_WIDTH, DEFAULT_HEIGHT):
                    img = img.resize((DEFAULT_WIDTH, DEFAULT_HEIGHT), Image.LANCZOS)
                
                # Metadata for caption
                meta = {
                    "seed": active_s,
                    "steps": default_steps,
                    "cfg": default_cfg,
                    "denoise": strength
                }

                # 1. Apply zoom AND spaceship movement (This is the BASE for both server and disk)
                img, zoom_data = apply_pil_zoom(
                    img, 
                    current_frame, 
                    frames_current,
                    overlay_png_path="static/logo.png",
                    overlay_opacity=0.8,
                    spaceship_path="static/spaceship.png"
                )
                
                # 2. Save feedback image locally
                feedback_img = img.copy()
                feedback_img = apply_seed_guide(feedback_img)
                
                if feedback_color_boost != 1.0 or feedback_contrast_boost != 1.0 or feedback_sharpness_boost != 1.0:
                    from PIL import ImageEnhance
                    try:
                        if feedback_color_boost != 1.0:
                            feedback_img = ImageEnhance.Color(feedback_img).enhance(feedback_color_boost)
                        if feedback_contrast_boost != 1.0:
                            feedback_img = ImageEnhance.Contrast(feedback_img).enhance(feedback_contrast_boost)
                        if feedback_sharpness_boost != 1.0:
                            feedback_img = ImageEnhance.Sharpness(feedback_img).enhance(feedback_sharpness_boost)
                    except Exception as ee:
                        logit(f"Feedback loop stabilization error: {ee}")
                
                clean_path = os.path.join(OUTPUT_DIR, f"clean_{current_frame:03d}.png")
                feedback_img.save(clean_path)
                
                # 3. CREATE LOCAL ARCHIVE IMAGE (With Overlays)
                local_img = img.copy()

                if use_metadata_caption:
                    local_img = draw_metadata_caption(local_img, current_frame, frames_current, meta, *zoom_data)
                
                # Apply Border
                local_img = apply_border(local_img)
                
                # Apply Temporary Top Caption
                with state_lock:
                    if caption_remaining > 0:
                        local_img = draw_top_caption(local_img, active_caption, active_caption_font, active_caption_font_size)
                        caption_remaining -= 1
                        if caption_remaining == 0:
                            logit("Temporary caption finished.")

                # Save the clean base frame for dynamic logo overlays
                latest_base_path = os.path.join(OUTPUT_DIR, "latest_base.png")
                local_img.copy().convert("RGBA").save(latest_base_path)
                
                # Also save a per-frame clean base to support robust local watermarking
                per_frame_base = os.path.join(OUTPUT_DIR, f"clean_base_{current_frame:03d}.png")
                local_img.copy().convert("RGBA").save(per_frame_base)

                # Apply the custom logo
                local_img = apply_custom_logo(local_img)

                local_path = os.path.join(OUTPUT_DIR, f"frame_{current_frame:03d}.png")
                local_img.save(local_path)
                
                # Visual Director prompt evolution
                if use_visual_director and (current_frame > 0) and (current_frame % visual_director_interval == 0):
                    logit(f"Visual Director: Evolving prompt based on frame {current_frame}...")
                    system_instruction = "Describe what you see in this image in one brief sentence."
                    img_desc = query_llava(clean_path, system_instruction)
                    if img_desc:
                        cleaned_desc = clean_desc(img_desc)
                        logit(f"Visual Director: Image description -> '{cleaned_desc}'")
                        
                        with state_lock:
                            latest_vision_desc = cleaned_desc
                            current_prompt = f"{parse_prompt(original_starting_prompt)[0]}, {cleaned_desc} {' '.join(list(dict.fromkeys(parse_prompt(original_starting_prompt)[1])))}".strip()
                        logit(f"Visual Director: New base prompt set -> '{current_prompt}'")
                    else:
                        logit("Visual Director: Vision model returned no description, keeping previous prompt.")
                
                # Cleanup
                del local_img # Free memory

                current_frame += 1
                save_state()
            except Exception as e:
                logit(f"Error processing frame {current_frame}: {e}")
                break

    except Exception as e:
        logit(f"Render loop error: {e}")
    finally:
        running = False

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
app.config["OVERLAYS_FOLDER"] = os.path.join(BASE_DIR, "static", "overlays")
os.makedirs(app.config["OVERLAYS_FOLDER"], exist_ok=True)

def get_available_logos():
    logos_dir = os.path.join(BASE_DIR, "static", "overlays")
    if not os.path.exists(logos_dir):
        os.makedirs(logos_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(logos_dir) if f.lower().endswith(".png")])
    return files

def get_available_bgm():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if (f.lower().endswith((".mp3", ".wav")) and not f.startswith("narration_") and not f.startswith("_pad"))])
    return files

@app.route("/")
def index():
    fonts = get_available_fonts()
    logos = get_available_logos()
    bgms = get_available_bgm()
    models = get_available_checkpoints()
    loras = get_available_loras()
    return render_template_string(
        HTML_UI, 
        MODELS=models, 
        LORAS=loras, 
        FONTS=fonts, 
        LOGOS=logos, 
        BGMS=bgms,
        CURRENT_MODEL=model_name if model_name else "None",
        CURRENT_LORA1=lora1_name if lora1_name else "None",
        CURRENT_LORA2=lora2_name if lora2_name else "None",
        CURRENT_LORA3=lora3_name if lora3_name else "None",
        CURRENT_LOGO=logo_filename,
        CURRENT_SEED_GUIDE=seed_guide_filename,
        USE_SEED_GUIDE=use_seed_guide,
        seed_guide_x=seed_guide_x,
        seed_guide_y=seed_guide_y,
        seed_guide_w=seed_guide_w,
        seed_guide_h=seed_guide_h,
        seed_guide_opacity=seed_guide_opacity,
        feedback_color_boost=feedback_color_boost,
        feedback_contrast_boost=feedback_contrast_boost,
        feedback_sharpness_boost=feedback_sharpness_boost,
        caption_font_size=caption_font_size,
        caption_x=caption_x,
        caption_y=caption_y,
        caption_bg_r=caption_bg_r,
        caption_bg_g=caption_bg_g,
        caption_bg_b=caption_bg_b,
        caption_bg_a=caption_bg_a,
        temp_caption_font_size=temp_caption_font_size,
        temp_caption_x=temp_caption_x,
        temp_caption_y=temp_caption_y,
        temp_caption_bg_r=temp_caption_bg_r,
        temp_caption_bg_b=temp_caption_bg_b,
        temp_caption_bg_a=temp_caption_bg_a,
        current_seed=current_seed,
        default_steps=default_steps,
        default_cfg=default_cfg,
        denoise_current=denoise_current,
        frames_current=frames_current,
        zoom_strength=zoom_strength,
        zoom_guidance_scale=zoom_guidance_scale,
        zoom_start=zoom_start,
        zoom_end=zoom_end
    )

# --------------------------------------------------
# JSON KEYFRAME BUILDER ENDPOINTS
# --------------------------------------------------
@app.route("/json_builder")
def json_builder():
    fonts = get_available_fonts()
    logos = get_available_logos()
    bgms = get_available_bgm()
    models = get_available_checkpoints()
    loras = get_available_loras()
    return render_template_string(
        HTML_JSON_BUILDER,
        MODELS=models,
        LORAS=loras,
        FONTS=fonts,
        LOGOS=logos,
        BGMS=bgms
    )

@app.route("/get_config")
def get_config():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    # Fallback to current memory state if file not present
    save_state()
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Config file not found"}), 404

@app.route("/save_config", methods=["POST"])
def save_config():
    try:
        new_state = request.json
        if not new_state:
            return jsonify({"error": "No data provided"}), 400
        
        with open(STATE_FILE, "w") as f:
            json.dump(new_state, f, indent=2)
            
        load_state()
        logit("Configuration updated via JSON Builder.")
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate_keyframes_ollama", methods=["POST"])
def generate_keyframes_ollama():
    try:
        d = request.json
        outline = d.get("outline", "").strip()
        model = d.get("model", "llama3.2").strip()
        model = resolve_model_name(model)
        instructions = d.get("instructions", "").strip()
        
        if not outline:
            return jsonify({"error": "Outline is empty"}), 400
            
        system_prompt = (
            "You are a Stable Diffusion keyframe generator. Your goal is to generate structured keyframes for an AI video animation based on a rough user outline. "
            "For each event in the outline, determine a frame number (e.g. 0, 50, 100) and create a detailed visual prompt (30-50 words) that describes the scene, focusing on visual details, lighting, atmosphere, and textures. "
            "Also choose a denoise value (between 0.3 and 0.85, depending on how much change there is from the previous scene; higher denoise for dramatic changes) and a seed_offset (typically between -10 and 10).\\n\\n"
            "Format your output strictly as a JSON object inside a single markdown code block (using ```json and ```). No conversation, explanations, or filler. The JSON structure must be:\\n"
            "{\\n"
            "  \\\"keyframes\\\": {\\n"
            "    \\\"0\\\": {\\n"
            "      \\\"prompt\\\": \\\"highly detailed visual description...\\\",\\n"
            "      \\\"denoise\\\": 0.35,\\n"
            "      \\\"seed_offset\\\": 0\\n"
            "    },\\n"
            "    \\\"50\\\": {\\n"
            "      \\\"prompt\\\": \\\"highly detailed visual description...\\\",\\n"
            "      \\\"denoise\\\": 0.55,\\n"
            "      \\\"seed_offset\\\": 3\\n"
            "    }\\n"
            "  }\\n"
            "}"
        )
        
        prompt = f"{system_prompt}\\n\\nOutline:\\n{outline}\\n\\nAdditional Instructions:\\n{instructions}\\n\\nJSON output:"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.7,
                "max_tokens": 1500
            }
        }
        
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=2000)
        if r.status_code != 200:
            return jsonify({"error": f"Ollama error: status code {r.status_code}"}), 500
            
        response_text = r.json().get("response", "").strip()
        
        try:
            parsed_json = json.loads(response_text)
        except Exception:
            # Fallback to regex-based extraction if there's any surrounding text
            import re
            json_match = re.search(r'(\{.*?\})', response_text, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(1))
            else:
                raise ValueError(f"Could not find valid JSON in response: {response_text}")
                
        if "keyframes" not in parsed_json:
            parsed_json = {"keyframes": parsed_json}
            
        return jsonify({"status": "ok", "keyframes": parsed_json.get("keyframes", {})})
    except Exception as e:
        return jsonify({"error": f"Failed to generate keyframes: {str(e)}"}), 500

@app.route("/generate_story", methods=["POST"])
def generate_story():
    try:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1000)
            available = [m["name"] for m in r.json().get("models", [])] if r.status_code == 200 else []
        except:
            available = []
            
        #vision_model = "moondream:latest"
        vision_model = "LlaVa:latest"
        if vision_model not in available:
            v_candidates = [m for m in available if "llava" in m or "moondream" in m]
            if v_candidates: vision_model = v_candidates[0]
            
        text_model = "llama3.2:3b"
        if text_model not in available:
            preferred_text = ["llama3.2:3b", "deepseek-r1:1.5b", "dolphin3:8b", "llama2-uncensored:7b"]
            for p in preferred_text:
                if p in available:
                    text_model = p
                    break
            else:
                t_candidates = [m for m in available if any(x in m.lower() for x in ["llama", "dolphin", "deepseek", "qwen", "mistral"])]
                if t_candidates: text_model = t_candidates[0]
            
        files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("clean_base_") and f.endswith(".png")])
        if not files:
            files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("frame_") and f.endswith(".png")])
            
        if not files:
            return jsonify({"error": "No generated frames found in static/spaceexplorer2/ to process."}), 400
            
        cache_path = os.path.join(OUTPUT_DIR, "descriptions_cache.json")
        cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cache = json.load(f)
            except:
                pass
                
        import base64
        import re
        descriptions = []
        for idx, filename in enumerate(files):
            match = re.search(r'_(\d+)\.png$', filename)
            frame_num = int(match.group(1)) if match else idx
            file_path = os.path.join(OUTPUT_DIR, filename)
            
            if filename in cache:
                desc = cache[filename]
            else:
                try:
                    with open(file_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                    payload = {
                        "model": vision_model,
                        "prompt": "Describe this liminal space journey image in one short sentence.",
                        "images": [img_data],
                        "stream": False,
                        "options": {"temperature": 0.4, "max_tokens": 100}
                    }
                    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=6000)
                    if r.status_code == 200:
                        desc = r.json().get("response", "").strip().replace('"', '').replace("'", "")
                        cache[filename] = desc
                        with open(cache_path, "w") as f:
                            json.dump(cache, f, indent=2)
                    else:
                        desc = "An empty liminal corridor."
                except Exception as e:
                    desc = f"An empty liminal corridor. Error: {str(e)}"
            descriptions.append({"frame": frame_num, "file": filename, "description": desc})
            
        # Prepare stateful timeline combining intent (keyframes) and visual descriptions
        desc_str = ""
        max_frame = max(descriptions, key=lambda x: x["frame"])["frame"] if descriptions else 1
        for item in descriptions:
            f = item["frame"]
            
            # 1. Get interpolated prompt (Director Intent)
            intent = get_interpolated_prompt(f, original_starting_prompt)
            clean_intent, _ = parse_prompt(intent)
            
            # 2. Interpolate emotional parameters
            progress = f / max_frame if max_frame > 0 else 0.0
            tension = 1.0 + progress * 8.5
            sanity = 100.0 - progress * 90.0
            
            if tension < 3.0:
                emotion = "Calm and observant, trying to find a way out."
                inflection_hint = "Write in complete, slow, descriptive sentences."
                speed = 0.85
            elif tension < 6.0:
                emotion = "Growing paranoiac, feeling like the environment is shifting."
                inflection_hint = "Use occasional ellipses (...) to simulate hesitation, pausing, and whispering."
                speed = 1.00
            elif tension < 8.0:
                emotion = "Deep dread and anxiety. Suspects something is hunting them."
                inflection_hint = "Use frequent ellipses (...) and shorter, breathy, fragmented sentences."
                speed = 1.10
            else:
                emotion = "Extreme terror, panic, running for survival."
                inflection_hint = "Use very short, fragmented, frantic phrases, repetition, and exclamation marks!"
                speed = 1.25
                
            item["tension"] = tension
            item["sanity"] = sanity
            item["emotion"] = emotion
            item["inflection_hint"] = inflection_hint
            item["speed"] = speed
            item["intent"] = clean_intent
            
            desc_str += (
                f"Frame {f}:\n"
                f"  - Actual Visuals: {item['description']}\n"
                f"  - Director Intent / Story Beat: {clean_intent}\n"
                f"  - Emotional State: {emotion}\n"
                f"  - Speech Inflection Style: {inflection_hint}\n\n"
            )

        if use_saved_script:
            diary_data = []
            story_map = {int(x["frame"]): x.get("story", "") for x in saved_diary}
            for item in descriptions:
                f = item["frame"]
                item["story"] = story_map.get(f, f"Wandering deeper. The visual match points to: {item['description']}.")
                diary_data.append(item)
            
            with open(diary_path, "w") as f:
                json.dump(diary_data, f, indent=2)
            logit("Preserved saved script from preview diary!")
        else:
            prompt = (
                "You are an expert space captain's log narrator. You will write a mission log of a traveler mining platinum in the Kepler asteroid belt.\n\n"
                "Below is a sequence of frames representing the journey. For each frame, you are given:\n"
                "1. Actual Visuals: What is physically visible in the space image frame.\n"
                "2. Director Intent / Story Beat: The thematic concept, maneuver, or mining stage that must happen.\n"
                "3. Emotional State: The captain's current tone and state of awe or efficiency.\n"
                "4. Speech Inflection Style: How the text must be formatted to guide voice inflection.\n\n"
                "Your task is to write a cohesive, continuous first-person space mission log matching this sequence.\n"
                f"You MUST write exactly {len(descriptions)} log entries, one for each frame. "
                "Keep each entry brief (15-30 words).\n\n"
                "Format your output strictly using '[ENTRY]' as a separator before each frame entry, like this:\n"
                "[ENTRY] Entry 0 text...\n"
                "[ENTRY] Entry 1 text...\n\n"
                "Do not output any introductory or concluding text. Output ONLY the log texts with their [ENTRY] separators.\n\n"
                f"Timeline Sequence:\n{desc_str}"
            )
            
            payload = {
                "model": text_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "max_tokens": 2048}
            }
            
            story_raw = ""
            r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=1200)
            if r.status_code == 200:
                story_raw = r.json().get("response", "").strip()
                
            if not story_raw:
                return jsonify({"error": "Ollama story generation failed."}), 500
                
            entries = [e.strip() for e in story_raw.split("[ENTRY]") if e.strip()]
            
            if len(entries) < len(descriptions):
                while len(entries) < len(descriptions):
                    idx = len(entries)
                    entries.append(f"Wandering deeper. The visual match points to: {descriptions[idx]['description']}.")
            elif len(entries) > len(descriptions):
                entries = entries[:len(descriptions)]
                
            diary_data = []
            for idx, item in enumerate(descriptions):
                item["story"] = entries[idx]
                diary_data.append(item)
                
            with open(diary_path, "w") as f:
                json.dump(diary_data, f, indent=2)
            
        voice = "am_adam"
        for item in diary_data:
            frame_num = item["frame"]
            text = item["story"]
            speed_val = item.get("speed", 1.0)
            output_name = f"narration_{frame_num:03d}.wav"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            
            try:
                payload = {
                    "model": "kokoro",
                    "voice": voice,
                    "input": text,
                    "speed": speed_val,
                    "response_format": "wav"
                }
                r = requests.post("http://localhost:8880/v1/audio/speech", json=payload, timeout=600)
                if r.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(r.content)
                        
                    # Apply VHS / Lofi audio distortion and hum using pydub
                    try:
                        from pydub.generators import Sine, WhiteNoise
                        sound = AudioSegment.from_wav(output_path)
                        sound = sound.low_pass_filter(2800).high_pass_filter(150)
                        hum = Sine(60).to_audio_segment(duration=len(sound), volume=-32)
                        hiss = WhiteNoise().to_audio_segment(duration=len(sound), volume=-38)
                        processed_sound = sound.overlay(hum).overlay(hiss)
                        processed_sound.export(output_path, format="wav")
                    except Exception as ae:
                        logit(f"Lofi audio post-processing failed for frame {frame_num}: {ae}")
            except Exception as e:
                logit(f"TTS generation error for frame {frame_num}: {e}")
                
        return jsonify({"status": "ok", "diary": diary_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_available_text_model():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1000)
        available = [m["name"] for m in r.json().get("models", [])] if r.status_code == 200 else []
    except:
        available = []
    
    text_model = "llama3.2:3b"
    if text_model not in available:
        preferred_text = ["llama3.2:3b", "deepseek-r1:1.5b", "dolphin3:8b", "llama2-uncensored:7b"]
        for p in preferred_text:
            if p in available:
                text_model = p
                break
        else:
            t_candidates = [m for m in available if any(x in m.lower() for x in ["llama", "dolphin", "deepseek", "qwen", "mistral"])]
            if t_candidates: text_model = t_candidates[0]
    return text_model

@app.route("/preview_story", methods=["POST"])
def preview_story():
    try:
        text_model = get_available_text_model()
        
        total_frames = frames_current
        if total_frames <= 0:
            total_frames = 120
            
        desc_str = ""
        descriptions = []
        
        for f in range(total_frames):
            intent = get_interpolated_prompt(f, original_starting_prompt)
            clean_intent, _ = parse_prompt(intent)
            
            progress = f / (total_frames - 1) if total_frames > 1 else 0.0
            tension = 1.0 + progress * 8.5
            sanity = 100.0 - progress * 90.0
            
            if tension < 3.0:
                emotion = "Calm and observant, trying to find a way out."
                inflection_hint = "Write in complete, slow, descriptive sentences."
                speed = 0.85
            elif tension < 6.0:
                emotion = "Growing paranoiac, feeling like the environment is shifting."
                inflection_hint = "Use occasional ellipses (...) to simulate hesitation, pausing, and whispering."
                speed = 1.00
            elif tension < 8.0:
                emotion = "Deep dread and anxiety. Suspects something is hunting them."
                inflection_hint = "Use frequent ellipses (...) and shorter, breathy, fragmented sentences."
                speed = 1.10
            else:
                emotion = "Extreme terror, panic, running for survival."
                inflection_hint = "Use very short, fragmented, frantic phrases, repetition, and exclamation marks!"
                speed = 1.25
                
            item = {
                "frame": f,
                "description": "Placeholder (Visuals not generated yet)",
                "tension": tension,
                "sanity": sanity,
                "emotion": emotion,
                "inflection_hint": inflection_hint,
                "speed": speed,
                "intent": clean_intent,
                "story": ""
            }
            descriptions.append(item)
            
            desc_str += (
                f"Frame {f}:\n"
                f"  - Actual Visuals: (Not generated yet. Focus entirely on the story beat / intent below)\n"
                f"  - Director Intent / Story Beat: {clean_intent}\n"
                f"  - Emotional State: {emotion}\n"
                f"  - Speech Inflection Style: {inflection_hint}\n\n"
            )
            
        prompt = (
            "You are an expert space captain's log narrator. You will write a mission log of a traveler mining platinum in the Kepler asteroid belt.\n\n"
            "Below is a sequence of frames representing the journey. For each frame, you are given:\n"
            "1. Director Intent / Story Beat: The thematic concept, maneuver, or mining stage that must happen.\n"
            "2. Emotional State: The captain's current tone and state of awe or efficiency.\n"
            "3. Speech Inflection Style: How the text must be formatted to guide voice inflection.\n\n"
            "Your task is to write a cohesive, continuous first-person space mission log matching this sequence.\n"
            f"You MUST write exactly {total_frames} log entries, one for each frame. "
            "Keep each entry brief (15-30 words).\n\n"
            "Format your output strictly using '[ENTRY]' as a separator before each frame entry, like this:\n"
            "[ENTRY] Entry 0 text...\n"
            "[ENTRY] Entry 1 text...\n\n"
            "Do not output any introductory or concluding text. Output ONLY the log texts with their [ENTRY] separators.\n\n"
            f"Timeline Sequence:\n{desc_str}"
        )
        
        payload = {
            "model": text_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "max_tokens": 2048}
        }
        
        story_raw = ""
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=1200)
        if r.status_code == 200:
            story_raw = r.json().get("response", "").strip()
            
        if not story_raw:
            return jsonify({"error": "Ollama story generation failed."}), 500
            
        entries = [e.strip() for e in story_raw.split("[ENTRY]") if e.strip()]
        
        if len(entries) < total_frames:
            while len(entries) < total_frames:
                idx = len(entries)
                entries.append(f"Wandering deeper. Story beat: {descriptions[idx]['intent']}.")
        elif len(entries) > total_frames:
            entries = entries[:total_frames]
            
        for idx, item in enumerate(descriptions):
            item["story"] = entries[idx]
            
        diary_path = os.path.join(OUTPUT_DIR, "space_diary.json")
        with open(diary_path, "w") as f:
            json.dump(descriptions, f, indent=2)
            
        return jsonify({"status": "ok", "diary": descriptions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/save_diary", methods=["POST"])
def save_diary():
    try:
        data = request.json
        diary_list = data.get("diary", [])
        if not diary_list:
            return jsonify({"error": "No diary entries provided"}), 400
            
        diary_path = os.path.join(OUTPUT_DIR, "space_diary.json")
        
        existing_map = {}
        if os.path.exists(diary_path):
            try:
                with open(diary_path, "r") as f:
                    existing = json.load(f)
                    for item in existing:
                        existing_map[int(item["frame"])] = item
            except:
                pass
                
        for item in diary_list:
            f = int(item["frame"])
            if f in existing_map:
                existing_map[f]["story"] = item["story"]
            else:
                existing_map[f] = item
                
        sorted_diary = [existing_map[k] for k in sorted(existing_map.keys())]
        
        with open(diary_path, "w") as f:
            json.dump(sorted_diary, f, indent=2)
            
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload_logo", methods=["POST"])
def upload_logo():
    if "logo" not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files["logo"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400
    if file and file.filename.lower().endswith(".png"):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config["OVERLAYS_FOLDER"], filename)
        try:
            from PIL import Image
            img = Image.open(file)
            img = img.convert("RGBA")
            img.save(path, "PNG")
            logit(f"Uploaded and converted logo to RGBA: {filename}")
            return jsonify({"status": "ok", "filename": filename})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to process image: {str(e)}"}), 500
    return jsonify({"status": "error", "message": "Only transparent PNGs allowed"}), 400

@app.route("/save_logo_position", methods=["POST"])
def save_logo_position():
    global logo_filename, logo_x, logo_y, logo_w, logo_h, logo_opacity
    d = request.json
    filename = d.get("logo_filename", "None")
    with state_lock:
        if filename == "None":
            logo_filename = "None"
        else:
            logo_filename = filename
            logo_x = int(d.get("x", 0))
            logo_y = int(d.get("y", 0))
            logo_w = int(d.get("w", 100))
            logo_h = int(d.get("h", 100))
            logo_opacity = float(d.get("opacity", 1.0))
    save_state()
    logit(f"Logo position saved: {logo_filename} at ({logo_x}, {logo_y}) {logo_w}x{logo_h} (Opacity: {logo_opacity})")
    return jsonify({"status": "ok"})

@app.route("/save_logo_local", methods=["POST"])
def save_logo_local():
    d = request.json
    filename = d.get("logo_filename", "None")
    lx = int(d.get("x", 0))
    ly = int(d.get("y", 0))
    lw = int(d.get("w", 100))
    lh = int(d.get("h", 100))
    l_opacity = float(d.get("opacity", 1.0))

    # Find last generated frame
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("frame_") and f.endswith(".png")])
    if not files:
        return jsonify({"status": "error", "message": "No frames generated yet"}), 400

    last_frame_file = files[-1]
    last_frame_path = os.path.join(OUTPUT_DIR, last_frame_file)
    
    # Try to find the per-frame clean base first to avoid stacking logos
    frame_num_str = last_frame_file.replace("frame_", "").replace(".png", "")
    per_frame_base_file = f"clean_base_{frame_num_str}.png"
    per_frame_base_path = os.path.join(OUTPUT_DIR, per_frame_base_file)
    
    if os.path.exists(per_frame_base_path):
        base_img_path = per_frame_base_path
    else:
        # Fallback to latest_base.png or the frame itself
        base_img_path = os.path.join(OUTPUT_DIR, "latest_base.png")
        if not os.path.exists(base_img_path):
            base_img_path = last_frame_path

    try:
        img = Image.open(base_img_path)
        img = apply_logo_to_image(img, filename, lx, ly, lw, lh, l_opacity)
        img.save(last_frame_path)
        logit(f"Saved logo locally on frame: {last_frame_file}")
        return jsonify({"status": "ok"})
    except Exception as e:
        logit(f"Error saving logo locally: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def get_ollama_models():
    """
    Queries local Ollama instance for installed models.
    """
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2000)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return models
    except:
        pass
    return []

def resolve_model_name(model_name):
    """
    Resolves a requested model name to one of the installed Ollama models.
    """
    available = get_ollama_models()
    if not available:
        return model_name
    
    if model_name in available:
        return model_name
        
    # Case-insensitive match
    for m in available:
        if m.lower() == model_name.lower():
            return m
            
    # Try common alias mappings (e.g. llama3.2 -> llama3.2:3b)
    cleaned = model_name.lower().replace(":latest", "")
    for m in available:
        m_base = m.lower().split(":")[0]
        if m_base == cleaned:
            return m
            
    # Substring match (e.g. llama3.2 matching llama3.2:3b)
    for m in available:
        if cleaned in m.lower():
            return m
            
    return model_name


@app.route("/get_ollama_models")
def get_ollama_route():
    models = get_ollama_models()
    return jsonify({"models": models})

@app.route("/inspect_image_llava", methods=["POST"])
def inspect_image_llava():
    d = request.json
    filename = d.get("filename")
    prompt = d.get("prompt", "Describe this image.")
    model = d.get("model", "LlaVa:latest")
    
    click_x = d.get("click_x")
    click_y = d.get("click_y")
    container_w = d.get("container_w", 1)
    container_h = d.get("container_h", 1)
    
    if not filename or filename == "latest_frame" or filename == "None":
        files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("frame_") and f.endswith(".png")])
        if not files:
            return jsonify({"error": "No frames generated yet"}), 400
        img_path = os.path.join(OUTPUT_DIR, files[-1])
    elif os.path.exists(os.path.join(OUTPUT_DIR, filename)):
        img_path = os.path.join(OUTPUT_DIR, filename)
    elif os.path.exists(os.path.join(BASE_DIR, "static", "overlays", filename)):
        img_path = os.path.join(BASE_DIR, "static", "overlays", filename)
    else:
        return jsonify({"error": f"Image file not found: {filename}"}), 400
        
    try:
        img = Image.open(img_path)
        
        # Click-to-crop visualizer mode
        if click_x is not None and click_y is not None:
            w, h = img.size
            real_x = int(click_x / container_w * w)
            real_y = int(click_y / container_h * h)
            size = 150
            left, top = max(0, real_x - size//2), max(0, real_y - size//2)
            right, bottom = min(w, real_x + size//2), min(h, real_y + size//2)
            img = img.crop((left, top, right, bottom))
            
        from io import BytesIO
        import base64
        buf = BytesIO()
        img.save(buf, format="PNG")
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False,
            "options": {
                "temperature": 0.5,
                "max_tokens": 150
            }
        }
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=2000)
        if r.status_code == 200:
            res = r.json().get("response", "").strip()
            return jsonify({"status": "ok", "description": res})
        else:
            return jsonify({"error": f"Ollama error: status code {r.status_code}"}), 500
    except Exception as e:
        return jsonify({"error": f"Scanner failed: {str(e)}"}), 500

@app.route("/templates_editor", methods=["GET", "POST"])
def templates_editor():
    templates_dir = os.path.join(BASE_DIR, "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    if request.method == "POST":
        d = request.json
        filename = secure_filename(d.get("filename", "").strip())
        content = d.get("content", "")
        if not filename:
            return jsonify({"status": "error", "message": "Filename is required"}), 400
        if not filename.endswith(".html"):
            filename += ".html"
            
        path = os.path.join(templates_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return jsonify({"status": "ok"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
            
    # GET method
    files = sorted([f for f in os.listdir(templates_dir) if f.lower().endswith((".html", ".md", ".txt"))])
    
    # Load specific file contents if requested
    selected_file = request.args.get("file", "")
    content = ""
    if selected_file:
        selected_file = secure_filename(selected_file)
        path = os.path.join(templates_dir, selected_file)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                content = f"Error reading file: {str(e)}"
                
    # Get available Ollama models to expose in the editor
    models = get_ollama_models()
    
    return render_template_string(
        HTML_EDITOR,
        files=files,
        selected_file=selected_file,
        content=content,
        MODELS=models
    )

@app.route("/delete_template", methods=["POST"])
def delete_template():
    d = request.json
    filename = secure_filename(d.get("filename", "").strip())
    if not filename:
        return jsonify({"status": "error", "message": "Filename is required"}), 400
    path = os.path.join(BASE_DIR, "templates", filename)
    if os.path.exists(path):
        try:
            os.remove(path)
            return jsonify({"status": "ok"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "error", "message": "File not found"}), 404

@app.route("/view_template/<filename>")
def view_template(filename):
    filename = secure_filename(filename)
    path = os.path.join(BASE_DIR, "templates", filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}", 500
    return "Page not found", 404

@app.route("/draft_template_ollama", methods=["POST"])
def draft_template_ollama():
    d = request.json
    prompt = d.get("prompt", "").strip()
    model = d.get("model", "").strip()
    model = resolve_model_name(model)
    if not prompt:
        return jsonify({"error": "Prompt is empty"}), 400
        
    system_prompt = (
        "You are an expert web developer. Generate clean, modern, and responsive HTML code for the requested page. "
        "Include CSS styles inside a <style> tag in the head. Use dark-mode visual theme with purple accents (#8b5cf6), "
        "matching a premium dark room engine aesthetic. "
        "Do NOT write any introduction or explanation. Do NOT wrap the code in markdown code fences. "
        "Return ONLY the raw HTML code."
    )
    
    try:
        payload = {
            "model": model,
            "prompt": f"{system_prompt}\n\nRequested Page Concept: {prompt}\nHTML Output:",
            "stream": False,
            "options": {
                "temperature": 0.6,
                "max_tokens": 1500
            }
        }
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=3000)
        if r.status_code == 200:
            res = r.json().get("response", "").strip()
            if res.startswith("```html"):
                res = res[7:]
            if res.startswith("```"):
                res = res[3:]
            if res.endswith("```"):
                res = res[:-3]
            return jsonify({"status": "ok", "html": res.strip()})
        else:
            return jsonify({"error": f"Ollama error: status code {r.status_code}"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to reach Ollama: {str(e)}"}), 500

@app.route("/chat_ollama", methods=["POST"])
def chat_ollama():
    d = request.json
    messages = d.get("messages", [])
    model = d.get("model", "llama3.2:3b")
    model = resolve_model_name(model)
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=2000)
        if r.status_code == 200:
            res_msg = r.json().get("message", {})
            return jsonify({"status": "ok", "message": res_msg})
        else:
            return jsonify({"error": f"Ollama error: status code {r.status_code}"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to reach Ollama: {str(e)}"}), 500

@app.route("/enhance_prompt", methods=["POST"])
def enhance_prompt():
    d = request.json
    user_prompt = d.get("prompt", "").strip()
    model = d.get("model", "").strip()
    model = resolve_model_name(model)
    if not user_prompt:
        return jsonify({"error": "Prompt is empty"}), 400
    
    system_prompt = (
        "Act as an expert Stable Diffusion prompt generator. "
        "Expand the following concept into a highly descriptive visual prompt. "
        "Focus on atmospheric lighting, artistic style, camera lens details, and vivid textures. "
        "Keep the output under 60 words and return ONLY the final prompt, with no intro, outro, or conversational filler."
    )
    
    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\nConcept: {user_prompt}\nExpanded Prompt:",
        "stream": False,
        "options": {
            "temperature": 0.7,
            "max_tokens": 150
        }
    }
    
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=1500)
        if r.status_code == 200:
            enhanced = r.json().get("response", "").strip()
            if enhanced.startswith('"') and enhanced.endswith('"'):
                enhanced = enhanced[1:-1]
            
            # Clean quotes and punctuation
            enhanced = clean_desc(enhanced)
            
            global current_prompt, original_starting_prompt
            with state_lock:
                current_prompt = enhanced
                original_starting_prompt = enhanced
            save_state()
            
            logit(f"Prompt enhanced by {model}: '{user_prompt}' -> '{enhanced}'")
            return jsonify({"status": "ok", "enhanced_prompt": enhanced})
        else:
            return jsonify({"error": f"Ollama error: status code {r.status_code}"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to reach Ollama: {str(e)}"}), 500

@app.route("/compile_movie", methods=["POST"])
def compile_movie():
    success = create_movie_from_frames()
    return jsonify({"status": "success" if success else "failed"})

@app.route("/control", methods=["POST"])
def control():
    global current_prompt, paused; d = request.json; act = d.get("action")
    if act == "start":
        current_prompt = d.get("prompt", current_prompt)
        Thread(target=lambda: render_video(False), daemon=True).start()
    elif act == "resume":
        Thread(target=lambda: render_video(True), daemon=True).start()
    elif act == "pause": paused = not paused
    return jsonify({"status": "ok"})

@app.route("/update_params", methods=["POST"])
def update_params():
    global model_name, negative_prompt, lora1_name, lora2_name, lora3_name, current_seed, denoise_current, frames_current, current_prompt, original_starting_prompt
    global zoom_start, zoom_end, pan_start_x, pan_end_x, pan_start_y, pan_end_y, default_steps, default_cfg, use_motion_zoom, use_metadata_caption
    global logo_filename, logo_x, logo_y, logo_w, logo_h, logo_opacity
    global roll_mode
    global feedback_color_boost, feedback_contrast_boost, feedback_sharpness_boost
    global lora1_strength, lora2_strength, lora3_strength
    global use_visual_director, visual_director_interval, visual_director_model
    global use_prompt_interpolation, use_video_interpolation
    global caption_font_size, caption_x, caption_y, caption_bg_r, caption_bg_g, caption_bg_b, caption_bg_a
    global zoom_mode, zoom_shrink, zoom_blur, zoom_strength, zoom_guidance_scale, zoom_num_inference_steps
    global use_seed_guide, seed_guide_filename, seed_guide_x, seed_guide_y, seed_guide_w, seed_guide_h, seed_guide_opacity
    
    d = request.json
    if not d:
        return jsonify({"status": "error", "message": "No JSON payload received"}), 400

    def safe_int(v, default_val):
        try:
            if v is None or v == "":
                return default_val
            return int(float(v))
        except Exception:
            return default_val

    def safe_float(v, default_val):
        try:
            if v is None or v == "":
                return default_val
            return float(v)
        except Exception:
            return default_val

    new_prompt = d.get("prompt")
    if new_prompt:
        if new_prompt != current_prompt:
            current_prompt = new_prompt
            original_starting_prompt = new_prompt
            logit(f"Prompt manually updated during stream. New baseline: '{current_prompt}'")
            
    caption_font_size = safe_int(d.get("caption_font_size"), caption_font_size)
    caption_x = safe_int(d.get("caption_x"), caption_x)
    caption_y = safe_int(d.get("caption_y"), caption_y)
    caption_bg_r = safe_int(d.get("caption_bg_r"), caption_bg_r)
    caption_bg_g = safe_int(d.get("caption_bg_g"), caption_bg_g)
    caption_bg_b = safe_int(d.get("caption_bg_b"), caption_bg_b)
    caption_bg_a = safe_float(d.get("caption_bg_a"), caption_bg_a)
    
    model_name = d.get("model")
    negative_prompt = d.get("negative_prompt", negative_prompt)
    lora1_name = d.get("lora1")
    lora2_name = d.get("lora2")
    lora3_name = d.get("lora3", "None")
    
    current_seed = safe_int(d.get("seed"), current_seed)
    denoise_current = safe_float(d.get("denoise"), denoise_current)
    frames_current = safe_int(d.get("frames"), frames_current)
    use_motion_zoom = bool(d.get("use_motion_zoom"))
    use_metadata_caption = bool(d.get("use_metadata_caption"))
    
    zoom_start = safe_float(d.get("zoom_start"), zoom_start)
    zoom_end = safe_float(d.get("zoom_end"), zoom_end)
    pan_start_x = safe_float(d.get("pan_start_x"), pan_start_x)
    pan_end_x = safe_float(d.get("pan_end_x"), pan_end_x)
    pan_start_y = safe_float(d.get("pan_start_y"), pan_start_y)
    pan_end_y = safe_float(d.get("pan_end_y"), pan_end_y)
    roll_mode = d.get("roll_mode", "none")
    
    default_steps = safe_int(d.get("steps"), default_steps)
    default_cfg = safe_float(d.get("cfg"), default_cfg)
    
    feedback_color_boost = safe_float(d.get("feedback_color"), feedback_color_boost)
    feedback_contrast_boost = safe_float(d.get("feedback_contrast"), feedback_contrast_boost)
    feedback_sharpness_boost = safe_float(d.get("feedback_sharpness"), feedback_sharpness_boost)
    
    lora1_strength = safe_float(d.get("lora1_strength"), lora1_strength)
    lora2_strength = safe_float(d.get("lora2_strength"), lora2_strength)
    lora3_strength = safe_float(d.get("lora3_strength"), lora3_strength)
    
    use_visual_director = bool(d.get("use_visual_director"))
    visual_director_interval = safe_int(d.get("visual_director_interval"), visual_director_interval)
    visual_director_model = d.get("visual_director_model", "moondream")
    
    use_prompt_interpolation = bool(d.get("use_prompt_interpolation"))
    use_video_interpolation = bool(d.get("use_video_interpolation"))
    
    zoom_mode = d.get("zoom_mode", "in")
    zoom_shrink = safe_int(d.get("zoom_shrink"), zoom_shrink)
    zoom_blur = safe_int(d.get("zoom_blur"), zoom_blur)
    zoom_strength = safe_float(d.get("zoom_strength"), zoom_strength)
    zoom_guidance_scale = safe_float(d.get("zoom_guidance_scale"), zoom_guidance_scale)
    zoom_num_inference_steps = default_steps

    use_seed_guide = bool(d.get("use_seed_guide"))
    seed_guide_filename = d.get("seed_guide_filename", "None")
    seed_guide_x = safe_int(d.get("seed_guide_x"), seed_guide_x)
    seed_guide_y = safe_int(d.get("seed_guide_y"), seed_guide_y)
    seed_guide_w = safe_int(d.get("seed_guide_w"), seed_guide_w)
    seed_guide_h = safe_int(d.get("seed_guide_h"), seed_guide_h)
    seed_guide_opacity = safe_float(d.get("seed_guide_opacity"), seed_guide_opacity)
    
    logit(f"ENGINE UPDATE: use_zoom={use_motion_zoom}, steps={default_steps}, zoom_steps={zoom_num_inference_steps}, zoom_strength={zoom_strength}")
    save_state()
    return jsonify({"status": "ok"})

@app.route("/status")
def status_route():
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("frame_") and f.endswith(".png")])
    history = files[-5:] if len(files) > 0 else []
    history.reverse()
    return jsonify({
        "running": running, "paused": paused, "frame": current_frame, "total": frames_current,
        "history": history, "keyframes": keyframes, "injections": injection_lines, "zoom": use_motion_zoom,
        "metadata_caption": use_metadata_caption,
        "prompt": current_prompt,
        "rendering_prompt": rendering_prompt,
        "progress": comfy_progress,
        "max_steps": comfy_max_steps,
        "width": DEFAULT_WIDTH,
        "height": DEFAULT_HEIGHT,
        "steps": default_steps,
        "seed": current_seed,
        "cfg": default_cfg,
        "denoise": denoise_current,
        "frames": frames_current,
        "logo_filename": logo_filename,
        "logo_x": logo_x,
        "logo_y": logo_y,
        "logo_w": logo_w,
        "logo_h": logo_h,
        "logo_opacity": logo_opacity,
        "lora1_strength": lora1_strength,
        "lora2_strength": lora2_strength,
        "lora3_strength": lora3_strength,
        "use_visual_director": use_visual_director,
        "visual_director_interval": visual_director_interval,
        "visual_director_model": visual_director_model,
        "use_prompt_interpolation": use_prompt_interpolation,
        "use_video_interpolation": use_video_interpolation,
        "caption_font_size": caption_font_size,
        "caption_x": caption_x,
        "caption_y": caption_y,
        "caption_bg_r": caption_bg_r,
        "caption_bg_g": caption_bg_g,
        "caption_bg_b": caption_bg_b,
        "caption_bg_a": caption_bg_a,
        "zoom_mode": zoom_mode,
        "zoom_shrink": zoom_shrink,
        "zoom_blur": zoom_blur,
        "zoom_strength": zoom_strength,
        "zoom_guidance_scale": zoom_guidance_scale,
        "zoom_num_inference_steps": zoom_num_inference_steps,
        "zoom_start": zoom_start,
        "zoom_end": zoom_end,
        "latest_vision_desc": latest_vision_desc,
        "use_seed_guide": use_seed_guide,
        "seed_guide_filename": seed_guide_filename,
        "seed_guide_x": seed_guide_x,
        "seed_guide_y": seed_guide_y,
        "seed_guide_w": seed_guide_w,
        "seed_guide_h": seed_guide_h,
        "seed_guide_opacity": seed_guide_opacity
    })

@app.route("/add_keyframe", methods=["POST"])
def add_keyframe():
    d = request.json; f_idx = str(d.get("frame", 0))
    keyframes[f_idx] = {"prompt": d.get("prompt", "") or current_prompt, "denoise": float(d.get("denoise", 0.5)), "seed_offset": int(d.get("seed_offset", 0))}
    save_state(); return jsonify({"status": "ok", "keyframes": keyframes})

@app.route("/latest_frame")
def latest():
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("frame_") and f.endswith(".png")])
    if not files: return "none", 404
    return send_file(os.path.join(OUTPUT_DIR, files[-1]), max_age=0)

@app.route("/inject", methods=["POST"])
def inject():
    t = request.json.get("text", "").strip(); 
    if t: injection_lines.append(t); save_state()
    return jsonify({"status": "ok"})

@app.route("/set_caption", methods=["POST"])
def set_caption():
    global active_caption, caption_remaining, active_caption_font
    global temp_caption_font_size, temp_caption_x, temp_caption_y
    global temp_caption_bg_r, temp_caption_bg_g, temp_caption_bg_b, temp_caption_bg_a
    d = request.json
    t = d.get("text", "").strip()
    font = d.get("font", "Default")
    if t:
        with state_lock:
            active_caption = t
            caption_remaining = 5
            active_caption_font = font
            temp_caption_font_size = int(d.get("font_size", temp_caption_font_size))
            temp_caption_x = int(d.get("x", temp_caption_x))
            temp_caption_y = int(d.get("y", temp_caption_y))
            temp_caption_bg_r = int(d.get("bg_r", temp_caption_bg_r))
            temp_caption_bg_g = int(d.get("bg_g", temp_caption_bg_g))
            temp_caption_bg_b = int(d.get("bg_b", temp_caption_bg_b))
            temp_caption_bg_a = float(d.get("bg_a", temp_caption_bg_a))
        save_state()
        logit(f"Caption set: {active_caption} (Font: {font}, Size: {temp_caption_font_size}, Remaining: 5)")
    return jsonify({"status": "ok"})

@app.route("/teleport", methods=["POST"])
def teleport():
    global teleport_image
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image part"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    try:
        # Resize image to match project dimensions (DEFAULT_WIDTH x DEFAULT_HEIGHT)
        img = Image.open(file).convert("RGB")
        if img.size != (DEFAULT_WIDTH, DEFAULT_HEIGHT):
            logit(f"Resizing teleport image from {img.size} to {(DEFAULT_WIDTH, DEFAULT_HEIGHT)}")
            img = img.resize((DEFAULT_WIDTH, DEFAULT_HEIGHT), Image.LANCZOS)
        
        # Save locally in the output/spaceexplorer2 folder
        teleport_filename = f"teleport_{int(time.time())}.png"
        teleport_path = os.path.join(OUTPUT_DIR, teleport_filename)
        img.save(teleport_path)
        
        with state_lock:
            teleport_image = teleport_path
        
        logit(f"Teleport image set: {teleport_image}")
        return jsonify({"status": "ok", "filename": teleport_filename})
    except Exception as e:
        logit(f"Teleport upload error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ============================================================
# JSON BUILDER TEMPLATE
# ============================================================
HTML_JSON_BUILDER = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EPOCH Liminal Spaces - Keyframe JSON Architect</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg-primary: #0a0a0c;
            --bg-secondary: #121216;
            --bg-tertiary: #1a1a22;
            --accent: #e5a93b;
            --accent-glow: rgba(229, 169, 59, 0.2);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --border-color: rgba(255, 255, 255, 0.08);
            --border-focus: rgba(229, 169, 59, 0.5);
            --success: #10b981;
            --error: #ef4444;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }

        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--bg-tertiary);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent);
        }

        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            z-index: 10;
        }

        .header h1 {
            margin: 0;
            font-size: 20px;
            font-weight: 700;
            color: var(--accent);
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .header-links {
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .header-link {
            color: var(--text-muted);
            text-decoration: none;
            font-size: 13px;
            font-weight: 500;
            transition: color 0.2s;
        }

        .header-link:hover {
            color: var(--accent);
        }

        .btn {
            background: var(--bg-tertiary);
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }

        .btn:hover {
            border-color: var(--accent);
            box-shadow: 0 0 10px var(--accent-glow);
        }

        .btn-primary {
            background: var(--accent);
            color: #000;
            border: none;
        }

        .btn-primary:hover {
            background: #f0b852;
            box-shadow: 0 0 15px rgba(229, 169, 59, 0.4);
        }

        .btn-danger {
            background: rgba(239, 68, 68, 0.15);
            color: var(--error);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .btn-danger:hover {
            background: var(--error);
            color: #fff;
            border-color: var(--error);
        }

        .btn-success {
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .btn-success:hover {
            background: var(--success);
            color: #fff;
            border-color: var(--success);
        }

        .status-badge {
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }

        .status-badge.running {
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .status-badge.paused {
            background: rgba(229, 169, 59, 0.15);
            color: var(--accent);
            border: 1px solid rgba(229, 169, 59, 0.3);
        }

        .container {
            display: grid;
            grid-template-columns: 28% 44% 28%;
            height: calc(100vh - 57px);
            overflow: hidden;
        }

        .panel {
            background: var(--bg-secondary);
            display: flex;
            flex-direction: column;
            height: 100%;
            border-right: 1px solid var(--border-color);
            min-height: 0;
        }

        .panel:last-child {
            border-right: none;
        }

        .panel-header {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
            font-size: 14px;
            color: var(--text-main);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-tertiary);
        }

        .panel-body {
            padding: 16px;
            overflow-y: auto;
            flex-grow: 1;
        }

        .form-group {
            margin-bottom: 14px;
        }

        .form-row {
            display: flex;
            gap: 10px;
        }

        .form-row .form-group {
            flex: 1;
        }

        label {
            display: block;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }

        input, select, textarea {
            width: 100%;
            box-sizing: border-box;
            background: var(--bg-tertiary);
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 13px;
            transition: all 0.2s;
            font-family: inherit;
        }

        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px var(--accent-glow);
        }

        textarea {
            resize: vertical;
        }

        .kf-card {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
            position: relative;
            transition: all 0.2s;
        }

        .kf-card:hover {
            border-color: var(--accent-glow);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        .kf-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .kf-badge {
            background: var(--accent);
            color: #000;
            font-weight: 700;
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 4px;
        }

        .json-textarea {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 12px;
            line-height: 1.5;
            background: #050507;
            height: 100%;
            resize: none;
        }

        .live-preview {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            background: #000;
            aspect-ratio: 340/512;
            max-width: 150px;
            margin: 10px auto;
            position: relative;
        }

        .live-preview img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        .live-preview-label {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.6);
            color: var(--accent);
            font-size: 9px;
            text-align: center;
            padding: 2px;
            font-weight: bold;
        }

        .notification {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 24px;
            border-radius: 8px;
            background: var(--bg-tertiary);
            border-left: 4px solid var(--accent);
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            display: none;
            z-index: 100;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        .loading-spinner {
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 50%;
            border-top: 2px solid var(--accent);
            width: 12px;
            height: 12px;
            animation: spin 0.8s linear infinite;
            display: inline-block;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--accent);"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line><line x1="15" y1="3" x2="15" y2="21"></line><line x1="3" y1="9" x2="21" y2="9"></line><line x1="3" y1="15" x2="21" y2="15"></line></svg>
            EPOCH Liminal Spaces - Keyframe Architect
        </h1>
        <div class="header-links">
            <div id="status-container" class="status-badge paused">Offline</div>
            <a href="/" class="header-link">Main Dashboard</a>
            <a href="/media" class="header-link">Media Library</a>
            <button class="btn btn-primary" id="save-btn" onclick="saveConfigToServer()">Save to Server</button>
        </div>
    </div>

    <div class="container">
        <!-- LEFT PANEL: Global Configuration -->
        <div class="panel">
            <div class="panel-header">
                <span>Global Configuration</span>
            </div>
            <div class="panel-body">
                <div class="form-group">
                    <label>Stable Diffusion Model</label>
                    <select id="model" onchange="updateConfigField('model', this.value)">
                        {% for m in MODELS %}
                        <option value="{{ m }}">{{ m }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label>Base Prompt</label>
                    <textarea id="prompt" rows="3" oninput="updateConfigField('prompt', this.value)"></textarea>
                </div>

                <div class="form-group">
                    <label>Negative Prompt</label>
                    <textarea id="negative_prompt" rows="2" oninput="updateConfigField('negative_prompt', this.value)"></textarea>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label>Seed</label>
                        <input type="number" id="seed" oninput="updateConfigField('seed', parseInt(this.value) || 0)">
                    </div>
                    <div class="form-group">
                        <label>CFG Scale</label>
                        <input type="number" id="cfg" step="0.5" oninput="updateConfigField('cfg', parseFloat(this.value) || 0)">
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label>Steps</label>
                        <input type="number" id="steps" oninput="updateConfigField('steps', parseInt(this.value) || 0)">
                    </div>
                    <div class="form-group">
                        <label>Default Denoise</label>
                        <input type="number" id="denoise" step="0.05" oninput="updateConfigField('denoise', parseFloat(this.value) || 0)">
                    </div>
                </div>

                <div style="border-top: 1px solid var(--border-color); margin: 15px 0; padding-top: 10px;"></div>

                <label>LoRA 1 & Strength</label>
                <div class="form-row" style="margin-bottom: 8px;">
                    <select id="lora1" style="flex:2;" onchange="updateConfigField('lora1', this.value)">
                        {% for l in LORAS %}
                        <option value="{{ l }}">{{ l }}</option>
                        {% endfor %}
                    </select>
                    <input type="number" id="lora1_strength" step="0.1" style="flex:1;" oninput="updateConfigField('lora1_strength', parseFloat(this.value) || 0)">
                </div>

                <label>LoRA 2 & Strength</label>
                <div class="form-row" style="margin-bottom: 8px;">
                    <select id="lora2" style="flex:2;" onchange="updateConfigField('lora2', this.value)">
                        {% for l in LORAS %}
                        <option value="{{ l }}">{{ l }}</option>
                        {% endfor %}
                    </select>
                    <input type="number" id="lora2_strength" step="0.1" style="flex:1;" oninput="updateConfigField('lora2_strength', parseFloat(this.value) || 0)">
                </div>

                <label>LoRA 3 & Strength</label>
                <div class="form-row" style="margin-bottom: 8px;">
                    <select id="lora3" style="flex:2;" onchange="updateConfigField('lora3', this.value)">
                        {% for l in LORAS %}
                        <option value="{{ l }}">{{ l }}</option>
                        {% endfor %}
                    </select>
                    <input type="number" id="lora3_strength" step="0.1" style="flex:1;" oninput="updateConfigField('lora3_strength', parseFloat(this.value) || 0)">
                </div>

                <div style="border-top: 1px solid var(--border-color); margin: 15px 0; padding-top: 10px;"></div>

                <div class="form-group" style="display:flex; justify-content:space-between; align-items:center;">
                    <label style="margin-bottom:0;">Use Motion Zoom</label>
                    <input type="checkbox" id="use_motion_zoom" style="width:auto;" onchange="updateConfigField('use_motion_zoom', this.checked)">
                </div>

                <div class="form-group">
                    <label>Zoom Mode</label>
                    <select id="zoom_mode" onchange="updateConfigField('zoom_mode', this.value); toggleBuilderZoomFields();">
                        <option value="in">Zoom In (Crop & Pan)</option>
                        <option value="out">Zoom Out (Blur, Shrink & Extend)</option>
                    </select>
                </div>

                <div id="builder_zoom_in_params">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Zoom Start</label>
                            <input type="number" id="z_s" step="0.01" oninput="updateConfigField('z_s', parseFloat(this.value) || 0)">
                        </div>
                        <div class="form-group">
                            <label>Zoom End</label>
                            <input type="number" id="z_e" step="0.01" oninput="updateConfigField('z_e', parseFloat(this.value) || 0)">
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label>Pan X Start/End</label>
                            <div style="display:flex; gap:5px;">
                                <input type="number" id="px_s" step="0.05" oninput="updateConfigField('px_s', parseFloat(this.value) || 0)">
                                <input type="number" id="px_e" step="0.05" oninput="updateConfigField('px_e', parseFloat(this.value) || 0)">
                            </div>
                        </div>
                        <div class="form-group">
                            <label>Pan Y Start/End</label>
                            <div style="display:flex; gap:5px;">
                                <input type="number" id="py_s" step="0.05" oninput="updateConfigField('py_s', parseFloat(this.value) || 0)">
                                <input type="number" id="py_e" step="0.05" oninput="updateConfigField('py_e', parseFloat(this.value) || 0)">
                            </div>
                        </div>
                    </div>
                </div>

                <div id="builder_zoom_out_params" style="display:none;">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Shrink (px)</label>
                            <input type="number" id="zoom_shrink" oninput="updateConfigField('zoom_shrink', parseInt(this.value) || 10)">
                        </div>
                        <div class="form-group">
                            <label>Blur Radius</label>
                            <input type="number" id="zoom_blur" oninput="updateConfigField('zoom_blur', parseInt(this.value) || 8)">
                        </div>
                    </div>
                </div>
                <div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Zoom Strength</label>
                            <input type="number" id="zoom_strength" step="0.01" oninput="updateConfigField('zoom_strength', parseFloat(this.value) || 0.28)">
                        </div>
                        <div class="form-group">
                            <label>Zoom CFG Scale</label>
                            <input type="number" id="zoom_guidance_scale" step="0.1" oninput="updateConfigField('zoom_guidance_scale', parseFloat(this.value) || 7.0)">
                        </div>
                        <div class="form-group">
                            <label>Zoom Steps</label>
                            <input type="number" id="zoom_num_inference_steps" oninput="updateConfigField('zoom_num_inference_steps', parseInt(this.value) || 25)">
                        </div>
                    </div>
                </div>

                <div class="form-group">
                    <label>Roll Mode</label>
                    <select id="roll_mode" onchange="updateConfigField('roll_mode', this.value)">
                        <option value="none">None</option>
                        <option value="left">Left Roll</option>
                        <option value="right">Right Roll</option>
                    </select>
                </div>

                <div style="border-top: 1px solid var(--border-color); margin: 15px 0; padding-top: 10px;"></div>
                
                <label>Stream Preview</label>
                <div class="live-preview">
                    <img id="stream-preview-img" src="/latest_frame" onerror="this.src='/static/border.png'">
                    <div class="live-preview-label" id="preview-label">Frame: Loading</div>
                </div>
            </div>
        </div>

        <!-- CENTER PANEL: Keyframes Sequence Manager -->
        <div class="panel">
            <div class="panel-header">
                <span>Keyframes Sequence</span>
                <span id="kf-count" style="font-size: 11px; background: var(--bg-tertiary); padding: 2px 6px; border-radius: 4px; border: 1px solid var(--border-color);">0 Keyframes</span>
            </div>
            <div class="panel-body">
                <!-- Add Keyframe Form -->
                <div style="background: var(--bg-tertiary); border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                    <h3 style="margin-top:0; margin-bottom:12px; font-size:13px; color:var(--accent); text-transform:uppercase; letter-spacing:0.5px;">Add New Keyframe</h3>
                    <div class="form-row">
                        <div class="form-group" style="flex:0.8;">
                            <label>Frame #</label>
                            <input type="number" id="new-kf-frame" min="0" placeholder="e.g. 50">
                        </div>
                        <div class="form-group" style="flex:1.2;">
                            <label>Denoise</label>
                            <input type="number" id="new-kf-denoise" step="0.05" min="0.1" max="1.0" value="0.55">
                        </div>
                        <div class="form-group" style="flex:1;">
                            <label>Seed Offset</label>
                            <input type="number" id="new-kf-offset" value="3">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Keyframe Prompt</label>
                        <textarea id="new-kf-prompt" rows="2" placeholder="Leave empty to use base prompt..."></textarea>
                    </div>
                    <button class="btn btn-primary" style="width:100%;" onclick="addKeyframeLocal()">Add Keyframe</button>
                </div>

                <div id="keyframes-container">
                    <!-- Keyframe cards will render here -->
                </div>
            </div>
        </div>

        <!-- RIGHT PANEL: AI Generation & Raw JSON -->
        <div class="panel">
            <div class="panel-header">
                <span>AI Generator & Raw JSON</span>
            </div>
            <div class="panel-body" style="display:flex; flex-direction:column; gap:15px; height:100%; box-sizing:border-box;">
                <!-- AI Keyframe Generator Section -->
                <div style="background: var(--bg-tertiary); border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; flex-shrink:0;">
                    <h3 style="margin-top:0; margin-bottom:8px; font-size:13px; color:var(--accent); text-transform:uppercase; letter-spacing:0.5px; display:flex; justify-content:space-between; align-items:center;">
                        <span>Liminal AI Keyframe Generator</span>
                        <div class="loading-spinner" id="ai-spinner" style="display:none;"></div>
                    </h3>
                    
                    <div class="form-group">
                        <label>AI Outline / Script</label>
                        <textarea id="ai-outline" rows="3" value="frame 0: lost in liminal yellow corridor.&#10;frame 100: dark flickering fluorescent lights.&#10;frame 200: distant shadow moves.&#10;frame 300: sprinting through tiled hallways."></textarea>
                    </div>
                    <div class="form-group">
                        <label>Ollama Model</label>
                        <select id="ai-model">
                            <option value="loading">Loading Ollama Models...</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Custom Instructions</label>
                        <input type="text" id="ai-instructions" value="Make it visual, photorealistic, and extremely unsettling." value="Style settings...">
                    </div>
                    <button class="btn btn-success" style="width:100%;" id="ai-gen-btn" onclick="generateKeyframesAI()">Generate & Merge Keyframes</button>
                </div>

                <!-- Liminal Storyteller & Voiceover Section -->
                <div style="background: var(--bg-tertiary); border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; flex-shrink:0; max-height: 480px; display: flex; flex-direction: column;">
                    <h3 style="margin-top:0; margin-bottom:8px; font-size:13px; color:var(--accent); text-transform:uppercase; letter-spacing:0.5px; display:flex; justify-content:space-between; align-items:center;">
                        <span>Liminal Storyteller & TTS</span>
                        <div class="loading-spinner" id="story-spinner" style="display:none;"></div>
                    </h3>
                    <p style="font-size: 11px; color: var(--text-muted); margin-top: 0; margin-bottom: 10px;">
                        Plan and edit your horror diary script, apply voice inflection, and compile the final BGM-ducked movie.
                    </p>
                    
                    <div class="form-group" style="margin-bottom: 10px;">
                        <label style="margin-top:0; font-size: 10px; color:var(--text-muted);">Background Music (BGM)</label>
                        <select id="story-bgm" style="width:100%; font-size:11px; padding: 6px; background: var(--bg-primary); color:#fff; border:1px solid var(--border-color); border-radius:6px; box-sizing:border-box;">
                            <option value="None">None (Voice Only)</option>
                            {% for b in BGMS %}
                            <option value="{{b}}">{{b}}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div style="display: flex; gap: 6px; margin-bottom: 8px;">
                        <button class="btn" style="flex:1; margin-top:0; padding:6px;" id="story-prev-btn" onclick="previewStory()">Preview Script</button>
                        <button class="btn btn-success" style="flex:1; margin-top:0; padding:6px; display:none;" id="story-save-btn" onclick="saveStoryScript()">Save Script</button>
                    </div>
                    
                    <button class="btn btn-primary" style="width:100%; margin-bottom:8px; margin-top:0;" id="story-btn" onclick="generateStoryDiary()">Compile Story & Audio Tracks</button>
                    <button class="btn btn-success" style="width:100%; margin-bottom:10px; margin-top:0;" id="story-compile-btn" onclick="compileStoryMovie()">Compile Synchronized Video</button>
                    
                    <div id="story-entries-container" style="flex-grow: 1; overflow-y: auto; gap: 8px; display: flex; flex-direction: column; max-height: 180px; padding-right: 2px; box-sizing:border-box;">
                        <!-- Story paragraphs will render here -->
                    </div>
                </div>

                <!-- Raw JSON Section -->
                <div style="display:flex; flex-direction:column; flex-grow:1; min-height:150px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <label style="margin-bottom:0;">Raw Config JSON</label>
                        <div style="display:flex; gap:6px;">
                            <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="formatRawJson()">Format</button>
                            <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="applyRawJsonToForm()">Apply</button>
                        </div>
                    </div>
                    <textarea id="raw-json" class="json-textarea" style="flex-grow:1; width:100%;"></textarea>
                </div>
            </div>
        </div>
    </div>

    <!-- Notification system -->
    <div class="notification" id="notification">Config saved successfully.</div>

    <script>
        let configData = {};
        let unsavedChanges = false;
        let selectedOllamaModel = '';

        window.addEventListener('DOMContentLoaded', () => {
            fetchConfigFromServer();
            fetchOllamaModels();
            
            setInterval(pollStatus, 3000);
            setInterval(updatePreview, 2500);
        });

        function showNotification(text, duration = 3000) {
            const el = document.getElementById('notification');
            el.innerText = text;
            el.style.display = 'block';
            setTimeout(() => {
                el.style.display = 'none';
            }, duration);
        }

        function fetchConfigFromServer() {
            fetch('/get_config')
                .then(res => {
                    if (!res.ok) throw new Error("Failed to load config");
                    return res.json();
                })
                .then(data => {
                    configData = data;
                    if(!configData.keyframes) configData.keyframes = {};
                    populateForm();
                    renderKeyframes();
                    updateRawJson();
                    unsavedChanges = false;
                    updateSaveButtonState();
                })
                .catch(err => {
                    console.error(err);
                    showNotification("Error loading configuration: " + err.message);
                });
        }

        function fetchOllamaModels() {
            fetch('/get_ollama_models')
                .then(res => res.json())
                .then(data => {
                    const select = document.getElementById('ai-model');
                    select.innerHTML = '';
                    if (data.models && data.models.length > 0) {
                        data.models.forEach(model => {
                            const opt = document.createElement('option');
                            opt.value = model;
                            opt.text = model;
                            select.appendChild(opt);
                        });
                        const preferred = ["dolphin3:8b", "mistral:7b-instruct", "llama3.2:3b", "qwen3:8b", "llama3.2", "llama3.2:latest"];
                        let foundPreferred = false;
                        for (let p of preferred) {
                            for (let opt of select.options) {
                                if (opt.value.includes(p)) {
                                    select.value = opt.value;
                                    foundPreferred = true;
                                    break;
                                }
                            }
                        }
                        if (!foundPreferred) {
                            select.value = data.models[0];
                        }
                    } else {
                        const opt = document.createElement('option');
                        opt.value = 'llama3.2';
                        opt.text = 'llama3.2 (Fallback)';
                        select.appendChild(opt);
                    }
                })
                .catch(err => {
                    console.error("Failed to load Ollama models", err);
                    const select = document.getElementById('ai-model');
                    select.innerHTML = '<option value="llama3.2">llama3.2 (Fallback)</option>';
                });
        }

        function pollStatus() {
            fetch('/status')
                .then(res => res.json())
                .then(data => {
                    const badge = document.getElementById('status-container');
                    if (data.running) {
                        badge.className = 'status-badge running';
                        badge.innerText = 'Running';
                    } else if (data.paused) {
                        badge.className = 'status-badge paused';
                        badge.innerText = 'Paused';
                    } else {
                        badge.className = 'status-badge';
                        badge.style.background = '#333';
                        badge.style.color = '#ccc';
                        badge.style.border = '1px solid #444';
                        badge.innerText = 'Idle';
                    }
                    
                    document.getElementById('preview-label').innerText = `Frame: ${data.frame} / ${data.total}`;
                })
                .catch(err => {
                    document.getElementById('status-container').className = 'status-badge';
                    document.getElementById('status-container').innerText = 'Offline';
                });
        }

        function updatePreview() {
            const img = document.getElementById('stream-preview-img');
            img.src = '/latest_frame?t=' + Date.now();
        }

        function toggleBuilderZoomFields() {
            const elMode = document.getElementById('zoom_mode');
            if (!elMode) return;
            const mode = elMode.value;
            if (mode === 'in') {
                document.getElementById('builder_zoom_in_params').style.display = 'block';
                document.getElementById('builder_zoom_out_params').style.display = 'none';
            } else {
                document.getElementById('builder_zoom_in_params').style.display = 'none';
                document.getElementById('builder_zoom_out_params').style.display = 'block';
            }
        }

        function populateForm() {
            const fields = [
                'model', 'prompt', 'negative_prompt', 'seed', 'cfg', 
                'steps', 'denoise', 'lora1', 'lora1_strength', 
                'lora2', 'lora2_strength', 'lora3', 'lora3_strength',
                'z_s', 'z_e', 'px_s', 'px_e', 'py_s', 'py_e', 'roll_mode',
                'zoom_mode', 'zoom_shrink', 'zoom_blur', 'zoom_strength', 'zoom_guidance_scale', 'zoom_num_inference_steps'
            ];
            
            fields.forEach(field => {
                const el = document.getElementById(field);
                if (el) {
                    if (configData[field] !== undefined) {
                        el.value = configData[field];
                    }
                }
            });

            const check = document.getElementById('use_motion_zoom');
            if (check) {
                check.checked = !!configData.use_motion_zoom;
            }
            toggleBuilderZoomFields();
        }

        function updateConfigField(field, val) {
            configData[field] = val;
            unsavedChanges = true;
            updateSaveButtonState();
            updateRawJson();
        }

        function renderKeyframes() {
            const container = document.getElementById('keyframes-container');
            container.innerHTML = '';
            
            const kfObj = configData.keyframes || {};
            const frames = Object.keys(kfObj).sort((a, b) => parseInt(b) - parseInt(a));
            
            document.getElementById('kf-count').innerText = `${frames.length} Keyframes`;
            
            if (frames.length === 0) {
                container.innerHTML = '<div style="text-align:center; padding:30px; color:var(--text-muted); font-size:13px; border:1px dashed var(--border-color); border-radius:8px;">No keyframes defined yet. Add some above!</div>';
                return;
            }
            
            frames.forEach(frame => {
                const kf = kfObj[frame];
                const card = document.createElement('div');
                card.className = 'kf-card';
                card.innerHTML = `
                    <div class="kf-card-header">
                        <span class="kf-badge">Frame ${frame}</span>
                        <button class="btn btn-danger" style="padding:2px 8px; font-size:11px;" onclick="deleteKeyframeLocal('${frame}')">Delete</button>
                    </div>
                    <div class="form-group">
                        <label>Prompt</label>
                        <textarea rows="2" style="font-size:12px; line-height:1.4;" oninput="updateKeyframeField('${frame}', 'prompt', this.value)">${kf.prompt || ''}</textarea>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Denoise</label>
                            <input type="number" step="0.05" min="0.1" max="1.0" value="${kf.denoise !== undefined ? kf.denoise : 0.5}" oninput="updateKeyframeField('${frame}', 'denoise', parseFloat(this.value) || 0.5)">
                        </div>
                        <div class="form-group">
                            <label>Seed Offset</label>
                            <input type="number" value="${kf.seed_offset !== undefined ? kf.seed_offset : 0}" oninput="updateKeyframeField('${frame}', 'seed_offset', parseInt(this.value) || 0)">
                        </div>
                    </div>
                `;
                container.appendChild(card);
            });
        }

        function updateKeyframeField(frame, field, value) {
            if (configData.keyframes && configData.keyframes[frame]) {
                configData.keyframes[frame][field] = value;
                unsavedChanges = true;
                updateSaveButtonState();
                updateRawJson();
            }
        }

        function addKeyframeLocal() {
            const frameInput = document.getElementById('new-kf-frame');
            const promptInput = document.getElementById('new-kf-prompt');
            const denoiseInput = document.getElementById('new-kf-denoise');
            const offsetInput = document.getElementById('new-kf-offset');
            
            const frame = frameInput.value.trim();
            if (!frame || isNaN(parseInt(frame))) {
                alert("Please enter a valid frame number.");
                return;
            }
            
            const frameStr = parseInt(frame).toString();
            const prompt = promptInput.value.trim();
            const denoise = parseFloat(denoiseInput.value) || 0.5;
            const seed_offset = parseInt(offsetInput.value) || 0;
            
            if (!configData.keyframes) configData.keyframes = {};
            
            configData.keyframes[frameStr] = {
                prompt: prompt,
                denoise: denoise,
                seed_offset: seed_offset
            };
            
            frameInput.value = '';
            promptInput.value = '';
            
            unsavedChanges = true;
            updateSaveButtonState();
            renderKeyframes();
            updateRawJson();
            showNotification(`Keyframe ${frameStr} added locally.`);
        }

        function deleteKeyframeLocal(frame) {
            if (configData.keyframes && configData.keyframes[frame]) {
                delete configData.keyframes[frame];
                unsavedChanges = true;
                updateSaveButtonState();
                renderKeyframes();
                updateRawJson();
                showNotification(`Keyframe ${frame} deleted locally.`);
            }
        }

        function updateRawJson() {
            const txt = document.getElementById('raw-json');
            txt.value = JSON.stringify(configData, null, 2);
        }

        function formatRawJson() {
            const txt = document.getElementById('raw-json');
            try {
                const parsed = JSON.parse(txt.value);
                configData = parsed;
                if(!configData.keyframes) configData.keyframes = {};
                txt.value = JSON.stringify(configData, null, 2);
                populateForm();
                renderKeyframes();
                showNotification("JSON Formatted.");
            } catch (e) {
                alert("Invalid JSON: " + e.message);
            }
        }

        function applyRawJsonToForm() {
            const txt = document.getElementById('raw-json');
            try {
                const parsed = JSON.parse(txt.value);
                configData = parsed;
                if(!configData.keyframes) configData.keyframes = {};
                populateForm();
                renderKeyframes();
                unsavedChanges = true;
                updateSaveButtonState();
                showNotification("JSON parsed and loaded into editor.");
            } catch (e) {
                alert("Invalid JSON: " + e.message);
            }
        }

        function updateSaveButtonState() {
            const btn = document.getElementById('save-btn');
            if (unsavedChanges) {
                btn.innerText = "Save to Server *";
                btn.className = "btn btn-primary";
                btn.style.boxShadow = "0 0 15px rgba(229,169,59,0.5)";
            } else {
                btn.innerText = "Saved to Server";
                btn.className = "btn";
                btn.style.boxShadow = "none";
            }
        }

        function saveConfigToServer() {
            const btn = document.getElementById('save-btn');
            btn.disabled = true;
            btn.innerText = "Saving...";
            
            const rawText = document.getElementById('raw-json').value;
            try {
                configData = JSON.parse(rawText);
            } catch (e) {
                alert("Cannot save: Invalid JSON in editor box. " + e.message);
                btn.disabled = false;
                updateSaveButtonState();
                return;
            }
            
            fetch('/save_config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(configData)
            })
            .then(res => {
                if(!res.ok) throw new Error("Failed to save configuration.");
                return res.json();
            })
            .then(data => {
                unsavedChanges = false;
                updateSaveButtonState();
                btn.disabled = false;
                showNotification("Configuration saved and reloaded on server.");
            })
            .catch(err => {
                alert("Error saving config: " + err.message);
                btn.disabled = false;
                updateSaveButtonState();
            });
        }

        function generateKeyframesAI() {
            const outline = document.getElementById('ai-outline').value.trim();
            const model = document.getElementById('ai-model').value;
            const instructions = document.getElementById('ai-instructions').value;
            const genBtn = document.getElementById('ai-gen-btn');
            const spinner = document.getElementById('ai-spinner');
            
            if (!outline) {
                alert("Please enter a rough outline or script first.");
                return;
            }
            
            genBtn.disabled = true;
            genBtn.innerText = "Generating with Ollama...";
            spinner.style.display = 'inline-block';
            
            fetch('/generate_keyframes_ollama', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    outline: outline,
                    model: model,
                    instructions: instructions
                })
            })
            .then(res => {
                if (!res.ok) {
                    return res.json().then(errData => {
                        throw new Error(errData.error || "Ollama generation failed.");
                    });
                }
                return res.json();
            })
            .then(data => {
                if (data.status === 'ok' && data.keyframes) {
                    if (!configData.keyframes) configData.keyframes = {};
                    
                    const newKfs = data.keyframes;
                    let count = 0;
                    for (const frame in newKfs) {
                        configData.keyframes[frame] = newKfs[frame];
                        count++;
                    }
                    
                    unsavedChanges = true;
                    updateSaveButtonState();
                    renderKeyframes();
                    updateRawJson();
                    showNotification(`AI generated & merged ${count} keyframes successfully!`);
                } else {
                    throw new Error(data.error || "Unknown error during keyframe generation.");
                }
            })
            .catch(err => {
                alert("Ollama Error: " + err.message);
            })
            .finally(() => {
                genBtn.disabled = false;
                genBtn.innerText = "Generate & Merge Keyframes";
                spinner.style.display = 'none';
            });
        }

        function previewStory() {
            const btn = document.getElementById('story-prev-btn');
            const saveBtn = document.getElementById('story-save-btn');
            const spinner = document.getElementById('story-spinner');
            const container = document.getElementById('story-entries-container');
            
            btn.disabled = true;
            btn.innerText = "Planning Script...";
            spinner.style.display = 'inline-block';
            container.innerHTML = '<div style="text-align:center; font-size:12px; color:var(--text-muted); padding:10px;">Running text LLM to draft the diary script based on keyframe prompts...</div>';
            
            fetch('/preview_story', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(res => {
                if(!res.ok) throw new Error("Story script preview failed.");
                return res.json();
            })
            .then(data => {
                if (data.status === 'ok' && data.diary) {
                    renderDiaryScriptEditor(data.diary);
                    saveBtn.style.display = 'inline-flex';
                    showNotification("Script drafted successfully! You can edit any frame's text below.");
                } else {
                    throw new Error(data.error || "Failed to generate story preview.");
                }
            })
            .catch(err => {
                alert("Storyteller Preview Error: " + err.message);
                container.innerHTML = `<div style="text-align:center; font-size:12px; color:var(--error); padding:10px;">Error: ${err.message}</div>`;
            })
            .finally(() => {
                btn.disabled = false;
                btn.innerText = "Preview Script";
                spinner.style.display = 'none';
            });
        }

        function renderDiaryScriptEditor(diary) {
            const container = document.getElementById('story-entries-container');
            container.innerHTML = '';
            diary.forEach(item => {
                const entryEl = document.createElement('div');
                entryEl.style.background = 'var(--bg-secondary)';
                entryEl.style.padding = '8px';
                entryEl.style.borderRadius = '6px';
                entryEl.style.border = '1px solid var(--border-color)';
                entryEl.style.fontSize = '12px';
                
                entryEl.innerHTML = `
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-weight:bold; color:var(--accent);">Frame ${item.frame}</span>
                        <span style="font-size:10px; color:var(--text-muted); font-style:italic;">Tension: ${item.tension.toFixed(1)} | Sanity: ${item.sanity.toFixed(0)}%</span>
                    </div>
                    <textarea class="story-edit-input" data-frame="${item.frame}" rows="2" style="width:100%; background:var(--bg-primary); color:#fff; border:1px solid var(--border-color); padding:4px; border-radius:4px; font-size:11px; font-family:inherit; resize:vertical; box-sizing:border-box; margin-top:3px;">${item.story}</textarea>
                `;
                container.appendChild(entryEl);
            });
        }

        function saveStoryScript() {
            const saveBtn = document.getElementById('story-save-btn');
            const inputs = document.querySelectorAll('.story-edit-input');
            const diary = [];
            
            inputs.forEach(input => {
                diary.push({
                    frame: parseInt(input.getAttribute('data-frame')),
                    story: input.value
                });
            });
            
            saveBtn.disabled = true;
            saveBtn.innerText = "Saving...";
            
            fetch('/save_diary', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ diary: diary })
            })
            .then(res => {
                if(!res.ok) throw new Error("Failed to save script.");
                return res.json();
            })
            .then(data => {
                if (data.status === 'ok') {
                    showNotification("Story script saved successfully!");
                } else {
                    throw new Error(data.error);
                }
            })
            .catch(err => {
                alert("Save Script Error: " + err.message);
            })
            .finally(() => {
                saveBtn.disabled = false;
                saveBtn.innerText = "Save Script";
            });
        }

        function compileStoryMovie() {
            const btn = document.getElementById('story-compile-btn');
            const bgm = document.getElementById('story-bgm').value;
            const spinner = document.getElementById('story-spinner');
            
            btn.disabled = true;
            btn.innerText = "Compiling Video...";
            spinner.style.display = 'inline-block';
            
            fetch(`/compile_story_movie?bgm=${encodeURIComponent(bgm)}`)
            .then(res => {
                if (res.redirected) {
                    window.open(res.url, '_blank');
                    showNotification("Compilation complete! Video opened in a new tab.");
                } else if (res.ok) {
                    return res.text().then(url => {
                        window.open(url, '_blank');
                        showNotification("Compilation complete!");
                    });
                } else {
                    throw new Error("Compilation endpoint failed.");
                }
            })
            .catch(err => {
                alert("Movie Compilation Error: " + err.message);
            })
            .finally(() => {
                btn.disabled = false;
                btn.innerText = "Compile Synchronized Video";
                spinner.style.display = 'none';
            });
        }

        function generateStoryDiary() {
            const btn = document.getElementById('story-btn');
            const spinner = document.getElementById('story-spinner');
            const container = document.getElementById('story-entries-container');
            
            btn.disabled = true;
            btn.innerText = "Analyzing & Synthesizing...";
            spinner.style.display = 'inline-block';
            container.innerHTML = '<div style="text-align:center; font-size:12px; color:var(--text-muted); padding:10px;">Running vision analysis, story composition, and Kokoro TTS speech generation... (this may take 1-2 minutes)</div>';
            
            fetch('/generate_story', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(res => {
                if(!res.ok) throw new Error("Story teller pipeline failed.");
                return res.json();
            })
            .then(data => {
                if (data.status === 'ok' && data.diary) {
                    renderDiaryScriptEditor(data.diary);
                    document.getElementById('story-save-btn').style.display = 'inline-flex';
                    showNotification("Story diary and voice narration tracks compiled!");
                } else {
                    throw new Error(data.error || "Failed to compile story.");
                }
            })
            .catch(err => {
                alert("Storyteller Error: " + err.message);
                container.innerHTML = `<div style="text-align:center; font-size:12px; color:var(--error); padding:10px;">Error: ${err.message}</div>`;
            })
            .finally(() => {
                btn.disabled = false;
                btn.innerText = "Compile Story & Audio Tracks";
                spinner.style.display = 'none';
            });
        }
    </script>
</body>
</html>
"""

# ============================================================
# UI TEMPLATE
# ============================================================
HTML_UI = """
<!DOCTYPE html><html><head><title>spaceexplorer2 PIL AI Director</title>
<style>
    body { 
        margin: 0; 
        background-color: #08090c; 
        background-image: 
            radial-gradient(at 50% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
            linear-gradient(rgba(18, 24, 38, 0.5) 1px, transparent 1px),
            linear-gradient(90deg, rgba(18, 24, 38, 0.5) 1px, transparent 1px);
        background-size: 100% 100%, 20px 20px, 20px 20px;
        color: #e2e8f0; 
        font-family: 'Courier New', Courier, monospace; 
        display: flex; 
        height: 100vh; 
        overflow: hidden; 
    }
    .column { 
        padding: 15px; 
        box-sizing: border-box; 
        overflow-y: auto; 
        border-right: 2px solid #1e293b; 
        box-shadow: inset 0 0 15px rgba(0,0,0,0.5);
    }
    .left { padding: 12px; width: 30%; background: rgba(15, 23, 42, 0.95); border-left: 4px solid #0ea5e9; }
    .center { 
        width: 40%; 
        background: rgba(2, 6, 23, 0.98); 
        text-align: center; 
        display: flex; 
        flex-direction: column; 
        align-items: center;
        border-left: 2px solid #1e293b;
        border-right: 2px solid #1e293b;
    }
    .right { padding: 12px; width: 30%; background: rgba(15, 23, 42, 0.95); border-right: 4px solid #38bdf8; }
    label { font-size: 11px; color: #888; font-weight: bold; display: block; margin-top: 8px; }
    input, select, textarea { width: 96%; background: #0f172a; color: #38bdf8; border: 1px solid #334155; padding: 6px; border-radius: 4px; font-size: 12px; margin-top: 4px; margin-right: 10px; font-family: inherit; }
    button { width: 98%; padding: 8px; margin-top: 8px; cursor: pointer; border: none; border-radius: 4px; font-weight: bold; transition: all 0.2s; font-family: inherit; }
    .btn-green { background: #059669; color: white; box-shadow: 0 0 10px rgba(5, 150, 105, 0.3); }
    .btn-blue { background: #0284c7; color: white; box-shadow: 0 0 10px rgba(2, 132, 199, 0.3); }
    .btn-orange { background: #d97706; color: white; box-shadow: 0 0 10px rgba(217, 119, 6, 0.3); }
    #preview { max-width: 100%; max-height: 100%; border: none; border-radius: 22px; }
    .thumb-strip { display: flex; justify-content: center; gap: 10px; margin-top: 10px; min-height: 100px;}
    .thumb-strip img { width: 18%; height: auto; border: 2px solid #333; border-radius: 4px; opacity: 0.6; }
    .tag { background: blue; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin: 2px; display: inline-block; }
    .kf-item { background: #1c1c21; padding: 5px; margin-top: 5px; border-radius: 4px; border-left: 3px solid #8b5cf6; font-size: 10px; text-align: left;}
</style></head>
<body>
    <div class="column left">
        <h3>FlaskArchitect's spaceexplorer2 Engine Config</h3>
        <div style="margin-bottom: 12px; display: flex; gap: 5px;">
            <a href="/json_builder" target="_blank" style="flex: 1; text-align: center; color: #fff; background: #8b5cf6; padding: 6px; border-radius: 4px; text-decoration: none; font-weight: bold; font-size: 11px; transition: background 0.2s;" onmouseover="this.style.background='#7c3aed'" onmouseout="this.style.background='#8b5cf6'">Open JSON Keyframe Builder</a>
            <a href="/templates_editor" target="_blank" style="flex: 1; text-align: center; color: #fff; background: #3b82f6; padding: 6px; border-radius: 4px; text-decoration: none; font-weight: bold; font-size: 11px; transition: background 0.2s;" onmouseover="this.style.background='#2563eb'" onmouseout="this.style.background='#3b82f6'">Open Templates Editor</a>
        </div>
        <label>Base Prompt</label><textarea id="prompt" rows="3">Highly detailed Centered Science fiction image of a star-gate with semi transparent space creatures swimming in space similar to mythical sea monsters, surrounded with space, stars, planets, nebula, dust and space debris <lora:more_details:.8>, 8k resolution, dramatic clouds in the atmosphere</textarea>
        
        <!-- Ollama Prompt Enhancer integration -->
        <label>AI Prompt Enhancer (Ollama)</label>
        <div style="display:flex; gap:5px; margin-top:4px; margin-bottom:8px;">
            <select id="ollama_model" style="flex:1; margin-top:0;">
                <option value="None">Loading Ollama models...</option>
            </select>
            <button class="btn-blue" onclick="enhancePrompt()" style="width:auto; margin-top:0; padding: 4px 8px; font-size:11px;">Enhance</button>
        </div>

        <label>Negative Prompt</label><textarea id="neg_prompt" rows="2">low quality, blurry, bare breasts, nipples, large breasts, NSFW, deformed fingers</textarea>
        <label>Model</label><select id="model">{% for m in MODELS %}<option {% if m == CURRENT_MODEL %}selected{% endif %}>{{m}}</option>{% endfor %}</select>
        
        <label>LoRA 1 & Strength</label>
        <div style="display:flex; gap:5px; margin-top:4px;">
            <select id="lora1" style="flex:2; margin-top:0;">{% for l in LORAS %}<option {% if l == CURRENT_LORA1 %}selected{% endif %}>{{l}}</option>{% endfor %}</select>
            <input type="number" id="lora1_str" value="0.8" step="0.1" min="0" max="2.0" style="flex:1; margin-top:0;">
        </div>

        <label>LoRA 2 & Strength</label>
        <div style="display:flex; gap:5px; margin-top:4px;">
            <select id="lora2" style="flex:2; margin-top:0;">{% for l in LORAS %}<option {% if l == CURRENT_LORA2 %}selected{% endif %}>{{l}}</option>{% endfor %}</select>
            <input type="number" id="lora2_str" value="0.8" step="0.1" min="0" max="2.0" style="flex:1; margin-top:0;">
        </div>

        <label>LoRA 3 & Strength</label>
        <div style="display:flex; gap:5px; margin-top:4px;">
            <select id="lora3" style="flex:2; margin-top:0;">{% for l in LORAS %}<option {% if l == CURRENT_LORA3 %}selected{% endif %}>{{l}}</option>{% endfor %}</select>
            <input type="number" id="lora3_str" value="0.8" step="0.1" min="0" max="2.0" style="flex:1; margin-top:0;">
        </div>
        
        <!-- Row 1 -->
        <div style="display:flex; gap:5px;">
            <div style="flex:1;">
                <label>Seed</label>
                <input type="number" id="seed" value="{{ current_seed }}">
            </div>

            <div style="flex:1;">
                <label>Steps</label>
                <input type="number" id="steps" value="{{ default_steps }}">
            </div>

            <div style="flex:1;">
                <label>CFG</label>
                <input type="number" id="cfg" value="{{ default_cfg }}" step="0.1">
            </div>
        </div>

        <!-- Row 2 -->
        <div style="display:flex; gap:5px; margin-top:5px;">
            <div style="flex:1;">
                <label>Denoise</label>
                <input type="number" id="denoise" step="0.01" value="{{ denoise_current }}">
            </div>

            <div style="flex:1;">
                <label>Frames</label>
                <input type="number" id="frames" value="{{ frames_current }}" oninput="updateZoomRateLabel()">
            </div>
        </div>
<button class="btn-green" onclick="window.open('/media', '_blank')">
    MAKE VIDEOS
</button>
        <hr style="border:0; border-top:1px solid #333; margin:20px 0;">
        <h3>Motion Zoom</h3>
        <label><input type="checkbox" id="use_zoom" checked style="width:auto;"> ENABLE ZOOM</label>
        <label><input type="checkbox" id="use_caption" style="width:auto;"> SHOW METADATA CAPTION</label>
        
        <!-- Metadata Caption Style Customization -->
        <div style="background: #1c1c21; padding: 8px; border-radius: 4px; margin-top: 5px; border: 1px solid #333; margin-bottom: 5px;">
            <div style="display:flex; gap:5px;">
                <div style="flex:1;">
                    <label>Font Size</label>
                    <input type="number" id="cap_font_size" value="{{caption_font_size}}">
                </div>
                <div style="flex:1;">
                    <label>Loc X (px)</label>
                    <input type="number" id="cap_x" value="{{caption_x}}">
                </div>
                <div style="flex:1;">
                    <label>Loc Y (px)</label>
                    <input type="number" id="cap_y" value="{{caption_y}}">
                </div>
            </div>
            <div style="display:flex; gap:5px; margin-top:5px; align-items:center;">
                <div style="flex:3;">
                    <label>BG Color (R, G, B)</label>
                    <div style="display:flex; gap:2px;">
                        <input type="number" id="cap_bg_r" value="{{caption_bg_r}}" min="0" max="255" style="padding: 4px; margin-right: 0;">
                        <input type="number" id="cap_bg_g" value="{{caption_bg_g}}" min="0" max="255" style="padding: 4px; margin-right: 0;">
                        <input type="number" id="cap_bg_b" value="{{caption_bg_b}}" min="0" max="255" style="padding: 4px; margin-right: 0;">
                    </div>
                </div>
                <div style="flex:1;">
                    <label>Opacity</label>
                    <input type="number" id="cap_bg_a" value="{{caption_bg_a}}" step="0.1" min="0" max="1" style="padding: 4px;">
                </div>
            </div>
        </div>
        <label>Zoom Mode</label>
        <select id="zoom_mode" onchange="toggleZoomFields()">
            <option value="in">Zoom In (Crop & Pan)</option>
            <option value="out">Zoom Out (Blur, Shrink & Extend)</option>
        </select>

        <div id="zoom_in_params">
            <label>Zoom Start/End <span id="zoom_rate_label" style="font-size: 0.85em; color: #38bdf8; font-weight: 600; margin-left: 8px;"></span></label>
            <div style="display:flex; gap:5px;">
                <input type="number" id="zs" value="{{ zoom_start }}" step="0.01" oninput="updateZoomRateLabel()">
                <input type="number" id="ze" value="{{ zoom_end }}" step="0.01" oninput="updateZoomRateLabel()">
            </div>
            <label>Yaw S/E</label><div style="display:flex; gap:5px;"><input type="number" id="pxs" value="0.5" step="0.01"><input type="number" id="pxe" value="0.5" step="0.01"></div>
            <label>Pitch S/E</label><div style="display:flex; gap:5px;"><input type="number" id="pys" value="0.5" step="0.01"><input type="number" id="pye" value="0.5" step="0.01"></div>
        </div>

        <div id="zoom_out_params" style="display:none; background: #1c1c21; padding: 8px; border-radius: 4px; margin-top: 5px; border: 1px solid #333; margin-bottom: 5px;">
            <div style="display:flex; gap:5px;">
                <div style="flex:1;">
                    <label>Shrink (px)</label>
                    <input type="number" id="zoom_shrink" value="10">
                </div>
                <div style="flex:1;">
                    <label>Blur Radius</label>
                    <input type="number" id="zoom_blur" value="8">
                </div>
            </div>
        </div>

        <div id="zoom_shared_params" style="background: #1c1c21; padding: 8px; border-radius: 4px; margin-top: 5px; border: 1px solid #333; margin-bottom: 5px;">
            <div style="display:flex; gap:5px;">
                <div style="flex:1;">
                    <label>Zoom Strength</label>
                    <input type="number" id="zoom_strength" value="{{ zoom_strength }}" step="0.01">
                </div>
                <div style="flex:1;">
                    <label>Zoom CFG Scale</label>
                    <input type="number" id="zoom_guidance_scale" value="{{ zoom_guidance_scale }}" step="0.1">
                </div>
            </div>
        </div>
        <label>Ship Roll</label>
        <select id="roll_mode">
            <option value="none">Level (Stop)</option>
            <option value="right">Roll Right</option>
            <option value="left">Roll Left</option>
        </select>

        <hr style="border:0; border-top:1px solid #333; margin:10px 0;">
        <h3 style="margin-top:10px; margin-bottom:5px;">AI Visual Director (moondream)</h3>
        <label><input type="checkbox" id="use_director" style="width:auto;"> ENABLE VISUAL DIRECTOR</label>
        <label>Decision Interval (Frames)</label>
        <input type="number" id="director_interval" value="25" min="5" max="200">
        <label>Description Model</label>
        <select id="director_model">
            <option value="LlaVa:latest">LlaVa:latest</option>
        </select>
        <label>Latest Vision Observation</label>
        <div id="latest_vision_desc" style="background:#1c1c21; border:1px solid #333; padding:8px; border-radius:4px; font-size:11px; color:#aaa; min-height:18px; margin-top:4px; word-wrap:break-word;">No vision data yet.</div>

        <hr style="border:0; border-top:1px solid #333; margin:10px 0;">
        <h3 style="margin-top:10px; margin-bottom:5px;">Morphing & Video Options</h3>
        <label><input type="checkbox" id="use_prompt_interpolation" style="width:auto;"> ENABLE PROMPT INTERPOLATION</label>
        <label><input type="checkbox" id="use_video_interpolation" style="width:auto;"> SMOOTH VIDEO INTERPOLATION (RIFE)</label>
        
        <hr style="border:0; border-top:1px solid #333; margin:10px 0;">
        <h3 style="margin-top:10px; margin-bottom:5px;">Feedback Stabilizer</h3>
        <div style="display:flex; gap:5px;">
            <div style="flex:1;">
                <label>Color</label>
                <input type="number" id="fb_color" step="0.01" value="{{ feedback_color_boost }}">
            </div>
            <div style="flex:1;">
                <label>Contrast</label>
                <input type="number" id="fb_contrast" step="0.01" value="{{ feedback_contrast_boost }}">
            </div>
            <div style="flex:1;">
                <label>Sharpness</label>
                <input type="number" id="fb_sharpness" step="0.05" value="{{ feedback_sharpness_boost }}">
            </div>
        </div>
        
        <div style="margin: 10px 0; display: flex; align-items: center; gap: 8px;">
            <input type="checkbox" id="pause_ui_sync" style="width: auto; margin: 0; cursor: pointer;">
            <label for="pause_ui_sync" style="margin: 0; cursor: pointer; font-weight: bold; color: #f59e0b; font-size: 11px;">Pause Parameter Syncing (Manual edit mode)</label>
        </div>
        <button class="btn-blue" onclick="update(this)">UPDATE ENGINE</button>
    </div>

    <div class="column center">
        <h2 id="status_text">IDLE</h2>
        <div id="injections"></div>
        <div id="observation_window_frame" style="position: relative; display: inline-block; padding: 18px; background: linear-gradient(135deg, #4d3d2d, #291f16, #3d3023, #1f160e); border: 4px solid #1a130e; border-radius: 40px; box-shadow: 0 15px 35px rgba(0,0,0,0.9), inset 0 0 25px rgba(0,0,0,0.8); margin-top: 15px;">
            <div id="preview_container" style="position: relative; overflow: hidden; border-radius: 24px; border: 3px solid #1c1510; box-shadow: inset 0 0 20px rgba(0,0,0,0.95); background: #020205; width: 340px; height: 512px; display: flex; justify-content: center; align-items: center;">
                <img id="preview" src="" style="width: 100%; height: 100%; display: block; border-radius: 22px; object-fit: cover;">
                <div id="overlay_logo_wrapper" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></div>
                <div id="overlay_stencil_wrapper" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></div>
                <div id="overlay_seed_wrapper" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></div>
            </div>
            <!-- Metallic rivets on the aged bronze frame to look like a spaceship window -->
            <div style="position: absolute; top: 6px; left: 50%; transform: translateX(-50%); width: 8px; height: 8px; border-radius: 50%; background: #6e573f; box-shadow: 1px 1px 1px #000, inset -1px -1px 2px #1f160e;"></div>
            <div style="position: absolute; bottom: 6px; left: 50%; transform: translateX(-50%); width: 8px; height: 8px; border-radius: 50%; background: #6e573f; box-shadow: 1px 1px 1px #000, inset -1px -1px 2px #1f160e;"></div>
            <div style="position: absolute; left: 6px; top: 50%; transform: translateY(-50%); width: 8px; height: 8px; border-radius: 50%; background: #6e573f; box-shadow: 1px 1px 1px #000, inset -1px -1px 2px #1f160e;"></div>
            <div style="position: absolute; right: 6px; top: 50%; transform: translateY(-50%); width: 8px; height: 8px; border-radius: 50%; background: #6e573f; box-shadow: 1px 1px 1px #000, inset -1px -1px 2px #1f160e;"></div>
            <div style="position: absolute; top: 12px; left: 12px; width: 8px; height: 8px; border-radius: 50%; background: #6e573f; box-shadow: 1px 1px 1px #000, inset -1px -1px 2px #1f160e;"></div>
            <div style="position: absolute; top: 12px; right: 12px; width: 8px; height: 8px; border-radius: 50%; background: #6e573f; box-shadow: 1px 1px 1px #000, inset -1px -1px 2px #1f160e;"></div>
            <div style="position: absolute; bottom: 12px; left: 12px; width: 8px; height: 8px; border-radius: 50%; background: #6e573f; box-shadow: 1px 1px 1px #000, inset -1px -1px 2px #1f160e;"></div>
            <div style="position: absolute; bottom: 12px; right: 12px; width: 8px; height: 8px; border-radius: 50%; background: #6e573f; box-shadow: 1px 1px 1px #000, inset -1px -1px 2px #1f160e;"></div>
        </div>
        <div id="live_prompt_container" style="background: #18181b; border: 1px solid #27272a; border-radius: 4px; padding: 10px; font-family: monospace; font-size: 12px; color: #38bdf8; text-align: left; max-width: 98%; margin: 10px auto 0 auto; white-space: pre-wrap; word-break: break-all; box-sizing: border-box;">
            <span style="color: #a1a1aa; font-weight: bold; display: block; margin-bottom: 4px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;">Active Inference Prompt:</span>
            <span id="live_prompt_display">Waiting for engine to start...</span>
        </div>
        <div id="thumb_strip" class="thumb-strip"></div>
        <div id="feedback" style="margin-top:10px; color:#10b981; font-weight:bold; height:20px;"></div>
    </div>

    <div class="column right">
        <h3>Director Tools -rendering takes about 3min- </h3>
        <button class="btn-green" onclick="ctrl('start', this)">NEW PRODUCTION</button>
        <button class="btn-blue" onclick="ctrl('resume', this)">RESUME SESSION</button>
        <button class="btn-orange" onclick="ctrl('pause', this)">PAUSE / UNPAUSE</button>
        <label>Inject Keyword</label><input id="inj" placeholder="colorful"><button class="btn-blue" onclick="inject(this)">INJECT</button>
        
        <hr style="border:0; border-top:1px solid #333; margin:20px 0;">
        <h3>Temporary Caption</h3>
        <label>Caption Text</label>
        <textarea id="cap_text" rows="2" placeholder="Enter single line caption"></textarea>
        <label>Caption Font</label>
        <select id="cap_font">
            <option value="Default">Default PIL Font</option>
            {% for f in FONTS %}
            <option value="{{ f }}">{{ f }}</option>
            {% endfor %}
        </select>
        
        <!-- Temporary Caption Style Customization -->
        <div style="background: #1c1c21; padding: 8px; border-radius: 4px; margin-top: 5px; border: 1px solid #333; margin-bottom: 5px;">
            <div style="display:flex; gap:5px;">
                <div style="flex:1;">
                    <label>Font Size</label>
                    <input type="number" id="temp_cap_font_size" value="{{temp_caption_font_size}}">
                </div>
                <div style="flex:1;">
                    <label>Loc X (px)</label>
                    <input type="number" id="temp_cap_x" value="{{temp_caption_x}}">
                </div>
                <div style="flex:1;">
                    <label>Loc Y (px)</label>
                    <input type="number" id="temp_cap_y" value="{{temp_caption_y}}">
                </div>
            </div>
            <div style="display:flex; gap:5px; margin-top:5px; align-items:center;">
                <div style="flex:3;">
                    <label>BG Color (R, G, B)</label>
                    <div style="display:flex; gap:2px;">
                        <input type="number" id="temp_cap_bg_r" value="{{temp_caption_bg_r}}" min="0" max="255" style="padding: 4px; margin-right: 0;">
                        <input type="number" id="temp_cap_bg_g" value="{{temp_caption_bg_g}}" min="0" max="255" style="padding: 4px; margin-right: 0;">
                        <input type="number" id="temp_cap_bg_b" value="{{temp_caption_bg_b}}" min="0" max="255" style="padding: 4px; margin-right: 0;">
                    </div>
                </div>
                <div style="flex:1;">
                    <label>Opacity</label>
                    <input type="number" id="temp_cap_bg_a" value="{{temp_caption_bg_a}}" step="0.1" min="0" max="1" style="padding: 4px;">
                </div>
            </div>
        </div>
        <button class="btn-blue" onclick="setCaption(this)">INSERT CAPTION (5 FRAMES)</button>

        <hr style="border:0; border-top:1px solid #333; margin:20px 0;">
        <h3>Drag-and-Drop Logo Overlay</h3>
        <label>Upload Logo (transparent PNG)</label>
        <div style="display:flex; gap:5px;">
            <input type="file" id="logo_file_input" accept="image/png" style="flex:1;">
            <button class="btn-blue" onclick="uploadLogo(this)" style="width:auto; margin-top:4px;">Upload</button>
        </div>
        <label>Select Logo</label>
        <select id="logo_select" onchange="changeLogo(this.value)">
            <option value="None">None</option>
            {% for l in LOGOS %}
            <option value="{{ l }}" {% if l == CURRENT_LOGO %}selected{% endif %}>{{ l }}</option>
            {% endfor %}
        </select>
        <label>Logo Width (px)</label>
        <input type="number" id="logo_width_input" value="100" min="10" max="1000" oninput="changeLogoSize()" onchange="saveLogoLocally()">
        <label>Opacity</label>
        <input type="range" id="logo_opacity_slider" min="0" max="1" step="0.05" value="1.0" oninput="updateLogoOpacity(this.value)" onchange="saveLogoLocally()">
        <div id="logo_control_buttons" style="display:none; margin-top: 8px; gap: 4px; flex-wrap: wrap;">
            <button class="btn-blue" onclick="saveLogoLocally()" style="flex: 1; min-width: 120px;">Save Locally Only</button>
            <button class="btn-green" onclick="saveLogoToServer()" style="flex: 1; min-width: 120px;">Save to Server</button>
            <button class="btn-orange" onclick="cancelLogoPlacement()" style="flex: 1; min-width: 120px;">Clear Logo</button>
        </div>

        <hr style="border:0; border-top:1px solid #333; margin:20px 0;">
        <h3>Guides & Stencils</h3>
        
        <!-- Local Stencil Section -->
        <div style="background: #1c1c21; padding: 10px; border-radius: 6px; margin-top: 5px; border: 1px solid #333; margin-bottom: 10px;">
            <h4 style="margin: 0 0 8px 0; color: #8b5cf6;">Local Stencil (Browser Only)</h4>
            <div style="margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
                <input type="checkbox" id="show_stencil" style="width: auto; margin: 0; cursor: pointer;" onchange="toggleStencilDisplay()">
                <label for="show_stencil" style="margin: 0; cursor: pointer; color: #d1d1d1; font-size: 11px;">Enable Stencil Overlay</label>
            </div>
            
            <label>Select Stencil Image</label>
            <select id="stencil_select" onchange="changeStencil(this.value)">
                <option value="None">None</option>
                {% for l in LOGOS %}
                <option value="{{ l }}">{{ l }}</option>
                {% endfor %}
            </select>
            
            <div style="display:flex; gap:5px; margin-top: 5px;">
                <div style="flex:1;">
                    <label>Width (px)</label>
                    <input type="number" id="stencil_width_input" value="100" min="10" max="1000" oninput="changeStencilSize()">
                </div>
                <div style="flex:1;">
                    <label>X (px)</label>
                    <input type="number" id="stencil_x_input" value="0" oninput="changeStencilCoords()">
                </div>
                <div style="flex:1;">
                    <label>Y (px)</label>
                    <input type="number" id="stencil_y_input" value="0" oninput="changeStencilCoords()">
                </div>
            </div>
            
            <label style="margin-top: 5px;">Opacity</label>
            <input type="range" id="stencil_opacity_slider" min="0" max="1" step="0.05" value="0.5" oninput="updateStencilOpacity(this.value)">
        </div>

        <!-- AI Seed Guide Section -->
        <div style="background: #1c1c21; padding: 10px; border-radius: 6px; margin-top: 5px; border: 1px solid #333; margin-bottom: 10px;">
            <h4 style="margin: 0 0 8px 0; color: #8b5cf6;">AI Seed Guide (Server Feedback)</h4>
            <div style="margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
                <input type="checkbox" id="use_seed_guide" style="width: auto; margin: 0; cursor: pointer;" onchange="toggleSeedGuide(this.checked)">
                <label for="use_seed_guide" style="margin: 0; cursor: pointer; color: #d1d1d1; font-size: 11px;">Enable AI Seed Guide</label>
            </div>
            
            <label>Select Guide Image</label>
            <select id="seed_select" onchange="changeSeedGuide(this.value)">
                <option value="None">None</option>
                {% for l in LOGOS %}
                <option value="{{ l }}" {% if l == CURRENT_SEED_GUIDE %}selected{% endif %}>{{ l }}</option>
                {% endfor %}
            </select>
            
            <div style="display:flex; gap:5px; margin-top: 5px;">
                <div style="flex:1;">
                    <label>Width (px)</label>
                    <input type="number" id="seed_width_input" value="100" min="10" max="1000" oninput="changeSeedGuideSize()">
                </div>
                <div style="flex:1;">
                    <label>X (px)</label>
                    <input type="number" id="seed_x_input" value="0" oninput="changeSeedGuideCoords()">
                </div>
                <div style="flex:1;">
                    <label>Y (px)</label>
                    <input type="number" id="seed_y_input" value="0" oninput="changeSeedGuideCoords()">
                </div>
            </div>
            
            <label style="margin-top: 5px;">Opacity</label>
            <input type="range" id="seed_opacity_slider" min="0" max="1" step="0.01" value="0.10" oninput="updateSeedGuideOpacity(this.value)">
            
            <div style="margin-top: 8px; display: flex;">
                <button class="btn-green" onclick="saveSeedGuideToServer()" style="flex: 1; margin-top: 5px;">Set & Send to Server</button>
            </div>
        </div>

        <!-- LlaVa Vision Scanner Section -->
        <hr style="border:0; border-top:1px solid #333; margin:20px 0;">
        <h3>LlaVa Vision Scanner</h3>
        <div style="background: #1c1c21; padding: 10px; border-radius: 6px; margin-top: 5px; border: 1px solid #333; margin-bottom: 10px;">
            <label>Scan Target Image</label>
            <select id="scanner_image_select" style="width:96%;">
                <option value="latest_frame">Latest Generated Frame</option>
                {% for l in LOGOS %}
                <option value="{{ l }}">{{ l }} (Overlay)</option>
                {% endfor %}
            </select>
            
            <label>Vision Model</label>
            <select id="scanner_model_select" style="width:96%;">
                <option value="LlaVa:latest">LlaVa:latest</option>
                <option value="moondream:latest">moondream:latest</option>
            </select>
            
            <label>Custom Query / Prompt</label>
            <textarea id="scanner_prompt" rows="2" style="width: 96%;">Describe this image. Be specific about objects and details.</textarea>
            
            <button class="btn-green" onclick="runVisionScanner(null, null)" style="margin-top: 8px; width: 98%;">Run Vision Scan (Full Image)</button>
            
            <p style="font-size: 10px; color: #aaa; margin-top: 8px; font-style: italic; line-height: 1.3;">
                💡 Tip: Click anywhere directly on the main preview window to scan a 150px cropped detail around that spot!
            </p>
            
            <div id="scanner_status" style="margin-top: 8px; color: #3b82f6; font-size: 11px; font-weight: bold; display: none;">Scanning image...</div>
            
            <label style="margin-top: 8px; color: #8b5cf6;">Scanner Description Output</label>
            <div id="scanner_result" style="background: #0c0c0e; border: 1px solid #333; border-radius: 4px; padding: 8px; font-size: 11px; color: #fff; min-height: 50px; margin-top: 4px; overflow-y: auto; text-align: left; white-space: pre-wrap;">No scan data yet. Click preview or click button to analyze.</div>
        </div>

        <hr style="border:0; border-top:1px solid #333; margin:20px 0;">
        <h3>Teleport (One-Time)</h3>
        <label>Upload Planet Image</label>
        <input type="file" id="tele_file" accept="image/*">
        <button class="btn-orange" onclick="teleport(this)">TELEPORT NOW</button>

        <hr style="border:0; border-top:1px solid #333; margin:20px 0;">
        <h3>Keyframe Editor</h3>
        <label>Frame</label>
        <input type="number" id="kf_f" value="0">

        <label>Prompt Override</label>
        <textarea id="kf_p" rows="2" placeholder="optional prompt"></textarea>

        <label>Denoise</label>
        <input type="number" id="kf_d" step="0.05" value="0.5">

        <label>Seed Offset</label>
        <input type="number" id="kf_s" value="0">

        <button class="btn-blue" onclick="addKF(this)">ADD KEYFRAME</button>
        <div id="kf_list"></div>
    </div>

    <script>
        function showFeedback(t){ const f=document.getElementById('feedback'); f.innerText=t; setTimeout(()=>f.innerText='',7000); }
        function updateZoomRateLabel() {
            const zsEl = document.getElementById('zs');
            const zeEl = document.getElementById('ze');
            const framesEl = document.getElementById('frames');
            const labelEl = document.getElementById('zoom_rate_label');
            if (!zsEl || !zeEl || !framesEl || !labelEl) return;

            const zs = parseFloat(zsEl.value) || 1.0;
            const ze = parseFloat(zeEl.value) || 1.1;
            const frames = parseInt(framesEl.value) || 10;
            const mode = document.getElementById('zoom_mode').value;

            if (mode !== 'in') {
                labelEl.innerText = '';
                return;
            }

            const total_frames_val = Math.max(frames - 1, 1);
            if (zs > 0 && ze > 0) {
                const r = Math.pow(ze / zs, 1.0 / total_frames_val);
                const percentage = (r - 1.0) * 100;
                labelEl.innerText = `(~+${percentage.toFixed(2)}% zoom/frame)`;
            } else {
                labelEl.innerText = '';
            }
        }

        function toggleZoomFields() {
            const mode = document.getElementById('zoom_mode').value;
            if (mode === 'in') {
                document.getElementById('zoom_in_params').style.display = 'block';
                document.getElementById('zoom_out_params').style.display = 'none';
            } else {
                document.getElementById('zoom_in_params').style.display = 'none';
                document.getElementById('zoom_out_params').style.display = 'block';
            }
            updateZoomRateLabel();
        }
        async function ctrl(a,b){
            if (a === 'start' || a === 'resume' || a === 'pause') {
                try {
                    await fetch('/update_params',{
                        method:'POST',
                        headers:{'Content-Type':'application/json'},
                        body:JSON.stringify({
                            model:document.getElementById('model').value,
                            negative_prompt:document.getElementById('neg_prompt').value,
                            lora1:document.getElementById('lora1').value,
                            lora2:document.getElementById('lora2').value,
                            lora3:document.getElementById('lora3').value,
                            lora1_strength:document.getElementById('lora1_str').value,
                            lora2_strength:document.getElementById('lora2_str').value,
                            lora3_strength:document.getElementById('lora3_str').value,
                            seed:document.getElementById('seed').value,
                            denoise:document.getElementById('denoise').value,
                            frames:document.getElementById('frames').value,
                            steps:document.getElementById('steps').value,
                            cfg:document.getElementById('cfg').value,
                            use_motion_zoom: document.getElementById('use_zoom').checked,
                            use_metadata_caption: document.getElementById('use_caption').checked,
                            caption_font_size: document.getElementById('cap_font_size').value,
                            caption_x: document.getElementById('cap_x').value,
                            caption_y: document.getElementById('cap_y').value,
                            caption_bg_r: document.getElementById('cap_bg_r').value,
                            caption_bg_g: document.getElementById('cap_bg_g').value,
                            caption_bg_b: document.getElementById('cap_bg_b').value,
                            caption_bg_a: document.getElementById('cap_bg_a').value,
                            zoom_start:document.getElementById('zs').value,
                            zoom_end:document.getElementById('ze').value,
                            pan_start_x:document.getElementById('pxs').value,
                            pan_end_x:document.getElementById('pxe').value,
                            pan_start_y:document.getElementById('pys').value,
                            pan_end_y:document.getElementById('pye').value,
                            roll_mode:document.getElementById('roll_mode').value,
                            feedback_color:document.getElementById('fb_color').value,
                            feedback_contrast:document.getElementById('fb_contrast').value,
                            feedback_sharpness:document.getElementById('fb_sharpness').value,
                            use_visual_director:document.getElementById('use_director').checked,
                            visual_director_interval:document.getElementById('director_interval').value,
                            visual_director_model:document.getElementById('director_model').value,
                            use_prompt_interpolation:document.getElementById('use_prompt_interpolation').checked,
                            use_video_interpolation:document.getElementById('use_video_interpolation').checked,
                            zoom_mode:document.getElementById('zoom_mode').value,
                            zoom_shrink:document.getElementById('zoom_shrink').value,
                            zoom_blur:document.getElementById('zoom_blur').value,
                            zoom_strength:document.getElementById('zoom_strength').value,
                            zoom_guidance_scale:document.getElementById('zoom_guidance_scale').value,
                            zoom_num_inference_steps:document.getElementById('steps').value
                        })
                    });
                } catch(e) { console.error("Autosave failed:", e); }
            }
            fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:a, prompt:document.getElementById('prompt').value})})
            .then(()=>showFeedback("Action Sent: " + a.toUpperCase()));
        }
        function inject(b){ fetch('/inject',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:document.getElementById('inj').value})}).then(()=>{document.getElementById('inj').value=''; showFeedback("Injected!");}); }
        function setCaption(b){
            fetch('/set_caption',{
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body:JSON.stringify({
                    text:document.getElementById('cap_text').value,
                    font:document.getElementById('cap_font').value,
                    font_size:document.getElementById('temp_cap_font_size').value,
                    x:document.getElementById('temp_cap_x').value,
                    y:document.getElementById('temp_cap_y').value,
                    bg_r:document.getElementById('temp_cap_bg_r').value,
                    bg_g:document.getElementById('temp_cap_bg_g').value,
                    bg_b:document.getElementById('temp_cap_bg_b').value,
                    bg_a:document.getElementById('temp_cap_bg_a').value
                })
            }).then(()=>{
                document.getElementById('cap_text').value='';
                showFeedback("Caption Armed!");
            });
        }
        function teleport(b){
            const fileInput = document.getElementById('tele_file');
            if (fileInput.files.length === 0) { alert("Please select an image first"); return; }
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            fetch('/teleport', { method: 'POST', body: formData })
            .then(r => r.json())
            .then(d => {
                if (d.status === 'ok') {
                    showFeedback("Teleport Armed!");
                    fileInput.value = '';
                } else {
                    alert("Error: " + d.message);
                }
            });
        }
        function addKF(b){
            fetch('/add_keyframe',{
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body:JSON.stringify({
                    frame:document.getElementById('kf_f').value,
                    prompt:document.getElementById('kf_p').value,
                    denoise:document.getElementById('kf_d').value,
                    seed_offset:document.getElementById('kf_s').value
                })
            })
            .then(r=>r.json())
            .then(d=>{
                showFeedback("KF Added!");
                updateKFList(d.keyframes);

                // clear inputs (feels much better when working fast)
                document.getElementById('kf_p').value = "";
                document.getElementById('kf_f').value = 0;
                document.getElementById('kf_d').value = 0.5;
                document.getElementById('kf_s').value = 0;
            });
        }
        function update(b){
            fetch('/update_params',{
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body:JSON.stringify({
                    prompt:document.getElementById('prompt').value,
                    model:document.getElementById('model').value,
                    negative_prompt:document.getElementById('neg_prompt').value,
                    lora1:document.getElementById('lora1').value,
                    lora2:document.getElementById('lora2').value,
                    lora3:document.getElementById('lora3').value,
                    lora1_strength:document.getElementById('lora1_str').value,
                    lora2_strength:document.getElementById('lora2_str').value,
                    lora3_strength:document.getElementById('lora3_str').value,
                    seed:document.getElementById('seed').value,
                    denoise:document.getElementById('denoise').value,
                    frames:document.getElementById('frames').value,
                    steps:document.getElementById('steps').value,
                    cfg:document.getElementById('cfg').value,
                    use_motion_zoom: document.getElementById('use_zoom').checked,
                    use_metadata_caption: document.getElementById('use_caption').checked,
                    caption_font_size: document.getElementById('cap_font_size').value,
                    caption_x: document.getElementById('cap_x').value,
                    caption_y: document.getElementById('cap_y').value,
                    caption_bg_r: document.getElementById('cap_bg_r').value,
                    caption_bg_g: document.getElementById('cap_bg_g').value,
                    caption_bg_b: document.getElementById('cap_bg_b').value,
                    caption_bg_a: document.getElementById('cap_bg_a').value,
                    zoom_start:document.getElementById('zs').value,
                    zoom_end:document.getElementById('ze').value,
                    pan_start_x:document.getElementById('pxs').value,
                    pan_end_x:document.getElementById('pxe').value,
                    pan_start_y:document.getElementById('pys').value,
                    pan_end_y:document.getElementById('pye').value,
                    roll_mode:document.getElementById('roll_mode').value,
                    feedback_color:document.getElementById('fb_color').value,
                    feedback_contrast:document.getElementById('fb_contrast').value,
                    feedback_sharpness:document.getElementById('fb_sharpness').value,
                    use_visual_director:document.getElementById('use_director').checked,
                    visual_director_interval:document.getElementById('director_interval').value,
                    visual_director_model:document.getElementById('director_model').value,
                    use_prompt_interpolation:document.getElementById('use_prompt_interpolation').checked,
                    use_video_interpolation:document.getElementById('use_video_interpolation').checked,
                    zoom_mode:document.getElementById('zoom_mode').value,
                    zoom_shrink:document.getElementById('zoom_shrink').value,
                    zoom_blur:document.getElementById('zoom_blur').value,
                    zoom_strength:document.getElementById('zoom_strength').value,
                    zoom_guidance_scale:document.getElementById('zoom_guidance_scale').value,
                    zoom_num_inference_steps:document.getElementById('steps').value,
                    use_seed_guide: document.getElementById('use_seed_guide').checked,
                    seed_guide_filename: document.getElementById('seed_select').value,
                    seed_guide_x: currentSeedX,
                    seed_guide_y: currentSeedY,
                    seed_guide_w: currentSeedW,
                    seed_guide_h: currentSeedH,
                    seed_guide_opacity: currentSeedOpacity
                })
            }).then(()=>showFeedback("Parameters Updated"));
        }
        function updateKFList(kfs){ const l = document.getElementById('kf_list');
        l.innerHTML = '<h4>Active KFs</h4>';
        Object.keys(kfs).sort((a,b)=>a-b).forEach(f=>{
            l.innerHTML += `<div class="kf-item">
                <b>F${f}</b>: D:${kfs[f].denoise}<br>
                <span style="color:#aaa;">${kfs[f].prompt || '(no prompt override)'}</span>
            </div>`;
        });  }

        let isDraggingLogo = false;
        let startX = 0, startY = 0;
        let logoLeft = 0, logoTop = 0;
        let dragLogoEl = null;
        let currentLogoFilename = "{{ CURRENT_LOGO }}";
        let currentLogoX = 0;
        let currentLogoY = 0;
        let currentLogoW = 100;
        let currentLogoH = 100;
        let currentLogoOpacity = 1.0;
        let currentFrameWidth = 340;
        let currentFrameHeight = 512;

        // Stencil variables
        let isDraggingStencil = false;
        let stencilLeft = 0, stencilTop = 0;
        let dragStencilEl = null;
        let currentStencilFilename = "None";
        let currentStencilX = 0;
        let currentStencilY = 0;
        let currentStencilW = 100;
        let currentStencilH = 100;
        let currentStencilOpacity = 0.5;
        let showStencil = false;

        // Seed Guide variables
        let isDraggingSeed = false;
        let seedLeft = 0, seedTop = 0;
        let dragSeedEl = null;
        let currentSeedFilename = "{{ CURRENT_SEED_GUIDE }}";
        let currentSeedX = {{ seed_guide_x }};
        let currentSeedY = {{ seed_guide_y }};
        let currentSeedW = {{ seed_guide_w }};
        let currentSeedH = {{ seed_guide_h }};
        let currentSeedOpacity = {{ seed_guide_opacity }};
        let useSeedGuide = {% if USE_SEED_GUIDE %}true{% else %}false{% endif %};

        // Load local logo parameters on startup if they exist
        try {
            const localParamsStr = localStorage.getItem("local_logo_params");
            if (localParamsStr) {
                const localParams = JSON.parse(localParamsStr);
                currentLogoFilename = localParams.logo_filename;
                currentLogoX = localParams.x;
                currentLogoY = localParams.y;
                currentLogoW = localParams.w;
                currentLogoH = localParams.h;
                currentLogoOpacity = localParams.opacity;
            }
        } catch(e) {}

        // Load local stencil parameters on startup if they exist
        try {
            const localStencilStr = localStorage.getItem("local_stencil_params");
            if (localStencilStr) {
                const localParams = JSON.parse(localStencilStr);
                currentStencilFilename = localParams.stencil_filename || "None";
                currentStencilX = localParams.x || 0;
                currentStencilY = localParams.y || 0;
                currentStencilW = localParams.w || 100;
                currentStencilH = localParams.h || 100;
                currentStencilOpacity = localParams.opacity !== undefined ? localParams.opacity : 0.5;
                showStencil = localParams.show !== undefined ? localParams.show : false;
            }
        } catch(e) {}

        function uploadLogo(btn) {
            const fileInput = document.getElementById('logo_file_input');
            if (fileInput.files.length === 0) { alert("Please select a PNG image first"); return; }
            const formData = new FormData();
            formData.append('logo', fileInput.files[0]);
            fetch('/upload_logo', { method: 'POST', body: formData })
            .then(r => r.json())
            .then(d => {
                if (d.status === 'ok') {
                    showFeedback("Logo uploaded successfully!");
                    fileInput.value = '';
                    setTimeout(() => location.reload(), 2000);
                } else {
                    alert("Error: " + d.message);
                }
            });
        }

        function initDraggableLogo() {
            dragLogoEl = document.getElementById("draggable_logo");
            if (!dragLogoEl) return;
            dragLogoEl.addEventListener("pointerdown", (e) => {
                e.preventDefault();
                isDraggingLogo = true;
                dragLogoEl.setPointerCapture(e.pointerId);
                const rect = dragLogoEl.getBoundingClientRect();
                const parentRect = document.getElementById("overlay_logo_wrapper").getBoundingClientRect();
                startX = e.clientX;
                startY = e.clientY;
                logoLeft = rect.left - parentRect.left;
                logoTop = rect.top - parentRect.top;
            });
            dragLogoEl.addEventListener("pointermove", (e) => {
                if (!isDraggingLogo) return;
                e.preventDefault();
                const parentRect = document.getElementById("overlay_logo_wrapper").getBoundingClientRect();
                const deltaX = e.clientX - startX;
                const deltaY = e.clientY - startY;
                let newX = logoLeft + deltaX;
                let newY = logoTop + deltaY;
                const maxW = parentRect.width - dragLogoEl.offsetWidth;
                const maxH = parentRect.height - dragLogoEl.offsetHeight;
                newX = Math.max(0, Math.min(newX, maxW));
                newY = Math.max(0, Math.min(newY, maxH));
                dragLogoEl.style.left = newX + "px";
                dragLogoEl.style.top = newY + "px";
            });
            dragLogoEl.addEventListener("pointerup", (e) => {
                if (isDraggingLogo) {
                    dragLogoEl.releasePointerCapture(e.pointerId);
                    isDraggingLogo = false;
                    
                    // Immediately calculate and sync local coordinates to prevent status polling snap-back
                    const parentRect = document.getElementById("overlay_logo_wrapper").getBoundingClientRect();
                    if (parentRect.width > 0 && parentRect.height > 0) {
                        const scaleX = currentFrameWidth / parentRect.width;
                        const scaleY = currentFrameHeight / parentRect.height;
                        currentLogoX = Math.round(parseFloat(dragLogoEl.style.left) * scaleX);
                        currentLogoY = Math.round(parseFloat(dragLogoEl.style.top) * scaleY);
                    }
                    
                    // Autosave locally on drop
                    saveLogoLocally();
                }
            });
        }

        function changeLogo(filename) {
            if (filename === "None") {
                currentLogoFilename = "None";
                localStorage.removeItem("local_logo_params");
                updateLogoOverlayDisplay();
                fetch("/save_logo_position", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ logo_filename: "None" })
                });
                return;
            }
            currentLogoFilename = filename;
            currentLogoX = 20;
            currentLogoY = 20;
            currentLogoW = 100;
            currentLogoH = 0; // set to 0 so onload handler calculates correct aspect ratio
            currentLogoOpacity = 1.0;
            initialSyncDone = true; // Mark sync done immediately to prevent status poll overriding
            localStorage.setItem("local_logo_params", JSON.stringify({
                logo_filename: filename,
                x: 20,
                y: 20,
                w: 100,
                h: 0,
                opacity: 1.0
            }));
            updateLogoOverlayDisplay();
        }

        function changeLogoSize() {
            const el = document.getElementById("draggable_logo");
            const widthInput = document.getElementById("logo_width_input");
            if (!el || !widthInput) return;
            const w = parseInt(widthInput.value) || 100;
            const ratio = (el.naturalWidth > 0) ? (el.naturalHeight / el.naturalWidth) : 1.0;
            const h = Math.round(w * ratio);
            el.style.width = w + "px";
            el.style.height = h + "px";
            
            // Sync local dimensions immediately to prevent polling reset
            currentLogoW = w;
            currentLogoH = h;
        }

        function updateLogoOpacity(val) {
            currentLogoOpacity = parseFloat(val);
            const el = document.getElementById("draggable_logo");
            if (el) el.style.opacity = currentLogoOpacity;
        }

        function cancelLogoPlacement() {
            localStorage.removeItem("local_logo_params");
            changeLogo("None");
        }

        function calculateCurrentLogoParams() {
            const select = document.getElementById("logo_select");
            const logo_filename = select ? select.value : currentLogoFilename;
            const opacitySlider = document.getElementById("logo_opacity_slider");
            const logo_opacity = opacitySlider ? parseFloat(opacitySlider.value) : currentLogoOpacity;
            
            return {
                logo_filename: logo_filename,
                x: currentLogoX,
                y: currentLogoY,
                w: currentLogoW,
                h: currentLogoH,
                opacity: logo_opacity
            };
        }

        async function saveLogoLocally() {
            const params = calculateCurrentLogoParams();
            if (!params) return;
            currentLogoFilename = params.logo_filename;
            currentLogoX = params.x;
            currentLogoY = params.y;
            currentLogoW = params.w;
            currentLogoH = params.h;
            currentLogoOpacity = params.opacity;
            
            // Save to browser's localStorage
            localStorage.setItem("local_logo_params", JSON.stringify(params));
            
            // Instantly overlay on the current frame saved locally on server's disk
            const resp = await fetch("/save_logo_local", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(params)
            });
            if (resp.ok) {
                showFeedback("Logo position saved locally & overlaid on current frame!");
                // Force preview update to show the newly overlaid logo
                const previewImg = document.getElementById("preview");
                if (previewImg) {
                    previewImg.src = "/latest_frame?t=" + Date.now();
                }
            } else {
                showFeedback("Logo position saved locally in browser!");
            }
        }

        async function saveLogoToServer() {
            const params = calculateCurrentLogoParams();
            if (!params) return;
            currentLogoFilename = params.logo_filename;
            currentLogoX = params.x;
            currentLogoY = params.y;
            currentLogoW = params.w;
            currentLogoH = params.h;
            currentLogoOpacity = params.opacity;
            
            // Save to browser's localStorage
            localStorage.setItem("local_logo_params", JSON.stringify(params));
            
            const resp = await fetch("/save_logo_position", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(params)
            });
            if (resp.ok) {
                showFeedback("Logo position saved to server!");
            } else {
                alert("Failed to save logo position to server.");
            }
        }

        // Stencil helper functions
        function initDraggableStencil() {
            dragStencilEl = document.getElementById("draggable_stencil");
            if (!dragStencilEl) return;
            dragStencilEl.addEventListener("pointerdown", (e) => {
                e.preventDefault();
                isDraggingStencil = true;
                dragStencilEl.setPointerCapture(e.pointerId);
                const rect = dragStencilEl.getBoundingClientRect();
                const parentRect = document.getElementById("overlay_stencil_wrapper").getBoundingClientRect();
                startX = e.clientX;
                startY = e.clientY;
                stencilLeft = rect.left - parentRect.left;
                stencilTop = rect.top - parentRect.top;
            });
            dragStencilEl.addEventListener("pointermove", (e) => {
                if (!isDraggingStencil) return;
                e.preventDefault();
                const parentRect = document.getElementById("overlay_stencil_wrapper").getBoundingClientRect();
                const deltaX = e.clientX - startX;
                const deltaY = e.clientY - startY;
                let newX = stencilLeft + deltaX;
                let newY = stencilTop + deltaY;
                const maxW = parentRect.width - dragStencilEl.offsetWidth;
                const maxH = parentRect.height - dragStencilEl.offsetHeight;
                newX = Math.max(0, Math.min(newX, maxW));
                newY = Math.max(0, Math.min(newY, maxH));
                dragStencilEl.style.left = newX + "px";
                dragStencilEl.style.top = newY + "px";
                
                if (parentRect.width > 0 && parentRect.height > 0) {
                    const scaleX = currentFrameWidth / parentRect.width;
                    const scaleY = currentFrameHeight / parentRect.height;
                    const currentX = Math.round(newX * scaleX);
                    const currentY = Math.round(newY * scaleY);
                    safeSyncValue("stencil_x_input", currentX);
                    safeSyncValue("stencil_y_input", currentY);
                }
            });
            dragStencilEl.addEventListener("pointerup", (e) => {
                if (isDraggingStencil) {
                    dragStencilEl.releasePointerCapture(e.pointerId);
                    isDraggingStencil = false;
                    
                    const parentRect = document.getElementById("overlay_stencil_wrapper").getBoundingClientRect();
                    if (parentRect.width > 0 && parentRect.height > 0) {
                        const scaleX = currentFrameWidth / parentRect.width;
                        const scaleY = currentFrameHeight / parentRect.height;
                        currentStencilX = Math.round(parseFloat(dragStencilEl.style.left) * scaleX);
                        currentStencilY = Math.round(parseFloat(dragStencilEl.style.top) * scaleY);
                        safeSyncValue("stencil_x_input", currentStencilX);
                        safeSyncValue("stencil_y_input", currentStencilY);
                    }
                    saveStencilLocally();
                }
            });
        }

        function toggleStencilDisplay() {
            const showEl = document.getElementById("show_stencil");
            showStencil = showEl ? showEl.checked : false;
            saveStencilLocally();
            updateStencilOverlayDisplay();
        }

        function changeStencil(filename) {
            currentStencilFilename = filename;
            if (filename !== "None") {
                showStencil = true;
                const showEl = document.getElementById("show_stencil");
                if (showEl) showEl.checked = true;
            }
            currentStencilX = 0;
            currentStencilY = 0;
            currentStencilW = 100;
            currentStencilH = 0; // calculated on load
            saveStencilLocally();
            updateStencilOverlayDisplay();
        }

        function changeStencilSize() {
            const widthEl = document.getElementById("stencil_width_input");
            if (!widthEl) return;
            currentStencilW = parseInt(widthEl.value) || 100;
            const el = document.getElementById("draggable_stencil");
            if (el) {
                const ratio = (el.naturalWidth > 0) ? (el.naturalHeight / el.naturalWidth) : 1.0;
                currentStencilH = Math.round(currentStencilW * ratio);
            }
            saveStencilLocally();
            updateStencilOverlayDisplay();
        }

        function changeStencilCoords() {
            const xEl = document.getElementById("stencil_x_input");
            const yEl = document.getElementById("stencil_y_input");
            if (xEl) currentStencilX = parseInt(xEl.value) || 0;
            if (yEl) currentStencilY = parseInt(yEl.value) || 0;
            saveStencilLocally();
            updateStencilOverlayDisplay();
        }

        function updateStencilOpacity(val) {
            currentStencilOpacity = parseFloat(val);
            const el = document.getElementById("draggable_stencil");
            if (el) el.style.opacity = currentStencilOpacity;
            saveStencilLocally();
        }

        function saveStencilLocally() {
            const params = {
                stencil_filename: currentStencilFilename,
                x: currentStencilX,
                y: currentStencilY,
                w: currentStencilW,
                h: currentStencilH,
                opacity: currentStencilOpacity,
                show: showStencil
            };
            localStorage.setItem("local_stencil_params", JSON.stringify(params));
        }

        function updateStencilOverlayDisplay() {
            if (isDraggingStencil) return;
            const container = document.getElementById("overlay_stencil_wrapper");
            if (!container) return;
            
            const stencilSelect = document.getElementById("stencil_select");
            if (stencilSelect && document.activeElement !== stencilSelect) {
                stencilSelect.value = currentStencilFilename;
            }
            const showEl = document.getElementById("show_stencil");
            if (showEl && document.activeElement !== showEl) {
                showEl.checked = showStencil;
            }
            
            if (!showStencil || currentStencilFilename === "None") {
                container.innerHTML = "";
                return;
            }
            
            let el = document.getElementById("draggable_stencil");
            if (!el) {
                el = document.createElement("img");
                el.id = "draggable_stencil";
                el.className = "draggable";
                el.style.position = "absolute";
                el.style.pointerEvents = "auto";
                el.style.cursor = "move";
                el.style.outline = "2px dashed #a78bfa";
                el.onload = function() {
                    if (el.naturalWidth > 0 && currentStencilH === 0) {
                        const ratio = el.naturalHeight / el.naturalWidth;
                        currentStencilH = Math.round(currentStencilW * ratio);
                        saveStencilLocally();
                        updateStencilOverlayDisplay();
                    }
                };
                container.appendChild(el);
                initDraggableStencil();
            }
            const expectedSrc = "/static/overlays/" + currentStencilFilename;
            if (!el.src.endsWith(expectedSrc)) {
                el.src = expectedSrc;
            }
            const parentRect = container.getBoundingClientRect();
            if (parentRect.width > 0 && parentRect.height > 0) {
                const scaleX = parentRect.width / currentFrameWidth;
                const scaleY = parentRect.height / currentFrameHeight;
                el.style.left = Math.round(currentStencilX * scaleX) + "px";
                el.style.top = Math.round(currentStencilY * scaleY) + "px";
                el.style.width = Math.round(currentStencilW * scaleX) + "px";
                el.style.height = Math.round(currentStencilH * scaleY) + "px";
                el.style.opacity = currentStencilOpacity;
                
                safeSyncValue("stencil_width_input", currentStencilW);
                safeSyncValue("stencil_x_input", currentStencilX);
                safeSyncValue("stencil_y_input", currentStencilY);
                safeSyncValue("stencil_opacity_slider", currentStencilOpacity);
            }
        }

        // AI Seed Guide helper functions
        function initDraggableSeed() {
            dragSeedEl = document.getElementById("draggable_seed_guide");
            if (!dragSeedEl) return;
            dragSeedEl.addEventListener("pointerdown", (e) => {
                e.preventDefault();
                isDraggingSeed = true;
                dragSeedEl.setPointerCapture(e.pointerId);
                const rect = dragSeedEl.getBoundingClientRect();
                const parentRect = document.getElementById("overlay_seed_wrapper").getBoundingClientRect();
                startX = e.clientX;
                startY = e.clientY;
                seedLeft = rect.left - parentRect.left;
                seedTop = rect.top - parentRect.top;
            });
            dragSeedEl.addEventListener("pointermove", (e) => {
                if (!isDraggingSeed) return;
                e.preventDefault();
                const parentRect = document.getElementById("overlay_seed_wrapper").getBoundingClientRect();
                const deltaX = e.clientX - startX;
                const deltaY = e.clientY - startY;
                let newX = seedLeft + deltaX;
                let newY = seedTop + deltaY;
                const maxW = parentRect.width - dragSeedEl.offsetWidth;
                const maxH = parentRect.height - dragSeedEl.offsetHeight;
                newX = Math.max(0, Math.min(newX, maxW));
                newY = Math.max(0, Math.min(newY, maxH));
                dragSeedEl.style.left = newX + "px";
                dragSeedEl.style.top = newY + "px";
                
                if (parentRect.width > 0 && parentRect.height > 0) {
                    const scaleX = currentFrameWidth / parentRect.width;
                    const scaleY = currentFrameHeight / parentRect.height;
                    const currentX = Math.round(newX * scaleX);
                    const currentY = Math.round(newY * scaleY);
                    safeSyncValue("seed_x_input", currentX);
                    safeSyncValue("seed_y_input", currentY);
                }
            });
            dragSeedEl.addEventListener("pointerup", (e) => {
                if (isDraggingSeed) {
                    dragSeedEl.releasePointerCapture(e.pointerId);
                    isDraggingSeed = false;
                    
                    const parentRect = document.getElementById("overlay_seed_wrapper").getBoundingClientRect();
                    if (parentRect.width > 0 && parentRect.height > 0) {
                        const scaleX = currentFrameWidth / parentRect.width;
                        const scaleY = currentFrameHeight / parentRect.height;
                        currentSeedX = Math.round(parseFloat(dragSeedEl.style.left) * scaleX);
                        currentSeedY = Math.round(parseFloat(dragSeedEl.style.top) * scaleY);
                        safeSyncValue("seed_x_input", currentSeedX);
                        safeSyncValue("seed_y_input", currentSeedY);
                    }
                    saveSeedGuideToServer();
                }
            });
        }

        function toggleSeedGuide(enabled) {
            useSeedGuide = enabled;
            saveSeedGuideToServer();
            updateSeedGuideOverlayDisplay();
        }

        function changeSeedGuide(filename) {
            currentSeedFilename = filename;
            if (filename !== "None") {
                useSeedGuide = true;
                const useEl = document.getElementById("use_seed_guide");
                if (useEl) useEl.checked = true;
            }
            currentSeedX = 0;
            currentSeedY = 0;
            currentSeedW = 100;
            currentSeedH = 0; // calculated on load
            saveSeedGuideToServer();
            updateSeedGuideOverlayDisplay();
        }

        function changeSeedGuideSize() {
            const widthEl = document.getElementById("seed_width_input");
            if (!widthEl) return;
            currentSeedW = parseInt(widthEl.value) || 100;
            const el = document.getElementById("draggable_seed_guide");
            if (el) {
                const ratio = (el.naturalWidth > 0) ? (el.naturalHeight / el.naturalWidth) : 1.0;
                currentSeedH = Math.round(currentSeedW * ratio);
            }
            saveSeedGuideToServer();
            updateSeedGuideOverlayDisplay();
        }

        function changeSeedGuideCoords() {
            const xEl = document.getElementById("seed_x_input");
            const yEl = document.getElementById("seed_y_input");
            if (xEl) currentSeedX = parseInt(xEl.value) || 0;
            if (yEl) currentSeedY = parseInt(yEl.value) || 0;
            saveSeedGuideToServer();
            updateSeedGuideOverlayDisplay();
        }

        function updateSeedGuideOpacity(val) {
            currentSeedOpacity = parseFloat(val);
            const el = document.getElementById("draggable_seed_guide");
            if (el) el.style.opacity = currentSeedOpacity;
            saveSeedGuideToServer();
        }

        async function saveSeedGuideToServer() {
            const resp = await fetch('/update_params', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: document.getElementById('prompt').value,
                    model: document.getElementById('model').value,
                    negative_prompt: document.getElementById('neg_prompt').value,
                    lora1: document.getElementById('lora1').value,
                    lora2: document.getElementById('lora2').value,
                    lora3: document.getElementById('lora3').value,
                    lora1_strength: document.getElementById('lora1_str').value,
                    lora2_strength: document.getElementById('lora2_str').value,
                    lora3_strength: document.getElementById('lora3_str').value,
                    seed: document.getElementById('seed').value,
                    denoise: document.getElementById('denoise').value,
                    frames: document.getElementById('frames').value,
                    steps: document.getElementById('steps').value,
                    cfg: document.getElementById('cfg').value,
                    use_motion_zoom: document.getElementById('use_zoom').checked,
                    use_metadata_caption: document.getElementById('use_caption').checked,
                    caption_font_size: document.getElementById('cap_font_size').value,
                    caption_x: document.getElementById('cap_x').value,
                    caption_y: document.getElementById('cap_y').value,
                    caption_bg_r: document.getElementById('cap_bg_r').value,
                    caption_bg_g: document.getElementById('cap_bg_g').value,
                    caption_bg_b: document.getElementById('cap_bg_b').value,
                    caption_bg_a: document.getElementById('cap_bg_a').value,
                    zoom_start: document.getElementById('zs').value,
                    zoom_end: document.getElementById('ze').value,
                    pan_start_x: document.getElementById('pxs').value,
                    pan_end_x: document.getElementById('pxe').value,
                    pan_start_y: document.getElementById('pys').value,
                    pan_end_y: document.getElementById('pye').value,
                    roll_mode: document.getElementById('roll_mode').value,
                    feedback_color: document.getElementById('fb_color').value,
                    feedback_contrast: document.getElementById('fb_contrast').value,
                    feedback_sharpness: document.getElementById('fb_sharpness').value,
                    use_visual_director: document.getElementById('use_director').checked,
                    visual_director_interval: document.getElementById('director_interval').value,
                    visual_director_model: document.getElementById('director_model').value,
                    use_prompt_interpolation: document.getElementById('use_prompt_interpolation').checked,
                    use_video_interpolation: document.getElementById('use_video_interpolation').checked,
                    zoom_mode: document.getElementById('zoom_mode').value,
                    zoom_shrink: document.getElementById('zoom_shrink').value,
                    zoom_blur: document.getElementById('zoom_blur').value,
                    zoom_strength: document.getElementById('zoom_strength').value,
                    zoom_guidance_scale: document.getElementById('zoom_guidance_scale').value,
                    zoom_num_inference_steps: document.getElementById('steps').value,
                    
                    use_seed_guide: useSeedGuide,
                    seed_guide_filename: currentSeedFilename,
                    seed_guide_x: currentSeedX,
                    seed_guide_y: currentSeedY,
                    seed_guide_w: currentSeedW,
                    seed_guide_h: currentSeedH,
                    seed_guide_opacity: currentSeedOpacity
                })
            });
            if (resp.ok) {
                showFeedback("AI Seed Guide updated on server!");
            }
        }

        function updateSeedGuideOverlayDisplay() {
            if (isDraggingSeed) return;
            const container = document.getElementById("overlay_seed_wrapper");
            if (!container) return;
            
            const seedSelect = document.getElementById("seed_select");
            if (seedSelect && document.activeElement !== seedSelect) {
                seedSelect.value = currentSeedFilename;
            }
            const useEl = document.getElementById("use_seed_guide");
            if (useEl && document.activeElement !== useEl) {
                useEl.checked = useSeedGuide;
            }
            
            if (!useSeedGuide || currentSeedFilename === "None") {
                container.innerHTML = "";
                return;
            }
            
            let el = document.getElementById("draggable_seed_guide");
            if (!el) {
                el = document.createElement("img");
                el.id = "draggable_seed_guide";
                el.className = "draggable";
                el.style.position = "absolute";
                el.style.pointerEvents = "auto";
                el.style.cursor = "move";
                el.style.outline = "2px dashed #ec4899";
                el.onload = function() {
                    if (el.naturalWidth > 0 && currentSeedH === 0) {
                        const ratio = el.naturalHeight / el.naturalWidth;
                        currentSeedH = Math.round(currentSeedW * ratio);
                        saveSeedGuideToServer();
                        updateSeedGuideOverlayDisplay();
                    }
                };
                container.appendChild(el);
                initDraggableSeed();
            }
            const expectedSrc = "/static/overlays/" + currentSeedFilename;
            if (!el.src.endsWith(expectedSrc)) {
                el.src = expectedSrc;
            }
            const parentRect = container.getBoundingClientRect();
            if (parentRect.width > 0 && parentRect.height > 0) {
                const scaleX = parentRect.width / currentFrameWidth;
                const scaleY = parentRect.height / currentFrameHeight;
                el.style.left = Math.round(currentSeedX * scaleX) + "px";
                el.style.top = Math.round(currentSeedY * scaleY) + "px";
                el.style.width = Math.round(currentSeedW * scaleX) + "px";
                el.style.height = Math.round(currentSeedH * scaleY) + "px";
                el.style.opacity = currentSeedOpacity;
                
                safeSyncValue("seed_width_input", currentSeedW);
                safeSyncValue("seed_x_input", currentSeedX);
                safeSyncValue("seed_y_input", currentSeedY);
                safeSyncValue("seed_opacity_slider", currentSeedOpacity);
            }
        }

        function safeSyncValue(id, val) {
            const el = document.getElementById(id);
            if (el && document.activeElement !== el) {
                el.value = val;
            }
        }

        function safeSyncChecked(id, checked) {
            const el = document.getElementById(id);
            if (el && document.activeElement !== el) {
                el.checked = checked;
            }
        }

        function updateLogoOverlayDisplay() {
            if (isDraggingLogo) return;
            const select = document.getElementById("logo_select");
            const container = document.getElementById("overlay_logo_wrapper");
            if (!select || !container) return;
            if (document.activeElement !== select && select.value !== currentLogoFilename) {
                select.value = currentLogoFilename;
            }
            const opacitySlider = document.getElementById("logo_opacity_slider");
            if (document.activeElement !== opacitySlider) {
                opacitySlider.value = currentLogoOpacity;
            }
            if (currentLogoFilename === "None") {
                container.innerHTML = "";
                document.getElementById("logo_control_buttons").style.display = "none";
                return;
            }
            document.getElementById("logo_control_buttons").style.display = "flex";
            let el = document.getElementById("draggable_logo");
            if (!el) {
                el = document.createElement("img");
                el.id = "draggable_logo";
                el.className = "draggable";
                el.style.position = "absolute";
                el.style.pointerEvents = "auto";
                el.style.cursor = "move";
                el.style.outline = "2px dashed #3b82f6";
                el.onload = function() {
                    if (el.naturalWidth > 0 && currentLogoH === 0) {
                        const ratio = el.naturalHeight / el.naturalWidth;
                        currentLogoH = Math.round(currentLogoW * ratio);
                        const parentRect = container.getBoundingClientRect();
                        if (parentRect.width > 0 && parentRect.height > 0) {
                            const scaleY = parentRect.height / currentFrameHeight;
                            el.style.height = Math.round(currentLogoH * scaleY) + "px";
                        }
                        saveLogoLocally();
                    }
                };
                container.appendChild(el);
                initDraggableLogo();
            }
            const expectedSrc = "/static/overlays/" + currentLogoFilename;
            if (!el.src.endsWith(expectedSrc)) {
                el.src = expectedSrc;
            }
            const parentRect = container.getBoundingClientRect();
            if (parentRect.width > 0 && parentRect.height > 0) {
                const scaleX = parentRect.width / currentFrameWidth;
                const scaleY = parentRect.height / currentFrameHeight;
                el.style.left = Math.round(currentLogoX * scaleX) + "px";
                el.style.top = Math.round(currentLogoY * scaleY) + "px";
                el.style.width = Math.round(currentLogoW * scaleX) + "px";
                el.style.height = Math.round(currentLogoH * scaleY) + "px";
                el.style.opacity = currentLogoOpacity;
                
                const widthInput = document.getElementById("logo_width_input");
                if (document.activeElement !== widthInput) {
                    widthInput.value = currentLogoW;
                }
            }
        }

        function loadOllamaModels() {
            fetch('/get_ollama_models')
            .then(r => r.json())
            .then(d => {
                const select = document.getElementById("ollama_model");
                const dirSelect = document.getElementById("director_model");
                const scanSelect = document.getElementById("scanner_model_select");
                select.innerHTML = "";
                dirSelect.innerHTML = "";
                if (scanSelect) scanSelect.innerHTML = "";
                
                if (d.models && d.models.length > 0) {
                    d.models.forEach(m => {
                        const opt = document.createElement("option");
                        opt.value = m;
                        opt.text = m;
                        select.appendChild(opt);
                        
                        const optDir = document.createElement("option");
                        optDir.value = m;
                        optDir.text = m;
                        dirSelect.appendChild(optDir);

                        if (scanSelect) {
                            const optScan = document.createElement("option");
                            optScan.value = m;
                            optScan.text = m;
                            scanSelect.appendChild(optScan);
                        }
                    });
                    const preferred = ["dolphin3:8b", "mistral:7b-instruct", "llama3.2:3b", "qwen3:8b"];
                    for (let p of preferred) {
                        for (let opt of select.options) {
                            if (opt.value.includes(p)) {
                                select.value = opt.value;
                                break;
                            }
                        }
                    }
                    const preferredDir = ["moondream", "llava", "dolphin3", "llama3.2", "mistral"];
                    for (let p of preferredDir) {
                        for (let opt of dirSelect.options) {
                            if (opt.value.includes(p)) {
                                dirSelect.value = opt.value;
                                break;
                            }
                        }
                    }
                    const preferredScan = ["llava", "moondream", "llama3.2-vision", "qwen2-vl", "vision"];
                    if (scanSelect) {
                        for (let p of preferredScan) {
                            for (let opt of scanSelect.options) {
                                if (opt.value.toLowerCase().includes(p)) {
                                    scanSelect.value = opt.value;
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    const opt = document.createElement("option");
                    opt.value = "None";
                    opt.text = "No Ollama models found / offline";
                    select.appendChild(opt);
                    
                    const optDir = document.createElement("option");
                    optDir.value = "None";
                    optDir.text = "No Ollama models found / offline";
                    dirSelect.appendChild(optDir);

                    if (scanSelect) {
                        const optScan = document.createElement("option");
                        optScan.value = "None";
                        optScan.text = "No Ollama models found / offline";
                        scanSelect.appendChild(optScan);
                    }
                }
            });
        }

        function enhancePrompt() {
            const promptArea = document.getElementById("prompt");
            const modelSelect = document.getElementById("ollama_model");
            if (modelSelect.value === "None") {
                alert("No Ollama model selected or Ollama is offline.");
                return;
            }
            showFeedback("Enhancing prompt with AI...");
            fetch("/enhance_prompt", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prompt: promptArea.value,
                    model: modelSelect.value
                })
            })
            .then(r => r.json())
            .then(d => {
                if (d.status === "ok") {
                    promptArea.value = d.enhanced_prompt;
                    showFeedback("Prompt enhanced successfully!");
                } else {
                    alert("Enhancement failed: " + d.error);
                }
            })
            .catch(e => {
                alert("Enhancement failed: " + e);
            });
        }

        function showScanMarker(x, y, cw, ch) {
            const container = document.getElementById("preview_container");
            if (!container) return;
            
            const existing = document.getElementById("scan_marker");
            if (existing) existing.remove();
            
            const marker = document.createElement("div");
            marker.id = "scan_marker";
            marker.style.position = "absolute";
            
            const img = document.getElementById("preview");
            marker.style.left = (img.offsetLeft + x - 25) + "px"; // center a 50px visual box
            marker.style.top = (img.offsetTop + y - 25) + "px";
            marker.style.width = "50px";
            marker.style.height = "50px";
            marker.style.border = "2px dashed #ef4444";
            marker.style.borderRadius = "4px";
            marker.style.pointerEvents = "none";
            marker.style.zIndex = "1000";
            
            container.appendChild(marker);
            setTimeout(() => { if (marker) marker.remove(); }, 3000);
        }

        async function runVisionScanner(clickX, clickY, cw, ch) {
            const statusEl = document.getElementById("scanner_status");
            const resultEl = document.getElementById("scanner_result");
            if (!statusEl || !resultEl) return;
            
            statusEl.style.display = "block";
            statusEl.innerText = clickX !== null ? "Scanning cropped detail..." : "Scanning full image...";
            resultEl.innerText = "Processing request on server...";
            
            const filename = document.getElementById("scanner_image_select").value;
            const model = document.getElementById("scanner_model_select").value;
            const prompt = document.getElementById("scanner_prompt").value;
            
            try {
                const resp = await fetch("/inspect_image_llava", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        filename: filename,
                        model: model,
                        prompt: prompt,
                        click_x: clickX,
                        click_y: clickY,
                        container_w: cw,
                        container_h: ch
                    })
                });
                
                const data = await resp.json();
                statusEl.style.display = "none";
                
                if (data.status === "ok") {
                    resultEl.innerText = data.description;
                    showFeedback("Scan complete!");
                } else {
                    resultEl.innerText = "Error: " + (data.error || "Unknown scanner error");
                }
            } catch(e) {
                statusEl.style.display = "none";
                resultEl.innerText = "Error connecting to server: " + e;
            }
        }

        document.addEventListener("DOMContentLoaded", () => {
            loadOllamaModels();
            
            // Initial sync of Stencil and Seed Guide DOM elements from loaded variables
            const showStencilEl = document.getElementById("show_stencil");
            if (showStencilEl) showStencilEl.checked = showStencil;
            const stencilSelectEl = document.getElementById("stencil_select");
            if (stencilSelectEl) stencilSelectEl.value = currentStencilFilename;
            const stencilWidthInput = document.getElementById("stencil_width_input");
            if (stencilWidthInput) stencilWidthInput.value = currentStencilW;
            const stencilXInput = document.getElementById("stencil_x_input");
            if (stencilXInput) stencilXInput.value = currentStencilX;
            const stencilYInput = document.getElementById("stencil_y_input");
            if (stencilYInput) stencilYInput.value = currentStencilY;
            const stencilOpacitySlider = document.getElementById("stencil_opacity_slider");
            if (stencilOpacitySlider) stencilOpacitySlider.value = currentStencilOpacity;

            const useSeedGuideEl = document.getElementById("use_seed_guide");
            if (useSeedGuideEl) useSeedGuideEl.checked = useSeedGuide;
            const seedSelectEl = document.getElementById("seed_select");
            if (seedSelectEl) seedSelectEl.value = currentSeedFilename;
            const seedWidthInput = document.getElementById("seed_width_input");
            if (seedWidthInput) seedWidthInput.value = currentSeedW;
            const seedXInput = document.getElementById("seed_x_input");
            if (seedXInput) seedXInput.value = currentSeedX;
            const seedYInput = document.getElementById("seed_y_input");
            if (seedYInput) seedYInput.value = currentSeedY;
            const seedOpacitySlider = document.getElementById("seed_opacity_slider");
            if (seedOpacitySlider) seedOpacitySlider.value = currentSeedOpacity;
            
            const previewImg = document.getElementById("preview");
            if (previewImg) {
                // Set placeholder sizing before loading
                previewImg.style.width = currentFrameWidth + "px";
                previewImg.style.height = currentFrameHeight + "px";
                previewImg.style.background = "#111";
                
                // Attempt to load the latest frame
                previewImg.src = "/latest_frame?t=" + Date.now();
                
                // Visual scan cursor and click handler
                previewImg.style.cursor = "crosshair";
                previewImg.addEventListener("click", (e) => {
                    const rect = previewImg.getBoundingClientRect();
                    const clickX = e.clientX - rect.left;
                    const clickY = e.clientY - rect.top;
                    const containerW = rect.width;
                    const containerH = rect.height;
                    showScanMarker(clickX, clickY, containerW, containerH);
                    runVisionScanner(clickX, clickY, containerW, containerH);
                });
                
                previewImg.onload = function() {
                    // Reset custom placeholder styles when image loads
                    previewImg.style.width = "";
                    previewImg.style.height = "";
                    previewImg.style.background = "";
                    updateLogoOverlayDisplay();
                    updateStencilOverlayDisplay();
                    updateSeedGuideOverlayDisplay();
                };
                
                previewImg.onerror = function() {
                    // Keep placeholder styles if image fails to load
                    previewImg.style.width = currentFrameWidth + "px";
                    previewImg.style.height = currentFrameHeight + "px";
                    previewImg.style.background = "#111";
                    updateLogoOverlayDisplay();
                    updateStencilOverlayDisplay();
                    updateSeedGuideOverlayDisplay();
                };
            }
        });

        let initialSyncDone = false;
        let lastLoadedFrame = -1;

        setInterval(()=>{
            fetch('/status').then(r=>r.json()).then(d=>{
                let statusMsg = d.running ? (d.paused ? "PAUSED" : "RENDERING") : "IDLE";
                
                // Add Step Progress if rendering
                if (d.running && !d.paused && d.max_steps > 0) {
                    statusMsg += ` (Step ${d.progress}/${d.max_steps})`;
                }
                
                const pauseSyncEl = document.getElementById('pause_ui_sync');
                const pauseSync = (pauseSyncEl && pauseSyncEl.checked) || !d.running || d.paused;
                
                const displayStatus = (pauseSync && d.running && !d.paused) ? statusMsg + " (Sync Paused)" : statusMsg;
                document.getElementById('status_text').innerText = displayStatus + " " + d.frame + "/" + d.total;
                
                if(d.frame > 0 && d.frame !== lastLoadedFrame) {
                    document.getElementById('preview').src = '/latest_frame?t=' + Date.now();
                    lastLoadedFrame = d.frame;
                }
                if(d.history) document.getElementById('thumb_strip').innerHTML = d.history.map(i=>`<img src="/static/spaceexplorer2/${i}?t=${Date.now()}">`).join('');
                if(d.injections) document.getElementById('injections').innerHTML = d.injections.map(i=>`<span class="tag">${i}</span>`).join('');
                if(d.rendering_prompt !== undefined) {
                    document.getElementById('live_prompt_display').innerText = d.rendering_prompt || "Idle / No active prompt";
                }
                
                // Sync dynamic outputs
                if (!pauseSync) {
                    if (d.prompt !== undefined) safeSyncValue('prompt', d.prompt);
                    if (d.latest_vision_desc !== undefined) {
                        document.getElementById('latest_vision_desc').innerText = d.latest_vision_desc || "No vision data yet.";
                    }
                }
                
                // Sync inputs (only once on initial load to prevent status poll from overriding user edits)
                if (!initialSyncDone) {
                    safeSyncChecked('use_caption', d.metadata_caption);
                    if (d.caption_font_size !== undefined) safeSyncValue('cap_font_size', d.caption_font_size);
                    if (d.caption_x !== undefined) safeSyncValue('cap_x', d.caption_x);
                    if (d.caption_y !== undefined) safeSyncValue('cap_y', d.caption_y);
                    if (d.caption_bg_r !== undefined) safeSyncValue('cap_bg_r', d.caption_bg_r);
                    if (d.caption_bg_g !== undefined) safeSyncValue('cap_bg_g', d.caption_bg_g);
                    if (d.caption_bg_b !== undefined) safeSyncValue('cap_bg_b', d.caption_bg_b);
                    if (d.caption_bg_a !== undefined) safeSyncValue('cap_bg_a', d.caption_bg_a);
                    
                    // Sync Logo (only on initial load to prevent status poll from overriding unsaved local placements)
                    if (d.logo_filename !== undefined) {
                        const localParamsStr = localStorage.getItem("local_logo_params");
                        if (localParamsStr) {
                            try {
                                const localParams = JSON.parse(localParamsStr);
                                currentLogoFilename = localParams.logo_filename;
                                currentLogoX = localParams.x;
                                currentLogoY = localParams.y;
                                currentLogoW = localParams.w;
                                currentLogoH = localParams.h;
                                currentLogoOpacity = localParams.opacity;
                            } catch(e) {
                                currentLogoFilename = d.logo_filename;
                                currentLogoX = d.logo_x;
                                currentLogoY = d.logo_y;
                                currentLogoW = d.logo_w;
                                currentLogoH = d.logo_h;
                                currentLogoOpacity = d.logo_opacity;
                            }
                        } else {
                            currentLogoFilename = d.logo_filename;
                            currentLogoX = d.logo_x;
                            currentLogoY = d.logo_y;
                            currentLogoW = d.logo_w;
                            currentLogoH = d.logo_h;
                            currentLogoOpacity = d.logo_opacity;
                        }
                    }
                    
                    // Sync AI Seed Guide (only on initial load to prevent status poll from overriding unsaved local placements)
                    if (d.seed_guide_filename !== undefined) {
                        useSeedGuide = d.use_seed_guide;
                        currentSeedFilename = d.seed_guide_filename;
                        currentSeedX = d.seed_guide_x;
                        currentSeedY = d.seed_guide_y;
                        currentSeedW = d.seed_guide_w;
                        currentSeedH = d.seed_guide_h;
                        currentSeedOpacity = d.seed_guide_opacity;
                    }
                    
                    if (d.width !== undefined) currentFrameWidth = d.width;
                    if (d.height !== undefined) currentFrameHeight = d.height;
                    updateLogoOverlayDisplay();
                    updateStencilOverlayDisplay();
                    updateSeedGuideOverlayDisplay();
                    
                    // Sync Ollama & Interpolation params
                    if (d.use_visual_director !== undefined) safeSyncChecked('use_director', d.use_visual_director);
                    if (d.visual_director_interval !== undefined) safeSyncValue('director_interval', d.visual_director_interval);
                    if (d.visual_director_model !== undefined) safeSyncValue('director_model', d.visual_director_model);
                    if (d.use_prompt_interpolation !== undefined) safeSyncChecked('use_prompt_interpolation', d.use_prompt_interpolation);
                    if (d.use_video_interpolation !== undefined) safeSyncChecked('use_video_interpolation', d.use_video_interpolation);
                    if (d.lora1_strength !== undefined) safeSyncValue('lora1_str', d.lora1_strength);
                    if (d.lora2_strength !== undefined) safeSyncValue('lora2_str', d.lora2_strength);
                    if (d.lora3_strength !== undefined) safeSyncValue('lora3_str', d.lora3_strength);
                    
                    if (d.zoom_mode !== undefined) {
                        safeSyncValue('zoom_mode', d.zoom_mode);
                        toggleZoomFields();
                    }
                    if (d.zoom_shrink !== undefined) safeSyncValue('zoom_shrink', d.zoom_shrink);
                    if (d.zoom_blur !== undefined) safeSyncValue('zoom_blur', d.zoom_blur);
                    if (d.zoom_strength !== undefined) safeSyncValue('zoom_strength', d.zoom_strength);
                    if (d.zoom_guidance_scale !== undefined) safeSyncValue('zoom_guidance_scale', d.zoom_guidance_scale);
                    if (d.zoom_num_inference_steps !== undefined) safeSyncValue('zoom_num_inference_steps', d.zoom_num_inference_steps);
                    if (d.steps !== undefined) safeSyncValue('steps', d.steps);
                    if (d.seed !== undefined) safeSyncValue('seed', d.seed);
                    if (d.cfg !== undefined) safeSyncValue('cfg', d.cfg);
                    if (d.denoise !== undefined) safeSyncValue('denoise', d.denoise);
                    if (d.frames !== undefined) safeSyncValue('frames', d.frames);
                    
                    if (d.zoom_start !== undefined) safeSyncValue('zs', d.zoom_start);
                    if (d.zoom_end !== undefined) safeSyncValue('ze', d.zoom_end);
                    updateZoomRateLabel();
                    initialSyncDone = true;
                }
            });
        }, 4000);
        setTimeout(updateZoomRateLabel, 500);
    </script>
</body></html>
"""
# -------------------------------------------------


# --------------------------------------------------
# The Sound Stage
# --------------------------------------------------


# --------------------------------------------------
# VOICES
# --------------------------------------------------
VOICES = ['af_bella','af_sarah','am_adam','bm_george']
DEFAULT_VOICE = "am_adam"

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

SHARED_PATH = os.path.join(BASE_DIR, "static", "spaceexplorer2")
FRAME_PATH = os.path.join(BASE_DIR, "static", "assets", "transtalks.png")

os.makedirs(SHARED_PATH, exist_ok=True)

UNIQUE = str(randint(100000, 999999))

# --------------------------------------------------
# HTML
# --------------------------------------------------
HTML = """
<!doctype html>
<html>
<head>
<style>
body { background:#111; color:#eee; font-family:Arial; text-align:center; }
.box { display:inline-block; margin:10px; padding:10px; border:1px solid #333; }
img, video { width:300px; }
button { padding:10px 20px; margin-top:20px; }
a { font-size:3vw;color:orange; }
</style>
</head>
<body>

<h1>Video Builder</h1>

<a href="/download_mp4">Download MP4</a><br>
<a href="/text_to_mp3">Generate MP3</a><br>
<a href="/compile_story_movie" style="color: #4ade80; font-size: 2.2vw; font-weight: bold; display: block; margin-top: 15px; text-decoration: underline;">Compile Synchronized Story Video (Images + Narrations)</a>

<form method="post" action="/process_video">

<h2>Videos</h2>
{% for v in videos %}
<div class="box">
<video controls src="{{ url_for('static', filename='spaceexplorer2/' + v) }}"></video><br>
<input type="radio" name="video" value="{{v}}">
{{v}}
</div>
{% endfor %}

<h2>Images</h2>
<div style="margin-bottom: 15px;">
    <button type="button" onclick="selectAllImages(true)" style="margin-right: 10px; padding: 6px 12px; margin-top: 0; width: auto; font-weight: bold; cursor: pointer;">Select All</button>
    <button type="button" onclick="selectAllImages(false)" style="padding: 6px 12px; margin-top: 0; width: auto; font-weight: bold; cursor: pointer;">Deselect All</button>
</div>
{% for img in images %}
<div class="box">
<img src="{{ url_for('static', filename='spaceexplorer2/' + img) }}"><br>
<input type="checkbox" name="images" value="{{ img }}">
</div>
{% endfor %}

<h2>Audio</h2>
{% for mp3 in mp3s %}
<div class="box" style="vertical-align: top;">
<audio controls style="width:250px; display:block; margin: 0 auto 8px auto;">
    <source src="{{ url_for('static', filename='spaceexplorer2/' + mp3) }}" type="audio/mpeg">
</audio>
<input type="radio" name="audio" value="{{ mp3 }}">
{{ mp3 }}
</div>
{% endfor %}

<div style="margin: 20px auto; max-width: 400px; text-align: left; background: #1c1c21; padding: 15px; border-radius: 4px; border: 1px solid #333;">
    <h3 style="margin-top: 0; font-size: 14px; color: orange;">Audio Trimming (Optional)</h3>
    <label style="display:inline-block; font-size:11px; color:#888; font-weight:bold; width:130px;">Start Time (sec):</label>
    <input type="number" name="audio_start" step="0.1" min="0" placeholder="0.0" style="width:80px; display:inline-block; margin-bottom: 8px;"><br>
    <label style="display:inline-block; font-size:11px; color:#888; font-weight:bold; width:130px;">End Time (sec):</label>
    <input type="number" name="audio_end" step="0.1" min="0" placeholder="End of file" style="width:80px; display:inline-block;">
</div>

<br>
<button type="submit">PROCESS</button>
</form>

<script>
function selectAllImages(checked) {
    const checkboxes = document.querySelectorAll('input[name="images"]');
    checkboxes.forEach(cb => cb.checked = checked);
}
</script>
</body>
</html>
"""

# --------------------------------------------------
# HOME
# --------------------------------------------------
@app.route("/media")
def media():
    import os
    from icecream import ic

    # --------------------------------------------------
    # GET FILES
    # --------------------------------------------------
    files = os.listdir(SHARED_PATH)
    ic("ALL FILES:", files)

    # --------------------------------------------------
    # HELPER: GET FILE TIME
    # --------------------------------------------------
    def get_mtime(filename):
        full_path = os.path.join(SHARED_PATH, filename)
        try:
            mtime = os.path.getmtime(full_path)
            ic(f"mtime for {filename}:", mtime)
            return mtime
        except Exception as e:
            ic(f"ERROR reading mtime for {filename}:", e)
            return 0

    # --------------------------------------------------
    # FILTER FILE TYPES
    # --------------------------------------------------
    images = [
        f for f in files 
        if f.lower().endswith((".jpg", ".png")) 
        and not f.startswith("temp_clean_")
        and not f.startswith("clean_")
    ]
    mp3s   = [f for f in files if f.lower().endswith(".mp3")]
    videos = [f for f in files if f.lower().endswith(".mp4")]

    ic("UNSORTED IMAGES:", images)
    ic("UNSORTED MP3S:", mp3s)
    ic("UNSORTED VIDEOS:", videos)

    # --------------------------------------------------
    # SORT BY DATE (NEWEST FIRST)
    # --------------------------------------------------
    images.sort(key=get_mtime, reverse=True)
    mp3s.sort(key=get_mtime, reverse=True)
    videos.sort(key=get_mtime, reverse=True)

    ic("SORTED IMAGES:", images)
    ic("SORTED MP3S:", mp3s)
    ic("SORTED VIDEOS:", videos)

    # --------------------------------------------------
    # RETURN TEMPLATE
    # --------------------------------------------------
    return render_template_string(
        HTML,
        images=images,
        mp3s=mp3s,
        videos=videos
    )
'''    
@app.route("/media")
def media():
    files = os.listdir(SHARED_PATH)

    images = [f for f in files if f.endswith((".jpg",".png"))]
    mp3s   = [f for f in files if f.endswith(".mp3")]
    videos = [f for f in files if f.endswith(".mp4")]

    ic(images, mp3s, videos)

    return render_template_string(
        HTML,
        images=images,
        mp3s=mp3s,
        videos=videos
    )
'''
# --------------------------------------------------
# DOWNLOAD MP4
# --------------------------------------------------
@app.route("/download_mp4", methods=["GET","POST"])
def download_mp4():
    if request.method == "POST":

        # file upload instead of URL
        if "file" not in request.files:
            return "No file part", 400

        file = request.files["file"]

        if file.filename == "":
            return "No selected file", 400

        filename = secure_filename(file.filename)
        out = os.path.join(SHARED_PATH, filename)

        ic("Uploading file:", filename)

        file.save(out)

        return redirect("/")

    return """
    <h2>Upload MP4 from your computer or LAN</h2>

    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".mp4"><br><br>
        <button>Upload</button>
    </form>
    """
# --------------------------------------------------
# AUDIO PAD
# --------------------------------------------------
def pad_audio(src, out, start_sec=None, end_sec=None):
    audio = AudioSegment.from_mp3(src)
    
    # Calculate millisecond trim points
    start_ms = int(start_sec * 1000) if start_sec is not None else 0
    end_ms = int(end_sec * 1000) if end_sec is not None else len(audio)
    
    # Clamp to audio bounds
    start_ms = max(0, min(start_ms, len(audio)))
    end_ms = max(start_ms, min(end_ms, len(audio)))
    
    # Slice the audio segment
    trimmed_audio = audio[start_ms:end_ms]
    
    silence = AudioSegment.silent(duration=200)
    (silence + trimmed_audio + silence).export(out, format="mp3")

# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------
@app.route("/process_video", methods=["POST"])
def process_video():

    selected_video = request.form.get("video")
    selected_audio = request.form.get("audio")
    selected_images = request.form.getlist("images")
    audio_start = request.form.get("audio_start")
    audio_end = request.form.get("audio_end")
    
    start_sec = float(audio_start) if audio_start and audio_start.strip() else None
    end_sec = float(audio_end) if audio_end and audio_end.strip() else None

    ic(selected_video, selected_audio, selected_images, start_sec, end_sec)

    audio_path = os.path.join(SHARED_PATH, selected_audio)
    padded_audio = os.path.join(SHARED_PATH, "_pad.mp3")

    pad_audio(audio_path, padded_audio, start_sec, end_sec)

    audio_clip = AudioFileClip(padded_audio)

    # ------------------------------------------
    # CASE 1: EXISTING VIDEO
    # ------------------------------------------
    if selected_video:

        video_path = os.path.join(SHARED_PATH, selected_video)
        video = VideoFileClip(video_path)

        ic("Original duration:", video.duration)
        ic("Audio duration:", audio_clip.duration)

        speed_factor = video.duration / audio_clip.duration
        ic("Speed factor:", speed_factor)

        new_video = video.fx(vfx.speedx, speed_factor)

        final = new_video.set_audio(audio_clip)

    # ------------------------------------------
    # CASE 2: IMAGE SLIDESHOW
    # ------------------------------------------
    else:
        clips = []
        duration = audio_clip.duration / len(selected_images)

        # Sort images chronologically (alphabetically) to prevent reverse playback
        for img in sorted(selected_images):
            p = os.path.join(SHARED_PATH, img)
            clip = ImageClip(p).set_duration(duration)
            clips.append(clip)

        final = concatenate_videoclips(clips).set_audio(audio_clip)

    # ------------------------------------------
    # WRITE OUTPUT
    # ------------------------------------------
    out = os.path.join(SHARED_PATH, f"{UNIQUE}_output.mp4")

    ic("Writing:", out)

    final.write_videofile(
        out,
        fps=24,
        codec="libx264",
        audio_codec="aac"
    )

    return redirect(url_for("serve_video", filename=os.path.basename(out)))

# --------------------------------------------------
# SERVE
# --------------------------------------------------
@app.route("/video/<filename>")
def serve_video(filename):
    return send_file(os.path.join(SHARED_PATH, filename))

# --------------------------------------------------
# COMPILE STORY MOVIE
# --------------------------------------------------
@app.route("/compile_story_movie")
def compile_story_movie():
    import re
    import math
    # Find all narration WAV files
    wav_files = sorted([f for f in os.listdir(SHARED_PATH) if f.startswith("narration_") and f.endswith(".wav")])
    if not wav_files:
        return "No narration WAV files found in static/spaceexplorer2/. Please generate your story first.", 400

    bgm_name = request.args.get("bgm", "None")
    bgm_clip = None
    if bgm_name and bgm_name != "None":
        bgm_path = os.path.join(SHARED_PATH, bgm_name)
        if os.path.exists(bgm_path):
            try:
                # Load and loop BGM to ensure it doesn't run out
                raw_bgm = AudioFileClip(bgm_path)
                bgm_clip = raw_bgm.fx(vfx.loop, duration=10000.0)
            except Exception as e:
                logit(f"Failed to load BGM track: {e}")

    # Load space_diary.json to get the tension level for each frame (for camera shake!)
    diary_path = os.path.join(SHARED_PATH, "space_diary.json")
    if os.path.exists(diary_path):
        try:
            with open(diary_path, "r") as f:
                diary_entries = json.load(f)
                for entry in diary_entries:
                    diary_map[int(entry["frame"])] = entry
        except Exception as e:
            logit(f"Failed to load diary json for camera shake: {e}")

    clips = []
    current_bgm_time = 0.0
    
    for wav_name in wav_files:
        match = re.search(r'narration_(\d+)\.wav$', wav_name)
        if not match:
            continue
        idx = int(match.group(1))

        img_candidates = [
            f"frame_{idx:03d}.png",
            f"clean_{idx:03d}.png"
        ]
        
        img_path = None
        for cand in img_candidates:
            cand_path = os.path.join(SHARED_PATH, cand)
            if os.path.exists(cand_path):
                img_path = cand_path
                break
                
        if not img_path:
            continue

        try:
            audio_clip = AudioFileClip(os.path.join(SHARED_PATH, wav_name))
            img_clip = ImageClip(img_path).set_duration(audio_clip.duration)
            
            # --- Audio-Reactive Camera Shake ---
            entry_info = diary_map.get(idx, {})
            tension = float(entry_info.get("tension", 1.0))
            
            if tension > 7.0:
                shake_amp = (tension - 7.0) * 1.5
                img_clip = img_clip.rotate(lambda t: shake_amp * math.sin(t * 25), resample="bicubic", expand=False)
            elif tension > 4.0:
                shake_amp = (tension - 4.0) * 0.4
                img_clip = img_clip.rotate(lambda t: shake_amp * math.sin(t * 12), resample="bicubic", expand=False)
                
            # --- Smart BGM Auto-Ducking ---
            segment_audio = audio_clip
            if bgm_clip:
                bgm_slice = bgm_clip.subclip(current_bgm_time, current_bgm_time + audio_clip.duration)
                ducked_bgm = bgm_slice.volumex(0.12)
                segment_audio = CompositeAudioClip([audio_clip, ducked_bgm])
                current_bgm_time += audio_clip.duration
                
            video_segment = img_clip.set_audio(segment_audio)
            clips.append(video_segment)
        except Exception as e:
            logit(f"Error processing index {idx:03d} for story video: {e}")

    if not clips:
        return "No valid video segments (matching image-audio pairs) could be created.", 400

    out = os.path.join(SHARED_PATH, "story_production.mp4")
    try:
        final_video = concatenate_videoclips(clips, method="compose")
        final_video.write_videofile(
            out,
            fps=24,
            codec="libx264",
            audio_codec="aac"
        )
        return redirect(url_for("serve_video", filename="story_production.mp4"))
    except Exception as e:
        return f"Error during video compilation: {str(e)}", 500

# --------------------------------------------------
# TTS
# --------------------------------------------------
@app.route("/text_to_mp3", methods=["GET","POST"])
def text_to_mp3():

    if request.method == "POST":
        text = request.form.get("text")
        voice = request.form.get("voice")

        out = os.path.join(SHARED_PATH, text[:20].replace(" ","_")+".mp3")

        payload = {
            "model": "kokoro",
            "voice": voice,
            "input": text
        }

        r = requests.post(
            "http://localhost:8880/v1/audio/speech",
            json=payload
        )

        with open(out, "wb") as f:
            f.write(r.content)

        return redirect("/")

    return """
    <form method="post">
    <textarea name="text" style="width:60%"></textarea><br>
    <select name="voice">
        <option>af_bella</option>
        <option>am_adam</option>
    </select><br>
    <button>Generate</button>
    </form>
    """



# --------------------------------------------------
# HTML_EDITOR Template String
# --------------------------------------------------
HTML_EDITOR = """
<!DOCTYPE html>
<html>
<head>
    <title>Templates & Notebook Editor</title>
    <style>
        body { margin: 0; background: #0c0c0e; color: #d1d1d1; font-family: sans-serif; display: flex; height: 100vh; overflow: hidden; }
        .column { padding: 15px; box-sizing: border-box; overflow-y: auto; border-right: 1px solid #333; }
        .left { padding: 8px; width: 25%; background: #141417; display: flex; flex-direction: column; }
        .right { width: 75%; background: #080808; display: flex; flex-direction: column; padding: 15px; }
        h3 { margin-top: 5px; color: #fff; font-size: 1.1em; border-bottom: 1px solid #333; padding-bottom: 5px; }
        label { font-size: 11px; color: #888; font-weight: bold; display: block; margin-top: 8px; }
        input, select, textarea { width: 96%; background: #1c1c21; color: #fff; border: 1px solid #333; padding: 6px; border-radius: 4px; font-size: 12px; margin-top: 4px; }
        button { width: 98%; padding: 8px; margin-top: 8px; cursor: pointer; border: none; border-radius: 4px; font-weight: bold; transition: all 0.2s; }
        .btn-green { background: #10b981; color: white; }
        .btn-blue { background: #3b82f6; color: white; }
        .btn-orange { background: #f59e0b; color: white; }
        .btn-red { background: #ef4444; color: white; }
        
        .file-item { background: #1c1c21; padding: 8px; margin-top: 5px; border-radius: 4px; border-left: 3px solid #8b5cf6; display: flex; justify-content: space-between; align-items: center; }
        .file-item a { color: #fff; text-decoration: none; font-size: 12px; font-weight: bold; flex-grow: 1; }
        .file-item a:hover { color: #8b5cf6; }
        .file-actions { display: flex; gap: 5px; }
        .file-actions button { width: auto; margin-top: 0; padding: 2px 6px; font-size: 10px; }
        
        /* Tabs Styling */
        .tabs-header { display: flex; gap: 6px; margin-bottom: 12px; border-bottom: 1px solid #222; padding-bottom: 8px; }
        .tab-btn { background: #141417; color: #888; border: 1px solid #2a2a30; padding: 8px 16px; font-size: 12px; border-radius: 4px; cursor: pointer; font-weight: bold; transition: all 0.2s; width: auto; margin-top: 0; }
        .tab-btn:hover { color: #fff; background: #1c1c21; border-color: #444; }
        .tab-btn.active { background: #8b5cf6; color: #fff; border-color: #8b5cf6; }
        
        .tab-content { display: none; flex: 1; flex-direction: column; min-height: 0; }
        .tab-content.active { display: flex; }
        
        .editor-container { display: flex; flex: 1; gap: 10px; min-height: 0; }
        .editor-pane { display: flex; flex-direction: column; flex: 1; min-width: 0; }
        .preview-pane { display: flex; flex-direction: column; flex: 1; min-width: 0; }
        
        #code_editor { flex: 1; font-family: monospace; font-size: 13px; line-height: 1.5; resize: none; background: #000; color: #9f9; padding: 10px; outline: none; border: 1px solid #333; }
        #preview_iframe { flex: 1; background: #fff; border: 1px solid #333; border-radius: 4px; }
        
        .feedback { margin-top: 5px; color: #10b981; font-weight: bold; font-size: 11px; height: 15px; }
        .editor-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        
        /* Chat Console Styling */
        .chat-container { display: flex; flex-direction: column; flex: 1; min-height: 0; background: #080808; border: 1px solid #222; border-radius: 6px; padding: 12px; }
        .chat-history { flex: 1; overflow-y: auto; padding: 8px; display: flex; flex-direction: column; gap: 12px; border-bottom: 1px solid #222; margin-bottom: 12px; }
        .chat-message { padding: 10px 14px; border-radius: 8px; max-width: 85%; font-size: 13.5px; line-height: 1.45; word-wrap: break-word; }
        .chat-message.user { background: #2563eb; color: #fff; align-self: flex-end; border-bottom-right-radius: 2px; }
        .chat-message.assistant { background: #141417; color: #e5e7eb; align-self: flex-start; border-bottom-left-radius: 2px; border: 1px solid #2a2a30; }
        .chat-input-area { display: flex; gap: 10px; align-items: flex-end; }
        .chat-input-area textarea { flex: 1; resize: none; margin-top: 0; background: #141417; border: 1px solid #2a2a30; border-radius: 4px; padding: 8px 12px; color: #fff; font-size: 13px; line-height: 1.4; }
        .chat-input-area textarea:focus { outline: none; border-color: #8b5cf6; }
    </style>
</head>
<body>
    <div class="column left">
        <a href="/" style="display: block; text-align: center; color: #fff; background: #3b82f6; padding: 8px; border-radius: 4px; text-decoration: none; font-weight: bold; font-size: 12px; margin-bottom: 12px;">← Back to Dashboard</a>
        
        <h3>Templates Directory</h3>
        <div style="margin-bottom: 10px;">
            <input type="text" id="new_filename" placeholder="e.g. ideas.html" style="width: 90%;">
            <button class="btn-green" onclick="createNewFile()">Create New Page</button>
            <input type="file" id="sidebar_upload_input" accept=".txt,.md,.html,.js,.css" style="display:none;" onchange="handleSidebarUpload(this)">
            <button class="btn-blue" onclick="document.getElementById('sidebar_upload_input').click()">📤 Load Text/MD to New Page</button>
        </div>
        
        <div style="overflow-y: auto; flex: 1; max-height: 35vh;">
            {% for f in files %}
            <div class="file-item">
                <a href="/templates_editor?file={{ f }}">{{ f }}</a>
                <div class="file-actions">
                    <button class="btn-blue" onclick="window.open('/view_template/{{ f }}', '_blank')">👁️</button>
                    <button class="btn-red" onclick="deleteFile('{{ f }}')">🗑️</button>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <hr style="border:0; border-top:1px solid #333; margin:15px 0;">
        <h3>Ollama AI Co-Writer</h3>
        <label>Select Ollama Model</label>
        <select id="editor_model" style="width: 96%;">
            {% for m in MODELS %}
            <option value="{{ m }}">{{ m }}</option>
            {% endfor %}
        </select>
        
        <label>Instructions for AI</label>
        <textarea id="ai_prompt" rows="4" placeholder="Describe the page layout or content you want the AI to draft..." style="width: 90%;"></textarea>
        <button class="btn-blue" onclick="draftWithOllama()">Draft with Ollama</button>
        <div id="ai_status" style="margin-top: 5px; color: #3b82f6; font-size: 11px; font-weight: bold; display: none;">AI drafting code...</div>
    </div>
    
    <div class="right">
        <!-- Tabs Header Bar -->
        <div class="tabs-header">
            <button class="tab-btn active" id="btn_editor_tab" onclick="switchTab('editor_tab')">📄 Page Editor</button>
            <button class="tab-btn" id="btn_chat_tab" onclick="switchTab('chat_tab')">💬 Ollama Chat Console</button>
        </div>
        
        <!-- Tab Content 1: Page Editor -->
        <div id="editor_tab" class="tab-content active">
            <div class="editor-header">
                <h2 style="margin: 0; font-size: 1.3em; color: #fff;">
                    {% if selected_file %}Editing: <span style="color: #8b5cf6;">{{ selected_file }}</span>{% else %}Select or create a file to edit{% endif %}
                </h2>
                {% if selected_file %}
                <div style="display: flex; gap: 10px;">
                    <input type="file" id="upload_file_input" accept=".txt,.md,.html,.js,.css" style="display:none;" onchange="handleFileUpload(this)">
                    <button class="btn-blue" onclick="document.getElementById('upload_file_input').click()" style="margin: 0; padding: 6px 12px; width: auto;">📤 Load Text/MD</button>
                    <button class="btn-green" onclick="saveFile()" style="margin: 0; padding: 6px 12px; width: auto;">💾 Save</button>
                    <button class="btn-green" onclick="saveFileAs()" style="margin: 0; padding: 6px 12px; width: auto; background: #059669;">💾 Save As...</button>
                    <button class="btn-orange" onclick="updatePreview()" style="margin: 0; padding: 6px 12px; width: auto;">🔄 Refresh Preview</button>
                </div>
                {% endif %}
            </div>
            
            <div class="feedback" id="feedback_msg"></div>
            
            <div class="editor-container">
                {% if selected_file %}
                <div class="editor-pane">
                    <label>HTML Editor</label>
                    <textarea id="code_editor" oninput="debouncePreview()">{{ content }}</textarea>
                </div>
                <div class="preview-pane">
                    <label>Live Preview</label>
                    <iframe id="preview_iframe"></iframe>
                </div>
                {% else %}
                <div style="flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #555;">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                    <p style="margin-top: 15px; font-size: 14px;">Select a page from the sidebar or click "Create New Page" to start writing.</p>
                </div>
                {% endif %}
            </div>
        </div>
        
        <!-- Tab Content 2: Ollama Chat Console -->
        <div id="chat_tab" class="tab-content">
            <div class="editor-header">
                <h2 style="margin: 0; font-size: 1.3em; color: #fff;">Ollama Chat Console</h2>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <label style="margin: 0; font-size: 12px; color: #888;">Model:</label>
                    <select id="chat_model" style="width: auto; margin-top: 0; padding: 4px 8px; font-size: 12px;">
                        {% for m in MODELS %}
                        <option value="{{ m }}">{{ m }}</option>
                        {% endfor %}
                    </select>
                    <button class="btn-orange" onclick="clearChatHistory()" style="margin: 0; padding: 6px 12px; width: auto; font-size: 12px;">🗑️ Clear Memory</button>
                </div>
            </div>
            
            <div class="chat-container">
                <div class="chat-history" id="chat_history">
                    <div class="chat-message assistant">
                        <strong>Ollama:</strong> Hello! I am your local AI co-pilot. How can I help you build templates or ideas today? You can ask me to write HTML code snippets, design pages, or troubleshoot concepts. (Memory is enabled!)
                    </div>
                </div>
                <div class="chat-input-area">
                    <textarea id="chat_input" placeholder="Ask Ollama something... (e.g. 'Draft a simple CSS grid layout for a photogallery')" rows="3"></textarea>
                    <button class="btn-blue" id="btn_send_chat" onclick="sendChatMessage()" style="width: auto; margin-top: 0; padding: 8px 20px; align-self: stretch;">Send 🚀</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const activeFile = "{{ selected_file }}";
        let chatHistory = [];
        
        function showFeedback(text) {
            const f = document.getElementById("feedback_msg");
            if (f) {
                f.innerText = text;
                setTimeout(() => f.innerText = "", 5000);
            }
        }
        
        function updatePreview() {
            try {
                const editor = document.getElementById("code_editor");
                const iframe = document.getElementById("preview_iframe");
                if (editor && iframe) {
                    let val = editor.value;
                    const isHtml = val.trim().toLowerCase().startsWith("<!doctype") || 
                                   val.trim().toLowerCase().startsWith("<html") || 
                                   val.trim().toLowerCase().startsWith("<div") || 
                                   val.trim().toLowerCase().startsWith("<p");
                    if (!isHtml) {
                        val = `<!DOCTYPE html><html><head><style>body { background: #0c0c0e; color: #d1d1d1; font-family: sans-serif; padding: 15px; font-size: 14px; line-height: 1.5; } h1, h2, h3 { color: #8b5cf6; } pre, code { background: #141417; border: 1px solid #333; padding: 2px 5px; border-radius: 3px; font-family: monospace; }</style></head><body>` + escapeHtml(val).replace(/\\n/g, "<br>") + `</body></html>`;
                    }
                    iframe.srcdoc = val;
                }
            } catch (e) {
                console.error("Preview error:", e);
            }
        }
        
        let previewTimeout;
        function debouncePreview() {
            clearTimeout(previewTimeout);
            previewTimeout = setTimeout(updatePreview, 600);
        }
        
        function createNewFile() {
            const input = document.getElementById("new_filename");
            let filename = input ? input.value.trim() : "";
            if (!filename) { alert("Please enter a filename"); return; }
            if (!filename.endsWith(".html")) filename += ".html";
            
            fetch("/templates_editor", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filename: filename,
                    content: `<!DOCTYPE html>\\n<html>\\n<head>\\n    <title>${filename}</title>\\n    <style>\\n        body { background: #0c0c0e; color: #fff; font-family: sans-serif; padding: 20px; }\\n        h1 { color: #8b5cf6; }\\n    </style>\\n</head>\\n<body>\\n    <h1>Hello World</h1>\\n    <p>This is your new template ${filename}.</p>\\n</body>\\n</html>`
                })
            })
            .then(r => r.json())
            .then(d => {
                if (d.status === "ok") {
                    location.href = "/templates_editor?file=" + filename;
                } else {
                    alert("Error: " + d.message);
                }
            });
        }
        
        function handleSidebarUpload(input) {
            const file = input.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const content = e.target.result;
                let filename = file.name;
                const extIndex = filename.lastIndexOf(".");
                if (extIndex !== -1) {
                    const ext = filename.substring(extIndex).toLowerCase();
                    if (ext === ".txt" || ext === ".md") {
                        filename = filename.substring(0, extIndex) + ".html";
                    }
                } else {
                    filename += ".html";
                }
                
                const finalFilename = prompt("Create new template file with this name?", filename);
                if (!finalFilename) return;
                
                fetch("/templates_editor", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        filename: finalFilename,
                        content: content
                    })
                })
                .then(r => r.json())
                .then(d => {
                    if (d.status === "ok") {
                        location.href = "/templates_editor?file=" + finalFilename;
                    } else {
                        alert("Error: " + d.message);
                    }
                });
            };
            reader.readAsText(file);
            input.value = "";
        }
        
        function handleFileUpload(input) {
            const file = input.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const editor = document.getElementById("code_editor");
                if (editor) {
                    editor.value = e.target.result;
                    updatePreview();
                    showFeedback("Loaded file content into editor!");
                }
            };
            reader.readAsText(file);
            input.value = "";
        }
        
        function saveFile() {
            if (!activeFile) return;
            const content = document.getElementById("code_editor").value;
            fetch("/templates_editor", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filename: activeFile,
                    content: content
                })
            })
            .then(r => r.json())
            .then(d => {
                if (d.status === "ok") {
                    showFeedback("File saved successfully!");
                    updatePreview();
                } else {
                    alert("Error saving: " + d.message);
                }
            });
        }
        
        function saveFileAs() {
            try {
                const editor = document.getElementById("code_editor");
                if (!editor) {
                    alert("No active editor found to save!");
                    return;
                }
                const content = editor.value;
                let defaultName = activeFile || "new_page.html";
                let filename = prompt("Enter new filename to save as:", defaultName);
                if (!filename) return;
                if (!filename.endsWith(".html")) filename += ".html";
                
                fetch("/templates_editor", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        filename: filename,
                        content: content
                    })
                })
                .then(r => r.json())
                .then(d => {
                    if (d.status === "ok") {
                        showFeedback("Saved as " + filename + " successfully!");
                        location.href = "/templates_editor?file=" + filename;
                    } else {
                        alert("Error saving: " + d.message);
                    }
                })
                .catch(err => {
                    alert("Network error saving file: " + err);
                });
            } catch (e) {
                console.error("Save As error:", e);
                alert("Save As error: " + e.message);
            }
        }
        
        function deleteFile(filename) {
            if (!confirm("Are you sure you want to delete " + filename + "?")) return;
            fetch("/delete_template", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: filename })
            })
            .then(r => r.json())
            .then(d => {
                if (d.status === "ok") {
                    if (activeFile === filename) {
                        location.href = "/templates_editor";
                    } else {
                        location.reload();
                    }
                } else {
                    alert("Error: " + d.message);
                }
            });
        }
        
        async function draftWithOllama() {
            const prompt = document.getElementById("ai_prompt").value.trim();
            const model = document.getElementById("editor_model").value;
            const status = document.getElementById("ai_status");
            if (!prompt) { alert("Please enter instructions for the AI"); return; }
            
            status.style.display = "block";
            try {
                const resp = await fetch("/draft_template_ollama", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ prompt: prompt, model: model })
                });
                const data = await resp.json();
                status.style.display = "none";
                
                if (data.status === "ok") {
                    const editor = document.getElementById("code_editor");
                    if (editor) {
                        editor.value = data.html;
                        updatePreview();
                        showFeedback("AI Draft populated in editor!");
                    } else {
                        alert("Please select or create a file first before drafting.");
                    }
                } else {
                    alert("AI Error: " + (data.error || "Failed to generate"));
                }
            } catch(e) {
                status.style.display = "none";
                alert("Connection failed: " + e);
            }
        }
        
        // Tab switching logic
        function switchTab(tabId) {
            document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
            document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
            
            document.getElementById(tabId).classList.add("active");
            if (tabId === "editor_tab") {
                document.getElementById("btn_editor_tab").classList.add("active");
            } else if (tabId === "chat_tab") {
                document.getElementById("btn_chat_tab").classList.add("active");
                scrollToBottom();
            }
        }
        
        // Ollama Chat functions
        async function sendChatMessage() {
            const input = document.getElementById("chat_input");
            const btn = document.getElementById("btn_send_chat");
            const text = input.value.trim();
            if (!text) return;
            
            input.value = "";
            chatHistory.push({ role: "user", content: text });
            appendMessage("user", text);
            
            const botMsgDiv = appendMessage("assistant", "Thinking...");
            botMsgDiv.id = "temp_loading_msg";
            
            btn.disabled = true;
            btn.innerText = "Thinking...";
            
            const model = document.getElementById("chat_model").value;
            
            try {
                const resp = await fetch("/chat_ollama", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        messages: chatHistory,
                        model: model
                    })
                });
                const data = await resp.json();
                
                const tempDiv = document.getElementById("temp_loading_msg");
                if (tempDiv) tempDiv.remove();
                
                if (data.status === "ok" && data.message) {
                    chatHistory.push(data.message);
                    appendMessage("assistant", data.message.content);
                } else {
                    appendMessage("assistant", "Error: " + (data.error || "Failed to get response"));
                }
            } catch(e) {
                const tempDiv = document.getElementById("temp_loading_msg");
                if (tempDiv) tempDiv.remove();
                appendMessage("assistant", "Error connecting to backend: " + e);
            } finally {
                btn.disabled = false;
                btn.innerText = "Send 🚀";
                scrollToBottom();
            }
        }
        
        function appendMessage(role, content) {
            const chatHistoryDiv = document.getElementById("chat_history");
            const msgDiv = document.createElement("div");
            msgDiv.className = `chat-message ${role}`;
            
            const label = role === "user" ? "You" : "Ollama";
            const formattedContent = role === "user" ? escapeHtml(content) : formatMessageContent(content);
            msgDiv.innerHTML = `<strong>${label}:</strong><div style="margin-top:4px;">${formattedContent}</div>`;
            
            chatHistoryDiv.appendChild(msgDiv);
            scrollToBottom();
            return msgDiv;
        }
        
        function formatMessageContent(text) {
            const parts = text.split(/(```[\\s\\S]*?```)/g);
            return parts.map(part => {
                if (part.startsWith("```") && part.endsWith("```")) {
                    const lines = part.split("\\n");
                    let lang = "code";
                    let codeStartIdx = 1;
                    if (lines[0].length > 3) {
                        lang = lines[0].substring(3).trim();
                    }
                    const code = lines.slice(codeStartIdx, lines.length - 1).join("\\n");
                    return `<div style="background:#0c0c0e; border:1px solid #333; border-radius:4px; padding:8px; margin:8px 0; font-family:monospace; white-space:pre; overflow-x:auto;"><div style="font-size:10px; color:#888; border-bottom:1px solid #222; margin-bottom:4px; padding-bottom:2px; text-transform:uppercase;">${lang}</div>${escapeHtml(code)}</div>`;
                }
                return escapeHtml(part);
            }).join("");
        }
        
        function escapeHtml(unsafe) {
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }
        
        function scrollToBottom() {
            const chatHistoryDiv = document.getElementById("chat_history");
            chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
        }
        
        function clearChatHistory() {
            chatHistory = [];
            const chatHistoryDiv = document.getElementById("chat_history");
            chatHistoryDiv.innerHTML = `<div class="chat-message assistant"><strong>Ollama:</strong> Memory cleared! How can I help you now?</div>`;
        }
        
        // Handle Enter in chat input
        document.getElementById("chat_input").addEventListener("keydown", function(e) {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
        
        // Initialize preview on load
        window.onload = function() {
            updatePreview();
        };
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import signal
    def handle_sigint(sig, frame):
        print("\nExiting immediately...")
        os._exit(0)
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    load_state()
    app.run(host="0.0.0.0", port=5004, debug=False, use_reloader=False)
