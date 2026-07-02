# Gradients images although, it also can be math like or terminal like titles

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1200, 300


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


pink = np.array(hex_to_rgb("ff6ec4"), dtype=float)  # pink
purple = np.array(hex_to_rgb("7873f5"), dtype=float)  # purple
cyan = np.array(hex_to_rgb("4ADEDE"), dtype=float)  # cyan
black = np.array(hex_to_rgb("0a0a0a"), dtype=float)  # near black

# Gradient colors (radial, top-left to bottom-right feel)
c1 = pink
c2 = purple
c3 = np.array(hex_to_rgb("4ADEDE"), dtype=float)  # cyan
c4 = np.array(hex_to_rgb("0a0a0a"), dtype=float)  # near blacku

yy, xx = np.mgrid[0:H, 0:W]
# normalize coords, radial gradient centered at ~30%,20%
cx, cy = 0.3 * W, 0.2 * H
dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
dist = dist / dist.max()

# blend across 3 stops then to dark edge
t = np.clip(dist * 1.6, 0, 1)
img = np.zeros((H, W, 3), dtype=float)

# piecewise blend c1->c2->c3->c4
stops = [0.0, 0.4, 0.7, 1.0]
colors = [c1, c2, c3, c4]
for i in range(len(stops) - 1):
    s0, s1 = stops[i], stops[i + 1]
    mask = (t >= s0) & (t <= s1)
    local_t = np.clip((t - s0) / (s1 - s0), 0, 1)
    for ch in range(3):
        img[..., ch] = np.where(
            mask,
            colors[i][ch] + (colors[i + 1][ch] - colors[i][ch]) * local_t,
            img[..., ch],
        )

img = np.clip(img, 0, 255).astype(np.uint8)
base = Image.fromarray(img, mode="RGB")

# rounded corners mask
radius = 32
mask = Image.new("L", (W, H), 0)
d = ImageDraw.Draw(mask)
d.rounded_rectangle([0, 0, W, H], radius=radius, fill=255)

# Generate grain noise
rng = np.random.default_rng(42)
noise = rng.normal(loc=128, scale=45, size=(H, W)).clip(0, 255).astype(np.uint8)
noise_img = Image.fromarray(noise, mode="L").convert("RGB")

# Overlay blend mode (manual, since PIL doesn't have 'overlay' built-in)
base_arr = np.asarray(base).astype(float) / 255.0
noise_arr = np.asarray(noise_img).astype(float) / 255.0


def overlay_blend(base, top):
    return np.where(base < 0.5, 2 * base * top, 1 - 2 * (1 - base) * (1 - top))


blended = overlay_blend(base_arr, noise_arr)
grain_strength = 0.35
final_arr = base_arr * (1 - grain_strength) + blended * grain_strength
final_arr = np.clip(final_arr, 0, 1)
final = Image.fromarray((final_arr * 255).astype(np.uint8), mode="RGB")

# Apply rounded mask onto a black canvas (for GitHub, transparency also works)
canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
final_rgba = final.convert("RGBA")
canvas.paste(final_rgba, (0, 0), mask)

# Draw title text
draw = ImageDraw.Draw(canvas)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font = ImageFont.truetype(font_path, 64)
text = "Physi"

bbox = draw.textbbox((0, 0), text, font=font)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
tx, ty = (W - tw) / 2 - bbox[0], (H - th) / 2 - bbox[1]

# subtle shadow for readability
draw.text((tx + 2, ty + 3), text, font=font, fill=(0, 0, 0, 120))
draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))

canvas.save("title-banner.png")
print("saved")
