#!/usr/bin/env python3
"""Generate PNG icons for the CRUMB browser extension using only stdlib."""

import struct
import zlib
import os

def make_png(width, height, pixels):
    """Create a PNG file from raw RGBA pixel data."""
    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack('>I', len(data)) + c + crc

    # PNG signature
    sig = b'\x89PNG\r\n\x1a\n'

    # IHDR
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA
    ihdr = chunk(b'IHDR', ihdr_data)

    # IDAT - raw pixel rows with filter byte
    raw = b''
    for y in range(height):
        raw += b'\x00'  # filter: none
        for x in range(width):
            idx = (y * width + x) * 4
            raw += bytes(pixels[idx:idx+4])
    compressed = zlib.compress(raw)
    idat = chunk(b'IDAT', compressed)

    # IEND
    iend = chunk(b'IEND', b'')

    return sig + ihdr + idat + iend


def draw_icon(size):
    """Draw the CRUMB icon at the given size. Returns RGBA pixel list."""
    pixels = [0] * (size * size * 4)

    bg_r, bg_g, bg_b = 0x1a, 0x1a, 0x2e
    fg_r, fg_g, fg_b = 0xf9, 0x73, 0x16

    corner_radius = max(2, size // 8)

    def set_pixel(x, y, r, g, b, a=255):
        if 0 <= x < size and 0 <= y < size:
            idx = (y * size + x) * 4
            pixels[idx] = r
            pixels[idx+1] = g
            pixels[idx+2] = b
            pixels[idx+3] = a

    def in_rounded_rect(x, y, x0, y0, x1, y1, rad):
        if x < x0 or x > x1 or y < y0 or y > y1:
            return False
        # Check corners
        corners = [
            (x0 + rad, y0 + rad),
            (x1 - rad, y0 + rad),
            (x0 + rad, y1 - rad),
            (x1 - rad, y1 - rad),
        ]
        for cx, cy in corners:
            dx = abs(x - cx)
            dy = abs(y - cy)
            if x < x0 + rad or x > x1 - rad:
                if y < y0 + rad or y > y1 - rad:
                    if dx * dx + dy * dy > rad * rad:
                        return False
        return True

    def dist(x1, y1, x2, y2):
        return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

    # Draw background rounded rectangle
    for y in range(size):
        for x in range(size):
            if in_rounded_rect(x, y, 0, 0, size-1, size-1, corner_radius):
                set_pixel(x, y, bg_r, bg_g, bg_b, 255)
            else:
                set_pixel(x, y, 0, 0, 0, 0)

    # Draw the icon: a bread crumb shape (rounded blob) with an arrow
    # Design: An orange circle/blob (the "crumb") with a small arrow pointing right
    # This represents a handoff crumb

    cx = size * 0.42
    cy = size * 0.5
    radius = size * 0.28

    # Draw the crumb (circle)
    for y in range(size):
        for x in range(size):
            d = dist(x, y, cx, cy)
            if d <= radius:
                if in_rounded_rect(x, y, 0, 0, size-1, size-1, corner_radius):
                    set_pixel(x, y, fg_r, fg_g, fg_b, 255)

    # Draw arrow pointing right from the crumb
    arrow_start_x = cx + radius * 0.5
    arrow_end_x = size * 0.82
    arrow_y = cy
    arrow_thickness = max(1, size // 16)
    arrow_head_size = max(2, size // 8)

    for y in range(size):
        for x in range(size):
            if not in_rounded_rect(x, y, 0, 0, size-1, size-1, corner_radius):
                continue
            # Arrow shaft
            if (arrow_start_x <= x <= arrow_end_x and
                abs(y - arrow_y) <= arrow_thickness):
                set_pixel(x, y, fg_r, fg_g, fg_b, 255)
            # Arrow head (triangle)
            dx = arrow_end_x - x
            dy = abs(y - arrow_y)
            if 0 <= dx <= arrow_head_size and dy <= dx * 0.9:
                set_pixel(x, y, fg_r, fg_g, fg_b, 255)

    # Add a small second crumb (dot) trailing behind the main one - "bread crumb trail"
    small_cx = size * 0.14
    small_cy = size * 0.5
    small_r = size * 0.08

    for y in range(size):
        for x in range(size):
            d = dist(x, y, small_cx, small_cy)
            if d <= small_r:
                if in_rounded_rect(x, y, 0, 0, size-1, size-1, corner_radius):
                    # Slightly dimmer orange
                    set_pixel(x, y, 0xd0, 0x60, 0x10, 255)

    return pixels


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for size in [16, 48, 128]:
        print(f"Generating icon-{size}.png ...")
        pixels = draw_icon(size)
        png_data = make_png(size, size, pixels)
        path = os.path.join(script_dir, f"icon-{size}.png")
        with open(path, 'wb') as f:
            f.write(png_data)
        print(f"  Wrote {path} ({len(png_data)} bytes)")

    print("Done.")


if __name__ == '__main__':
    main()
