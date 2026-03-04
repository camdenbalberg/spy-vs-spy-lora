"""
Add sharpie-and-ruler comic panel borders to Spy vs Spy animation frames.

Style: Drawn with a ruler and sharpie marker.
- Lines are STRAIGHT (ruler-guided), not wobbly
- Lines slightly overshoot at corners (hand-drawn imperfection)
- Line ends are rounded (sharpie tip)
- Slight thickness variation between sides (different pen pressure)
- Random asymmetric white padding (cropped-from-page look)

Usage:
  python add_borders.py --test s1_0001.png --input upscaled --output bordered
  python add_borders.py --input upscaled --output bordered
"""

from PIL import Image, ImageDraw
import random
import glob
import os
import argparse


def draw_sharpie_line(draw, start, end, thickness=5):
    """
    Draw a straight line with rounded ends (sharpie style).
    """
    x1, y1 = start
    x2, y2 = end
    
    # Draw the main line
    draw.line([start, end], fill='black', width=thickness)
    
    # Round the endpoints with filled circles (sharpie tip)
    r = thickness / 2
    draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill='black')
    draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill='black')


def add_comic_border(img, base_thickness=5, pad_range=(5, 15), margin_range=(2, 6), rot_max=1.5):
    """
    Add a sharpie-and-ruler comic panel border with random white padding.
    Slightly randomly rotates the result.
    """
    # Random white padding per side (asymmetric = cropped from larger page)
    pad_top = random.randint(*pad_range)
    pad_bottom = random.randint(*pad_range)
    pad_left = random.randint(*pad_range)
    pad_right = random.randint(*pad_range)
    
    # Small outer margin (page edge) — extra padding to survive rotation crop
    base_margin = random.randint(*margin_range)
    rot_buffer = 20  # extra space so rotation doesn't clip the border
    margin = base_margin + rot_buffer
    
    # Calculate new canvas size
    new_width = img.width + pad_left + pad_right + margin * 2
    new_height = img.height + pad_top + pad_bottom + margin * 2
    
    # Create white canvas
    canvas = Image.new('L', (new_width, new_height), 255)
    
    # Paste original image
    paste_x = margin + pad_left
    paste_y = margin + pad_top
    canvas.paste(img, (paste_x, paste_y))
    
    draw = ImageDraw.Draw(canvas)
    
    # Border rectangle coordinates — right at the image edges, no gap
    bx1 = paste_x
    by1 = paste_y
    bx2 = paste_x + img.width - 1
    by2 = paste_y + img.height - 1
    
    # Slight thickness variation per side (different pen pressure per stroke)
    t_top = base_thickness + random.randint(-1, 1)
    t_bottom = base_thickness + random.randint(-1, 1)
    t_left = base_thickness + random.randint(-1, 1)
    t_right = base_thickness + random.randint(-1, 1)
    
    # All lines touch the frame exactly, no overshoot
    draw_sharpie_line(draw, (bx1, by1), (bx2, by1), t_top)       # top
    draw_sharpie_line(draw, (bx1, by2), (bx2, by2), t_bottom)    # bottom
    draw_sharpie_line(draw, (bx1, by1), (bx1, by2), t_left)      # left
    draw_sharpie_line(draw, (bx2, by1), (bx2, by2), t_right)     # right
    
    # Slight random rotation (like a scanned comic page)
    angle = random.uniform(-rot_max, rot_max)
    canvas = canvas.rotate(angle, resample=Image.BICUBIC, fillcolor=255, expand=False)
    
    return canvas


def main():
    parser = argparse.ArgumentParser(description='Add comic panel borders to frames')
    parser.add_argument('--input', required=True, help='Input folder')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--test', default=None, help='Test on single filename')
    parser.add_argument('--thickness', type=int, default=5, help='Base border thickness (default: 5)')
    parser.add_argument('--pad-min', type=int, default=5, help='Min white padding per side (default: 5)')
    parser.add_argument('--pad-max', type=int, default=15, help='Max white padding per side (default: 15)')
    parser.add_argument('--margin-min', type=int, default=2, help='Min outer margin (default: 2)')
    parser.add_argument('--margin-max', type=int, default=6, help='Max outer margin (default: 6)')
    parser.add_argument('--rot-max', type=float, default=1.5, help='Max rotation degrees in either direction (default: 1.5)')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.test:
        filepath = os.path.join(args.input, args.test)
        img = Image.open(filepath).convert('L')
        result = add_comic_border(img, base_thickness=args.thickness, pad_range=(args.pad_min, args.pad_max), margin_range=(args.margin_min, args.margin_max), rot_max=args.rot_max)
        out_path = os.path.join(args.output, f"{os.path.splitext(args.test)[0]}_border_test.png")
        result.save(out_path)
        print(f"Test saved: {out_path}")
        print(f"Original: {img.width}x{img.height} -> Bordered: {result.width}x{result.height}")
    else:
        files = sorted(glob.glob(os.path.join(args.input, '*.png')))
        print(f"Processing {len(files)} images...")
        for f in files:
            img = Image.open(f).convert('L')
            result = add_comic_border(img, base_thickness=args.thickness, pad_range=(args.pad_min, args.pad_max), margin_range=(args.margin_min, args.margin_max), rot_max=args.rot_max)
            out_path = os.path.join(args.output, os.path.basename(f))
            result.save(out_path)
            print(f"  {os.path.basename(f)}")
        print(f"Done! {len(files)} images saved to {args.output}")


if __name__ == '__main__':
    main()