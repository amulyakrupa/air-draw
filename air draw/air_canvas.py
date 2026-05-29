import cv2
import numpy as np
import mediapipe as mp
import math
import random
import os
import urllib.request
from datetime import datetime

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Done!")

hand_landmarker = mp.tasks.vision.HandLandmarker
hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
vision_running_mode = mp.tasks.vision.RunningMode
base_options = mp.tasks.BaseOptions

latest_result = None
def callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = hand_landmarker_options(
    base_options=base_options(model_asset_path=model_path),
    running_mode=vision_running_mode.LIVE_STREAM,
    num_hands=1,
    result_callback=callback
)

cap = cv2.VideoCapture(0)
canvas = None
glow_canvas = None
prev_x, prev_y = 0, 0
drawing = False
timestamp = 0

brush_size = 5
rainbow_mode = False
rainbow_hue = 0
current_color = (255, 0, 0)
particles = []

COLORS = {
    ord('b'): (255, 50, 50),
    ord('g'): (50, 255, 50),
    ord('r'): (50, 50, 255),
    ord('y'): (0, 255, 255),
    ord('p'): (255, 50, 255),
    ord('w'): (255, 255, 255),
}

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-3, 3)
        self.life = random.randint(15, 30)
        self.max_life = self.life
        self.size = random.randint(2, 6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1
        self.life -= 1

    def draw(self, frame):
        if self.life > 0:
            alpha = self.life / self.max_life
            size = max(1, int(self.size * alpha))
            b, g, r = self.color
            bright_color = (
                min(255, int(b + (255 - b) * alpha)),
                min(255, int(g + (255 - g) * alpha)),
                min(255, int(r + (255 - r) * alpha))
            )
            cv2.circle(frame, (int(self.x), int(self.y)), size, bright_color, -1)

def get_rainbow_color(hue):
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr[0][0])

def draw_glow(canvas, glow_canvas, x1, y1, x2, y2, color, size):
    for thickness, alpha in [(size + 8, 0.15), (size + 4, 0.3), (size, 1.0)]:
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)
    cv2.line(glow_canvas, (x1, y1), (x2, y2), color, size + 10)

def draw_ui(frame, color, brush_size, rainbow_mode, drawing):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    mode = "RAINBOW" if rainbow_mode else "BRUSH"
    cv2.putText(frame, mode, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"SIZE:{brush_size}", (200, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    status = "DRAWING" if drawing else "READY"
    status_color = (0, 255, 0) if drawing else (100, 100, 100)
    cv2.putText(frame, status, (320, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    cv2.putText(frame, "B/G/R/Y/P/W=Color | +/-=Size | N=Rainbow | C=Clear | S=Save | Q=Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

print("🎨 Air Canvas ULTIMATE loaded!")
print("Controls:")
print("  B=Blue G=Green R=Red Y=Yellow P=Purple W=White")
print("  N = Rainbow mode")
print("  +/- = Brush size")
print("  C = Clear | S = Save | Q = Quit")

with hand_landmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)
            glow_canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp += 1
        landmarker.detect_async(mp_image, timestamp)

        if rainbow_mode:
            rainbow_hue = (rainbow_hue + 3) % 180
            current_color = get_rainbow_color(rainbow_hue)

        if latest_result and latest_result.hand_landmarks:
            lm = latest_result.hand_landmarks[0]
            index_tip = lm[8]
            middle_tip = lm[12]
            ring_tip = lm[16]

            cx = int(index_tip.x * w)
            cy = int(index_tip.y * h)
            index_y = int(index_tip.y * h)
            middle_y = int(middle_tip.y * h)

            cv2.circle(frame, (cx, cy), brush_size + 4, current_color, 2)
            cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

            if index_y < middle_y - 30:
                drawing = True
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = cx, cy

                draw_glow(canvas, glow_canvas, prev_x, prev_y, cx, cy, current_color, brush_size)

                for _ in range(3):
                    particles.append(Particle(cx, cy, current_color))

                prev_x, prev_y = cx, cy
            else:
                drawing = False
                prev_x, prev_y = 0, 0
        else:
            drawing = False
            prev_x, prev_y = 0, 0

        particles = [p for p in particles if p.life > 0]
        for p in particles:
            p.update()
            p.draw(frame)

        blurred_glow = cv2.GaussianBlur(glow_canvas, (21, 21), 0)
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        combined = cv2.add(frame_bg, canvas_fg)
        combined = cv2.addWeighted(combined, 1.0, blurred_glow, 0.4, 0)

        draw_ui(combined, current_color, brush_size, rainbow_mode, drawing)
        cv2.imshow("Air Canvas ULTIMATE", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros_like(frame)
            glow_canvas = np.zeros_like(frame)
            particles = []
        elif key == ord('n'):
            rainbow_mode = not rainbow_mode
            print("Rainbow mode:", "ON" if rainbow_mode else "OFF")
        elif key == ord('s'):
            filename = f"drawing_{datetime.now().strftime('%H%M%S')}.png"
            cv2.imwrite(filename, combined)
            print(f"Saved as {filename}!")
        elif key == ord('+') or key == ord('='):
            brush_size = min(30, brush_size + 2)
        elif key == ord('-'):
            brush_size = max(2, brush_size - 2)
        elif key in COLORS:
            current_color = COLORS[key]
            rainbow_mode = False

cap.release()
cv2.destroyAllWindows()