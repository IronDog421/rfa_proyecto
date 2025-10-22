from pathlib import Path
import time
import cv2
import numpy as np
import mss
import pygetwindow as gw
from ultralytics import YOLO
import ctypes
from ctypes import wintypes
from pynput.keyboard import Controller, Key
import threading

WIN_TITLE = "pygame window"
MODEL_PATH = "runs/segment/rfa_default_n/weights/best.pt"
DEVICE = "0"
H_PATH = Path("matrixes") / "homography_matrix.npy"
TOPDOWN_W, TOPDOWN_H = 450, 950
MIN_OVERLAP_PERCENT = 0.20
OVERLAP_MODE = "roi"   # "roi" | "min" | "mask"
H_IS_PLANE_TO_IMAGE = False
ROI_INNER = None
H_DEFINED_IN_CLIENT_COORDS = True
DRAW_GRID = True
GRID_STEP_PX = 100

ROI_SPACE = "plane"
DETECTION_ROI_FRAC = (0.10, 0.40, 0.90, 0.60)
BRAKE_COOLDOWN = 2

def make_dpi_aware():
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()

def get_client_bbox(hwnd):
    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                    ("right", wintypes.LONG), ("bottom", wintypes.LONG)]
    class POINT(ctypes.Structure):
        _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

    rc_client = RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rc_client))
    origin = POINT(rc_client.left, rc_client.top)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))

    left, top = origin.x, origin.y
    width  = rc_client.right - rc_client.left
    height = rc_client.bottom - rc_client.top
    return left, top, width, height

def load_homography(H_path: Path, invert: bool = False) -> np.ndarray:
    H = np.load(H_path)
    if H.shape != (3, 3):
        raise ValueError(f"La homografía en {H_path} debe ser 3x3, obtuve {H.shape}")
    H = H.astype(np.float32)
    if invert:
        H = np.linalg.inv(H).astype(np.float32)
    return H

def adjust_h_for_roi(H_img2plane: np.ndarray, roi_offset_xy) -> np.ndarray:
    ox, oy = roi_offset_xy
    T = np.array([[1, 0, ox],
                  [0, 1, oy],
                  [0, 0, 1 ]], dtype=np.float32)
    return H_img2plane @ T

def draw_grid(img, step=100):
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (80, 80, 80), 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (80, 80, 80), 1, cv2.LINE_AA)

def rect_from_frac(frac_rect, width, height):
    x1f, y1f, x2f, y2f = frac_rect
    return (
        int(x1f * width),
        int(y1f * height),
        int(x2f * width),
        int(y2f * height),
    )

def normalize_rect(r):
    x1, y1, x2, y2 = map(float, r)
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return x1, y1, x2, y2

def clip_rect_to_dims(r, w, h):
    x1, y1, x2, y2 = normalize_rect(r)
    x1 = min(max(0, x1), w);  x2 = min(max(0, x2), w)
    y1 = min(max(0, y1), h);  y2 = min(max(0, y2), h)
    return x1, y1, x2, y2

def mask_overlap_fraction(mask_bool: np.ndarray, roi_rect_xyxy, mode="mask") -> float:
    Hh, Ww = mask_bool.shape[:2]
    x1, y1, x2, y2 = clip_rect_to_dims(roi_rect_xyxy, Ww, Hh)
    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    roi_crop = mask_bool[y1i:y2i, x1i:x2i]
    inter = float(np.count_nonzero(roi_crop))
    mask_area = float(np.count_nonzero(mask_bool))
    roi_area = float((x2i - x1i) * (y2i - y1i))
    if mode == "roi":
        denom = roi_area
    elif mode == "min":
        denom = min(mask_area, roi_area)
    else:
        denom = mask_area
    return 0.0 if denom <= 0.0 else inter / denom

def warp_mask_to_plane(mask_bool_img: np.ndarray, H: np.ndarray) -> np.ndarray:
    mask_u8 = (mask_bool_img.astype(np.uint8) * 255)
    warped = cv2.warpPerspective(mask_u8, H, (TOPDOWN_W, TOPDOWN_H),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped >= 128

def project_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = pts_xy.astype(np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, H)
    return proj.reshape(-1, 2)

def bbox_image_to_plane_aabb(bbox_xyxy_img: np.ndarray, H: np.ndarray,
                             clip_w: int, clip_h: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy_img.astype(np.float32)
    x1 = np.clip(x1, 0, clip_w); x2 = np.clip(x2, 0, clip_w)
    y1 = np.clip(y1, 0, clip_h); y2 = np.clip(y2, 0, clip_h)
    corners = np.array([[x1, y1],[x2, y1],[x2, y2],[x1, y2]], dtype=np.float32)
    proj = project_points(H, corners)
    x_min, y_min = np.min(proj, axis=0); x_max, y_max = np.max(proj, axis=0)
    x_min = np.clip(x_min, 0, TOPDOWN_W); x_max = np.clip(x_max, 0, TOPDOWN_W)
    y_min = np.clip(y_min, 0, TOPDOWN_H); y_max = np.clip(y_max, 0, TOPDOWN_H)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

def overlap_fraction_rect(a, b, mode="box"):
    ax1, ay1, ax2, ay2 = normalize_rect(a)
    bx1, by1, bx2, by2 = normalize_rect(b)
    xL, yT = max(ax1, bx1), max(ay1, by1)
    xR, yB = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, xR - xL) * max(0.0, yB - yT)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if mode == "roi":
        denom = area_b
    elif mode == "union":
        denom = area_a + area_b - inter
    elif mode == "min":
        denom = min(area_a, area_b)
    else:
        denom = area_a
    return 0.0 if denom <= 0.0 else inter / denom

class BrakeController:
    def __init__(self, cooldown=3.0):
        self.keyboard = Controller()
        self.cooldown = cooldown
        self.last_brake_time = 0
        self.is_braking = False
        self.lock = threading.Lock()

    def can_brake(self):
        with self.lock:
            return not self.is_braking and (time.time() - self.last_brake_time) >= self.cooldown

    def execute_brake_sequence(self):
        if not self.can_brake():
            return False
        with self.lock:
            self.is_braking = True
            self.last_brake_time = time.time()
        threading.Thread(target=self._brake_sequence_thread, daemon=True).start()
        return True

    def _brake_sequence_thread(self):
        try:
            self.keyboard.press('q'); time.sleep(0.05); self.keyboard.release('q'); time.sleep(0.05)
            self.keyboard.press('w'); time.sleep(2);     self.keyboard.release('w'); time.sleep(0.05)
            self.keyboard.press('q'); time.sleep(0.05);  self.keyboard.release('q'); time.sleep(0.05)
        except Exception as e:
            print(f"Error en secuencia de frenada: {e}")
        finally:
            with self.lock:
                self.is_braking = False

def main():
    make_dpi_aware()

    wins = [w for w in gw.getAllWindows() if WIN_TITLE.lower() in (w.title or "").lower()]
    if not wins:
        titles = [w.title for w in gw.getAllWindows() if w.title]
        raise RuntimeError(f"No se encontró ventana que contenga '{WIN_TITLE}'. Ventanas disponibles (primeras 10): {titles[:10]}")
    win = wins[0]
    if not hasattr(win, "_hWnd"):
        raise RuntimeError("pygetwindow no expone _hWnd (¿no es Windows?).")

    hwnd = win._hWnd
    left_client, top_client, width_client, height_client = get_client_bbox(hwnd)

    if ROI_INNER is None:
        monitor = {"left": int(left_client), "top": int(top_client),
                   "width": int(width_client), "height": int(height_client)}
        roi_offset = (0, 0)
    else:
        x1, y1, x2, y2 = ROI_INNER
        monitor = {"left": int(left_client + x1), "top": int(top_client + y1),
                   "width": int(x2 - x1), "height": int(y2 - y1)}
        roi_offset = (x1, y1)

    print("Capturando bbox físico:", monitor)

    H = load_homography(H_PATH, invert=H_IS_PLANE_TO_IMAGE)

    if ROI_INNER is not None and H_DEFINED_IN_CLIENT_COORDS:
        H = adjust_h_for_roi(H, roi_offset)

    seg_model = YOLO(MODEL_PATH)
    brake_ctrl = BrakeController(cooldown=BRAKE_COOLDOWN)

    cv2.namedWindow("Seg on Simulator ROI", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Seg on Birds-Eye (Horizontal Plane)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seg on Simulator ROI", 1922, 1128)
    cv2.resizeWindow("Seg on Birds-Eye (Horizontal Plane)", TOPDOWN_W, TOPDOWN_H)

    t0 = time.perf_counter()
    with mss.mss() as sct:
        plane_roi_pixels = rect_from_frac(DETECTION_ROI_FRAC, TOPDOWN_W, TOPDOWN_H)

        while True:
            shot = sct.grab(monitor)
            frame = np.array(shot)[:, :, :3].copy()
            H_img, W_img = frame.shape[:2]

            if ROI_SPACE == "image":
                roi_pixels = rect_from_frac(DETECTION_ROI_FRAC, W_img, H_img)
            else:
                roi_pixels = plane_roi_pixels

            results = seg_model.predict(
                source=frame, imgsz=640, conf=0.25, iou=0.7,
                device=DEVICE, verbose=False
            )

            any_hit = False
            plane_masks_for_draw = []

            if results[0].masks is not None and len(results[0].masks) > 0:
                masks_tensor = results[0].masks.data
                N = masks_tensor.shape[0]
                for i in range(N):
                    m = masks_tensor[i].detach().cpu().numpy()
                    m = (m > 0.5).astype(np.uint8)
                    m_resized = cv2.resize(m, (W_img, H_img), interpolation=cv2.INTER_NEAREST).astype(bool)

                    if ROI_SPACE == "image":
                        frac = mask_overlap_fraction(m_resized, roi_pixels, mode=OVERLAP_MODE)
                        if frac >= MIN_OVERLAP_PERCENT:
                            any_hit = True
                    else:
                        m_plane = warp_mask_to_plane(m_resized, H)
                        plane_masks_for_draw.append(m_plane)
                        frac = mask_overlap_fraction(m_plane, roi_pixels, mode=OVERLAP_MODE)
                        if frac >= MIN_OVERLAP_PERCENT:
                            any_hit = True

                    if any_hit:
                        break

            else:
                vis_result = results[0]
                for box in (vis_result.boxes or []):
                    bbox_img = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
                    if ROI_SPACE == "image":
                        frac = overlap_fraction_rect(bbox_img, roi_pixels, mode="box")
                        if frac >= MIN_OVERLAP_PERCENT:
                            any_hit = True
                    else:
                        bbox_plane = bbox_image_to_plane_aabb(bbox_img, H, W_img, H_img)
                        frac = overlap_fraction_rect(bbox_plane, roi_pixels, mode="min")

                        if frac >= MIN_OVERLAP_PERCENT:
                            any_hit = True
                    if any_hit:
                        break

            if any_hit and brake_ctrl.execute_brake_sequence():
                print(f"[{time.perf_counter() - t0:.2f}s] MÁSCARA dentro del ROI - Frenada")

            vis = results[0].plot()

            if ROI_SPACE == "image":
                x1i, y1i, x2i, y2i = map(int, normalize_rect(roi_pixels))
                color_roi_img = (0, 0, 255) if any_hit else (0, 255, 0)
                cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), color_roi_img, 2)
                cv2.putText(vis, "DETECTION ROI (IMAGE)",
                            (x1i, max(25, y1i - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_roi_img, 2)

            birds = cv2.warpPerspective(vis, H, (TOPDOWN_W, TOPDOWN_H), flags=cv2.INTER_LINEAR)
            if DRAW_GRID:
                draw_grid(birds, GRID_STEP_PX)

            if ROI_SPACE == "plane":
                x1p, y1p, x2p, y2p = map(int, normalize_rect(roi_pixels))
                color_roi_plane = (0, 0, 255) if any_hit else (0, 255, 0)
                cv2.rectangle(birds, (x1p, y1p), (x2p, y2p), color_roi_plane, 2)
                cv2.putText(birds, "DETECTION ROI (PLANE)",
                            (x1p, max(25, y1p - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_roi_plane, 2)

                for mp in plane_masks_for_draw:
                    cnts, _ = cv2.findContours((mp.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(birds, cnts, -1, (255, 255, 0), 2)

            elapsed = time.perf_counter() - t0
            status = "BRAKING" if brake_ctrl.is_braking else "MONITORING"
            cv2.putText(vis,   f"{elapsed:6.2f}s | {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(birds, f"{elapsed:6.2f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            cv2.imshow("Seg on Simulator ROI", vis)
            cv2.imshow("Seg on Birds-Eye (Horizontal Plane)", birds)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
