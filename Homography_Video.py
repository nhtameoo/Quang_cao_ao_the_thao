
import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
DEFAULT_VIDEO = r"C:\Users\nhta4\Desktop\video.mp4"
DEFAULT_AD = r"D:\QCAA\IPCV-VirtualAdvertisement-main\data\ad_sign_example1.png"
DEFAULT_OUTPUT = r"C:\Users\nhta4\Desktop\train\Ket_qua_moi.mp4"

# --- YOLO phát hiện biển (tự huấn luyện). Không có file -> chạy bằng màu + chuyển động.
BILLBOARD_MODEL_PATH = r"D:\QCA\Virtual-Advertising-with-Homography-main\weights (1).pt"
BILLBOARD_CONF = 0.3
BILLBOARD_IMGSZ = 640
BILLBOARD_INTERVAL = 1        # chạy YOLO biển cách N khung (tăng để nhanh hơn)

# --- YOLO phân đoạn người/bóng
YOLO_MODEL_PATH = "yolov8n-seg.pt"
YOLO_DEVICE = "cpu"
YOLO_CONF = 0.20
YOLO_IMGSZ = 640
YOLO_INTERVAL = 2
YOLO_CLASSES = [0, 32, 29, 27, 28, 24, 26, 39]
SEG_HIGH = 0.6
SEG_LOW = 0.3
SEG_EXPAND_PX = 20            # lấy mặt nạ người trong vùng biển nới thêm chừng này px
PERSON_DILATE_PX = 1          # 0 = không giãn (hết viền đỏ quanh người); 1-2 nếu còn sót viền
MASK_COMP_MODE = 'object'     # khung bỏ qua phân đoạn: 'object' = dịch theo chuyển động CỦA NGƯỜI (nên dùng),
                              # 'camera' = dịch theo camera (sai khi camera bám theo cầu thủ), 'none' = giữ nguyên

# --- Màu biển (OpenCV: hue 0..179). Biển đỏ -> hai khoảng ở hai đầu dải hue.
BOARD_HUE_RANGES = [(0, 10), (170, 179)]
BOARD_SAT_MIN = 120
BOARD_VAL_MIN = 80

# --- Chữ trắng của biển (Coca-Cola) và cỏ: để đo mép trái/phải và đáy biển chính xác
WHITE_SAT_MAX = 60
WHITE_VAL_MIN = 150
GRASS_HUE = (35, 85)
GRASS_SAT_MIN = 60
GRASS_VAL_MIN = 40
GRASS_ROW_FRAC = 0.5          # hàng là cỏ nếu >= 50% pixel xanh
GRASS_MAX_BELOW = 45          # chỉ nhận mép cỏ nếu cách đáy đỏ không quá chừng này px
EDGE_SNAP = 40                # mép biển cách mép khung hình < chừng này px -> coi như biển kéo dài ra ngoài khung

# --- Đo mép biển
SEARCH_UP = 90                # px dải tìm phía trên hộp gợi ý
SEARCH_DOWN = 50              # px dải tìm phía dưới hộp gợi ý
ROW_FRAC_REL = 0.5            # hàng là biển nếu tỉ lệ đỏ >= 50% của hàng đỏ nhất
ROW_GAP = 8                   # nối khe hở tối đa (hàng) do chữ trắng
COL_FRAC_MIN = 0.5            # cột là biển nếu >= 50% pixel đỏ hoặc trắng (chữ)
COL_GAP = 250                 # nối khe hở tối đa (cột) do cầu thủ che
MIN_ASPECT = 1.5              # biển là dải dài; thấp hơn 3 để vẫn đo được khi chỉ thấy một đoạn biển
YOLO_TRUST_PX = 120           # chỉ tin hộp YOLO nếu lệch dự đoán < chừng này px

# --- Làm mượt, ghép
SMOOTH_ALPHA = 0.35
AD_ALPHA = 0.98
AD_TILE_GAP_RATIO = 0.0       # khoảng trống giữa hai ảnh lặp (tính theo bề cao biển)
AD_BLUR_SIGMA = 0.8           # làm mờ nhẹ cho gần với video gốc đã nén
LOST_LIMIT = 5                # mất biển quá số khung này -> xuất khung hình gốc
DUP_THRESH = 0.6              # khung gần trùng khung trước -> dùng lại kết quả

SHOW_DEBUG_ON_OUTPUT = False  # True: ghi cả viền xanh/chữ vào video xuất ra

WINDOW_NAME = "Thao tac chon 4 goc"
selected_points = []


# ==============================================================================
# GIAO DIỆN CHỌN GÓC
# ==============================================================================
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append((x, y))


def select_board(frame, initial_corners=None):
    """Click 4 góc (TL, TR, BR, BL), ENTER để chạy, R chọn lại.
    Nếu có initial_corners: ENTER ngay (không click) = giữ nguyên khung vàng."""
    global selected_points
    selected_points = []
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        preview = frame.copy()
        if initial_corners is not None and len(selected_points) == 0:
            ic = initial_corners.astype(np.int32)
            overlay = preview.copy()
            cv2.fillConvexPoly(overlay, ic, (0, 255, 255))
            preview = cv2.addWeighted(overlay, 0.25, preview, 0.75, 0)
            cv2.polylines(preview, [ic], True, (0, 255, 255), 2)

        for i, pt in enumerate(selected_points):
            cv2.circle(preview, pt, 5, (0, 0, 255), -1)
            cv2.putText(preview, str(i + 1), (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if i > 0:
                cv2.line(preview, selected_points[i - 1], selected_points[i], (0, 255, 0), 2)
        if len(selected_points) == 4:
            cv2.line(preview, selected_points[3], selected_points[0], (0, 255, 0), 2)

        msg = ("Click 4 goc moi, hoac ENTER de giu khung vang"
               if initial_corners is not None else "Click 4 goc. ENTER: Chay | R: Chon lai")
        cv2.putText(preview, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(20) & 0xFF
        if key == 13:
            if len(selected_points) == 4:
                break
            if len(selected_points) == 0 and initial_corners is not None:
                result = initial_corners.copy()
                cv2.destroyWindow(WINDOW_NAME)
                for _ in range(5):
                    cv2.waitKey(1)
                return result
        if key in (ord("r"), ord("R")):
            selected_points = []

    cv2.destroyWindow(WINDOW_NAME)
    for _ in range(5):          # trên Windows cửa sổ chỉ đóng thật sau khi waitKey chạy
        cv2.waitKey(1)
    return np.array(selected_points, dtype=np.float32)


def box_from_corners(corners):
    xs, ys = corners[:, 0], corners[:, 1]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def box_to_corners(box):
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


# ==============================================================================
# TÌM MÉP BIỂN BẰNG MÀU (sửa lỗi V16)
# ==============================================================================
def board_color_mask(bgr):
    """True ở pixel có màu biển (mặc định đỏ). KHÔNG dùng riêng độ bão hoà: áo vàng,
    cỏ, ghế khán đài cũng rất bão hoà và sẽ lọt vào mask."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_ok = np.zeros(h.shape, dtype=bool)
    for lo, hi in BOARD_HUE_RANGES:
        hue_ok |= (h >= lo) & (h <= hi)
    return hue_ok & (s >= BOARD_SAT_MIN) & (v >= BOARD_VAL_MIN)


def white_mask(bgr):
    """Chữ trắng trên biển (không bão hoà, sáng)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] <= WHITE_SAT_MAX) & (hsv[..., 2] >= WHITE_VAL_MIN)


def grass_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= GRASS_HUE[0]) & (h <= GRASS_HUE[1]) & (s >= GRASS_SAT_MIN) & (v >= GRASS_VAL_MIN)


def longest_run(mask1d, gap):
    """Đoạn True liên tục dài nhất (sau khi nối các khe hở <= gap). Trả (đầu, cuối) hoặc None."""
    m = mask1d.astype(np.uint8)[None, :]
    if gap > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((1, gap), np.uint8))
    xs = np.where(m[0] > 0)[0]
    if len(xs) == 0:
        return None
    runs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
    best = max(runs, key=len)
    return int(best[0]), int(best[-1])


def measure_board(frame, hint):
    """
    Đo hộp biển [x1,y1,x2,y2] TRỰC TIẾP từ màu ở khung hiện tại.
    hint = hộp gợi ý (từ YOLO hoặc dự đoán theo chuyển động), chỉ dùng để chọn dải tìm.
    - Cạnh trên: hàng đầu tiên có nhiều màu biển (đỏ HOẶC chữ trắng) theo chiều ngang.
    - Đáy: mép cỏ ngay dưới biển (chính xác hơn màu đỏ vì đáy biển có dải tối).
    - Trái/phải: đoạn cột liên tục dài nhất; sát mép khung hình thì dính hẳn vào mép.
    Trả (hộp, debug) hoặc (None, None).
    """
    h, w = frame.shape[:2]
    y1, y2 = hint[1], hint[3]
    ys = int(max(0, y1 - SEARCH_UP))
    ye = int(min(h, y2 + SEARCH_DOWN))
    if ye - ys < 20:
        return None, None

    bandimg = frame[ys:ye]
    red = board_color_mask(bandimg)
    board_like = red | white_mask(bandimg)          # chữ trắng cũng là biển

    # --- cạnh trên và đáy sơ bộ
    row_frac = board_like.mean(axis=1)
    if row_frac.max() < 0.15 or red.mean(axis=1).max() < 0.15:   # gần như không thấy màu biển
        return None, None
    r = longest_run(row_frac >= ROW_FRAC_REL * row_frac.max(), ROW_GAP)
    if r is None:
        return None, None
    top, bot = ys + r[0], ys + r[1] + 1

    # --- đáy chính xác: hàng cỏ đầu tiên ngay dưới biển
    gfrac = grass_mask(bandimg).mean(axis=1)
    start = max(r[1] - 5, 0)
    rows = np.where(gfrac[start:] >= GRASS_ROW_FRAC)[0]
    if len(rows):
        grass_top = ys + start + int(rows[0])
        if 0 <= grass_top - bot <= GRASS_MAX_BELOW:
            bot = grass_top

    # --- mép trái/phải: tỉ lệ theo CỘT (chia cho chiều cao, KHÔNG nhân 255)
    shrink = int(0.15 * (r[1] - r[0]))
    core = board_like[r[0] + shrink: r[1] - shrink + 1]
    if core.shape[0] < 3:
        return None, None
    col_frac = core.mean(axis=0)
    c = longest_run(col_frac > COL_FRAC_MIN, COL_GAP)  # nối qua cầu thủ đứng che
    if c is None:
        return None, None
    left, right = c[0], c[1] + 1
    if left <= EDGE_SNAP:                              # biển kéo dài ra ngoài khung hình
        left = 0
    if right >= w - EDGE_SNAP:
        right = w

    bw, bh = right - left, bot - top
    if bh < 15 or bw / max(bh, 1) < MIN_ASPECT:
        return None, None
    box = np.array([left, top, right, bot], dtype=np.float32)
    return box, {"ys": ys, "red": red}


# ==============================================================================
# CHUYỂN ĐỘNG CỦA CAMERA (để neo quảng cáo vào biển)
# ==============================================================================
def prep_gray(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (3, 3), 0)       # giảm răng lược/nhiễu nén


def phase_shift(prev_gray, gray, ys, ye, scale=0.5):
    """Dịch chuyển thô (dx, dy) của dải [ys, ye) bằng tương quan pha."""
    a = prev_gray[ys:ye]
    b = gray[ys:ye]
    a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return np.array([dx, dy], dtype=np.float32) / scale, resp


def estimate_board_motion(prev_gray, gray, box, person):
    """
    Ma trận 2x3 (tịnh tiến + co giãn + quay nhỏ) mô tả chuyển động của dải biển từ
    khung trước sang khung này. Trả None nếu không ước lượng được.
    Cách làm: điểm góc trong biển (bỏ vùng có người) -> LK có dự đoán trước bằng
    tương quan pha -> kiểm tra ngược -> RANSAC. Hỏng thì lùi về tương quan pha.
    """
    h, w = gray.shape
    x1, y1, x2, y2 = [int(v) for v in box]
    pad = int(0.1 * (y2 - y1))
    ys, ye = max(0, y1 + pad), min(h, y2 - pad)
    xs, xe = max(0, x1), min(w, x2)
    if ye - ys < 10 or xe - xs < 50:
        return None

    shift, resp = phase_shift(prev_gray, gray, ys, ye)
    fallback = None
    if resp > 0.1:
        fallback = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float64)

    mask = np.zeros((h, w), np.uint8)
    mask[ys:ye, xs:xe] = 255
    if person is not None:
        pm = (person > 0.2).astype(np.uint8)
        pm = cv2.dilate(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        mask[pm > 0] = 0

    p0 = cv2.goodFeaturesToTrack(prev_gray, 250, 0.01, 7, mask=mask, blockSize=7)
    if p0 is None or len(p0) < 12:
        return fallback

    lk = dict(winSize=(31, 31), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    guess = p0 + shift.reshape(1, 1, 2) if resp > 0.03 else p0.copy()
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, guess.copy(),
                                         flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
    if p1 is None:
        return fallback
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk)
    if back is None:
        return fallback
    fb = np.linalg.norm((back - p0).reshape(-1, 2), axis=1)
    good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.0)
    if np.count_nonzero(good) < 10:
        return fallback

    a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                         ransacReprojThreshold=1.5, maxIters=2000,
                                         confidence=0.995)
    if M is None or inl is None:
        return fallback
    if np.count_nonzero(inl) < max(8, 0.5 * len(a)):
        return fallback
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    if not (0.9 <= scale <= 1.1):
        return fallback
    return M


def shift_person_mask(prev_gray, gray, person):
    """
    Dịch mặt nạ người theo chuyển động CỦA CHÍNH NGƯỜI (trung vị luồng quang học của các điểm
    nằm trong mặt nạ). Không dùng chuyển động camera: khi camera bám theo cầu thủ, người gần như
    đứng yên trên màn hình còn biển trượt đi; dịch theo camera sẽ kéo mặt nạ lệch khỏi người.
    """
    h, w = gray.shape
    pm = (person > 0.5).astype(np.uint8)
    if int(pm.sum()) < 400:
        return person
    core = cv2.erode(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    p0 = cv2.goodFeaturesToTrack(prev_gray, 120, 0.01, 5, mask=core * 255, blockSize=5)
    if p0 is None or len(p0) < 8:
        return person
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(31, 31), maxLevel=4)
    if p1 is None:
        return person
    ok = st.reshape(-1) == 1
    d = (p1 - p0).reshape(-1, 2)[ok]
    if len(d) < 6:
        return person
    dx, dy = np.median(d, axis=0)
    if float(np.hypot(dx, dy)) > 80:
        return person
    Mt = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(person, Mt, (w, h), flags=cv2.INTER_LINEAR)


def apply_M(M, x, y):
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2],
            M[1, 0] * x + M[1, 1] * y + M[1, 2])


def propagate_box(box, M):
    """Dự đoán hộp biển ở khung mới bằng chuyển động camera."""
    x1, y1, x2, y2 = box
    xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx1, _ = apply_M(M, x1, ym)
    nx2, _ = apply_M(M, x2, ym)
    _, ny1 = apply_M(M, xm, y1)
    _, ny2 = apply_M(M, xm, y2)
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


# ==============================================================================
# YOLO
# ==============================================================================
_billboard_model = None
_seg_model = None


def get_billboard_model():
    global _billboard_model
    if _billboard_model is None:
        if _YOLO_AVAILABLE and os.path.exists(BILLBOARD_MODEL_PATH):
            _billboard_model = YOLO(BILLBOARD_MODEL_PATH)
        else:
            _billboard_model = "MISSING"
    return _billboard_model


def detect_billboard(frame):
    model = get_billboard_model()
    if model == "MISSING":
        return None
    res = model.predict(frame, conf=BILLBOARD_CONF, imgsz=BILLBOARD_IMGSZ,
                        device=YOLO_DEVICE, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    return boxes[int(np.argmax(confs))].astype(np.float32)


def get_seg_model():
    global _seg_model
    if _seg_model is None:
        _seg_model = YOLO(YOLO_MODEL_PATH)
    return _seg_model


def yolo_person_alpha(frame, box):
    """Mặt nạ người/bóng dạng mềm trong [0,1] (1 = chắc chắn là người). Chỉ giữ vùng quanh biển."""
    h, w = frame.shape[:2]
    res = get_seg_model().predict(frame, classes=YOLO_CLASSES, conf=YOLO_CONF,
                                  device=YOLO_DEVICE, verbose=False, iou=0.5,
                                  imgsz=YOLO_IMGSZ, retina_masks=True)[0]
    seg = np.zeros((h, w), np.float32)
    if res.masks is not None:
        for m in res.masks.data.cpu().numpy():
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            seg = np.maximum(seg, m)

    e = SEG_EXPAND_PX
    x1, y1, x2, y2 = [int(v) for v in box]
    region = np.zeros((h, w), np.float32)
    region[max(0, y1 - e):min(h, y2 + e), max(0, x1 - e):min(w, x2 + e)] = 1.0
    seg *= region

    person = np.clip((seg - SEG_LOW) / (SEG_HIGH - SEG_LOW), 0.0, 1.0)
    if PERSON_DILATE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * PERSON_DILATE_PX + 1, 2 * PERSON_DILATE_PX + 1))
        person = cv2.dilate(person, k)
    return person


# ==============================================================================
# THEO DÕI HỘP, LÀM MƯỢT, TRACKER
# ==============================================================================
class SimpleTracker:
    def __init__(self, max_lost=20):
        self.last_box = None
        self.lost = 0
        self.max_lost = max_lost

    def update(self, boxes, confs):
        if len(boxes) == 0:
            self.lost += 1
            if self.lost > self.max_lost:
                self.last_box = None
            return self.last_box
        if self.last_box is None:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            self.last_box = boxes[int(np.argmax(areas))]
            self.lost = 0
            return self.last_box
        ious = np.array([self._iou(self.last_box, b) for b in boxes])
        if ious.max() > 0.2:
            self.last_box = boxes[int(np.argmax(ious))]
            self.lost = 0
        else:
            self.lost += 1
        return self.last_box

    @staticmethod
    def _iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-6)


class BoxSmoother:
    """EMA kiểu 'dự đoán - hiệu chỉnh': trộn phép đo mới với hộp ĐÃ DỊCH theo chuyển động
    camera (không phải hộp cũ), nên làm mượt mà không bị trễ khi camera quay."""
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.smoothed = None

    def update(self, measured, predicted=None):
        measured = np.asarray(measured, dtype=np.float32)
        base = predicted if predicted is not None else self.smoothed
        if base is None:
            self.smoothed = measured.copy()
        else:
            self.smoothed = (self.alpha * measured + (1 - self.alpha) * base).astype(np.float32)
        return self.smoothed


# ==============================================================================
# GHÉP QUẢNG CÁO THEO HỆ TOẠ ĐỘ CỦA BIỂN
# ==============================================================================
def compose_ad(frame, ad, box, x0, person):
    """
    Vẽ ảnh quảng cáo lặp (tile) lên hộp biển. Cạnh trái của ảnh lặp đầu tiên nằm tại
    toạ độ màn hình x0 (x0 được cập nhật theo chuyển động camera -> quảng cáo bám biển).
    Bề rộng mỗi ảnh = tỉ lệ ảnh * bề cao biển, nên không méo và co giãn đúng khi zoom.
    """
    h, w = frame.shape[:2]
    x1, x2 = int(max(0, round(box[0]))), int(min(w, round(box[2])))
    y1, y2 = int(max(0, round(box[1]))), int(min(h, round(box[3])))
    bh = y2 - y1
    if x2 - x1 < 4 or bh < 4:
        return frame.copy()

    ad_h, ad_w = ad.shape[:2]
    tile_w = max(2, int(round(ad_w / ad_h * bh)))
    tile = cv2.resize(ad, (tile_w, bh), interpolation=cv2.INTER_AREA)
    if AD_BLUR_SIGMA > 0:
        tile = cv2.GaussianBlur(tile, (0, 0), AD_BLUR_SIGMA)

    gap = int(round(AD_TILE_GAP_RATIO * bh))
    period = float(tile_w + gap)
    xs = np.arange(x1, x2, dtype=np.float32) + 0.5
    phase = np.mod(xs - x0, period)                  # vị trí trong chu kỳ lặp
    c0 = np.floor(phase).astype(np.int32)
    in_tile = c0 < tile_w
    c0c = np.clip(c0, 0, tile_w - 1)
    c1 = (c0c + 1) % tile_w if gap == 0 else np.minimum(c0c + 1, tile_w - 1)
    f = (phase - c0).astype(np.float32)[None, :, None]
    strip = tile[:, c0c].astype(np.float32) * (1 - f) + tile[:, c1].astype(np.float32) * f

    A = np.repeat(in_tile[None, :].astype(np.float32), bh, axis=0) * AD_ALPHA
    A = cv2.GaussianBlur(A, (3, 3), 0)               # làm mềm viền hộp biển
    A *= (1.0 - person[y1:y2, x1:x2])                # nhân SAU khi làm mờ: vùng người giữ nguyên 100% hình gốc
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    out = frame.copy()
    out[y1:y2, x1:x2] = np.clip(strip * A[..., None] + roi * (1 - A[..., None]),
                                0, 255).astype(np.uint8)
    return out


# ==============================================================================
# PIPELINE
# ==============================================================================
class Pipeline:
    def __init__(self, ad, w, h):
        self.ad, self.w, self.h = ad, w, h
        self.tracker = SimpleTracker(max_lost=20)
        self.smoother = BoxSmoother(SMOOTH_ALPHA)
        self.box = None
        self.x0 = 0.0
        self.lost = 0
        self.prev_gray = None
        self.person = np.zeros((h, w), np.float32)
        self.last_seg = -10 ** 9
        self.frame_idx = 0
        self.last_result = None
        self.last_dbg = {}

    # ---- khởi tạo / chọn lại 4 góc
    def set_corners(self, frame, corners, keep_anchor=False):
        rect = box_from_corners(corners)
        measured, _ = measure_board(frame, rect)         # tinh chỉnh khung do người dùng chọn
        box = measured if measured is not None else rect
        self.box = box.copy()
        self.smoother.smoothed = box.copy()
        if not keep_anchor:
            self.x0 = float(box[0])                      # ảnh quảng cáo đầu tiên bắt đầu ở mép trái biển
        self.lost = 0
        self.tracker = SimpleTracker(max_lost=20)

    def _is_dup(self, gray):
        a = cv2.resize(gray, (160, 62)).astype(np.float32)
        b = cv2.resize(self.prev_gray, (160, 62)).astype(np.float32)
        return float(np.abs(a - b).mean()) < DUP_THRESH

    def process(self, frame):
        gray = prep_gray(frame)

        if self.prev_gray is None:                        # khung đầu tiên
            self.person = yolo_person_alpha(frame, self.box)
            self.last_seg = self.frame_idx
            result = compose_ad(frame, self.ad, self.box, self.x0, self.person)
            self.prev_gray = gray
            self.last_result, self.last_dbg = result, {"measured": self.box, "yolo": None}
            return result, self.last_dbg

        self.frame_idx += 1
        if self._is_dup(gray):                            # khung lặp: dùng lại kết quả
            return self.last_result, self.last_dbg

        # 1. chuyển động camera: dịch điểm neo quảng cáo và dự đoán hộp biển
        M = estimate_board_motion(self.prev_gray, gray, self.box, self.person)
        box_pred = self.box.copy()
        if M is not None:
            ym = (self.box[1] + self.box[3]) / 2.0
            self.x0, _ = apply_M(M, self.x0, ym)
            box_pred = propagate_box(self.box, M)

        # 2. YOLO biển: chỉ làm gợi ý. Bình thường tin dự đoán; chỉ dùng YOLO để TÌM LẠI khi mất dấu
        yolo_box = None
        if self.frame_idx % BILLBOARD_INTERVAL == 0:
            det = detect_billboard(frame)
            if det is not None:
                yolo_box = self.tracker.update(det.reshape(1, 4), np.array([1.0]))
        hint = box_pred
        if yolo_box is not None:
            consistent = (abs(yolo_box[1] - box_pred[1]) < YOLO_TRUST_PX and
                          abs(yolo_box[3] - box_pred[3]) < YOLO_TRUST_PX)
            if self.lost > 0 or consistent:
                hint = yolo_box

        # 3. đo biển trực tiếp bằng màu (thử lại bằng dự đoán nếu gợi ý YOLO không ra)
        measured, mdbg = measure_board(frame, hint)
        if measured is None and hint is not box_pred:
            measured, mdbg = measure_board(frame, box_pred)

        if measured is not None:
            box = self.smoother.update(measured, predicted=box_pred).copy()
            # Mép biển chạm khung hình: dính hẳn vào mép. Nếu không, phép đo (bị cắt tại mép) trộn với
            # dự đoán (đã trượt theo camera) sẽ cho một độ lệch cố định khi camera quay (lỗi mép phải đứt sớm).
            if measured[0] <= 0.5:
                box[0] = 0.0
            if measured[2] >= self.w - 0.5:
                box[2] = float(self.w)
            self.box = box
            self.smoother.smoothed = box.copy()
            self.lost = 0
        else:                                             # mất biển: đi tiếp theo dự đoán
            self.box = box_pred
            self.smoother.smoothed = box_pred.copy()
            self.lost += 1

        # 4. mặt nạ người
        if self.frame_idx - self.last_seg >= YOLO_INTERVAL:
            self.person = yolo_person_alpha(frame, self.box)
            self.last_seg = self.frame_idx
        elif MASK_COMP_MODE == 'object':
            self.person = shift_person_mask(self.prev_gray, gray, self.person)
        elif MASK_COMP_MODE == 'camera' and M is not None:
            self.person = cv2.warpAffine(self.person, M, (self.w, self.h), flags=cv2.INTER_LINEAR)

        # 5. ghép (mất biển lâu thì xuất hình gốc, không vẽ sai chỗ)
        if self.lost <= LOST_LIMIT:
            result = compose_ad(frame, self.ad, self.box, self.x0, self.person)
        else:
            result = frame.copy()

        self.prev_gray = gray
        self.last_result = result
        self.last_dbg = {"measured": measured, "yolo": yolo_box, "mdbg": mdbg, "M_ok": M is not None}
        return result, self.last_dbg


# ==============================================================================
# DASHBOARD (đúng tỉ lệ video) & MAIN
# ==============================================================================
def create_dashboard(frame, result, dbg, person, pipe, frame_idx, total):
    h, w = frame.shape[:2]
    cw = 640
    ch = int(round(cw * h / w))

    p1 = frame.copy()
    if dbg.get("yolo") is not None:
        x1, y1, x2, y2 = dbg["yolo"].astype(int)
        cv2.rectangle(p1, (x1, y1), (x2, y2), (0, 255, 255), 3)
    if pipe.box is not None:
        x1, y1, x2, y2 = pipe.box.astype(int)
        cv2.rectangle(p1, (x1, y1), (x2, y2), (0, 255, 0), 3)

    p2 = np.zeros((h, w, 3), np.uint8)
    mdbg = dbg.get("mdbg")
    if mdbg is not None:
        ys, red = mdbg["ys"], mdbg["red"]
        p2[ys:ys + red.shape[0]] = (red[..., None].astype(np.uint8) * 255)
    p3 = cv2.cvtColor((np.clip(person, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    p4 = result.copy()
    cv2.putText(p4, f"Frame {frame_idx}/{total} | Lost {pipe.lost} | motion {'OK' if dbg.get('M_ok', True) else 'FAIL'}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    if pipe.box is not None:
        cv2.polylines(p4, [box_to_corners(pipe.box).astype(np.int32)], True, (0, 255, 0), 2)

    cells = []
    for img, label, color in ((p1, "1. YOLO (vang) / Do duoc (xanh)", (0, 255, 255)),
                              (p2, "2. Mask mau do (dai biển)", (255, 255, 255)),
                              (p3, "3. Mat na nguoi", (255, 255, 255)),
                              (p4, "4. Ket qua (debug)", (0, 255, 0))):
        c = cv2.resize(img, (cw, ch))
        cv2.putText(c, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cells.append(c)
    return np.vstack((np.hstack(cells[:2]), np.hstack(cells[2:])))


def main():
    if not _YOLO_AVAILABLE:
        print("Chưa cài ultralytics. Chạy: pip install ultralytics")
        return
    if not os.path.exists(DEFAULT_VIDEO) or not os.path.exists(DEFAULT_AD):
        print("Không thấy video hoặc ảnh quảng cáo.")
        return

    cap = cv2.VideoCapture(DEFAULT_VIDEO)
    ok, frame = cap.read()
    if not ok:
        return
    ad_img = cv2.imread(DEFAULT_AD, cv2.IMREAD_COLOR)
    if ad_img is None:
        print("Không đọc được ảnh quảng cáo.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(os.path.dirname(os.path.abspath(DEFAULT_OUTPUT)), exist_ok=True)
    out = cv2.VideoWriter(DEFAULT_OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    get_seg_model()
    if get_billboard_model() == "MISSING":
        print("[Lưu ý] Không có model YOLO biển: chạy bằng màu + chuyển động camera.")

    pipe = Pipeline(ad_img, w, h)
    corners = select_board(frame)
    pipe.set_corners(frame, corners)

    frame_idx = 0
    print("Đang xử lý...  (S: chọn lại 4 góc | ESC: thoát)")
    while True:
        result, dbg = pipe.process(frame)

        if SHOW_DEBUG_ON_OUTPUT:                            # mặc định tắt: video xuất ra SẠCH
            out_frame = result.copy()
            cv2.polylines(out_frame, [box_to_corners(pipe.box).astype(np.int32)], True, (0, 255, 0), 2)
        else:
            out_frame = result
        out.write(out_frame)
        dashboard = create_dashboard(frame, result, dbg, pipe.person, pipe, frame_idx, total)
        cv2.imshow("Virtual Advertising - V17", dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord("s"), ord("S")):
            corners = select_board(frame, initial_corners=box_to_corners(pipe.box))
            pipe.set_corners(frame, corners, keep_anchor=True)

        frame_idx += 1
        ok, frame = cap.read()
        if not ok:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("HOÀN THÀNH! Video:", DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()