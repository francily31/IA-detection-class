import cv2
import torch
import numpy as np
from ultralytics import YOLO


# ----------------------------
# 1. CARREGAR MODELOS
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo = YOLO("yolov8n.pt")

# MiDaS oficial
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(device)
midas.eval()

# Transform MANUAL — 100% compatível
def midas_preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (384, 384))

    img = img_resized.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5            # Normalize
    img = np.transpose(img, (2, 0, 1)) # HWC → CHW

    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor


# ----------------------------
# 2. PROFUNDIDADE MIDAS
# ----------------------------
def get_depth(img):
    tensor = midas_preprocess(img)

    with torch.no_grad():
        depth = midas(tensor)[0].cpu().numpy()

    depth = cv2.resize(depth, (img.shape[1], img.shape[0]))

    # Normalize
    d_min = np.percentile(depth, 5)
    d_max = np.percentile(depth, 95)

    depth_norm = (depth - d_min) / (d_max - d_min + 1e-9)
    depth_norm = np.clip(depth_norm, 0, 1)

    return depth_norm


# ----------------------------
# 3. ESTIMATIVA DE DISTÂNCIA
# ----------------------------
# 114 m será usado como referência para calibrar
KNOWN_DISTANCE = 114  # metros
REFERENCE_CLASS = "car"  # carro foi sua referência


def calibrate_scale(depth_value):
    return KNOWN_DISTANCE / (1 - depth_value + 1e-9)


def estimate_distance(depth_map, x1, y1, x2, y2, scale):
    crop = depth_map[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        return None

    d = np.median(crop)
    dist = scale * (1 - d)
    return round(float(dist), 2)


# ----------------------------
# 4. PROCESSAR IMAGEM
# ----------------------------
def process_image(img_path, out_name):
    img = cv2.imread(img_path)
    if img is None:
        print("ERRO: não encontrou imagem:", img_path)
        return

    results = yolo(img)[0]
    depth_map = get_depth(img)

    # --------------
    # 4.1 CALIBRAR ESCALA
    # --------------
    scale = None
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]

        if label == REFERENCE_CLASS:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            crop = depth_map[int(y1):int(y2), int(x1):int(x2)]
            d = np.median(crop)
            scale = calibrate_scale(d)
            print(f"[CALIB] {label}: depth={d:.4f} → scale={scale:.4f}")
            break

    # se não achar referência, usar valor típico
    if scale is None:
        scale = 140
        print("[CALIB] usando escala padrão:", scale)

    # --------------
    # 4.2 DISTÂNCIAS PARA TODOS
    # --------------
    objects = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        label = results.names[cls_id]

        dist = estimate_distance(depth_map, x1, y1, x2, y2, scale)

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {dist} m", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        objects.append((label, dist))

    print("\nObjetos detectados:")
    for obj in objects:
        print(" →", obj)

    cv2.imwrite(out_name, img)
    print("Output salvo:", out_name)


# ----------------------------
# 5. EXECUTAR
# ----------------------------
process_image("manudist.jpeg", "novo12.jpeg")
