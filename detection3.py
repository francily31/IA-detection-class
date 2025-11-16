"""
distance_image.py

Requisitos (instalar antes):
pip install ultralytics opencv-python torch torchvision numpy pandas

Opcional (recomendado para melhores distâncias métricas):
pip install git+https://github.com/isl-org/ZoeDepth.git
-- ou instalar a implementação de ZoeDepth que preferir.

Uso:
python distance_image.py
Edite as variáveis no topo: IMAGE_INPUT, IMAGE_OUTPUT, CSV_OUTPUT, TRUE_DISTANCE_M (opcional)
"""

import os
import csv
import numpy as np
import cv2
from ultralytics import YOLO
import torch

# ---------------- CONFIGURAÇÃO ----------------
IMAGE_INPUT  = "manudist.jpeg"      # <<< coloque sua imagem aqui
IMAGE_OUTPUT = "saida_distancia.jpeg"
CSV_OUTPUT   = "distancias.csv"

# Se você souber uma distância real (m) para calibrar MiDaS, coloque aqui.
# Se usar ZoeDepth (métrico), não é necessário.
TRUE_DISTANCE_M = 114.0   # se None, não fará calibração quando usar MiDaS
REFERENCE_CLASS = "person"  # classe usada para calibrar (quando usando MiDaS)

# Escolhas/ajustes
USE_ZOED = True   # tenta usar ZoeDepth (se instalado). Se False, vai direto para MiDaS fallback.
LOWER_REGION_RATIO = 0.35   # usa 35% inferior da bbox para extrair profundidade (mais robusto)
PERCENTILE_REF = 20         # percentile para referência
PERCENTILE_OBJ = 50         # percentile para objetos (mediana)
# ------------------------------------------------

# ---------- Inicializa YOLO ----------
yolo = YOLO("yolov8n.pt")  # certifique-se de ter o peso yolov8n.pt na pasta ou baixe automatic.

# ---------- Tenta importar ZoeDepth (se disponível) ----------
zoe_available = False
if USE_ZOED:
    try:
        # implementação comum: pip install zoedepth (ou git clone do repo)
        # APIs variam; abaixo tentamos chamadas comuns.
        from zoedepth.predict import ZoeDepthPredictor  # tenta import "padrão" (pode variar)
        zoe = ZoeDepthPredictor()  # adaptador hipotético
        zoe_available = True
        print("ZoeDepth disponível: usando ZoeDepth para profundidade métrica (metros).")
    except Exception:
        zoe_available = False
        print("ZoeDepth não disponível. Vou usar MiDaS (fallback).")

# ---------- Se ZoeDepth não disponível, carrega MiDaS ----------
midas = None
midas_transform = None
if not zoe_available:
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        midas_transform = midas_transforms.dpt_transform
        midas.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        midas.eval()
        print("MiDaS carregado (fallback). Atenção: MiDaS retorna profundidade RELATIVA.")
    except Exception as e:
        raise RuntimeError("Nem ZoeDepth nem MiDaS puderam ser carregados. Instale ao menos MiDaS.") from e

# ---------- Funções auxiliares ----------
def compute_depth_zoe(img_bgr):
    """
    Função adaptadora para ZoeDepth.
    Atenção: implementações de ZoeDepth variam; ajuste conforme API instalada.
    Esta função deve retornar depth_map em METROS (H x W).
    """
    # Exemplo genérico - se sua instalação tiver API diferente, adapte aqui.
    # Se ZoeDepthPredictor.predict retorna depth em metros com mesma resolução, ótimo.
    try:
        depth = zoe.predict(img_bgr)  # muitos wrappers usam essa assinatura
        # garante ndarray e shape (H, W)
        if isinstance(depth, np.ndarray):
            return depth
        else:
            # se tiver estrutura diferente, adapte...
            return np.array(depth)
    except Exception:
        raise RuntimeError("ZoeDepth foi detectado, mas a API do wrapper não bateu com este script."
                           " Ajuste compute_depth_zoe conforme sua implementação de ZoeDepth.")


def compute_depth_midas(img_bgr):
    """Executa MiDaS e retorna mapa de profundidade (float) redimensionado para a imagem (H,W)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img_rgb).to(device)
    # garantir batch shape correto
    if len(input_batch.shape) == 3:
        input_batch = input_batch.unsqueeze(0)
    with torch.no_grad():
        prediction = midas(input_batch)
    # output shape: (1, H_out, W_out) possivelmente; pegamos [0]
    if isinstance(prediction, (list, tuple)):
        pred = prediction[0]
    else:
        pred = prediction
    pred = pred.squeeze().cpu().numpy()
    # redimensiona para o tamanho original
    H, W = img_bgr.shape[:2]
    pred_rs = cv2.resize(pred, (W, H), interpolation=cv2.INTER_CUBIC)
    return pred_rs


def get_lower_roi(depth_map, x1, y1, x2, y2, lower_ratio=LOWER_REGION_RATIO):
    H, W = depth_map.shape
    x1c, y1c = max(0, int(x1)), max(0, int(y1))
    x2c, y2c = min(W - 1, int(x2)), min(H - 1, int(y2))
    if x2c <= x1c or y2c <= y1c:
        return None
    h = y2c - y1c
    start = y1c + int(h * (1.0 - lower_ratio))
    if start >= y2c:
        start = y1c
    roi = depth_map[start:y2c, x1c:x2c]
    return roi


def safe_percentile(arr, p):
    a = np.array(arr)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return float(np.percentile(a, p))


# ---------- PROCESSAMENTO DA IMAGEM ----------
def process_image(image_path, image_output, csv_output, true_distance_m=None, reference_class="person"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    H, W = img.shape[:2]

    # 1) Depth map (zoe ou midas)
    if zoe_available:
        depth_map = compute_depth_zoe(img)
        # se ZoeDepth retornar em outra resolução, redimensione:
        if depth_map.shape != (H, W):
            depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_CUBIC)
        zoe_is_metric = True
    else:
        depth_map_rel = compute_depth_midas(img)  # relativo
        # normalizar robustamente para 0..1 (preserva ordem)
        p5 = np.percentile(depth_map_rel, 5)
        p95 = np.percentile(depth_map_rel, 95)
        depth_map = (depth_map_rel - p5) / (p95 - p5 + 1e-9)
        depth_map = np.clip(depth_map, 0.0, 1.0)
        zoe_is_metric = False

    # 2) Detectar objetos com YOLOv8n
    results = yolo(img)[0]

    # 3) Se MiDaS (relativo) e você forneceu true_distance_m, calcular escala usando referência
    scale = None
    if not zoe_is_metric and true_distance_m is not None:
        # procurar referência (maior bbox da classe reference_class)
        candidates = []
        for box in results.boxes:
            cls = int(box.cls[0])
            label = yolo.names[cls]
            if label == reference_class:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = get_lower_roi(depth_map, x1, y1, x2, y2)
                if roi is None or roi.size < 10:
                    continue
                val = safe_percentile(roi.flatten(), PERCENTILE_REF)
                if val is not None:
                    area = (x2 - x1) * (y2 - y1)
                    candidates.append((area, val, (x1, y1, x2, y2)))
        if len(candidates) > 0:
            # escolhe candidato com maior area
            candidates.sort(reverse=True, key=lambda x: x[0])
            depth_ref = candidates[0][1]
            # evitar depth_ref ~ 0
            if depth_ref <= 1e-6:
                depth_ref = np.mean([c[1] for c in candidates])
            scale = true_distance_m / (depth_ref + 1e-9)
            print(f"[Calib] referência detectada. depth_ref={depth_ref:.6f} scale={scale:.6f}")
        else:
            print("[Calib] Nenhuma referência para calibrar (usar true_distance_m). Distâncias serão relativas.")

    # 4) calcular distâncias por bbox e salvar
    out = img.copy()
    rows = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = yolo.names[cls]

        # extrair valor robusto da região inferior
        roi = get_lower_roi(depth_map, x1, y1, x2, y2)
        if roi is None or roi.size == 0:
            depth_val = None
        else:
            depth_val = safe_percentile(roi.flatten(), PERCENTILE_OBJ)

        # obter distância em metros (dependendo do modelo)
        if zoe_available:
            # ZoeDepth deve retornar profundidade métrica (metros) — se sua instalação devolver "relative", NÃO esqueça de calibrar
            if depth_val is None:
                dist_m = None
            else:
                dist_m = float(depth_val)
        else:
            # MiDaS: depth_map é relativo (0..1). Se scale disponível, convertemos para metros.
            if depth_val is None:
                dist_m = None
            else:
                if scale is not None:
                    dist_m = float(depth_val * scale)
                else:
                    # sem escala: retornar valor relativo (unidade arbitrária)
                    dist_m = float(depth_val)

        # desenhar retângulo e texto
        color = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        txt = f"{label}: {dist_m:.2f} m" if dist_m is not None else f"{label}: -"
        cv2.putText(out, txt, (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        rows.append({
            "label": label,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "depth_val": depth_val,
            "distance_m": dist_m
        })

    # salvar saída
    cv2.imwrite(image_output, out)
    print("Output salvo:", image_output)

    # salvar CSV
    with open(csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label","x1","y1","x2","y2","depth_val","distance_m"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print("CSV salvo:", csv_output)

    return rows

# ---------------- EXECUTA ----------------
if __name__ == "__main__":
    rows = process_image(IMAGE_INPUT, IMAGE_OUTPUT, CSV_OUTPUT, TRUE_DISTANCE_M, REFERENCE_CLASS)
    print("Processamento finalizado. Linhas:", len(rows))
