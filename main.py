# ╔══════════════════════════════════════════════════════════════════╗
# ║         YOLO26 Comparison — Versão Colab / Jupyter              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  PASSO 1 — rode esta célula primeiro (apenas uma vez):          ║
# ║    !pip install ultralytics opencv-python-headless -q           ║
# ║                                                                  ║
# ║  PASSO 2 — edite as configs abaixo e rode o script              ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Instalação automática (caso ainda não instalado) ─────────────
import importlib, subprocess, sys

def _ensure(pkg, import_name=None):
    import_name = import_name or pkg
    if importlib.util.find_spec(import_name) is None:
        print(f"[INFO] Instalando {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("ultralytics")
_ensure("opencv-python-headless", "cv2")

# ── Imports ───────────────────────────────────────────────────────
import os, time, urllib.request, collections
import cv2, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────
#  CONFIGURAÇÕES  ← edite aqui antes de rodar
# ──────────────────────────────────────────────────────────────────
VIDEO_ID   = 1       # 1-8 (veja catálogo abaixo) | None = abre menu
VIDEO_URL  = None    # ex: "https://example.com/video.mp4"
VIDEO_FILE = None    # ex: "/content/meu_video.mp4"

FRAMES  = 60         # quantos frames processar
CONF    = 0.8       # confiança mínima (0.0 – 1.0)
IMGSZ   = 640        # resolução de entrada (640 ou 1280)
DEVICE  = None       # None = auto | "cuda" | "cpu" | "cuda:0"

OUTPUT_PLOT   = "yolo26_comparison.png"
OUTPUT_REPORT = "yolo26_report.txt"
# ──────────────────────────────────────────────────────────────────

# ── Catálogo de vídeos ────────────────────────────────────────────
VIDEO_CATALOG = [
    {
        "id": 1,
        "name": "person-bicycle-car-detection",
        "desc": "Rua movimentada — pessoas, bicicletas, carros (Intel IoT, ~5.7 MB)",
        "url":  "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4",
        "file": "person-bicycle-car.mp4",
    },
    {
        "id": 2,
        "name": "store-aisle-detection",
        "desc": "Corredor de loja — pessoas, carrinhos, produtos (Intel IoT, ~2.7 MB)",
        "url":  "https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4",
        "file": "store-aisle.mp4",
    },
    {
        "id": 3,
        "name": "fruit-and-vegetable-detection",
        "desc": "Mesa com frutas e legumes — muitas classes (Intel IoT, ~2.1 MB)",
        "url":  "https://github.com/intel-iot-devkit/sample-videos/raw/master/fruit-and-vegetable-detection.mp4",
        "file": "fruit-vegetable.mp4",
    },
    {
        "id": 4,
        "name": "classroom",
        "desc": "Sala de aula — pessoas, cadeiras, mesas, objetos (Intel IoT, ~3.4 MB)",
        "url":  "https://github.com/intel-iot-devkit/sample-videos/raw/master/classroom.mp4",
        "file": "classroom.mp4",
    },
    {
        "id": 5,
        "name": "car-detection",
        "desc": "Estrada com tráfego — veículos variados (Intel IoT, ~2.9 MB)",
        "url":  "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
        "file": "car-detection.mp4",
    },
    {
        "id": 6,
        "name": "worker-zone-detection",
        "desc": "Zona de trabalho — pessoas com EPI, equipamentos (Intel IoT, ~3.1 MB)",
        "url":  "https://github.com/intel-iot-devkit/sample-videos/raw/master/worker-zone-detection.mp4",
        "file": "worker-zone.mp4",
    },
    {
        "id": 7,
        "name": "Big Buck Bunny",
        "desc": "Animação outdoor — animais, natureza, objetos variados (~10 MB)",
        "url":  "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        "file": "big-buck-bunny.mp4",
    },
    {
        "id": 8,
        "name": "Elephants Dream",
        "desc": "Animação Blender Foundation — sci-fi com muitos objetos (~13 MB)",
        "url":  "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
        "file": "elephants-dream.mp4",
    },
]

YOLO_VARIANTS  = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]
VARIANT_LABELS = ["Nano\n(n)", "Small\n(s)", "Medium\n(m)", "Large\n(l)", "XLarge\n(x)"]


# ── Funções ───────────────────────────────────────────────────────
def resolve_device(requested):
    if requested:
        print(f"[INFO] Device forçado: {requested.upper()}")
        return requested
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {name}  |  VRAM: {vram:.1f} GB  →  usando CUDA")
        return "cuda"
    print("[INFO] CUDA não encontrado  →  usando CPU")
    return "cpu"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct  = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    fill = int(pct / 2)
    print(f"\r  [{'█'*fill}{'░'*(50-fill)}] {pct:5.1f}%", end="", flush=True)


def download_video(url, path):
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1_048_576
        print(f"[INFO] Vídeo já existe: {path}  ({size_mb:.1f} MB)  — pulando download.")
        return
    print(f"[INFO] Baixando: {url}")
    urllib.request.urlretrieve(url, path, reporthook=_progress)
    print()


def interactive_menu():
    print("\n" + "="*65)
    print("  YOLO26 Comparison  —  Selecione o vídeo de teste")
    print("="*65)
    for v in VIDEO_CATALOG:
        print(f"  [{v['id']}] {v['name']}")
        print(f"       {v['desc']}")
    print("="*65)
    while True:
        try:
            choice = int(input("  Número do vídeo: ").strip())
            entry  = next((v for v in VIDEO_CATALOG if v["id"] == choice), None)
            if entry:
                return entry
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Opção inválida. Tente novamente.")


def extract_frames(video_path, max_frames):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    step  = max(1, total // max_frames)
    frames, idx = [], 0
    while len(frames) < max_frames and idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += step
    cap.release()
    dur = total / fps
    print(f"[INFO] {len(frames)} frames  |  duração: {dur:.1f}s  "
          f"|  total no vídeo: {total}  |  {fps:.0f} fps")
    return frames


def run_yolo26(variant, frames, conf, device, imgsz):
    model = YOLO(f"{variant}.pt")
    names = model.names

    # Warm-up: 3 inferências descartadas para aquecer a GPU/CUDA
    # e garantir que alocações de memória não contaminem os tempos
    if device != "cpu":
        print(f"   [warm-up GPU...]", flush=True)
        for _ in range(3):
            model(frames[0], conf=conf, device=device, imgsz=imgsz, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # garante que a GPU finalizou antes de medir

    unique   = set()
    counts   = collections.Counter()
    total_ms = 0.0
    for i, frame in enumerate(frames):
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.synchronize()  # sincroniza antes de iniciar o timer
        t0 = time.perf_counter()
        results = model(frame, conf=conf, device=device, imgsz=imgsz, verbose=False)
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.synchronize()  # aguarda GPU terminar antes de parar o timer
        total_ms += (time.perf_counter() - t0) * 1000
        for r in results:
            for cls_id in r.boxes.cls.tolist():
                name = names[int(cls_id)]
                unique.add(name)
                counts[name] += 1
        # progresso a cada 10 frames
        if (i + 1) % 10 == 0 or (i + 1) == len(frames):
            print(f"   frame {i+1}/{len(frames)}  "
                  f"|  classes até agora: {len(unique)}", flush=True)
    return unique, counts, total_ms / len(frames) if frames else 0.0


def write_report(summary, video_name, device, imgsz):
    lines = [
        "YOLO26 Version Comparison — Relatório",
        f"Vídeo  : {video_name}",
        f"Device : {device.upper()}",
        f"ImgSz  : {imgsz}",
        "=" * 60,
    ]
    for r in summary:
        lines += [
            f"\n{'─'*45}",
            f"Modelo             : {r['variant'].upper()}",
            f"Classes únicas     : {r['unique_classes']}",
            f"Total de detecções : {r['total_detections']}",
            f"Tempo médio/frame  : {r['avg_ms']:.1f} ms",
            "Top-5 classes      :",
        ]
        for cls, cnt in r["top_classes"]:
            lines.append(f"    {cls:<24} {cnt:>5} detecções")
        lines.append(f"Todas as classes   : {', '.join(r['class_names'])}")
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[✓] Relatório → {OUTPUT_REPORT}")


def plot_results(summary, video_name, conf, device, imgsz):
    variants   = [r["label"]            for r in summary]
    unique_cls = [r["unique_classes"]   for r in summary]
    total_det  = [r["total_detections"] for r in summary]
    avg_times  = [r["avg_ms"]           for r in summary]

    colors = ["#4CC9F0", "#4895EF", "#4361EE", "#7209B7", "#560BAD"]
    x = np.arange(len(variants))

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.patch.set_facecolor("#0F0E17")
    for ax in axes:
        ax.set_facecolor("#1A1A2E")
        ax.spines[:].set_color("#333355")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

    # Subplot 1 — Classes únicas
    bars1 = axes[0].bar(x, unique_cls, color=colors, edgecolor="#ffffff22",
                        linewidth=0.8, zorder=3)
    axes[0].set_xticks(x); axes[0].set_xticklabels(variants, fontsize=9)
    axes[0].set_ylabel("Classes únicas detectadas", fontsize=10)
    axes[0].set_title("Classes Únicas por Variante", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", color="#333355", linestyle="--", alpha=0.6, zorder=0)
    axes[0].set_ylim(0, max(unique_cls, default=1) * 1.28)
    for bar, val in zip(bars1, unique_cls):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.2, str(val),
                     ha="center", va="bottom", color="white",
                     fontsize=13, fontweight="bold")

    # Subplot 2 — Total detecções
    bars2 = axes[1].bar(x, total_det, color=colors, edgecolor="#ffffff22",
                        linewidth=0.8, zorder=3)
    axes[1].set_xticks(x); axes[1].set_xticklabels(variants, fontsize=9)
    axes[1].set_ylabel("Total de detecções acumuladas", fontsize=10)
    axes[1].set_title("Total de Detecções por Variante", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", color="#333355", linestyle="--", alpha=0.6, zorder=0)
    axes[1].set_ylim(0, max(total_det, default=1) * 1.25)
    for bar, val in zip(bars2, total_det):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1, str(val),
                     ha="center", va="bottom", color="white",
                     fontsize=12, fontweight="bold")

    # Subplot 3 — Velocidade
    axes[2].plot(x, avg_times, color="#F72585", marker="o",
                 markersize=9, linewidth=2.5, zorder=3)
    axes[2].fill_between(x, avg_times, alpha=0.15, color="#F72585")
    axes[2].set_xticks(x); axes[2].set_xticklabels(variants, fontsize=9)
    axes[2].set_ylabel("Tempo médio por frame (ms)", fontsize=10)
    axes[2].set_title(f"Velocidade de Inferência ({device.upper()})",
                      fontsize=12, fontweight="bold")
    axes[2].grid(axis="y", color="#333355", linestyle="--", alpha=0.6, zorder=0)
    max_t = max(avg_times, default=1)
    axes[2].set_ylim(0, max_t * 1.35)
    for xi, yi in zip(x, avg_times):
        axes[2].text(xi, yi + max_t * 0.05, f"{yi:.0f} ms",
                     ha="center", va="bottom", color="white",
                     fontsize=9, fontweight="bold")

    fig.suptitle(
        f"YOLO26: Comparação de Variantes  (nano → xl)\n"
        f"Vídeo: {video_name}  ·  conf≥{conf}  ·  imgsz={imgsz}  ·  device={device.upper()}",
        fontsize=12, fontweight="bold", color="white", y=1.03,
    )
    patches = [
        mpatches.Patch(color=c, label=f"YOLO26-{s}")
        for c, s in zip(colors, ["nano", "small", "medium", "large", "xlarge"])
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               facecolor="#1A1A2E", edgecolor="#333355",
               labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    # Exibe inline no Colab / Jupyter
    try:
        from IPython.display import Image, display
        display(Image(OUTPUT_PLOT))
    except ImportError:
        pass
    print(f"[✓] Gráfico → {OUTPUT_PLOT}")


# ── Pipeline principal ────────────────────────────────────────────
def main():
    device = resolve_device(DEVICE)

    # Resolver fonte de vídeo
    if VIDEO_FILE:
        video_path = VIDEO_FILE
        video_name = os.path.basename(VIDEO_FILE)
    elif VIDEO_URL:
        video_path = "custom_video.mp4"
        video_name = VIDEO_URL.split("/")[-1]
        download_video(VIDEO_URL, video_path)
    elif VIDEO_ID:
        entry = next((v for v in VIDEO_CATALOG if v["id"] == VIDEO_ID), None)
        if not entry:
            raise ValueError(f"VIDEO_ID inválido: {VIDEO_ID}. Use 1–{len(VIDEO_CATALOG)}.")
        video_path = entry["file"]
        video_name = entry["name"]
        download_video(entry["url"], video_path)
    else:
        entry      = interactive_menu()
        video_path = entry["file"]
        video_name = entry["name"]
        download_video(entry["url"], video_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")

    frames = extract_frames(video_path, FRAMES)
    if not frames:
        raise RuntimeError("Não foi possível extrair frames do vídeo.")

    summary = []
    print(f"\n{'='*65}")
    print(f"  YOLO26  |  device={device.upper()}  "
          f"|  {len(frames)} frames  |  conf≥{CONF}  |  imgsz={IMGSZ}")
    print(f"{'='*65}")

    for variant, label in zip(YOLO_VARIANTS, VARIANT_LABELS):
        print(f"\n▶  {variant.upper()}", flush=True)
        unique, counts, avg_ms = run_yolo26(variant, frames, CONF, device, IMGSZ)
        total = sum(counts.values())
        summary.append({
            "variant":          variant,
            "label":            label,
            "unique_classes":   len(unique),
            "class_names":      sorted(unique),
            "total_detections": total,
            "top_classes":      counts.most_common(5),
            "avg_ms":           avg_ms,
        })
        print(f"   ✓ Classes únicas  : {len(unique)}")
        print(f"   ✓ Total detecções : {total}")
        print(f"   ✓ Tempo/frame     : {avg_ms:.1f} ms")
        print(f"   ✓ Classes         : {', '.join(sorted(unique))}")

    write_report(summary, video_name, device, IMGSZ)
    plot_results(summary, video_name, CONF, device, IMGSZ)
    print(f"\n{'='*65}  Concluído!  {'='*10}\n")


main()
