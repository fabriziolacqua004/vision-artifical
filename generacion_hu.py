import cv2
import csv
import glob
import os
import numpy as np
import math

# --- CONFIG ---
LABELS = ["apple", "nike", "mercedes"]
INPUT_DIR = "logos"
OUTPUT_CSV = os.path.join("datos_generados", "hu.csv")
MAX_PER_LABEL = 15  # lee hasta 15 fotos por label
HEADLESS_SAVE = False  # si True NO abre ventanas, guarda las imágenes con contorno en datos_generados/contours/
# --------------

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
if HEADLESS_SAVE:
    os.makedirs(os.path.join(os.path.dirname(OUTPUT_CSV), "contours"), exist_ok=True)


def hu_moments_of_file(filename, show_window=True, save_path=None):
    image = cv2.imread(filename)
    if image is None:
        print(f"[WARN] No se pudo leer: {filename}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Binarización adaptativa similar a tu código original
    bin_img = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 67, 2)

    # Invertir (asumimos figuras negras sobre fondo claro)
    bin_img = 255 - bin_img

    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_ERODE, kernel)

    # Encontrar contornos
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"[WARN] No se encontraron contornos en: {filename}")
        return None

    # Tomar el contorno de mayor área
    shape_contour = max(contours, key=cv2.contourArea)

    # Dibujar el contorno para mostrar qué se está detectando
    disp = image.copy()
    cv2.drawContours(disp, [shape_contour], -1, (0, 255, 0), 2)

    if show_window:
        window_name = f"{os.path.basename(filename)}"
        cv2.imshow(window_name, disp)
        cv2.waitKey(0)  # espera tecla para pasar a la siguiente imagen
        cv2.destroyWindow(window_name)
    elif save_path:
        # guardar la imagen con contorno
        cv2.imwrite(save_path, disp)

    # Momentos y Hu Moments
    moments = cv2.moments(shape_contour)
    hu = cv2.HuMoments(moments).flatten()

    # Log scale seguro (evitar log10(0))
    for i in range(len(hu)):
        val = float(hu[i])
        if abs(val) > 0:
            hu[i] = -1.0 * math.copysign(1.0, val) * math.log10(abs(val))
        else:
            hu[i] = 0.0

    return hu


def generate_hu_csv(labels, input_dir, output_csv, max_per_label=15, headless_save=False):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # (opcional) escribir header:
        writer.writerow([f"hu{i+1}" for i in range(7)] + ["label"])

        for label in labels:
            pattern = os.path.join(input_dir, label, "*")
            files = sorted(glob.glob(pattern))
            if not files:
                print(f"[WARN] No se encontraron archivos para la etiqueta '{label}' en {pattern}")
                continue

            for idx, filepath in enumerate(files):
                if idx >= max_per_label:
                    break
                print(f"[INFO] Procesando {label} -> {os.path.basename(filepath)} ({idx+1}/{max_per_label})")

                save_path = None
                if headless_save:
                    # ruta para guardar las imágenes con contorno
                    basename = f"{label}__{os.path.basename(filepath)}"
                    save_path = os.path.join(os.path.dirname(output_csv), "contours", basename)

                hu = hu_moments_of_file(filepath, show_window=(not headless_save), save_path=save_path)
                if hu is None:
                    print(f"[WARN] Saltando {filepath}")
                    continue

                row = list(map(float, hu)) + [label]
                writer.writerow(row)

    print(f"[DONE] Archivo guardado en: {output_csv}")


if __name__ == "__main__":
    generate_hu_csv(LABELS, INPUT_DIR, OUTPUT_CSV, MAX_PER_LABEL, HEADLESS_SAVE)
