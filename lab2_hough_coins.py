import argparse
import os
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np


def check_inside_rect(x: float, y: float, x0: float, y0: float, x1: float, y1: float) -> bool:
    return bool(x0 < x < x1 and y0 < y < y1)


def find_image_paths(base_dir: str) -> List[str]:
    paths: List[str] = []
    for name in sorted(os.listdir(base_dir)):
        if name.lower().endswith(".jpg"):
            paths.append(os.path.join(base_dir, name))
    pictures = os.path.join(base_dir, "pictures")
    if os.path.isdir(pictures):
        for name in sorted(os.listdir(pictures)):
            if name.lower().endswith(".jpg"):
                paths.append(os.path.join(pictures, name))
    seen = set()
    unique: List[str] = []
    for p in paths:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return sorted(unique)


def bbox_from_hough_lines(
    lines: Optional[np.ndarray], shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    if lines is None or len(lines) == 0:
        margin = int(min(h, w) * 0.05)
        return margin, margin, w - margin, h - margin
    xs: List[int] = []
    ys: List[int] = []
    for line in lines:
        x2, y2, x3, y3 = line[0]
        xs.extend([x2, x3])
        ys.extend([y2, y3])
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def process_image(
    image_path: str,
    out_dir: Optional[str] = None,
    show: bool = False,
) -> None:
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    if img is None:
        print(f"Nie można wczytać pliku: {image_path}")
        return

    frame_colour = (255, 170, 255)
    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img_cir = cv.GaussianBlur(img, (5, 5), 0)
    img_cir = cv.medianBlur(img_cir, 5)
    img_cir = cv.cvtColor(img_cir, cv.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv.filter2D(img_cir, -1, kernel)

    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    x0, y0, x1, y1 = bbox_from_hough_lines(lines, (h, w))

    cv.line(img, (x0, y0), (x1, y0), frame_colour, 5)
    cv.line(img, (x0, y0), (x0, y1), frame_colour, 5)
    cv.line(img, (x1, y1), (x1, y0), frame_colour, 5)
    cv.line(img, (x1, y1), (x0, y1), frame_colour, 5)

    cv.circle(img, (x0, y0), 5, (255, 0, 0), -1)
    cv.circle(img, (x1, y1), 5, (0, 255, 0), -1)

    if lines is not None:
        for line in lines:
            x2, y2, x3, y3 = line[0]
            cv.line(img, (x2, y2), (x3, y3), (0, 255, 0), 2)

    circles = cv.HoughCircles(
        dst,
        cv.HOUGH_GRADIENT,
        1,
        20,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=40,
    )

    radius_size: List[int] = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            radius_size.append(int(i[2]))

    sum_in = 0.0
    count_in = 0
    sum_out = 0.0
    count_out = 0

    if circles is not None and radius_size:
        max_r = max(radius_size)
        for i in circles[0, :]:
            zm = 0.05
            rad_c = (255, 0, 0)
            if i[2] >= max_r - 3:
                rad_c = (0, 255, 255)
                zm = 5.0
            cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), rad_c, 2)
            center_c = (0, 255, 0)
            if not check_inside_rect(float(i[0]), float(i[1]), float(x0), float(y0), float(x1), float(y1)):
                center_c = (0, 0, 255)
                sum_out += zm
                count_out += 1
            else:
                sum_in += zm
                count_in += 1
            cv.circle(img, (int(i[0]), int(i[1])), 2, center_c, 3)

    base_name = os.path.basename(image_path)
    print(f"\n=== {base_name} ===")
    print("w srodku jest: ", count_in, " monet, suma: ", round(sum_in, 2))
    print("na zewnatrz jest: ", count_out, " monet, suma: ", round(sum_out, 2))
    print("suma: ", round(sum_in + sum_out, 2))

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(image_path), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"detected_{base_name}")
    cv.imwrite(out_path, img)
    print(f"Zapisano: {out_path}")

    if show:
        cv.imshow("img", img)
        cv.waitKey(0)
        cv.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hough: taca + monety, zapis do results/.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Katalog z .jpg (domyślnie: pictures/ obok skryptu, potem katalog skryptu).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Pokaż okno OpenCV po każdym obrazie.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.input_dir:
        input_dir = os.path.abspath(args.input_dir)
        if not os.path.isdir(input_dir):
            print(f"Brak katalogu: {input_dir}")
            return
        image_paths = sorted(
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(".jpg")
        )
    else:
        image_paths = find_image_paths(base_dir)

    if not image_paths:
        print(
            "Brak plików .jpg. Umieść obrazy w folderze projektu lub w pictures/, "
            "albo podaj --input-dir."
        )
        return

    for path in image_paths:
        process_image(path, show=args.show)


if __name__ == "__main__":
    main()
