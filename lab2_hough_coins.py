import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Coin:
    center: Tuple[int, int]
    radius: int
    value_gr: int  # coin value in grosz (5zl = 500gr, 5gr = 5gr)
    on_tray: bool


@dataclass
class Tray:
    # Four corners of the tray rectangle in image coordinates
    top_left: Tuple[int, int]
    top_right: Tuple[int, int]
    bottom_right: Tuple[int, int]
    bottom_left: Tuple[int, int]

    @property
    def polygon(self) -> np.ndarray:
        return np.array(
            [self.top_left, self.top_right, self.bottom_right, self.bottom_left],
            dtype=np.int32,
        )


def detect_tray_rect(image: np.ndarray) -> Optional[Tray]:
    """
    Detect tray rectangle using Hough line transform.
    We assume the tray is roughly axis-aligned (rectangular).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Standard Hough transform for lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
    if lines is None:
        return None

    vertical_rhos: List[float] = []
    horizontal_rhos: List[float] = []
    vertical_thetas: List[float] = []
    horizontal_thetas: List[float] = []

    for line in lines:
        rho, theta = line[0]
        # Rough separation: vertical vs horizontal
        if theta < np.pi / 4 or theta > 3 * np.pi / 4:
            vertical_rhos.append(rho)
            vertical_thetas.append(theta)
        else:
            horizontal_rhos.append(rho)
            horizontal_thetas.append(theta)

    if len(vertical_rhos) < 2 or len(horizontal_rhos) < 2:
        return None

    # Choose the two extreme vertical and horizontal lines
    left_rho = min(vertical_rhos)
    right_rho = max(vertical_rhos)
    top_rho = min(horizontal_rhos)
    bottom_rho = max(horizontal_rhos)

    # Approximate angles as the mean of corresponding groups
    left_theta = right_theta = float(np.mean(vertical_thetas))
    top_theta = bottom_theta = float(np.mean(horizontal_thetas))

    def line_intersection(rho1: float, theta1: float, rho2: float, theta2: float) -> Tuple[int, int]:
        # Convert from normal form to intersection point
        a1, b1 = np.cos(theta1), np.sin(theta1)
        a2, b2 = np.cos(theta2), np.sin(theta2)
        denom = a1 * b2 - a2 * b1
        if abs(denom) < 1e-6:
            return 0, 0
        x = (b2 * rho1 - b1 * rho2) / denom
        y = (a1 * rho2 - a2 * rho1) / denom
        return int(round(x)), int(round(y))

    top_left = line_intersection(left_rho, left_theta, top_rho, top_theta)
    top_right = line_intersection(right_rho, right_theta, top_rho, top_theta)
    bottom_left = line_intersection(left_rho, left_theta, bottom_rho, bottom_theta)
    bottom_right = line_intersection(right_rho, right_theta, bottom_rho, bottom_theta)

    return Tray(top_left=top_left, top_right=top_right, bottom_right=bottom_right, bottom_left=bottom_left)


def detect_coins(image: np.ndarray, tray: Optional[Tray]) -> List[Coin]:
    """
    Detect coins using Hough circle transform.
    Returns list of Coin objects with estimated value and tray membership.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Hough transform for circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=0,  # let OpenCV estimate upper bound
    )

    if circles is None:
        return []

    circles = np.uint16(np.around(circles[0, :]))

    radii = [int(c[2]) for c in circles]
    if not radii:
        return []

    # Separate two denominations based on radius.
    # We dynamically compute a threshold between small and large coins.
    min_r, max_r = float(min(radii)), float(max(radii))
    if max_r - min_r < 3:
        # Radii too similar, fall back to treating everything as small coins (5gr)
        threshold = max_r + 1.0
    else:
        threshold = (min_r + max_r) / 2.0

    # Prepare tray polygon for point-in-polygon test
    tray_poly = tray.polygon if tray is not None else None

    detected: List[Coin] = []
    for x, y, r in circles:
        radius = int(r)

        if tray_poly is not None:
            inside = cv2.pointPolygonTest(tray_poly, (int(x), int(y)), measureDist=False) >= 0
        else:
            # If tray not detected, assume everything is "on tray"
            inside = True

        # Classify coin value: larger radius -> 5zł, smaller -> 5gr
        if radius >= threshold:
            value_gr = 500
        else:
            value_gr = 5

        detected.append(Coin(center=(int(x), int(y)), radius=radius, value_gr=value_gr, on_tray=inside))

    return detected


def draw_results(image: np.ndarray, tray: Optional[Tray], coins: List[Coin]) -> np.ndarray:
    """
    Draw tray rectangle and detected coins on a copy of the image.
    """
    output = image.copy()

    # Draw tray
    if tray is not None:
        pts = tray.polygon.reshape((-1, 1, 2))
        cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw coins
    for coin in coins:
        x, y = coin.center
        color = (0, 255, 255) if coin.value_gr == 500 else (255, 255, 0)  # 5zł: yellow, 5gr: cyan
        cv2.circle(output, (x, y), coin.radius, color, 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

    return output


def summarize_coins(coins: List[Coin]) -> Tuple[int, int, int, int]:
    """
    Returns:
        count_on_tray, count_off_tray, value_on_tray_gr, value_off_tray_gr
    """
    count_on = sum(1 for c in coins if c.on_tray)
    count_off = len(coins) - count_on
    value_on = sum(c.value_gr for c in coins if c.on_tray)
    value_off = sum(c.value_gr for c in coins if not c.on_tray)
    return count_on, count_off, value_on, value_off


def find_image_paths(base_dir: str) -> List[str]:
    """
    Find tray1.jpg ... tray8.jpg either in the working directory or in a 'pictures' subdirectory.
    """
    image_paths: List[str] = []
    for i in range(1, 9):
        name = f"tray{i}.jpg"
        direct = os.path.join(base_dir, name)
        pictures = os.path.join(base_dir, "pictures", name)
        if os.path.exists(direct):
            image_paths.append(direct)
        elif os.path.exists(pictures):
            image_paths.append(pictures)
    return image_paths


def process_image(path: str) -> None:
    image = cv2.imread(path)
    if image is None:
        print(f"Nie można wczytać obrazu: {path}")
        return

    print(f"\n=== Przetwarzanie obrazu: {os.path.basename(path)} ===")

    tray = detect_tray_rect(image)
    if tray is None:
        print("UWAGA: Nie udało się jednoznacznie wykryć tacy – wszystkie monety zostaną potraktowane jako 'na tacy'.")

    coins = detect_coins(image, tray)
    count_on, count_off, value_on, value_off = summarize_coins(coins)

    print(f"Liczba monet na tacy: {count_on}")
    print(f"Liczba monet poza tacą: {count_off}")
    print(f"Suma wartości monet na tacy: {value_on / 100:.2f} zł")
    print(f"Suma wartości monet poza tacą: {value_off / 100:.2f} zł")

    output = draw_results(image, tray, coins)

    # Show image with annotations
    window_name = os.path.basename(path)
    cv2.imshow(window_name, output)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_paths = find_image_paths(base_dir)

    if not image_paths:
        print("Nie znaleziono obrazów tray1.jpg ... tray8.jpg w katalogu głównym ani w podkatalogu 'pictures'.")
        return

    for path in image_paths:
        process_image(path)


if __name__ == "__main__":
    main()

