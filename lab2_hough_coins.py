import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Coin:
    center: Tuple[int, int]
    radius: int
    value_gr: int
    on_tray: bool


@dataclass
class Tray:
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
    Detect tray rectangle.
    Najpierw używa transformacji Hougha dla linii (wymóg zadania),
    a następnie – aby mieć stabilny wynik – korzysta z segmentacji
    kolorystycznej (taca jest pomarańczowa na ciemnym tle) i wybiera
    największy prostokątny kontur.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1) Segmentacja koloru tacy (pomarańczowy) w przestrzeni HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Bardziej zawężony zakres dla pomarańczowego – żeby nie łapać tła.
    lower_orange = np.array([5, 120, 120])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Delikatna morfologia – domknięcie krawędzi tacy bez dużego „rozlewania”
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 2) Wykrywanie krawędzi tacy i Hough dla linii (wymóg zadania)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120, apertureSize=3)

    # Probabilistyczna transformata Hougha dla linii (dalej Hough dla linii)
    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(min(gray.shape[:2]) * 0.25),
        maxLineGap=20,
    )

    if lines_p is not None:
        vertical: List[Tuple[float, float]] = []  # (x_at_y0, x_at_yh)
        horizontal: List[Tuple[float, float]] = []  # (y_at_x0, y_at_xw)

        h, w = gray.shape[:2]
        for x1, y1, x2, y2 in lines_p[:, 0, :]:
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            angle = (angle + 180) % 180  # [0, 180)

            # Klasyfikacja orientacji
            if angle < 15 or angle > 165:
                # linia pozioma: y = a*x + b
                if dx == 0:
                    continue
                a = dy / dx
                b = y1 - a * x1
                y0 = b
                yw = a * (w - 1) + b
                horizontal.append((float(y0), float(yw)))
            elif 75 < angle < 105:
                # linia pionowa: x = a*y + b
                if dy == 0:
                    continue
                a = dx / dy
                b = x1 - a * y1
                x0 = b
                xh = a * (h - 1) + b
                vertical.append((float(x0), float(xh)))

        if len(vertical) >= 2 and len(horizontal) >= 2:
            # wybierz skrajne linie (lewa/prawa, góra/dół)
            left_x = min(min(v) for v in vertical)
            right_x = max(max(v) for v in vertical)
            top_y = min(min(hh) for hh in horizontal)
            bottom_y = max(max(hh) for hh in horizontal)

            # ogranicz do obrazu
            left_x = int(np.clip(round(left_x), 0, w - 1))
            right_x = int(np.clip(round(right_x), 0, w - 1))
            top_y = int(np.clip(round(top_y), 0, h - 1))
            bottom_y = int(np.clip(round(bottom_y), 0, h - 1))

            # sanity: prostokąt musi mieć sensowny rozmiar
            if right_x - left_x > 0.1 * w and bottom_y - top_y > 0.1 * h:
                return Tray(
                    top_left=(left_x, top_y),
                    top_right=(right_x, top_y),
                    bottom_right=(right_x, bottom_y),
                    bottom_left=(left_x, bottom_y),
                )

    # 3) Fallback: kontur z maski (gdy Hough nie wystarcza)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape[:2]
    image_area = h * w

    tray_candidate = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Taca powinna zajmować umiarkowanie duży obszar, ale nie prawie całe zdjęcie.
        if area < 0.03 * image_area or area > 0.5 * image_area:
            continue

        x, y, rw, rh = cv2.boundingRect(cnt)
        if rw <= 0 or rh <= 0:
            continue

        if max(rw, rh) / min(rw, rh) < 2.0:
            continue

        # Wybierz największy spełniający warunki kontur
        if tray_candidate is None or area > cv2.contourArea(tray_candidate):
            tray_candidate = np.array([[x, y], [x + rw, y], [x + rw, y + rh], [x, y + rh]], dtype=np.int32)

    if tray_candidate is None:
        return None

    top_left = tuple(tray_candidate[0])
    top_right = tuple(tray_candidate[1])
    bottom_right = tuple(tray_candidate[2])
    bottom_left = tuple(tray_candidate[3])

    return Tray(
        top_left=top_left,
        top_right=top_right,
        bottom_right=bottom_right,
        bottom_left=bottom_left,
    )


def detect_coins(image: np.ndarray, tray: Optional[Tray]) -> List[Coin]:
    """
    Detect coins using Hough circle transform.
    Zastosowano mocniejsze wstępne przetwarzanie oraz węższy,
    skalowany względem rozmiaru obrazu zakres promieni, aby
    zredukować liczbę fałszywych detekcji.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Wyrównanie kontrastu i redukcja szumu (stabilniej dla wszystkich tray1..tray8)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = gray.shape[:2]
    base = min(h, w)

    # Zakładamy, że promień monety to kilka procent wymiaru obrazu.
    # Minimalny promień lekko zmniejszony, żeby nie gubić małych monet.
    min_radius = int(base * 0.018)
    max_radius = int(base * 0.11)
    min_radius = max(min_radius, 6)

    min_dist = int(min_radius * 0.9)

    def hough_pass(min_r: int, max_r: int, param2: int) -> np.ndarray:
        c = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=130,
            param2=param2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if c is None:
            return np.empty((0, 3), dtype=np.int32)
        c = np.uint16(np.around(c[0, :]))
        return c.astype(np.int32)

    # Dwa przebiegi: małe i duże monety
    small = hough_pass(min_radius, int(base * 0.05), param2=35)
    large = hough_pass(int(base * 0.045), max_radius, param2=42)
    circles = np.vstack([small, large]) if (len(small) or len(large)) else np.empty((0, 3), dtype=np.int32)

    if circles.size == 0:
        return []

    # Scal duplikaty (ten sam okrąg wykryty w obu przebiegach)
    merged: List[Tuple[int, int, int]] = []
    for x, y, r in circles.tolist():
        keep = True
        for i, (mx, my, mr) in enumerate(merged):
            d2 = (x - mx) ** 2 + (y - my) ** 2
            if d2 < (0.4 * max(r, mr)) ** 2:
                # zachowaj większy (zwykle stabilniejszy) promień
                if r > mr:
                    merged[i] = (x, y, r)
                keep = False
                break
        if keep:
            merged.append((x, y, r))

    circles = np.array(merged, dtype=np.int32)

    # Odfiltruj oczywiste fałszywe detekcje:
    # - bardzo ciemne obszary (tło),
    # - koła o zbyt małym kontraście.
    filtered_circles = []
    # mapa gradientu do filtrowania po krawędziach
    grad = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    grad = np.abs(grad)

    for x, y, r in circles:
        r_int = int(r)
        if r_int <= 0:
            continue

        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r_int * 0.85), 255, -1)
        mean_val, std_val = cv2.meanStdDev(gray, mask=mask)
        mean_val = float(mean_val[0][0])
        std_val = float(std_val[0][0])

        # monety są stosunkowo jasne i mają zróżnicowaną teksturę
        if mean_val < 70 or std_val < 8:
            continue

        # krawędź monety: średni gradient w pierścieniu na obwodzie
        ring = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(ring, (int(x), int(y)), int(r_int * 1.05), 255, thickness=max(2, r_int // 12))
        ring_mean = float(cv2.mean(grad, mask=ring)[0])
        if ring_mean < 6.0:
            continue

        filtered_circles.append((x, y, r_int))

    if not filtered_circles:
        return []

    circles = np.array(filtered_circles, dtype=np.int32)

    radii = np.array([int(c[2]) for c in circles], dtype=np.float32)
    if radii.size == 0:
        return []

    # Rozróżnienie 5 zł / 5 gr:
    # Najpierw spróbuj k-średnich na promieniach (stabilne, gdy są oba nominały),
    # a gdy klastry są zbyt blisko – użyj progu zależnego od rozmiaru obrazu.
    coin_split_radius = base * 0.035
    if len(radii) >= 4:
        data = radii.reshape(-1, 1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        centers = sorted([float(c[0]) for c in centers])
        if (centers[1] - centers[0]) >= 3.0:
            coin_split_radius = (centers[0] + centers[1]) / 2.0

    tray_poly = tray.polygon if tray is not None else None

    detected: List[Coin] = []
    for x, y, r in circles:
        radius = int(r)

        if tray_poly is not None:
            inside = cv2.pointPolygonTest(tray_poly, (int(x), int(y)), measureDist=False) >= 0
        else:
            inside = True

        if radius >= coin_split_radius:
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

    if tray is not None:
        pts = tray.polygon.reshape((-1, 1, 2))
        cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    for coin in coins:
        x, y = coin.center
        color = (0, 255, 255) if coin.value_gr == 500 else (255, 255, 0)
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

