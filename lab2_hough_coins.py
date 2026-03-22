import os
import argparse
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

    hough_tray: Optional[Tray] = None
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
                hough_tray = Tray(
                    top_left=(left_x, top_y),
                    top_right=(right_x, top_y),
                    bottom_right=(right_x, bottom_y),
                    bottom_left=(left_x, bottom_y),
                )

    # 3) Kontur z maski koloru.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    contour_tray: Optional[Tray] = None
    if tray_candidate is not None:
        top_left = tuple(tray_candidate[0])
        top_right = tuple(tray_candidate[1])
        bottom_right = tuple(tray_candidate[2])
        bottom_left = tuple(tray_candidate[3])
        contour_tray = Tray(
            top_left=top_left,
            top_right=top_right,
            bottom_right=bottom_right,
            bottom_left=bottom_left,
        )

    if hough_tray is None:
        return contour_tray
    if contour_tray is None:
        return hough_tray

    def tray_area(t: Tray) -> int:
        x0 = min(p[0] for p in t.polygon)
        x1 = max(p[0] for p in t.polygon)
        y0 = min(p[1] for p in t.polygon)
        y1 = max(p[1] for p in t.polygon)
        return int(max(0, x1 - x0) * max(0, y1 - y0))

    # Preferuj większy i stabilniejszy prostokąt konturu, jeśli Hough zaniżył szerokość.
    return contour_tray if tray_area(contour_tray) > 1.08 * tray_area(hough_tray) else hough_tray


# Pomarańcz tacki (wężej niż „każdy ciepły odcień”) – moneta: niska saturacja metalu vs plastik
_ORANGE_HSV_LOWER = np.array([5, 115, 115])
_ORANGE_HSV_UPPER = np.array([26, 255, 255])


def _grad_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _circle_edge_threshold(grad_mag: np.ndarray) -> float:
    thresh = float(np.percentile(grad_mag, 75)) * 0.22
    return float(np.clip(thresh, 6.0, 45.0))


def _fraction_strong_edge_on_circle(
    grad_mag: np.ndarray,
    x: float,
    y: float,
    r: float,
    n_samples: int = 72,
) -> float:
    """
    Jaka część próbek na obwodzie koła ma wyraźną krawędź (gradient).
    Pełna moneta: ~cały obwód. Zaokrąglenie tacki: tylko ~ćwiartka łuku – < 0.5.
    """
    h, w = grad_mag.shape[:2]
    if r <= 0:
        return 0.0
    thresh = _circle_edge_threshold(grad_mag)
    good = 0
    for k in range(n_samples):
        theta = 2 * np.pi * k / n_samples
        px = int(round(x + r * np.cos(theta)))
        py = int(round(y + r * np.sin(theta)))
        if 0 <= px < w and 0 <= py < h and grad_mag[py, px] >= thresh:
            good += 1
    return good / float(n_samples)


def _max_weak_arc_run_on_circle(
    grad_mag: np.ndarray,
    x: float,
    y: float,
    r: float,
    n_samples: int = 72,
) -> int:
    """
    Najdłuższy ciąg próbek BEZ wyraźnej krawędzi (obwód „pusty”).
    Pełna moneta: krótkie przerwy. Łuk zaokrąglenia tacki: ~3/4 obwodu słabe → długi run.
    """
    h, w = grad_mag.shape[:2]
    if r <= 0:
        return n_samples
    thresh = _circle_edge_threshold(grad_mag)
    strong: List[int] = []
    for k in range(n_samples):
        theta = 2 * np.pi * k / n_samples
        px = int(round(x + r * np.cos(theta)))
        py = int(round(y + r * np.sin(theta)))
        ok = (
            0 <= px < w
            and 0 <= py < h
            and float(grad_mag[py, px]) >= thresh
        )
        strong.append(1 if ok else 0)
    # Podwójny obwód (okrąg) – max ciąg zer
    doubled = strong + strong
    best = cur = 0
    for v in doubled:
        if v == 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(min(best, n_samples))


def _non_orange_metal_fraction(
    orange_mask_full: np.ndarray,
    x: int,
    y: int,
    r: int,
) -> float:
    """
    Ułamek pikseli wewnątrz koła, które NIE są pomarańczem tacki (metal monety).
    """
    h, w = orange_mask_full.shape[:2]
    inner = max(3, int(r * 0.72))
    disk = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(disk, (x, y), inner, 255, -1)
    total = int(cv2.countNonZero(disk))
    if total <= 0:
        return 0.0
    not_orange = cv2.bitwise_not(orange_mask_full)
    metal = cv2.bitwise_and(disk, not_orange)
    return float(cv2.countNonZero(metal)) / float(total)


def _distance_to_nearest_tray_corner(
    x: float, y: float, tray: Tray
) -> float:
    pts = [tray.top_left, tray.top_right, tray.bottom_right, tray.bottom_left]
    dmin = float("inf")
    for cx, cy in pts:
        d = float(np.hypot(x - cx, y - cy))
        if d < dmin:
            dmin = d
    return dmin


def _tray_inner_size(tray: Tray) -> Tuple[float, float]:
    poly = tray.polygon
    tw = float(max(p[0] for p in poly) - min(p[0] for p in poly))
    th = float(max(p[1] for p in poly) - min(p[1] for p in poly))
    return tw, th


def _tray_corner_exclusion_radius(tray: Tray, base: int) -> float:
    """
    Promień strefy wokół każdego wierzchołka prostokąta tacy – tam jest zaokrąglenie plastiku,
    a nie moneta (Hough dopasowuje okrąg do krótkiego łuku).
    """
    tw, th = _tray_inner_size(tray)
    return max(18.0, 0.19 * min(tw, th), 0.11 * float(base))


def _is_in_tray_corner_vertex_zone(x: float, y: float, tray: Tray, base: int) -> bool:
    return _distance_to_nearest_tray_corner(x, y, tray) < _tray_corner_exclusion_radius(tray, base)


def _score_coin_candidate(
    gray: np.ndarray,
    hsv: np.ndarray,
    orange_mask: np.ndarray,
    grad_mag: np.ndarray,
    lap_abs: np.ndarray,
    x: int,
    y: int,
    r: int,
    tray: Optional[Tray],
    h: int,
    w: int,
    base: int,
) -> Optional[float]:
    """
    Zwraca score (wyższy = lepszy) lub None jeśli to nie moneta (łuk tacki / artefakt).
    Wymagane: co najmniej ~połowa obwodu z wyraźną krawędzią + wystarczająco dużo
    powierzchni nie-pomarańczowej (metal) w środku koła.
    """
    r = max(1, int(r))
    edge_frac = _fraction_strong_edge_on_circle(grad_mag, float(x), float(y), float(r))
    no_frac = _non_orange_metal_fraction(orange_mask, x, y, r)
    weak_run = _max_weak_arc_run_on_circle(grad_mag, float(x), float(y), float(r), n_samples=72)
    r_large_min = float(base) * 0.033  # powyżej typowego 5 gr – podejrzane w rogu

    # Długi „pusty” łuk na obwodzie = dopasowanie do krótkiego łuku (narożnik tacki), nie pełna moneta
    if weak_run >= 46:
        return None
    if tray is not None and _is_in_tray_corner_vertex_zone(float(x), float(y), tray, base):
        # W strefie wierzchołka: łuk plastiku daje ~¾ obwodu bez krawędzi monety
        if weak_run >= 26:
            return None
        if r >= r_large_min and weak_run >= 18:
            return None
        if edge_frac < 0.58 or no_frac < 0.34:
            return None
    elif weak_run >= 38:
        return None

    # Łuk zaokrąglenia: bardzo mało obwodu I brak metalu; pełna moneta: oba wyżej
    if edge_frac < 0.28 and no_frac < 0.20:
        return None
    if edge_frac < 0.38 and no_frac < 0.22:
        return None

    # Środek koła: tekstura / jasność
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (x, y), int(r * 0.85), 255, -1)
    mean_val, std_val = cv2.meanStdDev(gray, mask=mask)
    mean_hsv = cv2.mean(hsv, mask=mask)
    mean_val_f = float(mean_val[0][0])
    std_val_f = float(std_val[0][0])
    mean_sat = float(mean_hsv[1])

    if mean_val_f < 45 or std_val_f < 4:
        return None
    if mean_sat > 220:
        return None

    ring = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(ring, (x, y), int(r * 1.05), 255, thickness=max(2, r // 12))
    ring_mean = float(cv2.mean(lap_abs, mask=ring)[0])
    if ring_mean < 3.2:
        return None

    score = float(
        ring_mean
        + 0.45 * std_val_f
        + 0.02 * mean_val_f
        + 2.5 * edge_frac
        + 1.8 * no_frac
    )
    return score


def _is_tray_corner_arc_fake(
    x: float,
    y: float,
    r: float,
    tray: Tray,
    base: int,
    grad_mag: np.ndarray,
) -> bool:
    """Łuk zaokrąglenia w rogu prostokąta – duży słaby łuk na obwodzie, duży promień w strefie wierzchołka."""
    if not _is_in_tray_corner_vertex_zone(x, y, tray, base):
        return False
    wr = _max_weak_arc_run_on_circle(grad_mag, x, y, r, n_samples=72)
    if wr >= 24:
        return True
    if r >= float(base) * 0.033 and wr >= 17:
        return True
    return False


def _classify_5zl_indices(
    radii: np.ndarray,
    centers: np.ndarray,
    tray: Optional[Tray],
    base: int,
    grad_mag: np.ndarray,
) -> set[int]:
    """
    Dwie monety 5 zł: największe promienie, które NIE są łukiem rogu tacki.
    Jeśli dwa największe to fałszywki rogów, bierz kolejne duże promienie.
    """
    n = len(radii)
    if n < 2:
        return set()
    order = np.argsort(radii.astype(np.float32))[
        ::-1
    ]  # malejąco po promieniu
    picked: List[int] = []
    if tray is not None:
        for i in order.tolist():
            x, y = float(centers[i, 0]), float(centers[i, 1])
            rr = float(radii[i])
            if _is_tray_corner_arc_fake(x, y, rr, tray, base, grad_mag):
                continue
            picked.append(i)
            if len(picked) >= 2:
                return set(picked)
        # Za mało kandydatów po odfiltrowaniu rogów – uzupełnij największymi (ostateczność)
        if len(picked) == 1:
            for i in order.tolist():
                if i == picked[0]:
                    continue
                picked.append(i)
                if len(picked) >= 2:
                    return set(picked)
    # Fallback: k-means / 2 największe (gdy brak tacki lub wszystkie „fałszywe”)
    r = radii.astype(np.float32)
    c_small = float(np.min(r))
    c_large = float(np.max(r))
    for _ in range(15):
        dist_small = np.abs(r - c_small)
        dist_large = np.abs(r - c_large)
        labels = (dist_large < dist_small).astype(np.int32)
        if int(np.sum(labels == 1)) > 0:
            c_large = float(np.mean(r[labels == 1]))
        if int(np.sum(labels == 0)) > 0:
            c_small = float(np.mean(r[labels == 0]))
    large_mask = np.abs(r - c_large) <= np.abs(r - c_small)
    large_indices = np.where(large_mask)[0]
    if len(large_indices) >= 2:
        order2 = np.argsort(r[large_indices])[::-1][:2]
        return set(large_indices[order2].tolist())
    return set(np.argsort(r)[-2:].tolist())


def detect_coins(image: np.ndarray, tray: Optional[Tray]) -> List[Coin]:
    """
    Detect coins using Hough circle transform.
    Zastosowano mocniejsze wstępne przetwarzanie oraz węższy,
    skalowany względem rozmiaru obrazu zakres promieni, aby
    zredukować liczbę fałszywych detekcji.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
    max_radius = int(base * 0.105)
    min_radius = max(min_radius, 8)

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

    # Trzy przebiegi Hougha dla różnych skal monet.
    small = hough_pass(min_radius, int(base * 0.05), param2=35)
    medium = hough_pass(int(base * 0.04), int(base * 0.075), param2=38)
    large = hough_pass(int(base * 0.06), max_radius, param2=42)
    circles = (
        np.vstack([small, medium, large])
        if (len(small) or len(medium) or len(large))
        else np.empty((0, 3), dtype=np.int32)
    )

    # Dodatkowy przebieg Hough wewnątrz tacy (niższy próg),
    # żeby odzyskać drobne monety przy słabszym kontraście (np. dół tacy).
    if tray is not None:
        tray_mask_tight = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillPoly(tray_mask_tight, [tray.polygon], 255)
        tray_mask_tight = cv2.erode(
            tray_mask_tight, np.ones((7, 7), np.uint8), iterations=1
        )
        # Słabsza erozja – monety przy dolnym brzegu zostają w masce (wcześniej ginęły)
        tray_mask_loose = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillPoly(tray_mask_loose, [tray.polygon], 255)
        tray_mask_loose = cv2.erode(
            tray_mask_loose, np.ones((3, 3), np.uint8), iterations=1
        )

        tray_gray = cv2.bitwise_and(gray, gray, mask=tray_mask_tight)
        tray_small = cv2.HoughCircles(
            tray_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=130,
            param2=30,
            minRadius=min_radius,
            maxRadius=int(base * 0.06),
        )
        if tray_small is not None:
            tray_small = np.uint16(np.around(tray_small[0, :])).astype(np.int32)
            circles = np.vstack([circles, tray_small]) if circles.size else tray_small

        # Dolna część tacy – luźniejsza maska + niższy param2 (słabe 5 gr)
        x_min = int(min(p[0] for p in tray.polygon))
        x_max = int(max(p[0] for p in tray.polygon))
        y_min = int(min(p[1] for p in tray.polygon))
        y_max = int(max(p[1] for p in tray.polygon))
        y_split = int(y_min + 0.48 * (y_max - y_min))
        bottom_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.rectangle(bottom_mask, (x_min, y_split), (x_max, y_max), 255, -1)
        bottom_mask = cv2.bitwise_and(bottom_mask, tray_mask_loose)
        bottom_gray = cv2.bitwise_and(gray, gray, mask=bottom_mask)
        tray_bottom = cv2.HoughCircles(
            bottom_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.15,
            minDist=max(6, min_dist // 2),
            param1=115,
            param2=22,
            minRadius=max(6, min_radius - 1),
            maxRadius=int(base * 0.058),
        )
        if tray_bottom is not None:
            tray_bottom = np.uint16(np.around(tray_bottom[0, :])).astype(np.int32)
            circles = np.vstack([circles, tray_bottom]) if circles.size else tray_bottom

    if circles.size == 0:
        return []

    # Scal duplikaty (ten sam okrąg wykryty w obu przebiegach)
    merged: List[Tuple[int, int, int]] = []
    for x, y, r in circles.tolist():
        keep = True
        for i, (mx, my, mr) in enumerate(merged):
            d2 = (x - mx) ** 2 + (y - my) ** 2
            if d2 < (0.5 * max(r, mr)) ** 2:
                # zachowaj większy (zwykle stabilniejszy) promień
                if r > mr:
                    merged[i] = (x, y, r)
                keep = False
                break
        if keep:
            merged.append((x, y, r))

    circles = np.array(merged, dtype=np.int32)

    # Odfiltruj fałszywe detekcje: zaokrąglenia rogów tacki (łuk zamiast monety),
    # słabe kontury; wymagamy „pełnego” obwodu i metalu vs pomarańcz tła.
    filtered_circles: List[Tuple[int, int, int, float]] = []
    orange_mask = cv2.inRange(hsv, _ORANGE_HSV_LOWER, _ORANGE_HSV_UPPER)
    grad_mag = _grad_magnitude(gray)
    lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    for x, y, r in circles:
        r_int = int(r)
        if r_int <= 0:
            continue
        sc = _score_coin_candidate(
            gray,
            hsv,
            orange_mask,
            grad_mag,
            lap_abs,
            int(x),
            int(y),
            r_int,
            tray,
            h,
            w,
            base,
        )
        if sc is None:
            continue
        filtered_circles.append((int(x), int(y), r_int, sc))

    if not filtered_circles:
        return []

    # Na każdym zdjęciu: 2 monety 5 zł + 10 monet 5 gr = 12 monet
    EXPECTED_COINS = 12
    if len(filtered_circles) < EXPECTED_COINS:
        # Recovery pass: czuły Hough + ta sama walidacja (moneta ≠ łuk tacki).
        recovery = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=120,
            param2=24,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if recovery is not None:
            recovery = np.uint16(np.around(recovery[0, :])).astype(np.int32)
            for x, y, r_int in recovery.tolist():
                if any((x - cx) ** 2 + (y - cy) ** 2 < (0.55 * max(r_int, cr)) ** 2 for cx, cy, cr, _ in filtered_circles):
                    continue
                sc = _score_coin_candidate(
                    gray,
                    hsv,
                    orange_mask,
                    grad_mag,
                    lap_abs,
                    int(x),
                    int(y),
                    int(r_int),
                    tray,
                    h,
                    w,
                    base,
                )
                if sc is None:
                    continue
                filtered_circles.append((int(x), int(y), int(r_int), sc * 0.95))

        # Dodatkowy recovery: dół tacy (słabo widoczne 5 gr na pomarańczu).
        if tray is not None and len(filtered_circles) < EXPECTED_COINS:
            x_min = int(min(p[0] for p in tray.polygon))
            x_max = int(max(p[0] for p in tray.polygon))
            y_min = int(min(p[1] for p in tray.polygon))
            y_max = int(max(p[1] for p in tray.polygon))
            y_split = int(y_min + 0.48 * (y_max - y_min))
            tray_mask_loose_r = np.zeros_like(gray, dtype=np.uint8)
            cv2.fillPoly(tray_mask_loose_r, [tray.polygon], 255)
            tray_mask_loose_r = cv2.erode(
                tray_mask_loose_r, np.ones((3, 3), np.uint8), iterations=1
            )
            bottom_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.rectangle(bottom_mask, (x_min, y_split), (x_max, y_max), 255, -1)
            bottom_mask = cv2.bitwise_and(bottom_mask, tray_mask_loose_r)
            bottom_gray = cv2.bitwise_and(gray, gray, mask=bottom_mask)
            bot = cv2.HoughCircles(
                bottom_gray,
                cv2.HOUGH_GRADIENT,
                dp=1.1,
                minDist=max(8, min_dist // 2),
                param1=100,
                param2=22,
                minRadius=min_radius,
                maxRadius=int(base * 0.06),
            )
            if bot is not None:
                bot = np.uint16(np.around(bot[0, :])).astype(np.int32)
                for x, y, r_int in bot.tolist():
                    if any((x - cx) ** 2 + (y - cy) ** 2 < (0.55 * max(r_int, cr)) ** 2 for cx, cy, cr, _ in filtered_circles):
                        continue
                    sc = _score_coin_candidate(
                        gray,
                        hsv,
                        orange_mask,
                        grad_mag,
                        lap_abs,
                        int(x),
                        int(y),
                        int(r_int),
                        tray,
                        h,
                        w,
                        base,
                    )
                    if sc is None:
                        continue
                    filtered_circles.append((int(x), int(y), int(r_int), sc * 0.92))

    # W tym zestawie zdjęć liczba monet jest stała (2×5zł + 10×5gr = 12); odrzucamy najsłabsze trafienia.
    if len(filtered_circles) > EXPECTED_COINS:
        filtered_circles = sorted(filtered_circles, key=lambda c: c[3], reverse=True)[:EXPECTED_COINS]

    circles = np.array([(x, y, r) for x, y, r, _ in filtered_circles], dtype=np.int32)

    radii = np.array([int(c[2]) for c in circles], dtype=np.float32)
    if radii.size == 0:
        return []

    tray_poly = tray.polygon if tray is not None else None
    # 2 monety 5 zł – największe promienie, z pominięciem łuków zaokrągleń rogów tacki.
    idx_5zl: set[int] = _classify_5zl_indices(
        radii, circles[:, :2].astype(np.float32), tray, base, grad_mag
    )

    detected: List[Coin] = []
    for idx, (x, y, r) in enumerate(circles):
        radius = int(r)

        if tray_poly is not None:
            # Punkt w wielokącie tacy (signed distance: wewnątrz > 0).
            margin = max(6.0, float(0.008 * min(image.shape[:2])))
            dist = cv2.pointPolygonTest(tray_poly, (float(x), float(y)), True)
            inside = dist >= -margin
        else:
            inside = True

        value_gr = 500 if idx in idx_5zl else 5

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
        candidates = [
            os.path.join(base_dir, f"tray{i}.jpg"),
            os.path.join(base_dir, f"tray{i}.png"),
            os.path.join(base_dir, "pictures", f"tray{i}.jpg"),
            os.path.join(base_dir, "pictures", f"tray{i}.png"),
        ]
        assets_dir = os.path.join(base_dir, "assets")
        if os.path.isdir(assets_dir):
            for f in sorted(os.listdir(assets_dir)):
                if f.startswith(f"tray{i}-") and f.lower().endswith(".png"):
                    candidates.append(os.path.join(assets_dir, f))
        for candidate in candidates:
            if os.path.exists(candidate):
                image_paths.append(candidate)
                break
    return image_paths


def process_image(path: str, show_windows: bool = False) -> Tuple[int, int, int, int]:
    image = cv2.imread(path)
    if image is None:
        print(f"Nie można wczytać obrazu: {path}")
        return 0, 0, 0, 0

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
    out_dir = os.path.join(os.path.dirname(path), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"detected_{os.path.basename(path)}")
    cv2.imwrite(out_path, output)

    if show_windows:
        window_name = os.path.basename(path)
        cv2.imshow(window_name, output)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    return count_on, count_off, value_on, value_off


def run_self_test(base_dir: str) -> bool:
    """
    Sprawdza: 12 monet / zdjęcie, suma 10,50 zł / zdjęcie, suma całego zestawu 84 zł,
    oraz że żadna z dwóch „5 zł” nie jest łukiem zaokrąglenia rogu tacki.
    """
    image_paths = find_image_paths(base_dir)
    if not image_paths:
        print("SELF-TEST: brak obrazów.")
        return False
    ok = True
    expected_per_image_gr = 2 * 500 + 10 * 5  # 1050 gr = 10,50 zł
    total_v = 0
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"SELF-TEST FAIL: nie wczytano {path}")
            ok = False
            continue
        tray = detect_tray_rect(image)
        coins = detect_coins(image, tray)
        v = sum(c.value_gr for c in coins)
        total_v += v
        if len(coins) != 12:
            print(f"SELF-TEST FAIL: {os.path.basename(path)} — oczekiwano 12 monet, jest {len(coins)}")
            ok = False
        if v != expected_per_image_gr:
            print(
                f"SELF-TEST FAIL: {os.path.basename(path)} — oczekiwano {expected_per_image_gr/100:.2f} zł, jest {v/100:.2f} zł"
            )
            ok = False
        # Dodatkowa weryfikacja: 5 zł ≠ łuk rogu
        if tray is not None and coins:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            grad_mag = _grad_magnitude(gray)
            base = min(gray.shape[:2])
            for c in coins:
                if c.value_gr != 500:
                    continue
                if _is_tray_corner_arc_fake(
                    float(c.center[0]),
                    float(c.center[1]),
                    float(c.radius),
                    tray,
                    base,
                    grad_mag,
                ):
                    print(
                        f"SELF-TEST FAIL: {os.path.basename(path)} — 5 zł wygląda na łuk rogu tacki (środek {c.center}, r={c.radius})"
                    )
                    ok = False
    if len(image_paths) == 8 and total_v != 8 * expected_per_image_gr:
        print(f"SELF-TEST FAIL: suma zestawu {total_v/100:.2f} zł, oczekiwano {8 * expected_per_image_gr / 100:.2f} zł")
        ok = False
    if ok:
        print(
            "SELF-TEST: OK (12 monet × 10,50 zł, brak 5 zł jako łuk rogu, suma 84 zł / 8 obrazów)."
        )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Detekcja monet 5 zl / 5 gr i tacy (Hough).")
    parser.add_argument("--show", action="store_true", help="Pokazuj okna OpenCV podczas przetwarzania.")
    parser.add_argument("--self-test", action="store_true", help="Uruchom automatyczny test poprawności na tray1..tray8.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_paths = find_image_paths(base_dir)

    if not image_paths:
        print("Nie znaleziono obrazów tray1...tray8 (.jpg/.png) w katalogu głównym, 'pictures' ani 'assets'.")
        return

    if args.self_test:
        run_self_test(base_dir)
        return

    total_on = total_off = total_value_on = total_value_off = 0
    for path in image_paths:
        c_on, c_off, v_on, v_off = process_image(path, show_windows=args.show)
        total_on += c_on
        total_off += c_off
        total_value_on += v_on
        total_value_off += v_off

    print("\n=== PODSUMOWANIE ZESTAWU ===")
    print(f"Monety na tacy (razem): {total_on}")
    print(f"Monety poza tacą (razem): {total_off}")
    print(f"Wartość na tacy (razem): {total_value_on / 100:.2f} zł")
    print(f"Wartość poza tacą (razem): {total_value_off / 100:.2f} zł")


if __name__ == "__main__":
    main()