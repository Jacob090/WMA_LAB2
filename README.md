### WMA_LAB2 – Lab 2 (Hough)

**Opis**: Program wykrywa tacę (Hough dla linii) i monety 5 zł / 5 gr (Hough dla okręgów), liczy je na tacy i poza tacą oraz podaje ich łączną wartość. Wyniki wizualne zapisuje do katalogu `results/` obok przetwarzanego obrazu.

**Zgodność z wymaganiami LAB2 (PDF)**:
- Wczytuje tray1.jpg … tray8.jpg ✓
- Transformata Hougha dla linii (taca) ✓
- Transformata Hougha dla okręgów (monety) ✓
- Wyznaczenie krawędzi tacy (prostokąt) ✓
- Oznaczenie monet (okręgi), położenie i promień ✓
- Liczba monet na tacy / poza tacą ✓
- Suma wartości (5 zł, 5 gr) na tacy / poza tacą ✓
- Rozróżnienie monet po promieniu ✓

### Instalacja

- Python 3.8+  
- Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

### Uruchomienie i obrazy

- Umieść obrazy `tray1` ... `tray8` (`.jpg` lub `.png`) w katalogu głównym, `pictures/` lub `assets/`.
- Uruchom:

```bash
python lab2_hough_coins.py
```



```bash
python lab2_hough_coins.py --show
```

```bash
python lab2_hough_coins.py --self-test
```