### WMA_LAB2 – Lab 2 (Hough)

**Opis**: Program wykrywa tacę (Hough dla linii) i monety 5 zł / 5 gr (Hough dla okręgów), liczy je na tacy i poza tacą oraz podaje ich łączną wartość. Wyniki wizualne zapisuje do katalogu `results/` obok przetwarzanego obrazu.

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