# -*- coding: utf-8 -*-

"""
SKRYPT DO ZAAWANSOWANEJ KLASYFIKACJI STRON GAZET Z MANIFESTÓW IIIF
----------------------------------------------------------------------
Używa hybrydowego podejścia:
1.  KLASYFIKACJA WIZUALNA za pomocą modelu AI (OpenAI CLIP) do oceny ogólnego wyglądu strony.
2.  ANALIZA STRUKTURALNA za pomocą OCR (Google Tesseract) do wykrywania
    charakterystycznych cech, takich jak bardzo duży tekst w tytule.

Ta kombinacja daje znacznie wyższą skuteczność niż każda z tych metod osobno.
"""
import requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io
import json
import numpy as np
import cv2 # OpenCV do przetwarzania obrazów
import pytesseract

# --- GŁÓWNA KONFIGURACJA ---

# 1. Konfiguracja modelu CLIP (używamy większego modelu dla lepszej jakości)
MODEL_ID = "openai/clip-vit-large-patch14"

# 2. Konfiguracja Tesseract OCR
#    WAŻNE: Jeśli używasz Windows, podaj tutaj ścieżkę do pliku tesseract.exe.
TESSERACT_CMD_PATH = "C:\\Users\\praktyka\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
if TESSERACT_CMD_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH

# 3. Konfiguracja heurystyki dla OCR
MIN_LARGE_TEXT_HEIGHT_PIXELS = 50  # Minimalna wysokość w pikselach, aby uznać tekst za duży
LARGE_TEXT_TO_NORMAL_RATIO = 4.0   # Tekst jest duży, jeśli jest X razy wyższy niż mediana wysokości tekstu na stronie


# --- ŁADOWANIE MODELU CLIP ---
print(f"Ładowanie modelu: {MODEL_ID}...")
print("Uwaga: Przy pierwszym uruchomieniu może to potrwać, ponieważ model musi zostać pobrany.")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    clip_model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print(f"\nModel CLIP załadowany i działa na: {device.upper()}")
except Exception as e:
    print(f"\nBŁĄD KRYTYCZNY: Nie udało się pobrać modelu CLIP. Sprawdź połączenie internetowe. Szczegóły: {e}")
    exit()


def klasyfikuj_obraz_clip(image_bytes: bytes) -> dict:
    """
    Używa modelu CLIP do klasyfikacji wizualnej obrazu.
    Zwraca słownik z najlepszym opisem i jego prawdopodobieństwem.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        opisy = [
    "a photo of a newspaper cover with a title and masthead",
    "a photo of an internal page with articles and blocks of body text",
    "a photo of a page full of advertisements or announcements",
    "a photo of a page with a single large illustration or photograph",
    "a photo of a table of contents or an editorial page"
]
        teksty_do_modelu = [f"zdjęcie przedstawiające {opis}" for opis in opisy]

        inputs = clip_processor(text=teksty_do_modelu, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy().flatten()
        
        najlepszy_indeks = probs.argmax()
        return {
            "kategoria": opisy[najlepszy_indeks],
            "prawdopodobienstwo": probs[najlepszy_indeks],
            "jest_okladka_wg_clip": najlepszy_indeks == 0 # True, jeśli najlepszy opis to okładka
        }
    except Exception as e:
        return {"błąd": f"Błąd przetwarzania obrazu z CLIP: {e}"}


def analizuj_strukture_tekstu_ocr(image_bytes: bytes) -> dict:
    """
    Używa Tesseract OCR do analizy struktury tekstu na obrazie.
    Wykrywa obecność bardzo dużych czcionek, charakterystycznych dla strony tytułowej.
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        ocr_data = pytesseract.image_to_data(img_gray, lang='pol', output_type=pytesseract.Output.DICT)
        
        n_boxes = len(ocr_data['text'])
        all_heights = []
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 60:
                if ocr_data['text'][i].strip():
                    all_heights.append(ocr_data['height'][i])

        if not all_heights:
            return {"znaleziono_duzy_tekst": False, "info": "Nie znaleziono tekstu na stronie."}

        median_height = np.median(all_heights)
        
        large_text_count = 0
        for h in all_heights:
            is_large_by_size = h > MIN_LARGE_TEXT_HEIGHT_PIXELS
            is_large_by_ratio = h > (median_height * LARGE_TEXT_TO_NORMAL_RATIO)
            if is_large_by_size or is_large_by_ratio:
                large_text_count += 1
        
        return {
            "znaleziono_duzy_tekst": large_text_count > 0,
            "liczba_duzych_blokow": large_text_count,
            "mediana_wysokosci_tekstu": round(median_height, 2)
        }
    except Exception as e:
        return {"błąd": f"Błąd podczas analizy OCR: {e}"}


def analizuj_manifest(manifest_url: str, limit_stron: int = 5):
    """
    Główna funkcja orkiestrująca analizę.
    Pobiera manifest, iteruje po stronach i uruchamia na nich analizę CLIP i OCR,
    a następnie podejmuje ostateczną decyzję.
    """
    print("\n" + "="*80)
    print(f"ANALIZA MANIFESTU: {manifest_url}")
    print("="*80)
    
    try:
        manifest_data = requests.get(manifest_url).json()
        canvases = manifest_data.get('sequences', [{}])[0].get('canvases', [])
    except Exception as e:
        print(f"BŁĄD KRYTYCZNY: Nie udało się pobrać lub przetworzyć manifestu: {e}")
        return

    if not canvases:
        print("W manifeście nie znaleziono żadnych stron (canvases).")
        return
    
    print(f"Znaleziono {len(canvases)} stron. Analizuję pierwszych {limit_stron}.")

    ### ZMIANA ### Inicjalizujemy trzy osobne listy na wyniki
    wyniki_koncowe_okladki = [] 
    wyniki_clip_okladki = []
    wyniki_ocr_okladki = []

    for i, canvas in enumerate(canvases[:limit_stron]):
        label = canvas.get('label', '[Brak etykiety]')
        image_service_url = canvas.get('images', [{}])[0].get('resource', {}).get('service', {}).get('@id')
        
        print("-" * 60)
        print(f"Strona {i+1}/{limit_stron}: '{label}'")

        if not image_service_url:
            print("  -> POMINIĘTO (brak linku do serwisu obrazu w manifeście)")
            continue

        image_url = f"{image_service_url.rstrip('/')}/full/1200,/0/default.jpg"
        print(f"  Pobieranie obrazu: {image_url}")

        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
            
            # --- KROK 1: Analiza wizualna CLIP ---
            ocena_clip = klasyfikuj_obraz_clip(image_bytes)
            if "błąd" in ocena_clip:
                print(f"  [BŁĄD CLIP] {ocena_clip['błąd']}")
                continue
            print(f"  > Ocena CLIP: '{ocena_clip['kategoria']}' ({ocena_clip['prawdopodobienstwo']:.2%})")

            # --- KROK 2: Analiza strukturalna OCR ---
            analiza_ocr = analizuj_strukture_tekstu_ocr(image_bytes)
            if "błąd" in analiza_ocr:
                print(f"  [BŁĄD OCR] {analiza_ocr['błąd']}")
                continue
            if analiza_ocr['znaleziono_duzy_tekst']:
                print(f"  > Analiza OCR: Wykryto duży tekst! ({analiza_ocr['liczba_duzych_blokow']} bloków)")
            else:
                print(f"  > Analiza OCR: Nie wykryto dużego tekstu. ({analiza_ocr.get('info', '')})")

            # Tworzymy uniwersalny identyfikator strony do list
            identyfikator_strony = f"Strona {i+1} (Etykieta z manifestu: '{label}')"

            ### ZMIANA ### Zbieramy wyniki do poszczególnych list
            if ocena_clip['jest_okladka_wg_clip']:
                wyniki_clip_okladki.append(identyfikator_strony)
            
            if analiza_ocr['znaleziono_duzy_tekst']:
                wyniki_ocr_okladki.append(identyfikator_strony)

            # --- KROK 3: Ostateczna decyzja ---
            print("\n  ----------------- WYNIK KOŃCOWY -----------------")
            
            jest_okladka = False
            if analiza_ocr['znaleziono_duzy_tekst']:
                print("  >>> JEST NAGŁÓWKIEM (Potwierdzone przez analizę struktury tekstu OCR)")
                jest_okladka = True
            elif ocena_clip['jest_okladka_wg_clip']:
                print("  >>> JEST NAGŁÓWKIEM (Sugerowane przez analizę wizualną CLIP)")
                jest_okladka = True
            else:
                print("  >>> NIE JEST NAGŁÓWKIEM")
            print("  ---------------------------------------------------\n")

            if jest_okladka:
                wyniki_koncowe_okladki.append(identyfikator_strony)

        except requests.exceptions.RequestException as e:
            print(f"  BŁĄD: Nie udało się pobrać obrazu: {e}")
        except Exception as e:
            print(f"  BŁĄD: Wystąpił nieoczekiwany błąd podczas analizy strony: {e}")

    ### ZMIANA ### Całkowicie nowa, szczegółowa sekcja podsumowania
    print("\n" + "#"*80)
    print("### PODSUMOWANIE ANALIZY (WYNIKI SZCZEGÓŁOWE) ###")
    print("#"*80)

    # --- 1. Ostateczne wyniki ---
    print("\n--- 1. Strony uznane ostatecznie za okładki (logika OCR > CLIP) ---")
    if wyniki_koncowe_okladki:
        print(f"Liczba znalezionych okładek: {len(wyniki_koncowe_okladki)}")
        for okladka in wyniki_koncowe_okladki:
            print(f"  - {okladka}")
    else:
        print("  Brak stron ostatecznie sklasyfikowanych jako okładki.")

    # --- 2. Wyniki wg samego CLIP ---
    print("\n--- 2. Strony, które sam model CLIP uznał za okładki ---")
    if wyniki_clip_okladki:
        print(f"Liczba stron: {len(wyniki_clip_okladki)}")
        for okladka in wyniki_clip_okladki:
            print(f"  - {okladka}")
    else:
        print("  Model CLIP nie zidentyfikował żadnej strony jako okładki.")

    # --- 3. Wyniki wg samego OCR ---
    print("\n--- 3. Strony, na których sam OCR wykrył duży tekst (potencjalne okładki) ---")
    if wyniki_ocr_okladki:
        print(f"Liczba stron: {len(wyniki_ocr_okladki)}")
        for okladka in wyniki_ocr_okladki:
            print(f"  - {okladka}")
    else:
        print("  Analiza OCR nie wykryła dużego tekstu na żadnej ze stron.")
    
    print("\n" + "="*80)


### GŁÓWNA CZĘŚĆ PROGRAMU ###
if __name__ == "__main__":
    try:
        pytesseract.get_tesseract_version()
        print(f"Tesseract OCR jest dostępny (wersja: {pytesseract.get_tesseract_version()}).\n")
    except pytesseract.TesseractNotFoundError:
        print("\n" + "!"*80)
        print("BŁĄD KRYTYCZNY: Nie znaleziono Tesseract OCR.")
        print("Upewnij się, że jest zainstalowany i że ścieżka TESSERACT_CMD_PATH w kodzie jest poprawna (szczególnie na Windows).")
        print("!"*80 + "\n")
        exit()

    MANIFEST_DO_ANALIZY_1 = "https://glam.uni.wroc.pl/iiif/GSL_GSL_P_31520_IV_1915_32510/manifest" 
    
    analizuj_manifest(MANIFEST_DO_ANALIZY_1, limit_stron=14)