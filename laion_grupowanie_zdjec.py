# -*- coding: utf-8 -*-

import requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io
import json
import numpy as np
import cv2
import pytesseract

# --- GŁÓWNA KONFIGURACJA ---
MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
TESSERACT_CMD_PATH = "C:\\Users\\praktyka\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
if TESSERACT_CMD_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH

MIN_LARGE_TEXT_HEIGHT_PIXELS = 50
LARGE_TEXT_TO_NORMAL_RATIO = 4.0

# --- ŁADOWANIE MODELU CLIP ---
print(f"Ładowanie modelu: {MODEL_ID}...")
print("UWAGA: To bardzo duży model (ok. 2.5 GB).")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    clip_model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print(f"\nModel CLIP załadowany i działa na: {device.upper()}")
except Exception as e:
    print(f"\nBŁĄD KRYTYCZNY: {e}")
    exit()

# --- FUNKCJE POMOCNICZE (bez zmian) ---
def klasyfikuj_obraz_clip_wsadowo(images: list) -> list:
    """
    ### NOWA FUNKCJA WSADOWA ###
    Używa modelu CLIP do klasyfikacji wizualnej CAŁEJ PACZKI obrazów.
    Zwraca listę słowników z wynikami dla każdego obrazu.
    """
    try:
        opisy = [
            "a photo of a newspaper cover with a title and masthead",
            "a photo of an internal page with articles and blocks of body text (not a main title and masthead)",
            "a photo of an internal page full of advertisements or announcements (not a main title and masthead)",
            "a photo of an internal page with a large illustration or photograph (not a main title and masthead)",
            "a photo of a table of contents or an editorial page (not a main title and masthead)"
        ]
        
        inputs = clip_processor(text=opisy, images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        wyniki = []
        for i in range(len(images)):
            prawdopodobienstwa_obrazu = probs[i].cpu().numpy().flatten()
            najlepszy_indeks = prawdopodobienstwa_obrazu.argmax()
            wyniki.append({
                "kategoria": opisy[najlepszy_indeks],
                "prawdopodobienstwo": prawdopodobienstwa_obrazu[najlepszy_indeks],
                "jest_okladka_wg_clip": najlepszy_indeks == 0
            })
        return wyniki
    except Exception as e:
        # Zwracamy listę błędów, aby pętla mogła kontynuować
        return [{"błąd": f"Błąd przetwarzania wsadowego z CLIP: {e}"}] * len(images)


def analizuj_strukture_tekstu_ocr(image_bytes: bytes) -> dict:
    # Ta funkcja pozostaje bez zmian, OCR działa obraz po obrazie
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        ocr_data = pytesseract.image_to_data(img_gray, lang='pol', output_type=pytesseract.Output.DICT)
        all_heights = [ocr_data['height'][i] for i, conf in enumerate(ocr_data['conf']) if int(conf) > 60 and ocr_data['text'][i].strip()]
        if not all_heights:
            return {"znaleziono_duzy_tekst": False, "info": "Nie znaleziono tekstu na stronie."}
        median_height = np.median(all_heights)
        large_text_count = sum(1 for h in all_heights if h > MIN_LARGE_TEXT_HEIGHT_PIXELS or h > (median_height * LARGE_TEXT_TO_NORMAL_RATIO))
        return {"znaleziono_duzy_tekst": large_text_count > 0, "liczba_duzych_blokow": large_text_count, "mediana_wysokosci_tekstu": round(median_height, 2)}
    except Exception as e:
        return {"błąd": f"Błąd podczas analizy OCR: {e}"}

### ZMIANA ### Główna funkcja została przepisana, aby obsługiwać przetwarzanie wsadowe
def analizuj_manifest(manifest_url: str, limit_stron: int = 5, rozmiar_wsadu: int = 8):
    print("\n" + "="*80)
    print(f"ANALIZA MANIFESTU: {manifest_url}")
    print(f"Rozmiar wsadu (batch size): {rozmiar_wsadu}")
    print("="*80)
    
    try:
        manifest_data = requests.get(manifest_url).json()
        canvases = manifest_data.get('sequences', [{}])[0].get('canvases', [])
    except Exception as e:
        print(f"BŁĄD KRYTYCZNY: {e}")
        return

    if not canvases:
        print("W manifeście nie znaleziono stron.")
        return
    
    canvases_do_analizy = canvases[:limit_stron]
    print(f"Znaleziono {len(canvases)} stron. Analizuję pierwszych {len(canvases_do_analizy)}.")

    wyniki_koncowe_okladki = [] 
    wyniki_clip_okladki = []
    wyniki_ocr_okladki = []

    batch_danych = []

    for i, canvas in enumerate(canvases_do_analizy):
        label = canvas.get('label', '[Brak etykiety]')
        image_service_url = canvas.get('images', [{}])[0].get('resource', {}).get('service', {}).get('@id')
        
        print("-" * 60)
        print(f"Strona {i+1}/{len(canvases_do_analizy)}: '{label}'")

        if not image_service_url:
            print("  -> POMINIĘTO (brak linku)")
            continue

        image_url = f"{image_service_url.rstrip('/')}/full/1200,/0/default.jpg"
        print(f"  Pobieranie obrazu: {image_url}")

        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            # Zbieramy dane do paczki
            batch_danych.append({
                "image_bytes": response.content,
                "label": label,
                "numer_strony": i + 1
            })
        except requests.exceptions.RequestException as e:
            print(f"  BŁĄD POBIERANIA: {e}")

        # Jeśli paczka jest pełna LUB to ostatni element, przetwarzamy paczkę
        if len(batch_danych) == rozmiar_wsadu or i == len(canvases_do_analizy) - 1:
            print(f"\n--- Przetwarzanie wsadu {len(batch_danych)} obrazów... ---")
            
            # Krok 1: Klasyfikacja wizualna całej paczki za jednym razem
            obrazy_do_klasyfikacji = [Image.open(io.BytesIO(dane['image_bytes'])) for dane in batch_danych]
            wyniki_clip_wsadu = klasyfikuj_obraz_clip_wsadowo(obrazy_do_klasyfikacji)

            # Krok 2: Przetwarzanie wyników i OCR (nadal pojedynczo)
            for j, dane in enumerate(batch_danych):
                print(f"  Analiza wyniku dla Strony {dane['numer_strony']}: '{dane['label']}'")
                ocena_clip = wyniki_clip_wsadu[j]
                if "błąd" in ocena_clip:
                    print(f"    [BŁĄD MODELU] {ocena_clip['błąd']}")
                    continue
                print(f"    > Ocena modelu: '{ocena_clip['kategoria']}' ({ocena_clip['prawdopodobienstwo']:.2%})")

                analiza_ocr = analizuj_strukture_tekstu_ocr(dane['image_bytes'])
                if "błąd" in analiza_ocr:
                    print(f"    [BŁĄD OCR] {analiza_ocr['błąd']}")
                    continue
                if analiza_ocr['znaleziono_duzy_tekst']:
                    print(f"    > Analiza OCR: Wykryto duży tekst! ({analiza_ocr['liczba_duzych_blokow']} bloków)")
                else:
                    print(f"    > Analiza OCR: Nie wykryto dużego tekstu.")

                # Logika zbierania wyników (bez zmian)
                identyfikator_strony_podstawowy = f"Strona {dane['numer_strony']} (Etykieta: '{dane['label']}')"
                if ocena_clip['jest_okladka_wg_clip']:
                    prawdopodobienstwo_clip = ocena_clip['prawdopodobienstwo']
                    identyfikator_clip_z_procentem = f"{identyfikator_strony_podstawowy} (Prawdopodobieństwo: {prawdopodobienstwo_clip:.2%})"
                    wyniki_clip_okladki.append(identyfikator_clip_z_procentem)
                
                if analiza_ocr['znaleziono_duzy_tekst']:
                    wyniki_ocr_okladki.append(identyfikator_strony_podstawowy)
                
                if analiza_ocr['znaleziono_duzy_tekst'] or ocena_clip['jest_okladka_wg_clip']:
                    wyniki_koncowe_okladki.append(identyfikator_strony_podstawowy)

            # Wyczyść paczkę po przetworzeniu
            batch_danych.clear()

    # Sekcja podsumowania (bez zmian)
    print("\n" + "#"*80)
    print("### PODSUMOWANIE ANALIZY (WYNIKI SZCZEGÓŁOWE) ###")
    print("#"*80)
    print("\n--- 1. Strony uznane ostatecznie za okładki (logika OCR > Model) ---")
    if wyniki_koncowe_okladki: print(f"Liczba: {len(wyniki_koncowe_okladki)}\n  - " + "\n  - ".join(wyniki_koncowe_okladki))
    else: print("  Brak.")
    print("\n--- 2. Strony, które sam model wizualny uznał za okładki ---")
    if wyniki_clip_okladki: print(f"Liczba: {len(wyniki_clip_okladki)}\n  - " + "\n  - ".join(wyniki_clip_okladki))
    else: print("  Brak.")
    print("\n--- 3. Strony, na których sam OCR wykrył duży tekst ---")
    if wyniki_ocr_okladki: print(f"Liczba: {len(wyniki_ocr_okladki)}\n  - " + "\n  - ".join(wyniki_ocr_okladki))
    else: print("  Brak.")
    print("\n" + "="*80)


### GŁÓWNA CZĘŚĆ PROGRAMU ###
if __name__ == "__main__":
    try:
        pytesseract.get_tesseract_version()
        print(f"Tesseract OCR jest dostępny (wersja: {pytesseract.get_tesseract_version()}).\n")
    except pytesseract.TesseractNotFoundError:
        print("\n" + "!"*80 + "\nBŁĄD KRYTYCZNY: Nie znaleziono Tesseract OCR.\n" + "!"*80 + "\n")
        exit()

    MANIFEST_DO_ANALIZY_1 = "https://glam.uni.wroc.pl/iiif/GSL_GSL_P_31520_IV_1915_32510/manifest" 
    # Możesz eksperymentować z `rozmiar_wsadu`. Zacznij od małej wartości (4-8).
    # Jeśli nie masz błędów pamięci, możesz ją zwiększyć.
    analizuj_manifest(MANIFEST_DO_ANALIZY_1, limit_stron=30, rozmiar_wsadu=8)