import requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io
import json
import numpy as np
import cv2
import pytesseract

# 1. Konfiguracja modelu
MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

# 2. Konfiguracja Tesseract OCR
TESSERACT_CMD_PATH = "C:\\Users\\praktyka\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
if TESSERACT_CMD_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH

# 3. Konfiguracja heurystyki dla OCR
MIN_LARGE_TEXT_HEIGHT_PIXELS = 50
LARGE_TEXT_TO_NORMAL_RATIO = 4.0

# --- ŁADOWANIE MODELU CLIP ---
print(f"Ładowanie modelu: {MODEL_ID}...")
print("UWAGA: To bardzo duży model (ok. 2.5 GB). Przy pierwszym uruchomieniu pobieranie może potrwać bardzo długo.")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    clip_model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print(f"\nModel CLIP załadowany i działa na: {device.upper()}")
except Exception as e:
    print(f"\nBŁĄD KRYTYCZNY: Nie udało się pobrać modelu. Sprawdź połączenie internetowe. Szczegóły: {e}")
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
            "a photo of an internal page with articles and blocks of body tex (not title and masthead)",
            "a photo of an internal page full of advertisements or announcements (not title and masthead)",
            "a photo of an internal page with a large illustration or photograph (not title and masthead)",
            "a photo of a table of contents or an editorial page (not title and masthead)"
        ]
        
        inputs = clip_processor(text=opisy, images=image, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image

        probs = logits_per_image.softmax(dim=1).cpu().numpy().flatten()
        
        najlepszy_indeks = probs.argmax()
        return {
            "prawdopodobienstwo": float(probs[najlepszy_indeks]),
            "jest_okladka": najlepszy_indeks == 0
        }
    except Exception as e:
        return {"błąd": f"Błąd przetwarzania obrazu z CLIP: {e}"}


def analizuj_strukture_tekstu_ocr(image_bytes: bytes) -> dict:
    """
    Używa Tesseract OCR do analizy struktury tekstu.
    Zwraca informację, czy znaleziono duży tekst (nagłówek).
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        ocr_data = pytesseract.image_to_data(img_gray, lang='pol', output_type=pytesseract.Output.DICT)
        
        all_heights = [ocr_data['height'][i] for i, conf in enumerate(ocr_data['conf']) if int(conf) > 60 and ocr_data['text'][i].strip()]

        if not all_heights:
            return {"jest_okladka": False, "info": "Nie znaleziono tekstu na stronie."}

        median_height = np.median(all_heights)
        
        large_text_count = sum(1 for h in all_heights if h > MIN_LARGE_TEXT_HEIGHT_PIXELS or h > (median_height * LARGE_TEXT_TO_NORMAL_RATIO))
        
        return {
            "jest_okladka": large_text_count > 0,
            # Dla OCR prawdopodobieństwo jest binarne - albo znaleziono nagłówek, albo nie.
            "prawdopodobienstwo": 1.0 if large_text_count > 0 else 0.0 
        }
    except Exception as e:
        return {"błąd": f"Błąd podczas analizy OCR: {e}"}


def analizuj_manifest(manifest_url: str, metoda_analizy: str, limit_stron: int):
    print("\n" + "="*80)
    print(f"ANALIZA MANIFESTU: {manifest_url}")
    print(f"Wybrana metoda: {metoda_analizy.upper()}")
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
    
    print(f"Rozpoczynam analizę {limit_stron} z {len(canvases)} dostępnych stron...")

    znalezione_okladki = [] 

    for i, canvas in enumerate(canvases[:limit_stron]):
        label = canvas.get('label', '[Brak etykiety]')
        image_service_url = canvas.get('images', [{}])[0].get('resource', {}).get('service', {}).get('@id')
        
        print(f"  Analizuję stronę {i+1}/{limit_stron} ('{label}')...")

        if not image_service_url:
            print("   -> POMINIĘTO (brak linku do serwisu obrazu w manifeście)")
            continue

        image_url = f"{image_service_url.rstrip('/')}/full/1200,/0/default.jpg"
        
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
            
            wynik_analizy = {}
            if metoda_analizy == 'clip':
                wynik_analizy = klasyfikuj_obraz_clip(image_bytes)
            elif metoda_analizy == 'ocr':
                wynik_analizy = analizuj_strukture_tekstu_ocr(image_bytes)

            if "błąd" in wynik_analizy:
                print(f"   [BŁĄD ANALIZY] {wynik_analizy['błąd']}")
                continue
            
            if wynik_analizy.get('jest_okladka'):
                identyfikator_strony = f"Strona {i+1} (Etykieta: '{label}')"
                znalezione_okladki.append({
                    "identyfikator": identyfikator_strony,
                    "prawdopodobienstwo": wynik_analizy.get('prawdopodobienstwo', 0.0)
                })

        except requests.exceptions.RequestException as e:
            print(f"   BŁĄD: Nie udało się pobrać obrazu dla strony {i+1}: {e}")
        except Exception as e:
            print(f"   BŁĄD: Wystąpił nieoczekiwany błąd podczas analizy strony {i+1}: {e}")

    # --- PODSUMOWANIE ---
    print("\n" + "#"*80)
    print(f"### PODSUMOWANIE ANALIZY (METODA: {metoda_analizy.upper()}) ###")
    print("#"*80)

    if not znalezione_okladki:
        print("\nNie zidentyfikowano żadnej strony jako okładki przy użyciu wybranej metody.")
    else:
        print(f"\nZnaleziono {len(znalezione_okladki)} potencjalnych okładek:")
        for okladka in sorted(znalezione_okladki, key=lambda x: x['prawdopodobienstwo'], reverse=True):
            prawdopodobienstwo_str = f"{okladka['prawdopodobienstwo']:.2%}"
            print(f"  - {okladka['identyfikator']:<50} | Prawdopodobieństwo: {prawdopodobienstwo_str}")
            
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        pytesseract.get_tesseract_version()
        print(f"Tesseract OCR jest dostępny (wersja: {pytesseract.get_tesseract_version()}).\n")
    except pytesseract.TesseractNotFoundError:
        print("\n" + "!"*80)
        print("BŁĄD KRYTYCZNY: Nie znaleziono Tesseract OCR.")
        print("Upewnij się, że jest zainstalowany i że ścieżka TESSERACT_CMD_PATH w kodzie jest poprawna.")
        print("!"*80 + "\n")
        exit()

    MANIFEST_DO_ANALIZY_1 = "https://glam.uni.wroc.pl/iiif/GSL_GSL_P_31520_IV_1915_32510/manifest"

    # --- Pobranie informacji o manifeście przed pytaniami do użytkownika ---
    try:
        print(f"Pobieranie informacji z manifestu: {MANIFEST_DO_ANALIZY_1}")
        manifest_data = requests.get(MANIFEST_DO_ANALIZY_1).json()
        canvases = manifest_data.get('sequences', [{}])[0].get('canvases', [])
        liczba_dostepnych_stron = len(canvases)
        if liczba_dostepnych_stron == 0:
            print("BŁĄD: W manifeście nie znaleziono żadnych stron. Zakończono działanie.")
            exit()
        print(f"Znaleziono {liczba_dostepnych_stron} stron w manifeście.\n")
    except Exception as e:
        print(f"BŁĄD KRYTYCZNY: Nie udało się pobrać lub przetworzyć manifestu: {e}")
        exit()

    # --- Pytania do użytkownika ---
    wybrana_metoda = ""
    while wybrana_metoda not in ['ocr', 'clip']:
        wybrana_metoda = input("Wybierz metodę analizy (wpisz 'ocr' lub 'clip'): ").lower().strip()
        if wybrana_metoda not in ['ocr', 'clip']:
            print("Nieprawidłowy wybór. Proszę wpisać 'ocr' lub 'clip'.")

    limit_stron_do_analizy = 0
    while limit_stron_do_analizy <= 0 or limit_stron_do_analizy > liczba_dostepnych_stron:
        try:
            limit_stron_do_analizy = int(input(f"Ile stron przeanalizować? (max. {liczba_dostepnych_stron}): "))
            if limit_stron_do_analizy <= 0:
                print("Liczba musi być większa od zera.")
            elif limit_stron_do_analizy > liczba_dostepnych_stron:
                print(f"Podana liczba przekracza liczbę dostępnych stron. Wybierz wartość od 1 do {liczba_dostepnych_stron}.")
        except ValueError:
            print("Nieprawidłowa wartość. Proszę podać liczbę całkowitą.")
            
    analizuj_manifest(MANIFEST_DO_ANALIZY_1, metoda_analizy=wybrana_metoda, limit_stron=limit_stron_do_analizy)