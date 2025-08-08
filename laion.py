import requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io

# 1. Konfiguracja modelu
MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

# --- ŁADOWANIE MODELU CLIP ---
print(f"Ładowanie modelu: {MODEL_ID}...")
print("UWAGA: To bardzo duży model (ok. 2.5 GB). Przy pierwszym uruchomieniu pobieranie może potrwać bardzo długo.")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    # Ładowanie modelu i procesora CLIP z biblioteki transformers
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
            "a photo of an internal page with articles and blocks of body text (not title and masthead)",
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


def analizuj_strony(canvases: list, limit_stron: int, manifest_url: str):
    """
    Analizuje podaną listę stron (canvases), pobiera obrazy i klasyfikuje je za pomocą CLIP.
    """
    print("\n" + "="*80)
    print(f"ANALIZA MANIFESTU: {manifest_url}")
    print(f"Rozpoczynam analizę pierwszych {limit_stron} stron...")
    print("="*80)

    znalezione_okladki = []

    for i, canvas in enumerate(canvases[:limit_stron]):
        label = canvas.get('label', '[Brak etykiety]')
        image_service_url = canvas.get('images', [{}])[0].get('resource', {}).get('service', {}).get('@id')
        
        print(f"Analizuję stronę: {i+1}/{limit_stron} (Etykieta: '{label}')")

        if not image_service_url:
            print("   -> POMINIĘTO (brak linku do serwisu obrazu w manifeście)")
            continue

        image_url = f"{image_service_url.rstrip('/')}/full/1200,/0/default.jpg"
        
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
            
            wynik_analizy = klasyfikuj_obraz_clip(image_bytes)

            if "błąd" in wynik_analizy:
                print(f"   [BŁĄD ANALIZY] {wynik_analizy['błąd']}")
                continue
            
            if wynik_analizy.get('jest_okladka'):
                identyfikator_strony = f"Strona {i+1} (Etykieta: '{label}')"
                znalezione_okladki.append({
                    "identyfikator": identyfikator_strony,
                    "prawdopodobienstwo": wynik_analizy.get('prawdopodobienstwo', 0.0)
                })
                print(f"   -> ZNALEZIONO OKŁADKĘ! (Prawdopodobieństwo: {wynik_analizy.get('prawdopodobienstwo', 0.0):.2%})")

        except requests.exceptions.RequestException as e:
            print(f"   BŁĄD: Nie udało się pobrać obrazu dla strony {i+1}: {e}")
        except Exception as e:
            print(f"   BŁĄD: Wystąpił nieoczekiwany błąd podczas analizy strony {i+1}: {e}")

    # --- PODSUMOWANIE ---
    print("\n" + "#"*80)
    print("### PODSUMOWANIE ANALIZY (METODA: CLIP) ###")
    print("#"*80)

    if not znalezione_okladki:
        print("\nNie zidentyfikowano żadnej strony jako okładki.")
    else:
        print(f"\nZnaleziono {len(znalezione_okladki)} potencjalnych okładek:")
        for okladka in sorted(znalezione_okladki, key=lambda x: x['prawdopodobienstwo'], reverse=True):
            prawdopodobienstwo_str = f"{okladka['prawdopodobienstwo']:.2%}"
            print(f"  - {okladka['identyfikator']:<50} | Prawdopodobieństwo: {prawdopodobienstwo_str}")
            
    print("\n" + "="*80)


if __name__ == "__main__":
    
    # --- ETAP 1: Pytanie o link i pobranie manifestu ---
    manifest_url_uzytkownika = ""
    canvases = []
    liczba_wszystkich_stron = 0

    while True: # Pętla, dopóki nie uda się pobrać i przetworzyć manifestu
        manifest_url_uzytkownika = input("Podaj link do manifestu IIIF, który chcesz przeanalizować: ").strip()
        if not (manifest_url_uzytkownika.lower().startswith("http://") or manifest_url_uzytkownika.lower().startswith("https://")):
            print("BŁĄD: Podany link jest nieprawidłowy. Upewnij się, że zaczyna się od 'http://' lub 'https://'.")
            continue

        try:
            print(f"\nPobieranie informacji z manifestu: {manifest_url_uzytkownika}...")
            response = requests.get(manifest_url_uzytkownika, timeout=20)
            response.raise_for_status() # Sprawdza, czy zapytanie HTTP się powiodło
            manifest_data = response.json()
            canvases = manifest_data.get('sequences', [{}])[0].get('canvases', [])
            liczba_wszystkich_stron = len(canvases)
            
            if liczba_wszystkich_stron == 0:
                print("BŁĄD: W podanym manifeście nie znaleziono żadnych stron (canvases). Spróbuj z innym linkiem.")
                continue
            
            print(f"Sukces! Znaleziono {liczba_wszystkich_stron} stron w manifeście.")
            break # Wyjście z pętli, jeśli wszystko się udało
            
        except requests.exceptions.RequestException as e:
            print(f"BŁĄD KRYTYCZNY: Nie udało się pobrać manifestu. Sprawdź link lub połączenie internetowe. {e}")
        except Exception as e:
            print(f"BŁĄD KRYTYCZNY: Nie udało się przetworzyć manifestu. Upewnij się, że link prowadzi do poprawnego pliku JSON. {e}")

    # --- ETAP 2: Pytanie o liczbę stron do analizy ---
    limit_stron_uzytkownika = 0
    while True:
        try:
            prompt_text = f"\nPodaj, ile pierwszych stron chcesz przeanalizować (dostępnych: {liczba_wszystkich_stron}): "
            limit_stron_input = input(prompt_text)
            limit_stron_uzytkownika = int(limit_stron_input)
            
            if limit_stron_uzytkownika > liczba_wszystkich_stron:
                 print(f"UWAGA: Podano liczbę większą niż dostępna. Analiza zostanie ograniczona do {liczba_wszystkich_stron} stron.")
                 limit_stron_uzytkownika = liczba_wszystkich_stron
            
            if limit_stron_uzytkownika > 0:
                break
            else:
                print("BŁĄD: Proszę podać liczbę większą od zera.")
        except ValueError:
            print("BŁĄD: Nieprawidłowa wartość. Proszę podać liczbę całkowitą.")
    
    # --- ETAP 3: Uruchomienie właściwej analizy ---
    analizuj_strony(canvases, limit_stron=limit_stron_uzytkownika, manifest_url=manifest_url_uzytkownika)


    # 7 9