import io
import json
import requests
import threading
from PIL import Image
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import torch
from transformers import CLIPProcessor, CLIPModel

MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
clip_model = None
clip_processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def classify(image_bytes: bytes) -> dict:
    """Klasyfikuje obraz na podstawie podanych opisów tekstowych."""
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
        prob = logits_per_image.softmax(dim=1).cpu().numpy().flatten()
        best = prob.argmax()
        return {
            "prawdopodobienstwo": float(prob[best]),
            "jest_okladka": bool(best == 0)
        }
    except Exception as e:
        return {f"Błąd przetwarzania obrazu: {e}"}

class CoverFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizator Okładek Gazet (CLIP)")
        self.root.geometry("800x650")

        # --- Główne elementy interfejsu ---
        self.frame = ttk.Frame(root, padding="10")
        self.frame.pack(fill=tk.X)

        ttk.Label(self.frame, text="Link do manifestu IIIF:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.manifest_url_entry = ttk.Entry(self.frame, width=80)
        self.manifest_url_entry.grid(row=0, column=1, sticky=tk.EW)
        self.manifest_url_entry.insert(0, "")

        self.fetch_button = ttk.Button(self.frame, text="Pobierz informacje", command=self.start_fetch_thread)
        self.fetch_button.grid(row=0, column=2, padx=5)
        
        ttk.Label(self.frame, text="Zakres stron do analizy:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)

        range_frame = ttk.Frame(self.frame)
        range_frame.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(range_frame, text="Od:").pack(side=tk.LEFT, padx=(0, 5))
        self.start_page_entry = ttk.Entry(range_frame, width=10, state=tk.DISABLED)
        self.start_page_entry.pack(side=tk.LEFT)

        ttk.Label(range_frame, text="Do:").pack(side=tk.LEFT, padx=(10, 5))
        self.end_page_entry = ttk.Entry(range_frame, width=10, state=tk.DISABLED)
        self.end_page_entry.pack(side=tk.LEFT)

        self.analyze_button = ttk.Button(self.frame, text="Rozpocznij Analizę", command=self.start_analysis_thread, state=tk.DISABLED)
        self.analyze_button.grid(row=1, column=2, padx=5)

        # --- Przycisk do edycji i zapisu ---
        self.edit_button = ttk.Button(self.frame, text="Edytuj i Zapisz Manifest", command=self.open_edit_window, state=tk.DISABLED)
        self.edit_button.grid(row=2, column=1, columnspan=2, sticky=tk.E, pady=5)

        self.progress_frame = ttk.Frame(root, padding="0 10 10 10")
        self.progress_frame.pack(fill=tk.X)
        self.progress_label = ttk.Label(self.progress_frame, text="Postęp:")
        self.progressbar = ttk.Progressbar(self.progress_frame, orient='horizontal', mode='determinate')
        self.progress_percent_label = ttk.Label(self.progress_frame, text="0%")

        self.log_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=35)
        self.log_area.pack(pady=(0, 10), padx=10, fill=tk.BOTH, expand=True)

        self.frame.columnconfigure(1, weight=1)
        
        # --- Zmienne stanu aplikacji ---
        self.canvases = []
        self.liczba_wszystkich_stron = 0
        self.manifest_data = None
        self.analysed_pages = []

        self.log("Wklej link do manifestu i kliknij 'Pobierz informacje'.")
        self.log("Uwaga: pierwsze uruchomienie może potrwać długo (pobieranie modelu ok. 3.5 GB).")

    def log(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)

    def set_ui_state(self, state):
        for widget in [self.fetch_button, self.analyze_button, self.manifest_url_entry, self.start_page_entry, self.end_page_entry]:
            widget.config(state=state)
        
        self.edit_button.config(state=tk.DISABLED)
        if state == tk.NORMAL and self.analysed_pages:
            self.edit_button.config(state=tk.NORMAL)

        if state == tk.NORMAL and self.liczba_wszystkich_stron == 0:
            self.analyze_button.config(state=tk.DISABLED)
            self.start_page_entry.config(state=tk.DISABLED)
            self.end_page_entry.config(state=tk.DISABLED)

    def show_progress_bar(self, show=True):
        if show:
            self.progress_label.grid(row=0, column=0, padx=(0, 5))
            self.progressbar.grid(row=0, column=1, sticky=tk.EW)
            self.progress_percent_label.grid(row=0, column=2, padx=(5, 0))
            self.progress_frame.columnconfigure(1, weight=1)
        else:
            for widget in self.progress_frame.winfo_children():
                widget.grid_remove()

    def start_fetch_thread(self):
        self.set_ui_state(tk.DISABLED)
        self.analysed_pages = []
        threading.Thread(target=self.fetch_manifest_data, daemon=True).start()

    def fetch_manifest_data(self):
        manifest_url = self.manifest_url_entry.get().strip()
        if not manifest_url:
            self.log("Błąd: Pole z linkiem do manifestu jest puste.")
            self.set_ui_state(tk.NORMAL)
            return

        try:
            self.log(f"\nPobieranie informacji z manifestu: {manifest_url}...")
            response = requests.get(manifest_url, timeout=20)
            response.raise_for_status()
            self.manifest_data = response.json()
            self.canvases = self.manifest_data.get('sequences', [{}])[0].get('canvases', [])
            self.liczba_wszystkich_stron = len(self.canvases)

            if self.liczba_wszystkich_stron == 0:
                self.log("BŁĄD: W podanym manifeście nie znaleziono żadnych stron (canvases).")
            else:
                self.log(f"Znaleziono {self.liczba_wszystkich_stron} stron.")
                self.start_page_entry.delete(0, tk.END)
                self.start_page_entry.insert(0, "1")
                self.end_page_entry.delete(0, tk.END)
                self.end_page_entry.insert(0, str(self.liczba_wszystkich_stron))
        except Exception as e:
            self.log(f"Błąd pobierania manifestu: {e}")
            self.manifest_data = None
        finally:
            self.set_ui_state(tk.NORMAL)

    def start_analysis_thread(self):
        try:
            start_page = int(self.start_page_entry.get())
            end_page = int(self.end_page_entry.get())
            
            if not (1 <= start_page <= self.liczba_wszystkich_stron):
                self.log(f"BŁĄD: 'Od' musi być liczbą od 1 do {self.liczba_wszystkich_stron}.")
                return
            if not (1 <= end_page <= self.liczba_wszystkich_stron):
                self.log(f"BŁĄD: 'Do' musi być liczbą od 1 do {self.liczba_wszystkich_stron}.")
                return
            if start_page > end_page:
                self.log("BŁĄD: Strona początkowa ('Od') nie może być większa niż końcowa ('Do').")
                return
        except ValueError:
            self.log("BŁĄD: Wprowadź poprawne liczby w polach zakresu stron.")
            return

        self.set_ui_state(tk.DISABLED)
        self.show_progress_bar(True)
        self.progressbar['value'] = 0
        self.progress_percent_label.config(text="0%")
        threading.Thread(target=self.run_analysis, args=(start_page, end_page), daemon=True).start()

    def run_analysis(self, start_page, end_page):
        self.log("\n" + "="*80)
        self.log(f"Rozpoczynam analizę stron od {start_page} do {end_page}...")
        
        start_index = start_page - 1
        end_index = end_page
        
        canvases_to_analyze = self.canvases[start_index:end_index]
        total_to_process = len(canvases_to_analyze)
        
        self.analysed_pages = []
        for i, canvas in enumerate(canvases_to_analyze):
            current_page_number = start_page + i
            
            page_info = {
                "identyfikator": f"Strona {current_page_number} (Etykieta: '{canvas.get('label', '[Brak]')}')",
                "numer_strony": current_page_number,
                "canvas_id": canvas.get('@id'),
                "jest_okladka": False,
                "prawdopodobienstwo": 0.0
            }
            
            image_service_url = canvas.get('images', [{}])[0].get('resource', {}).get('service', {}).get('@id')
            if not image_service_url:
                self.analysed_pages.append(page_info)
                continue

            try:
                image_url = f"{image_service_url.rstrip('/')}/full/1200,/0/default.jpg"
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                wynik_analizy = classify(response.content)

                if 'błąd' not in wynik_analizy:
                    page_info.update(wynik_analizy)

            except Exception as e:
                self.log(f"Info: Pomijam stronę {current_page_number} z powodu błędu pobierania/analizy: {e}")
            
            self.analysed_pages.append(page_info)
            
            progress_value = (i + 1) / total_to_process * 100
            self.root.after(0, self.update_progress, progress_value)

        self.root.after(0, self.finalize_analysis)

    def update_progress(self, value):
        self.progressbar['value'] = value
        self.progress_percent_label.config(text=f"{int(value)}%")

    def finalize_analysis(self):
        self.show_progress_bar(False)
        self.log("\n### PODSUMOWANIE ANALIZY ###")

        okladki = [p for p in self.analysed_pages if p.get("jest_okladka")]

        if not okladki:
            self.log("\nNie zidentyfikowano żadnej strony jako okładki w podanym zakresie.")
        else:
            self.log(f"\nZnaleziono {len(okladki)} potencjalnych okładek (posortowane wg nr strony):")
            okladki.sort(key=lambda x: x['numer_strony'])
            for okladka in okladki:
                prawdopodobienstwo_str = f"{okladka.get('prawdopodobienstwo', 0):.2%}"
                self.log(f"- {okladka['identyfikator']:<50} | Prawdopodobieństwo: {prawdopodobienstwo_str}")

        self.log("\n" + "#"*80)
        self.log("Analiza zakończona. Możesz teraz edytować wyniki i zapisać manifest.")
        self.set_ui_state(tk.NORMAL)

    def open_edit_window(self):
        if not self.analysed_pages:
            self.log("BŁĄD: Najpierw przeprowadź analizę.")
            return

        edit_win = tk.Toplevel(self.root)
        edit_win.title("Edycja Struktury - Zaznacz Okładki")
        edit_win.geometry("600x500")

        analysed_results = {p['numer_strony']: p for p in self.analysed_pages}

        # --- Ramka na checkboxy z przewijaniem ---
        main_frame = ttk.Frame(edit_win)
        main_frame.pack(fill="both", expand=True)

        canvas_frame = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas_frame.yview)
        scrollable_frame = ttk.Frame(canvas_frame)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_frame.configure(scrollregion=canvas_frame.bbox("all"))
        )

        canvas_frame.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_frame.configure(yscrollcommand=scrollbar.set)

        checkbox_vars = {}

        for i, canvas in enumerate(self.canvases):
            current_page_number = i + 1
            identyfikator = f"Strona {current_page_number} (Etykieta: '{canvas.get('label', '[Brak]')}')"
            
            var = tk.BooleanVar()
            
            analysed_info = analysed_results.get(current_page_number)
            if analysed_info:
                var.set(bool(analysed_info.get("jest_okladka", False)))
            else:
                var.set(False)
            
            checkbox_vars[current_page_number] = var
            
            cb = ttk.Checkbutton(scrollable_frame, text=identyfikator, variable=var)
            cb.pack(anchor=tk.W, padx=10, pady=2)

        canvas_frame.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # --- Ramka na przyciski na dole ---
        button_frame = ttk.Frame(edit_win)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

        def toggle_all_off():
            for var in checkbox_vars.values():
                var.set(False)

        selection_buttons_frame = ttk.Frame(button_frame)
        selection_buttons_frame.pack()
        
        ### ZMIANA: Usunięto przycisk "Zaznacz wszystkie", zostawiono tylko "Odznacz wszystkie" ###
        deselect_all_btn = ttk.Button(selection_buttons_frame, text="Odznacz wszystkie", command=toggle_all_off)
        deselect_all_btn.pack(side=tk.LEFT, padx=5)
        
        # --- Przycisk zapisu ---
        save_button = ttk.Button(
            button_frame,
            text="Zapisz manifest.json",
            command=lambda: self.save_manifest_with_structure(checkbox_vars, edit_win)
        )
        save_button.pack(pady=(10, 0))


    def save_manifest_with_structure(self, checkbox_vars, window_to_close):
        if not self.manifest_data:
            self.log("BŁĄD: Brak danych manifestu do zapisu.")
            window_to_close.destroy()
            return

        cover_page_numbers = sorted([num for num, var in checkbox_vars.items() if var.get()])

        if not cover_page_numbers:
            self.log("INFO: Nie zaznaczono żadnych okładek. Zapisuję manifest bez pola 'structures'.")
            self.manifest_data.pop('structures', None)
        else:
            self.log(f"\nWybrane okładki to strony: {cover_page_numbers}")
            self.log("Generowanie nowej struktury manifestu...")
            
            manifest_base_id = self.manifest_data.get('@id', 'http://example.com/manifest')
            if not manifest_base_id.strip():
                manifest_base_id = 'http://example.com/manifest'
                self.log("OSTRZEŻENIE: Brak '@id' w manifeście. Używam domyślnego ID dla zakresów.")

            structures = []
            for i, start_page_num in enumerate(cover_page_numbers):
                ### ZMIANA: Zmiana etykiety na "zakres od strony X" ###
                range_label = f"zakres od strony {start_page_num}"
                range_id = f"{manifest_base_id.rstrip('/')}/range/r{i}"
                
                if i + 1 < len(cover_page_numbers):
                    end_page_num = cover_page_numbers[i+1] - 1
                else:
                    end_page_num = self.liczba_wszystkich_stron
                
                start_index = start_page_num - 1
                end_index = end_page_num 
                
                canvases_in_range_ids = [
                    c['@id'] for c in self.canvases[start_index:end_index] if '@id' in c
                ]

                if canvases_in_range_ids:
                    structures.append({
                        "@id": range_id,
                        "@type": "sc:Range",
                        "label": range_label,
                        "canvases": canvases_in_range_ids
                    })
            
            self.manifest_data['structures'] = structures
            self.log("Struktura została wygenerowana.")

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="manifest.json",
            title="Zapisz manifest jako..."
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.manifest_data, f, indent=4, ensure_ascii=False)
                self.log(f"Manifest został pomyślnie zapisany w: {file_path}")
            except Exception as e:
                self.log(f"BŁĄD: Nie udało się zapisać pliku. Szczegóły: {e}")
        else:
            self.log("Zapis anulowany przez użytkownika.")

        window_to_close.destroy()

if __name__ == "__main__":
    print(f"Ładowanie modelu: {MODEL_ID}...")
    try:
        clip_model = CLIPModel.from_pretrained(MODEL_ID).to(device)
        clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
        print(f"\nModel CLIP załadowany i działa na: {device.upper()}")

        root = tk.Tk()
        app = CoverFinderApp(root)
        root.mainloop()

    except Exception as e:
        print(f"\nBŁĄD KRYTYCZNY: Nie udało się załadować lub pobrać modelu. Sprawdź połączenie internetowe. Szczegóły: {e}")
        input("Naciśnij Enter, aby zamknąć...")