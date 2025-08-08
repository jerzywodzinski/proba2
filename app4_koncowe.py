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
    try:
        image = Image.open(io.BytesIO(image_bytes))
        texts = [
            "a photo of a newspaper cover with a title and masthead",
            "a photo of an internal page with articles and blocks of body text (not title and masthead)",
            "a photo of an internal page full of advertisements or announcements (not title and masthead)",
            "a photo of an internal page with a large illustration or photograph (not title and masthead)",
            "a photo of a table of contents or an editorial page (not title and masthead)"
        ]
        inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            prob = logits_per_image.softmax(dim=1).cpu().numpy().flatten()
        best = prob.argmax()
        return {
            "prob": float(prob[best]),
            "is_cover": bool(best == 0)
        }
    except Exception as e:
        return {"error": f"Błąd przetwarzania obrazu: {e}"}

class ManifestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizator Okładek Gazet (CLIP)")
        self.root.geometry("800x650")

        self.frame = ttk.Frame(root, padding="10")
        self.frame.pack(fill=tk.X)

        ttk.Label(self.frame, text="Link do manifestu IIIF:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.url_entry = ttk.Entry(self.frame, width=80)
        self.url_entry.grid(row=0, column=1, sticky=tk.EW)
        self.url_entry.insert(0, "")

        self.fetch_btn = ttk.Button(self.frame, text="Pobierz informacje", command=self.start_fetch)
        self.fetch_btn.grid(row=0, column=2, padx=5)
        
        ttk.Label(self.frame, text="Zakres stron do analizy:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)

        range_frame = ttk.Frame(self.frame)
        range_frame.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(range_frame, text="Od:").pack(side=tk.LEFT, padx=(0, 5))
        self.start_entry = ttk.Entry(range_frame, width=10, state=tk.DISABLED)
        self.start_entry.pack(side=tk.LEFT)

        ttk.Label(range_frame, text="Do:").pack(side=tk.LEFT, padx=(10, 5))
        self.end_entry = ttk.Entry(range_frame, width=10, state=tk.DISABLED)
        self.end_entry.pack(side=tk.LEFT)

        self.analyze_btn = ttk.Button(self.frame, text="Rozpocznij Analizę", command=self.start_analysis, state=tk.DISABLED)
        self.analyze_btn.grid(row=1, column=2, padx=5)

        # --- Przycisk do edycji i zapisu ---
        self.edit_btn = ttk.Button(self.frame, text="Edytuj i Zapisz Manifest", command=self.open_editor, state=tk.DISABLED)
        self.edit_btn.grid(row=2, column=1, columnspan=2, sticky=tk.E, pady=5)

        self.progress_frame = ttk.Frame(root, padding="0 10 10 10")
        self.progress_frame.pack(fill=tk.X)
        self.progress_lbl = ttk.Label(self.progress_frame, text="Postęp:")
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient='horizontal', mode='determinate')
        self.progress_percent_lbl = ttk.Label(self.progress_frame, text="0%")

        self.log_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=35)
        self.log_box.pack(pady=(0, 10), padx=10, fill=tk.BOTH, expand=True)

        self.frame.columnconfigure(1, weight=1)
        
        # --- Zmienne stanu aplikacji ---
        self.canvases = []
        self.total_pages = 0
        self.manifest = None
        self.analysis_results = []

        self.log("Wklej link do manifestu i kliknij 'Pobierz informacje'.")
        self.log("Uwaga: pierwsze uruchomienie może potrwać długo (pobieranie modelu ok. 3.5 GB).")

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)

    def toggle_ui(self, state):
        for widget in [self.fetch_btn, self.analyze_btn, self.url_entry, self.start_entry, self.end_entry]:
            widget.config(state=state)
        
        self.edit_btn.config(state=tk.DISABLED)
        if state == tk.NORMAL and self.analysis_results:
            self.edit_btn.config(state=tk.NORMAL)

        if state == tk.NORMAL and self.total_pages == 0:
            self.analyze_btn.config(state=tk.DISABLED)
            self.start_entry.config(state=tk.DISABLED)
            self.end_entry.config(state=tk.DISABLED)

    def toggle_progress_bar(self, show=True):
        if show:
            self.progress_lbl.grid(row=0, column=0, padx=(0, 5))
            self.progress_bar.grid(row=0, column=1, sticky=tk.EW)
            self.progress_percent_lbl.grid(row=0, column=2, padx=(5, 0))
            self.progress_frame.columnconfigure(1, weight=1)
        else:
            for widget in self.progress_frame.winfo_children():
                widget.grid_remove()

    def start_fetch(self):
        self.toggle_ui(tk.DISABLED)
        self.analysis_results = []
        threading.Thread(target=self.fetch_manifest, daemon=True).start()

    def fetch_manifest(self):
        url = self.url_entry.get().strip()
        if not url:
            self.log("Błąd: Pole z linkiem do manifestu jest puste.")
            self.toggle_ui(tk.NORMAL)
            return

        try:
            self.log(f"\nPobieranie informacji z manifestu: {url}...")
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            self.manifest = response.json()
            self.canvases = self.manifest.get('sequences', [{}])[0].get('canvases', [])
            self.total_pages = len(self.canvases)

            if self.total_pages == 0:
                self.log("BŁĄD: W podanym manifeście nie znaleziono żadnych stron (canvases).")
            else:
                self.log(f"Znaleziono {self.total_pages} stron.")
                self.start_entry.delete(0, tk.END)
                self.start_entry.insert(0, "1")
                self.end_entry.delete(0, tk.END)
                self.end_entry.insert(0, str(self.total_pages))
        except Exception as e:
            self.log(f"Błąd pobierania manifestu: {e}")
            self.manifest = None
        finally:
            self.toggle_ui(tk.NORMAL)

    def start_analysis(self):
        try:
            start_page = int(self.start_entry.get())
            end_page = int(self.end_entry.get())
            
            if not (1 <= start_page <= self.total_pages):
                self.log(f"BŁĄD: 'Od' musi być liczbą od 1 do {self.total_pages}.")
                return
            if not (1 <= end_page <= self.total_pages):
                self.log(f"BŁĄD: 'Do' musi być liczbą od 1 do {self.total_pages}.")
                return
            if start_page > end_page:
                self.log("BŁĄD: Strona początkowa ('Od') nie może być większa niż końcowa ('Do').")
                return
        except ValueError:
            self.log("BŁĄD: Wprowadź poprawne liczby w polach zakresu stron.")
            return

        self.toggle_ui(tk.DISABLED)
        self.toggle_progress_bar(True)
        self.progress_bar['value'] = 0
        self.progress_percent_lbl.config(text="0%")
        threading.Thread(target=self.run_analysis, args=(start_page, end_page), daemon=True).start()

    def run_analysis(self, start_page, end_page):
        self.log("\n" + "="*80)
        self.log(f"Rozpoczynam analizę stron od {start_page} do {end_page}...")
        
        start_index = start_page - 1
        end_index = end_page
        
        canvases_subset = self.canvases[start_index:end_index]
        total = len(canvases_subset)
        
        self.analysis_results = []
        for i, canvas in enumerate(canvases_subset):
            page_num = start_page + i
            
            page_data = {
                "id_text": f"Strona {page_num} (Etykieta: '{canvas.get('label', '[Brak]')}')",
                "page_num": page_num,
                "canvas_id": canvas.get('@id'),
                "is_cover": False,
                "prob": 0.0
            }
            
            img_service_url = canvas.get('images', [{}])[0].get('resource', {}).get('service', {}).get('@id')
            if not img_service_url:
                self.analysis_results.append(page_data)
                continue

            try:
                image_url = f"{img_service_url.rstrip('/')}/full/1200,/0/default.jpg"
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                result = classify(response.content)

                if 'error' not in result:
                    page_data.update(result)

            except Exception as e:
                self.log(f"Info: Pomijam stronę {page_num} z powodu błędu pobierania/analizy: {e}")
            
            self.analysis_results.append(page_data)
            
            progress = (i + 1) / total * 100
            self.root.after(0, self.update_progress, progress)

        self.root.after(0, self.show_summary)

    def update_progress(self, value):
        self.progress_bar['value'] = value
        self.progress_percent_lbl.config(text=f"{int(value)}%")

    def show_summary(self):
        self.toggle_progress_bar(False)
        self.log("\n### PODSUMOWANIE ANALIZY ###")

        covers = [p for p in self.analysis_results if p.get("is_cover")]

        if not covers:
            self.log("\nNie zidentyfikowano żadnej strony jako okładki w podanym zakresie.")
        else:
            self.log(f"\nZnaleziono {len(covers)} potencjalnych okładek (posortowane wg nr strony):")
            covers.sort(key=lambda x: x['page_num'])
            for cover in covers:
                prob_str = f"{cover.get('prob', 0):.2%}"
                self.log(f"- {cover['id_text']:<50} | Prawdopodobieństwo: {prob_str}")

        self.log("\n" + "#"*80)
        self.log("Analiza zakończona. Możesz teraz edytować wyniki i zapisać manifest.")
        self.toggle_ui(tk.NORMAL)

    def open_editor(self):
        if not self.analysis_results:
            self.log("BŁĄD: Najpierw przeprowadź analizę.")
            return

        editor_win = tk.Toplevel(self.root)
        editor_win.title("Edycja Struktury - Zaznacz Okładki")
        editor_win.geometry("600x500")

        results_map = {p['page_num']: p for p in self.analysis_results}

        # --- Ramka na checkboxy z przewijaniem ---
        main_frame = ttk.Frame(editor_win)
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

        check_vars = {}

        for i, canvas in enumerate(self.canvases):
            page_num = i + 1
            id_text = f"Strona {page_num} (Etykieta: '{canvas.get('label', '[Brak]')}')"
            
            var = tk.BooleanVar()
            
            result_data = results_map.get(page_num)
            if result_data:
                var.set(bool(result_data.get("is_cover", False)))
            else:
                var.set(False)
            
            check_vars[page_num] = var
            
            cb = ttk.Checkbutton(scrollable_frame, text=id_text, variable=var)
            cb.pack(anchor=tk.W, padx=10, pady=2)

        canvas_frame.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # --- Ramka na przyciski na dole ---
        button_frame = ttk.Frame(editor_win)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

        def deselect_all():
            for var in check_vars.values():
                var.set(False)

        selection_buttons_frame = ttk.Frame(button_frame)
        selection_buttons_frame.pack()
        
        ### ZMIANA: Usunięto przycisk "Zaznacz wszystkie", zostawiono tylko "Odznacz wszystkie" ###
        deselect_btn = ttk.Button(selection_buttons_frame, text="Odznacz wszystkie", command=deselect_all)
        deselect_btn.pack(side=tk.LEFT, padx=5)
        
        # --- Przycisk zapisu ---
        save_btn = ttk.Button(
            button_frame,
            text="Zapisz manifest.json",
            command=lambda: self.save_manifest(check_vars, editor_win)
        )
        save_btn.pack(pady=(10, 0))

    def save_manifest(self, check_vars, editor_win):
        if not self.manifest:
            self.log("BŁĄD: Brak danych manifestu do zapisu.")
            editor_win.destroy()
            return

        cover_pages = sorted([num for num, var in check_vars.items() if var.get()])

        if not cover_pages:
            self.log("INFO: Nie zaznaczono żadnych okładek. Zapisuję manifest bez pola 'structures'.")
            self.manifest.pop('structures', None)
        else:
            self.log(f"\nWybrane okładki to strony: {cover_pages}")
            self.log("Generowanie nowej struktury manifestu...")
            
            base_id = self.manifest.get('@id', 'http://example.com/manifest')
            if not base_id.strip():
                base_id = 'http://example.com/manifest'
                self.log("OSTRZEŻENIE: Brak '@id' w manifeście. Używam domyślnego ID dla zakresów.")

            structures = []
            for i, start_page in enumerate(cover_pages):
                ### ZMIANA: Zmiana etykiety na "zakres od strony X" ###
                label = f"zakres od strony {start_page}"
                range_id = f"{base_id.rstrip('/')}/range/r{i}"
                
                if i + 1 < len(cover_pages):
                    end_page = cover_pages[i+1] - 1
                else:
                    end_page = self.total_pages
                
                start_index = start_page - 1
                end_index = end_page
                
                range_canvas_ids = [
                    c['@id'] for c in self.canvases[start_index:end_index] if '@id' in c
                ]

                if range_canvas_ids:
                    structures.append({
                        "@id": range_id,
                        "@type": "sc:Range",
                        "label": label,
                        "canvases": range_canvas_ids
                    })
            
            self.manifest['structures'] = structures
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
                    json.dump(self.manifest, f, indent=4, ensure_ascii=False)
                self.log(f"Manifest został pomyślnie zapisany w: {file_path}")
            except Exception as e:
                self.log(f"BŁĄD: Nie udało się zapisać pliku. Szczegóły: {e}")
        else:
            self.log("Zapis anulowany przez użytkownika.")

        editor_win.destroy()

if __name__ == "__main__":
    print(f"Ładowanie modelu: {MODEL_ID}...")
    try:
        clip_model = CLIPModel.from_pretrained(MODEL_ID).to(device)
        clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
        print(f"\nModel CLIP załadowany i działa na: {device.upper()}")

        root = tk.Tk()
        app = ManifestApp(root)
        root.mainloop()

    except Exception as e:
        print(f"\nBŁĄD KRYTYCZNY: Nie udało się załadować lub pobrać modelu. Sprawdź połączenie internetowe. Szczegóły: {e}")
        input("Naciśnij Enter, aby zamknąć...")