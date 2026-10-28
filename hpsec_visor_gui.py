#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
"""
HPSEC · Visor de cromatogrames — llançador per a tècnics.

Finestra senzilla (sense terminal). El tècnic:
  1. Clica "Tria seqüència…" (carpeta) o "Tria MasterFile…" (.xlsx)
  2. El programa processa el DOC (segmenta + alinea amb el 254)
  3. Desa un HTML a la MATEIXA carpeta de la seqüència i l'obre al navegador

No calibra ni quantifica: només genera el visor de cromatogrames.
Doble clic a aquest fitxer (o al .bat) per obrir-lo.
"""

import os
import sys
import queue
import threading
import traceback

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Assegurar que trobem els mòduls de la Suite (mateixa carpeta que aquest fitxer)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def _default_start_dir():
    try:
        import hpsec_config
        fs = hpsec_config.get_data_folders()
        if fs and os.path.isdir(fs[0]):
            return fs[0]
    except Exception:
        pass
    for cand in (r"C:/Users/maria/Proyectos/Dades3",
                 os.path.join(APP_DIR, "Dades3")):
        if os.path.isdir(cand):
            return cand
    return APP_DIR


class VisorApp:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.busy = False
        root.title("HPSEC · Visor de cromatogrames")
        root.geometry("560x420")
        root.minsize(480, 360)

        pad = {"padx": 16, "pady": 6}

        title = ttk.Label(root, text="Visor de cromatogrames HPSEC",
                          font=("Segoe UI", 14, "bold"))
        title.pack(anchor="w", padx=16, pady=(16, 2))
        sub = ttk.Label(
            root,
            text="Tria una seqüència o un MasterFile. Es generarà un HTML\n"
                 "amb els cromatogrames (DOC + 254) a la mateixa carpeta.",
            foreground="#555", justify="left")
        sub.pack(anchor="w", padx=16, pady=(0, 10))

        btns = ttk.Frame(root)
        btns.pack(fill="x", **pad)
        self.btn_folder = ttk.Button(btns, text="📁  Tria seqüència (carpeta)…",
                                     command=self.pick_folder)
        self.btn_folder.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.btn_master = ttk.Button(btns, text="📄  Tria MasterFile (.xlsx)…",
                                     command=self.pick_master)
        self.btn_master.pack(side="left", expand=True, fill="x", padx=(6, 0))

        self.pb = ttk.Progressbar(root, mode="indeterminate")
        self.pb.pack(fill="x", padx=16, pady=(8, 4))

        self.status = ttk.Label(root, text="Preparat.", foreground="#333")
        self.status.pack(anchor="w", padx=16)

        logframe = ttk.Frame(root)
        logframe.pack(fill="both", expand=True, padx=16, pady=(8, 16))
        self.log = tk.Text(logframe, height=8, wrap="word", font=("Consolas", 9),
                           relief="solid", borderwidth=1)
        self.log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(logframe, command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=sb.set, state="disabled")

        self._poll()

    # --- UI helpers ---
    def _append(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_busy(self, busy, status=None):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.btn_folder.configure(state=state)
        self.btn_master.configure(state=state)
        if busy:
            self.pb.start(12)
        else:
            self.pb.stop()
        if status is not None:
            self.status.configure(text=status)

    # --- selecció ---
    def pick_folder(self):
        if self.busy:
            return
        d = filedialog.askdirectory(title="Tria la carpeta de la seqüència",
                                    initialdir=_default_start_dir())
        if d:
            self._run(d)

    def pick_master(self):
        if self.busy:
            return
        f = filedialog.askopenfilename(
            title="Tria el MasterFile",
            initialdir=_default_start_dir(),
            filetypes=[("MasterFile Excel", "*.xlsx"), ("Tots els fitxers", "*.*")])
        if f:
            self._run(f)

    # --- feina en segon pla ---
    def _run(self, target):
        self._set_busy(True, "Processant…")
        self._append(f"▶ {os.path.basename(target)}")

        def work():
            try:
                import hpsec_viewer_build as B
                out = B.build_one(target, open_after=True,
                                  progress=lambda m: self.q.put(("log", m)))
                self.q.put(("done", out))
            except Exception as e:
                self.q.put(("error", f"{e}\n\n{traceback.format_exc()}"))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._append("   " + payload)
                    self.status.configure(text=payload)
                elif kind == "done":
                    self._set_busy(False, "Fet. S'ha obert al navegador.")
                    self._append(f"✔ Desat i obert:\n   {payload}")
                elif kind == "error":
                    self._set_busy(False, "Error.")
                    short = payload.splitlines()[0] if payload else "Error desconegut"
                    self._append(f"✗ Error: {short}")
                    messagebox.showerror("Error generant el visor", short)
        except queue.Empty:
            pass
        self.root.after(120, self._poll)


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    VisorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
