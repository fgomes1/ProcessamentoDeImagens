"""
Ferramenta de Processamento de Imagens — Tkinter + OpenCV
Vetor de Mat's com CRUD, sliders, e 5 categorias de métodos.
Uso: python ferramenta_imagens.py
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

# ── Dados ────────────────────────────────────────────────────
class ImagemProcessada:
    def __init__(self, caminho: str, mat: np.ndarray):
        self.caminho = caminho
        self.nome = os.path.basename(caminho)
        self.mat_original = mat
        self.mat_processada = None
        self.metodo = ""

# ── Tema Escuro ──────────────────────────────────────────────
TEMAS = {
    "Escuro": {
        "bg": "#1e1e2e", "fg": "#cdd6f4", "accent": "#89b4fa",
        "panel": "#181825", "entry": "#313244", "select": "#45475a",
        "btn": "#585b70", "btn_fg": "#cdd6f4", "success": "#a6e3a1",
        "danger": "#f38ba8", "warning": "#f9e2af",
    },
    "Claro": {
        "bg": "#eff1f5", "fg": "#4c4f69", "accent": "#1e66f5",
        "panel": "#dce0e8", "entry": "#ccd0da", "select": "#bcc0cc",
        "btn": "#8c8fa1", "btn_fg": "#eff1f5", "success": "#40a02b",
        "danger": "#d20f39", "warning": "#df8e1d",
    },
}

# ── App Principal ────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🖼️ Ferramenta de Processamento de Imagens")
        self.geometry("1200x750")
        self.minsize(1000, 650)
        self.vetor_imagens: list[ImagemProcessada] = []
        self.imagem_selecionada: ImagemProcessada | None = None
        self.tema_atual = "Escuro"
        self.photo_orig = None
        self.photo_proc = None
        # Histórico: lista de tuplas (descricao, mat_copia)
        self.historico: list[tuple[str, np.ndarray]] = []
        self.photo_hist = None
        self._construir_ui()
        self._aplicar_tema()

    # ── Construção da UI ─────────────────────────────────────
    def _construir_ui(self):
        # Top bar
        self.topbar = tk.Frame(self, height=40)
        self.topbar.pack(fill="x", padx=8, pady=(8, 0))
        self.lbl_titulo = tk.Label(self.topbar, text="🖼️  Ferramenta de Processamento de Imagens",
                                   font=("Segoe UI", 14, "bold"))
        self.lbl_titulo.pack(side="left", padx=8)
        # Tema selector
        tk.Label(self.topbar, text="Tema:", font=("Segoe UI", 10)).pack(side="right", padx=(0,4))
        self.combo_tema = ttk.Combobox(self.topbar, values=list(TEMAS.keys()), width=10, state="readonly")
        self.combo_tema.set(self.tema_atual)
        self.combo_tema.pack(side="right", padx=4)
        self.combo_tema.bind("<<ComboboxSelected>>", lambda e: self._mudar_tema())

        # Main paned
        self.main_pane = tk.PanedWindow(self, orient="horizontal", sashwidth=6)
        self.main_pane.pack(fill="both", expand=True, padx=8, pady=8)

        # ── Painel Esquerdo: Lista + Histórico ────────────────
        self.frame_lista = tk.Frame(self.main_pane, width=240)
        self.main_pane.add(self.frame_lista, minsize=200)

        # Sub-painel: Imagens Carregadas
        tk.Label(self.frame_lista, text="📋 Imagens Carregadas", font=("Segoe UI", 11, "bold")).pack(pady=(4,2), anchor="w", padx=6)
        self.listbox = tk.Listbox(self.frame_lista, font=("Segoe UI", 10), activestyle="none", selectmode="single", height=6)
        self.listbox.pack(fill="x", padx=4)
        self.listbox.bind("<<ListboxSelect>>", self._ao_selecionar)
        # Botões CRUD
        fr_btns = tk.Frame(self.frame_lista)
        fr_btns.pack(fill="x", padx=4, pady=2)
        self.btn_add = tk.Button(fr_btns, text="＋ Adicionar", font=("Segoe UI", 9), command=self.adicionar_imagens)
        self.btn_add.pack(fill="x", pady=1)
        self.btn_rem = tk.Button(fr_btns, text="－ Remover", font=("Segoe UI", 9), command=self.remover_imagem)
        self.btn_rem.pack(fill="x", pady=1)
        self.btn_clear = tk.Button(fr_btns, text="🗑 Limpar Tudo", font=("Segoe UI", 9), command=self.limpar_tudo)
        self.btn_clear.pack(fill="x", pady=1)

        # Sub-painel: Histórico de Processamento
        tk.Label(self.frame_lista, text="📜 Histórico", font=("Segoe UI", 11, "bold")).pack(pady=(8,2), anchor="w", padx=6)
        self.listbox_hist = tk.Listbox(self.frame_lista, font=("Segoe UI", 9), activestyle="none", selectmode="single")
        self.listbox_hist.pack(fill="both", expand=True, padx=4)
        self.listbox_hist.bind("<<ListboxSelect>>", self._ao_selecionar_hist)
        # Botões do histórico
        fr_hist_btns = tk.Frame(self.frame_lista)
        fr_hist_btns.pack(fill="x", padx=4, pady=2)
        self.btn_usar_base = tk.Button(fr_hist_btns, text="⬆ Usar como Base", font=("Segoe UI", 9), command=self._usar_hist_como_base)
        self.btn_usar_base.pack(fill="x", pady=1)
        self.btn_rem_hist = tk.Button(fr_hist_btns, text="－ Remover Sel.", font=("Segoe UI", 9), command=self._remover_hist_selecionado)
        self.btn_rem_hist.pack(fill="x", pady=1)
        self.btn_limpar_hist = tk.Button(fr_hist_btns, text="🗑 Limpar Histórico", font=("Segoe UI", 9), command=self._limpar_historico)
        self.btn_limpar_hist.pack(fill="x", pady=1)

        # ── Painel Direito ───────────────────────────────────
        self.frame_dir = tk.Frame(self.main_pane)
        self.main_pane.add(self.frame_dir, minsize=700)

        # Preview area (top)
        self.frame_preview = tk.Frame(self.frame_dir)
        self.frame_preview.pack(fill="both", expand=True, pady=(0,4))
        # Original
        fr_orig = tk.LabelFrame(self.frame_preview, text=" Original ", font=("Segoe UI", 10, "bold"))
        fr_orig.pack(side="left", fill="both", expand=True, padx=(0,2))
        self.canvas_orig = tk.Canvas(fr_orig, bg="#11111b")
        self.canvas_orig.pack(fill="both", expand=True)
        # Processada
        fr_proc = tk.LabelFrame(self.frame_preview, text=" Processada ", font=("Segoe UI", 10, "bold"))
        fr_proc.pack(side="left", fill="both", expand=True, padx=(2,0))
        self.canvas_proc = tk.Canvas(fr_proc, bg="#11111b")
        self.canvas_proc.pack(fill="both", expand=True)

        # ── Painel Métodos (bottom) ──────────────────────────
        self.frame_metodos = tk.LabelFrame(self.frame_dir, text=" Métodos de Processamento ", font=("Segoe UI", 10, "bold"))
        self.frame_metodos.pack(fill="x", pady=(4,0))

        self._criar_metodos()
        # Salvar
        self.btn_salvar = tk.Button(self.frame_metodos, text="💾 Salvar Resultado", font=("Segoe UI", 10, "bold"), command=self.salvar_resultado)
        self.btn_salvar.grid(row=5, column=0, columnspan=6, sticky="ew", padx=6, pady=(6,8))
        # Bind resize
        self.canvas_orig.bind("<Configure>", lambda e: self._exibir_preview())
        self.canvas_proc.bind("<Configure>", lambda e: self._exibir_preview())

    def _criar_metodos(self):
        fm = self.frame_metodos
        pad = {"padx": 4, "pady": 3}
        font_lbl = ("Segoe UI", 9)
        font_btn = ("Segoe UI", 9, "bold")

        # Row 0: Conversão de Cor
        tk.Label(fm, text="🎨 Cor:", font=font_lbl).grid(row=0, column=0, sticky="w", **pad)
        self.combo_cor = ttk.Combobox(fm, values=["Cinza", "HSV", "Lab", "YCrCb", "HLS", "★Cinza-Manual"], state="readonly", width=14)
        self.combo_cor.set("Cinza")
        self.combo_cor.grid(row=0, column=1, **pad)
        tk.Button(fm, text="Aplicar", font=font_btn, command=self.aplicar_conversao).grid(row=0, column=2, **pad)

        # Row 1: Filtro
        tk.Label(fm, text="🔵 Filtro:", font=font_lbl).grid(row=1, column=0, sticky="w", **pad)
        self.combo_filtro = ttk.Combobox(fm, values=["Gaussiano", "Mediana", "Bilateral", "Cartoon", "★Média-Manual"], state="readonly", width=14)
        self.combo_filtro.set("Gaussiano")
        self.combo_filtro.grid(row=1, column=1, **pad)
        tk.Label(fm, text="Kernel:", font=font_lbl).grid(row=1, column=3, sticky="e", **pad)
        self.slider_filtro = tk.Scale(fm, from_=1, to=31, orient="horizontal", length=150, resolution=2,
                                      command=lambda v: self._live_update("filtro"))
        self.slider_filtro.set(5)
        self.slider_filtro.grid(row=1, column=4, **pad)
        tk.Button(fm, text="Aplicar", font=font_btn, command=self.aplicar_filtro).grid(row=1, column=2, **pad)

        # Row 2: Borda
        tk.Label(fm, text="📐 Borda:", font=font_lbl).grid(row=2, column=0, sticky="w", **pad)
        self.combo_borda = ttk.Combobox(fm, values=["Canny", "Laplaciano", "Sobel X", "Sobel Y", "★Sobel-Manual"], state="readonly", width=14)
        self.combo_borda.set("Canny")
        self.combo_borda.grid(row=2, column=1, **pad)
        tk.Label(fm, text="Limiar:", font=font_lbl).grid(row=2, column=3, sticky="e", **pad)
        self.slider_borda = tk.Scale(fm, from_=10, to=250, orient="horizontal", length=150,
                                     command=lambda v: self._live_update("borda"))
        self.slider_borda.set(100)
        self.slider_borda.grid(row=2, column=4, **pad)
        tk.Button(fm, text="Aplicar", font=font_btn, command=self.aplicar_borda).grid(row=2, column=2, **pad)

        # Row 3: Binarização
        tk.Label(fm, text="⬛ Binário:", font=font_lbl).grid(row=3, column=0, sticky="w", **pad)
        self.combo_bin = ttk.Combobox(fm, values=["Otsu", "Adaptativo", "Global", "★Otsu-Manual"], state="readonly", width=14)
        self.combo_bin.set("Otsu")
        self.combo_bin.grid(row=3, column=1, **pad)
        tk.Label(fm, text="Limiar:", font=font_lbl).grid(row=3, column=3, sticky="e", **pad)
        self.slider_bin = tk.Scale(fm, from_=0, to=255, orient="horizontal", length=150,
                                   command=lambda v: self._live_update("bin"))
        self.slider_bin.set(127)
        self.slider_bin.grid(row=3, column=4, **pad)
        tk.Button(fm, text="Aplicar", font=font_btn, command=self.aplicar_binarizacao).grid(row=3, column=2, **pad)

        # Row 4: Morfologia
        tk.Label(fm, text="🔷 Morfol.:", font=font_lbl).grid(row=4, column=0, sticky="w", **pad)
        self.combo_morf = ttk.Combobox(fm, values=["Abertura", "Fechamento", "Erosão", "Dilatação", "Gradiente", "★Erosão-Manual"], state="readonly", width=14)
        self.combo_morf.set("Abertura")
        self.combo_morf.grid(row=4, column=1, **pad)
        tk.Label(fm, text="Kernel:", font=font_lbl).grid(row=4, column=3, sticky="e", **pad)
        self.slider_morf = tk.Scale(fm, from_=1, to=21, orient="horizontal", length=150, resolution=2,
                                    command=lambda v: self._live_update("morf"))
        self.slider_morf.set(5)
        self.slider_morf.grid(row=4, column=4, **pad)
        tk.Button(fm, text="Aplicar", font=font_btn, command=self.aplicar_morfologia).grid(row=4, column=2, **pad)

        fm.columnconfigure(4, weight=1)

    # ── Tema ─────────────────────────────────────────────────
    def _mudar_tema(self):
        self.tema_atual = self.combo_tema.get()
        self._aplicar_tema()

    def _aplicar_tema(self):
        t = TEMAS[self.tema_atual]
        bg, fg, panel, entry, select = t["bg"], t["fg"], t["panel"], t["entry"], t["select"]
        self.configure(bg=bg)
        for w in [self.topbar, self.frame_lista, self.frame_dir, self.frame_preview, self.frame_metodos]:
            w.configure(bg=bg)
        for w in [self.frame_metodos]:
            w.configure(fg=fg)
        self.lbl_titulo.configure(bg=bg, fg=t["accent"])
        self.listbox.configure(bg=entry, fg=fg, selectbackground=t["accent"], selectforeground=bg, borderwidth=0, highlightthickness=1, highlightcolor=t["accent"])
        for btn in [self.btn_add, self.btn_rem, self.btn_clear, self.btn_salvar, self.btn_usar_base, self.btn_rem_hist, self.btn_limpar_hist]:
            btn.configure(bg=t["btn"], fg=t["btn_fg"], activebackground=t["accent"], activeforeground=bg, relief="flat", borderwidth=0)
        self.btn_add.configure(bg=t["success"], fg="#1e1e2e")
        self.btn_rem.configure(bg=t["danger"], fg="#1e1e2e")
        self.btn_usar_base.configure(bg=t["warning"], fg="#1e1e2e")
        self.btn_rem_hist.configure(bg=t["danger"], fg="#1e1e2e")
        self.btn_limpar_hist.configure(bg=t["danger"], fg="#1e1e2e")
        self.btn_salvar.configure(bg=t["accent"], fg="#1e1e2e")
        canvas_bg = "#11111b" if self.tema_atual == "Escuro" else "#e6e9ef"
        self.canvas_orig.configure(bg=canvas_bg)
        self.canvas_proc.configure(bg=canvas_bg)
        self.listbox_hist.configure(bg=entry, fg=fg, selectbackground=t["warning"], selectforeground=bg, borderwidth=0, highlightthickness=1, highlightcolor=t["warning"])
        self.main_pane.configure(bg=bg)
        # Apply to all labels and frames recursively
        self._tema_recursivo(self.frame_metodos, bg, fg)
        self._tema_recursivo(self.topbar, bg, fg)
        for child in self.frame_lista.winfo_children():
            if isinstance(child, (tk.Label, tk.Frame)):
                child.configure(bg=bg, fg=fg) if isinstance(child, tk.Label) else child.configure(bg=bg)
        for child in self.frame_preview.winfo_children():
            if isinstance(child, tk.LabelFrame):
                child.configure(bg=bg, fg=fg)
        # Sliders
        for s in [self.slider_filtro, self.slider_borda, self.slider_bin, self.slider_morf]:
            s.configure(bg=bg, fg=fg, troughcolor=entry, activebackground=t["accent"], highlightthickness=0)

    def _tema_recursivo(self, widget, bg, fg):
        for child in widget.winfo_children():
            try:
                if isinstance(child, tk.Label):
                    child.configure(bg=bg, fg=fg)
                elif isinstance(child, tk.Button):
                    pass  # already styled
                elif isinstance(child, tk.Frame):
                    child.configure(bg=bg)
                    self._tema_recursivo(child, bg, fg)
            except tk.TclError:
                pass

    # ── CRUD ─────────────────────────────────────────────────
    def adicionar_imagens(self):
        caminhos = filedialog.askopenfilenames(
            title="Selecionar Imagens",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("Todos", "*.*")]
        )
        for c in caminhos:
            mat = cv2.imread(c)
            if mat is not None:
                img = ImagemProcessada(c, mat)
                self.vetor_imagens.append(img)
                self.listbox.insert("end", img.nome)
        if caminhos:
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set("end")
            self.listbox.event_generate("<<ListboxSelect>>")

    def remover_imagem(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.vetor_imagens.pop(idx)
        self.listbox.delete(idx)
        self.imagem_selecionada = None
        self.canvas_orig.delete("all")
        self.canvas_proc.delete("all")

    def limpar_tudo(self):
        self.vetor_imagens.clear()
        self.listbox.delete(0, "end")
        self.imagem_selecionada = None
        self.canvas_orig.delete("all")
        self.canvas_proc.delete("all")
        self._limpar_historico()

    def _ao_selecionar(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.imagem_selecionada = self.vetor_imagens[sel[0]]
        self._exibir_preview()

    # ── Preview ──────────────────────────────────────────────
    def _mat_para_photo(self, mat, canvas):
        if mat is None:
            return None
        cw = max(canvas.winfo_width(), 100)
        ch = max(canvas.winfo_height(), 100)
        h, w = mat.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(mat, (nw, nh), interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 2:
            img_pil = Image.fromarray(resized, mode="L")
        else:
            img_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        return ImageTk.PhotoImage(img_pil)

    def _exibir_preview(self):
        if not self.imagem_selecionada:
            return
        img = self.imagem_selecionada
        self.photo_orig = self._mat_para_photo(img.mat_original, self.canvas_orig)
        if self.photo_orig:
            self.canvas_orig.delete("all")
            self.canvas_orig.create_image(
                self.canvas_orig.winfo_width() // 2,
                self.canvas_orig.winfo_height() // 2,
                anchor="center", image=self.photo_orig
            )
        if img.mat_processada is not None:
            self.photo_proc = self._mat_para_photo(img.mat_processada, self.canvas_proc)
            if self.photo_proc:
                self.canvas_proc.delete("all")
                self.canvas_proc.create_image(
                    self.canvas_proc.winfo_width() // 2,
                    self.canvas_proc.winfo_height() // 2,
                    anchor="center", image=self.photo_proc
                )
        else:
            self.canvas_proc.delete("all")

    def _checar_selecao(self) -> bool:
        if not self.imagem_selecionada:
            messagebox.showwarning("Atenção", "Selecione uma imagem na lista primeiro!")
            return False
        return True

    # ── Histórico ────────────────────────────────────────────
    def _adicionar_ao_historico(self, descricao: str, mat: np.ndarray):
        """Salva uma cópia da Mat processada no histórico."""
        self.historico.append((descricao, mat.copy()))
        idx = len(self.historico)
        self.listbox_hist.insert("end", f"{idx}. {descricao}")
        self.listbox_hist.see("end")

    def _ao_selecionar_hist(self, event=None):
        """Ao clicar no histórico, mostra a imagem no painel Processada."""
        sel = self.listbox_hist.curselection()
        if not sel:
            return
        descricao, mat = self.historico[sel[0]]
        self.photo_hist = self._mat_para_photo(mat, self.canvas_proc)
        if self.photo_hist:
            self.canvas_proc.delete("all")
            self.canvas_proc.create_image(
                self.canvas_proc.winfo_width() // 2,
                self.canvas_proc.winfo_height() // 2,
                anchor="center", image=self.photo_hist
            )

    def _usar_hist_como_base(self):
        """Usa o item selecionado do histórico como imagem base para novos processamentos."""
        sel = self.listbox_hist.curselection()
        if not sel:
            messagebox.showwarning("Atenção", "Selecione um item do histórico primeiro!")
            return
        if not self.imagem_selecionada:
            messagebox.showwarning("Atenção", "Selecione uma imagem na lista primeiro!")
            return
        descricao, mat = self.historico[sel[0]]
        self.imagem_selecionada.mat_original = mat.copy()
        self.imagem_selecionada.mat_processada = None
        self.imagem_selecionada.metodo = ""
        self._exibir_preview()
        messagebox.showinfo("Base Atualizada", f"A imagem base agora é:\n{descricao}\n\nAplique um novo método em cima dela!")

    def _limpar_historico(self):
        """Limpa toda a lista de histórico."""
        self.historico.clear()
        self.listbox_hist.delete(0, "end")

    def _remover_hist_selecionado(self):
        """Remove o item selecionado do histórico."""
        sel = self.listbox_hist.curselection()
        if not sel:
            return
        idx = sel[0]
        self.historico.pop(idx)
        self.listbox_hist.delete(idx)
        self.canvas_proc.delete("all")
        
        # Atualiza a numeração do histórico
        self.listbox_hist.delete(0, "end")
        for i, (descricao, _) in enumerate(self.historico):
            self.listbox_hist.insert("end", f"{i + 1}. {descricao}")

    def _live_update(self, categoria):
        """Chamado pelos sliders — reaplica o último método da categoria."""
        if not self.imagem_selecionada:
            return
        m = self.imagem_selecionada.metodo
        if categoria == "filtro" and m.startswith("Filtro"):
            self.aplicar_filtro()
        elif categoria == "borda" and m.startswith("Borda"):
            self.aplicar_borda()
        elif categoria == "bin" and m.startswith("Bin"):
            self.aplicar_binarizacao()
        elif categoria == "morf" and m.startswith("Morf"):
            self.aplicar_morfologia()

    # ── Implementação Manual: Cinza ───────────────────────────
    def _cinza_manual(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Conversão manual BGR → Escala de Cinza.
        Fórmula ITU-R BT.601: Y = 0.299*R + 0.587*G + 0.114*B
        Implementada sem usar cv2.cvtColor.
        """
        b = img_bgr[:, :, 0].astype(np.float64)
        g = img_bgr[:, :, 1].astype(np.float64)
        r = img_bgr[:, :, 2].astype(np.float64)
        cinza = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
        return cinza

    # ── Implementação Manual: Filtro de Média ─────────────────
    def _media_manual(self, img: np.ndarray, k: int) -> np.ndarray:
        """
        Filtro de média (box blur) implementado manualmente.
        Cria um kernel k×k onde todos os pesos = 1/(k*k),
        depois aplica convolução 2D com cv2.filter2D.
        """
        kernel = np.ones((k, k), dtype=np.float32) / (k * k)
        return cv2.filter2D(img, -1, kernel)

    # ── Implementação Manual: Sobel ──────────────────────────
    def _sobel_manual(self, gray: np.ndarray) -> np.ndarray:
        """
        Detector de bordas Sobel implementado manualmente.
        Define os kernels Gx e Gy explicitamente e calcula
        a magnitude do gradiente: M = sqrt(Gx² + Gy²).
        """
        # Kernels de Sobel definidos manualmente
        kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)
        # Convolução com cada kernel
        gx = cv2.filter2D(gray, cv2.CV_64F, kx)
        gy = cv2.filter2D(gray, cv2.CV_64F, ky)
        # Magnitude do gradiente
        magnitude = cv2.magnitude(gx, gy)
        return cv2.convertScaleAbs(magnitude)

    # ── Implementação Manual: Otsu ───────────────────────────
    def _otsu_manual(self, gray: np.ndarray) -> np.ndarray:
        """
        Binarização de Otsu implementada manualmente.
        Calcula o histograma e encontra o limiar que maximiza
        a variância entre classes (foreground vs background).
        Não usa cv2.THRESH_OTSU.
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        total = gray.size
        soma_total = np.dot(np.arange(256), hist)

        soma_bg, peso_bg = 0.0, 0.0
        max_variancia, melhor_limiar = 0.0, 0

        for t in range(256):
            peso_bg += hist[t]
            if peso_bg == 0:
                continue
            peso_fg = total - peso_bg
            if peso_fg == 0:
                break
            soma_bg += t * hist[t]
            media_bg = soma_bg / peso_bg
            media_fg = (soma_total - soma_bg) / peso_fg
            variancia = peso_bg * peso_fg * (media_bg - media_fg) ** 2
            if variancia > max_variancia:
                max_variancia = variancia
                melhor_limiar = t

        # Aplica o limiar encontrado
        resultado = np.zeros_like(gray)
        resultado[gray > melhor_limiar] = 255
        return resultado

    # ── Implementação Manual: Erosão ─────────────────────────
    def _erosao_manual(self, img: np.ndarray, k: int) -> np.ndarray:
        """
        Erosão morfológica implementada manualmente.
        Para cada pixel, verifica a vizinhança k×k:
        o pixel de saída recebe o valor MÍNIMO da vizinhança.
        Usa um kernel retangular de 1's (elemento estruturante).
        """
        h, w = img.shape[:2]
        pad = k // 2
        # Adiciona borda (padding) com valor 255 para não afetar as bordas
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        resultado = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                # Extrai a vizinhança k×k
                vizinhanca = padded[i:i + k, j:j + k]
                resultado[i, j] = vizinhanca.min()
        return resultado

    # ── Conversão de Cor ─────────────────────────────────────
    def aplicar_conversao(self):
        if not self._checar_selecao():
            return
        img = self.imagem_selecionada
        metodo = self.combo_cor.get()

        if metodo == "★Cinza-Manual":
            # Implementação própria da conversão para cinza
            img.mat_processada = self._cinza_manual(img.mat_original)
        else:
            codigos = {
                "Cinza": cv2.COLOR_BGR2GRAY, "HSV": cv2.COLOR_BGR2HSV,
                "Lab": cv2.COLOR_BGR2Lab, "YCrCb": cv2.COLOR_BGR2YCrCb,
                "HLS": cv2.COLOR_BGR2HLS,
            }
            img.mat_processada = cv2.cvtColor(img.mat_original, codigos[metodo])

        img.metodo = f"Cor: {metodo}"
        self._adicionar_ao_historico(img.metodo, img.mat_processada)
        self._exibir_preview()

    # ── Filtros ──────────────────────────────────────────────
    def aplicar_filtro(self):
        if not self._checar_selecao():
            return
        img = self.imagem_selecionada
        metodo = self.combo_filtro.get()
        k = self.slider_filtro.get()
        k = k if k % 2 == 1 else k + 1  # garante ímpar
        src = img.mat_original

        if metodo == "Gaussiano":
            img.mat_processada = cv2.GaussianBlur(src, (k, k), 0)
        elif metodo == "Mediana":
            img.mat_processada = cv2.medianBlur(src, k)
        elif metodo == "Bilateral":
            img.mat_processada = cv2.bilateralFilter(src, k, 75, 75)
        elif metodo == "Cartoon":
            # Efeito cartoon: bilateral forte + bordas
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 9)
            color = src
            for _ in range(k // 2 + 1):
                color = cv2.bilateralFilter(color, 9, 300, 300)
            img.mat_processada = cv2.bitwise_and(color, color, mask=edges)
        elif metodo == "★Média-Manual":
            # Implementação própria do filtro de média
            img.mat_processada = self._media_manual(src, k)

        img.metodo = f"Filtro: {metodo} (k={k})"
        self._adicionar_ao_historico(img.metodo, img.mat_processada)
        self._exibir_preview()

    # ── Detector de Borda ────────────────────────────────────
    def aplicar_borda(self):
        if not self._checar_selecao():
            return
        img = self.imagem_selecionada
        metodo = self.combo_borda.get()
        limiar = self.slider_borda.get()
        gray = cv2.cvtColor(img.mat_original, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if metodo == "Canny":
            img.mat_processada = cv2.Canny(gray, limiar // 2, limiar)
        elif metodo == "Laplaciano":
            lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            img.mat_processada = cv2.convertScaleAbs(lap)
        elif metodo == "Sobel X":
            sob = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            img.mat_processada = cv2.convertScaleAbs(sob)
        elif metodo == "Sobel Y":
            sob = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            img.mat_processada = cv2.convertScaleAbs(sob)
        elif metodo == "★Sobel-Manual":
            # Implementação própria do Sobel com kernels definidos à mão
            img.mat_processada = self._sobel_manual(gray)

        img.metodo = f"Borda: {metodo} (t={limiar})"
        self._adicionar_ao_historico(img.metodo, img.mat_processada)
        self._exibir_preview()

    # ── Binarização ──────────────────────────────────────────
    def aplicar_binarizacao(self):
        if not self._checar_selecao():
            return
        img = self.imagem_selecionada
        metodo = self.combo_bin.get()
        limiar = self.slider_bin.get()
        gray = cv2.cvtColor(img.mat_original, cv2.COLOR_BGR2GRAY)

        if metodo == "Otsu":
            _, img.mat_processada = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif metodo == "Adaptativo":
            img.mat_processada = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv2.THRESH_BINARY, 31, 8)
        elif metodo == "Global":
            _, img.mat_processada = cv2.threshold(gray, limiar, 255, cv2.THRESH_BINARY)
        elif metodo == "★Otsu-Manual":
            # Implementação própria do algoritmo de Otsu
            img.mat_processada = self._otsu_manual(gray)

        img.metodo = f"Bin: {metodo} (t={limiar})"
        self._adicionar_ao_historico(img.metodo, img.mat_processada)
        self._exibir_preview()

    # ── Morfologia ───────────────────────────────────────────
    def aplicar_morfologia(self):
        if not self._checar_selecao():
            return
        img = self.imagem_selecionada
        metodo = self.combo_morf.get()
        k = self.slider_morf.get()
        k = k if k % 2 == 1 else k + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # Trabalha sobre cinza se não tiver processada binária
        src = img.mat_original
        if img.mat_processada is not None and len(img.mat_processada.shape) == 2:
            src = img.mat_processada
        elif len(src.shape) == 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        if metodo == "★Erosão-Manual":
            # Implementação própria da erosão (mínimo na vizinhança)
            img.mat_processada = self._erosao_manual(src, k)
        else:
            ops = {
                "Abertura": cv2.MORPH_OPEN, "Fechamento": cv2.MORPH_CLOSE,
                "Erosão": cv2.MORPH_ERODE, "Dilatação": cv2.MORPH_DILATE,
                "Gradiente": cv2.MORPH_GRADIENT,
            }
            img.mat_processada = cv2.morphologyEx(src, ops[metodo], kernel, iterations=1)

        img.metodo = f"Morf: {metodo} (k={k})"
        self._adicionar_ao_historico(img.metodo, img.mat_processada)
        self._exibir_preview()

    # ── Salvar ───────────────────────────────────────────────
    def salvar_resultado(self):
        if not self.imagem_selecionada or self.imagem_selecionada.mat_processada is None:
            messagebox.showwarning("Atenção", "Aplique um método antes de salvar!")
            return
        caminho = filedialog.asksaveasfilename(
            title="Salvar Resultado",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
            initialfile=f"{os.path.splitext(self.imagem_selecionada.nome)[0]}_resultado.png"
        )
        if caminho:
            cv2.imwrite(caminho, self.imagem_selecionada.mat_processada)
            messagebox.showinfo("Sucesso", f"Salvo em:\n{caminho}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
