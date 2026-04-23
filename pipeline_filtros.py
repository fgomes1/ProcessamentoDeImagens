"""
=============================================================
  PIPELINE DE PROCESSAMENTO EM TEMPO REAL — WEBCAM
  Exercício de Visão Computacional / HighGUI

  Etapas do pipeline:
    1. Captura frame da webcam
    2. Clique do mouse → exibe cor BGR do pixel na imagem original
    3. TrackBar PB1 → filtro Passa-Baixa 1 (Gaussiano) no canal V
    4. TrackBar PB2 → filtro Passa-Baixa 2 (Mediana)   no canal V
    5. Filtro Passa-Alta (Laplaciano) aplicado sobre PB1
    6. Binarização automática pelo método de Otsu
    
  Layout da janela (grade 3 colunas × 2 linhas):
    ┌──────────────┬──────────────┬──────────────┐
    │  Original    │  Canal V     │  PB1 Gauss   │
    │  (BGR click) │  (HSV)       │  (suavizado) │
    ├──────────────┼──────────────┼──────────────┤
    │  PB2 Mediana │  Passa-Alta  │  Binarizada  │
    │  (suavizado) │  (bordas)    │  (Otsu)      │
    └──────────────┴──────────────┴──────────────┘
=============================================================
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# CONSTANTES DE LAYOUT
# ─────────────────────────────────────────────────────────────
THUMB_W = 320    # largura de cada miniatura na grade
THUMB_H = 240    # altura  de cada miniatura na grade
COLS    = 3      # número de colunas da grade
ROWS    = 2      # número de linhas  da grade

# ─────────────────────────────────────────────────────────────
# ESTADO GLOBAL (compartilhado com o callback do mouse)
# ─────────────────────────────────────────────────────────────
click_x     : int   = -1       # coluna no frame original
click_y     : int   = -1       # linha  no frame original
click_color : tuple = None     # (B, G, R) do pixel clicado
frame_w     : int   = 640      # largura real do frame da câmera
frame_h     : int   = 480      # altura  real do frame da câmera


# ─────────────────────────────────────────────────────────────
# CALLBACK DO MOUSE
# Só responde a cliques dentro do quadrante ORIGINAL (col=0, row=0).
# ─────────────────────────────────────────────────────────────
def on_mouse(event, gx, gy, flags, userdata):
    """
    Detecta cliques na grade e mapeia de volta ao frame original.

    O quadrante Original ocupa x ∈ [0, THUMB_W) e y ∈ [0, THUMB_H)
    na janela da grade. Fazemos a proporção inversa para encontrar
    a posição real no frame da câmera (que pode ser maior).
    """
    global click_x, click_y

    if event == cv2.EVENT_LBUTTONDOWN:
        # Verifica se o clique foi dentro do quadrante Original
        if 0 <= gx < THUMB_W and 0 <= gy < THUMB_H:
            # Escala de volta: thumb → frame real
            click_x = int(gx * (frame_w / THUMB_W))
            click_y = int(gy * (frame_h / THUMB_H))


# ─────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────────────────────

def kernel_impar(valor_tb: int, minimo: int = 1) -> int:
    """
    Converte o valor do TrackBar (0–N) em um tamanho de kernel SEMPRE ÍMPAR.
    
    Regra: k = 2 * valor + 1
      valor=0 → k=1  (identidade — sem filtro)
      valor=1 → k=3
      valor=2 → k=5  ...
    
    GaussianBlur e medianBlur exigem kernel ímpar e ≥ 1.
    """
    return max(2 * valor_tb + 1, minimo)


def adicionar_label(img, titulo: str, detalhe: str = "") -> np.ndarray:
    """
    Redimensiona a imagem para (THUMB_W × THUMB_H) e adiciona
    uma faixa semi-transparente no topo com título e detalhe.

    Aceita imagens grayscale (2D) ou BGR (3D).
    Retorna sempre uma imagem BGR.
    """
    # Converte grayscale para BGR se necessário
    if len(img.shape) == 2:
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        display = img.copy()

    # Redimensiona para o tamanho da miniatura
    display = cv2.resize(display, (THUMB_W, THUMB_H))

    # Faixa de fundo semi-transparente no topo
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (THUMB_W, 32), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    # Título (esquerda)
    cv2.putText(display, titulo, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

    # Detalhe (direita) — ex.: "kernel=5"
    if detalhe:
        # getTextSize retorna ((largura, altura), baseline)
        # usamos [0][0] para pegar só a largura como int
        tw = cv2.getTextSize(detalhe, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        cv2.putText(display, detalhe, (THUMB_W - tw - 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 140), 1, cv2.LINE_AA)
    return display


def desenhar_info_cor(frame: np.ndarray, cx: int, cy: int, cor: tuple) -> np.ndarray:
    """
    Desenha sobre o frame original:
      - Crosshair e círculo na posição clicada
      - Caixa de texto com BSG=(...) no canto inferior esquerdo
      - Quadrado preenchido com a cor clicada ao lado do texto
    """
    img = frame.copy()
    B, G, R = cor

    # Marcador na posição do clique
    cv2.circle(img, (cx, cy), 14, (B, G, R), -1)
    cv2.circle(img, (cx, cy), 14, (255, 255, 255), 2)
    cv2.drawMarker(img, (cx, cy), (255, 255, 255),
                   cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)

    # Caixa de informação na parte inferior
    BOX_Y1 = frame_h - 44
    BOX_Y2 = frame_h - 4
    cv2.rectangle(img, (6, BOX_Y1), (310, BOX_Y2), (20, 20, 20), -1)

    # Quadrado colorido
    cv2.rectangle(img, (10, BOX_Y1 + 4), (36, BOX_Y2 - 4), (B, G, R), -1)
    cv2.rectangle(img, (10, BOX_Y1 + 4), (36, BOX_Y2 - 4), (200, 200, 200), 1)

    # Texto
    texto = f"BGR = ({B:3d}, {G:3d}, {R:3d})"
    cv2.putText(img, texto, (42, BOX_Y2 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 210, 210), 1, cv2.LINE_AA)
    return img


# ─────────────────────────────────────────────────────────────
# FLUXO PRINCIPAL
# ─────────────────────────────────────────────────────────────
def main():
    global frame_w, frame_h, click_color

    NOME_JANELA = "Pipeline de Filtros — HighGUI"
    TB_PB1      = "PB1 Gauss  (kernel)"
    TB_PB2      = "PB2 Median (kernel)"
    TB_MODO     = "Binariz.: 0=Otsu  1=Adaptativo"

    print("=" * 58)
    print("  PIPELINE DE FILTROS — WEBCAM")
    print("=" * 58)
    print("  Clique no quadrante ORIGINAL (topo-esq.) para ver")
    print("  a cor BGR do pixel clicado.")
    print("  Ajuste os TrackBars para controlar os filtros.")
    print("  Pressione qualquer tecla ou feche a janela p/ sair.\n")

    # ── 1. ABRE A WEBCAM ──────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ❌ Não foi possível abrir a webcam.")
        print("     Verifique se ela está conectada e não está em uso.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ✅ Webcam aberta — {frame_w}×{frame_h} px\n")

    # ── 2. CRIA JANELA, CALLBACK E TRACKBARS ─────────────────
    cv2.namedWindow(NOME_JANELA, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(NOME_JANELA, on_mouse)

    # TrackBar PB1 — GaussianBlur
    cv2.createTrackbar(TB_PB1, NOME_JANELA, 1, 20, lambda v: None)

    # TrackBar PB2 — medianBlur
    cv2.createTrackbar(TB_PB2, NOME_JANELA, 1, 20, lambda v: None)

    # TrackBar MODO — escolha do método de binarização
    # 0 = Otsu (limiar global automático)
    # 1 = Adaptativo (limiar local por região — melhor com iluminação variada)
    cv2.createTrackbar(TB_MODO, NOME_JANELA, 0, 1, lambda v: None)

    click_color = None

    # ── LOOP PRINCIPAL ────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("  ⚠️  Falha ao capturar frame. Encerrando...")
            break

        # Espelha para comportamento natural de webcam
        frame = cv2.flip(frame, 1)

        # ── Etapa 1: Lê cor do pixel clicado ──────────────────
        if 0 <= click_x < frame_w and 0 <= click_y < frame_h:
            b, g, r = frame[click_y, click_x]
            click_color = (int(b), int(g), int(r))

        # ── Etapa 1 exibição: Original com marcador ───────────
        if click_color and click_x >= 0:
            orig = desenhar_info_cor(frame, click_x, click_y, click_color)
        else:
            orig = frame.copy()

        # ── Etapa 2: Converte para HSV e extrai canal V ────────
        # V (Value/Valor) representa o brilho — boa escolha para
        # filtragem pois concentra a informação estrutural da cena.
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        canal = cv2.split(hsv)[2]   # índice 2 = canal V

        # ── Etapa 3: Filtro Passa-Baixa 1 — GaussianBlur ──────
        # GaussianBlur: cada pixel recebe média ponderada dos
        # vizinhos por uma gaussiana 2D. Reduz ruído de forma suave.
        # Quanto maior o kernel, mais borrada (suavizada) fica a imagem.
        v1 = cv2.getTrackbarPos(TB_PB1, NOME_JANELA)
        k1 = kernel_impar(v1)
        pb1 = cv2.GaussianBlur(canal, (k1, k1), sigmaX=0)
        # sigmaX=0 → OpenCV calcula sigma automaticamente a partir do kernel

        # ── Etapa 4: Filtro Passa-Baixa 2 — medianBlur ────────
        # medianBlur: cada pixel recebe a MEDIANA dos pixels vizinhos.
        # Muito eficaz com ruído "sal e pimenta" (impulsos aleatórios).
        # Preserva bordas melhor que o Gaussiano.
        v2 = cv2.getTrackbarPos(TB_PB2, NOME_JANELA)
        k2 = kernel_impar(v2)
        pb2 = cv2.medianBlur(canal, k2)

        # ── Etapa 5: Filtro Passa-Alta — Laplaciano ───────────
        # O Laplaciano (∇²f) é a segunda derivada da intensidade.
        # Realça regiões de variação rápida (bordas e detalhes).
        # Aplicado sobre pb1 (já sem ruído) para resultados mais limpos.
        lap = cv2.Laplacian(pb1, cv2.CV_64F, ksize=3)
        # CV_64F captura respostas negativas (bordas em ambas as direções)
        pa  = cv2.convertScaleAbs(lap)   # converte de volta para uint8

        # POR QUE FICA ESCURO?
        # O Laplaciano é ≈0 em regiões uniformes e alto apenas nas bordas.
        # Como a maioria dos pixels é uniforme, quase tudo fica próximo
        # de 0 (preto). Normalizamos para esticar o intervalo real
        # [min, max] → [0, 255], tornando as bordas visíveis.
        pa = cv2.normalize(pa, None, 0, 255, cv2.NORM_MINMAX)

        # ── Etapa 6: Binarização ───────────────────────────────
        modo_bin = cv2.getTrackbarPos(TB_MODO, NOME_JANELA)

        if modo_bin == 0:
            # ── OTSU: limiar GLOBAL automático ────────────────
            # Analisa o histograma do frame inteiro e encontra o
            # limiar que maximiza a variância entre fundo e objeto.
            # PROBLEMA: recalcula a cada frame → quando um objeto
            # grande muda o histograma, o limiar se adapta e o
            # objeto pode "sumir" (é o que você observou com a mão).
            limiar_otsu, binaria = cv2.threshold(
                pb1, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            info_bin = f"Otsu t={int(limiar_otsu)}"

        else:
            # ── ADAPTATIVO: limiar LOCAL por região ───────────
            # Divide a imagem em blocos e calcula um limiar
            # diferente para cada bloco (média dos vizinhos - C).
            # Muito mais robusto a variações de iluminação:
            # a mão fica branca independente de quanto tempo
            # está na frente da câmera.
            #
            # blockSize = tamanho da vizinhança (deve ser ímpar)
            # C = constante subtraída da média local
            binaria = cv2.adaptiveThreshold(
                pb1,
                maxValue   = 255,
                adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType  = cv2.THRESH_BINARY,
                blockSize  = 31,   # analisa vizinhança 31×31
                C          = 5,    # subtrai 5 da média local
            )
            info_bin = "Adaptativo (local)"

        # ── Monta a grade 3×2 ─────────────────────────────────
        linha1 = np.hstack([
            adicionar_label(orig,    "Original",              "clique = cor BGR"),
            adicionar_label(canal,   "Canal V (HSV)",         "brilho / luminancia"),
            adicionar_label(pb1,     "PB1 - Gaussiano",       f"kernel={k1}"),
        ])
        linha2 = np.hstack([
            adicionar_label(pb2,     "PB2 - Mediana",         f"kernel={k2}"),
            adicionar_label(pa,      "Passa-Alta (Laplaciano)", "bordas / detalhes"),
            adicionar_label(binaria, "Binarizada", info_bin),
        ])
        grade = np.vstack([linha1, linha2])

        # ── Exibe a grade ──────────────────────────────────────
        # ATENÇÃO: waitKey é OBRIGATÓRIO para o HighGUI processar
        # eventos de mouse, teclado e trackbar!
        cv2.imshow(NOME_JANELA, grade)

        if cv2.waitKey(1) >= 0:
            print("\n  Tecla pressionada — encerrando.")
            break

        # Detecta fechamento pelo botão X da janela
        try:
            if cv2.getWindowProperty(NOME_JANELA, cv2.WND_PROP_VISIBLE) < 1:
                print("\n  Janela fechada — encerrando.")
                break
        except cv2.error:
            break

    # ── LIMPEZA ───────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("  ✅ Recursos liberados. Até logo!\n")


if __name__ == "__main__":
    main()
