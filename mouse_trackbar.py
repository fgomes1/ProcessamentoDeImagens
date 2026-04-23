"""
=============================================================
  HIGHGUI — MOUSE + TRACKBAR
  Exercício de Visão Computacional
  
  Funcionalidades:
    - Captura webcam em tempo real
    - Exibe coordenadas (x, y) do mouse sobre a janela
    - TrackBar para escolher a cor do texto (0–179 = matiz HSV)
=============================================================
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# ESTADO GLOBAL (compartilhado entre callback e loop principal)
# ─────────────────────────────────────────────────────────────
mouse_x: int = 0   # posição atual do cursor — coluna
mouse_y: int = 0   # posição atual do cursor — linha


# ─────────────────────────────────────────────────────────────
# CALLBACK DO MOUSE
# Registrado com cv2.setMouseCallback.
# Chamado automaticamente pelo HighGUI a cada evento de mouse.
# ─────────────────────────────────────────────────────────────
def on_mouse(event, x, y, flags, userdata):
    """
    Atualiza as coordenadas globais sempre que o mouse se mover
    dentro da janela. Também imprime cliques no terminal.

    Parâmetros (obrigatórios pelo OpenCV):
        event    : tipo do evento (EVENT_MOUSEMOVE, EVENT_LBUTTONDOWN, ...)
        x, y     : posição do cursor em pixels (coluna, linha)
        flags    : estado dos botões/modificadores
        userdata : dado extra passado em setMouseCallback (None aqui)
    """
    global mouse_x, mouse_y

    # Atualiza posição em qualquer movimento
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

    # Clique esquerdo — imprime no terminal
    elif event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        print(f"  🖱  Clique ESQUERDO  → ({x}, {y})")

    # Clique direito — imprime no terminal
    elif event == cv2.EVENT_RBUTTONDOWN:
        mouse_x, mouse_y = x, y
        print(f"  🖱  Clique DIREITO   → ({x}, {y})")


# ─────────────────────────────────────────────────────────────
# FUNÇÃO AUXILIAR: converte matiz HSV → BGR
# O TrackBar fornece H (0–179). Precisamos da cor em BGR
# para usar em cv2.putText.
# ─────────────────────────────────────────────────────────────
def hsv_hue_para_bgr(hue: int) -> tuple:
    """
    Converte um valor de Matiz (Hue) HSV em um pixel BGR.
    
    O OpenCV usa H: 0–179, S: 0–255, V: 0–255.
    Fixamos S=255 e V=255 para obter a cor mais viva possível.
    
    Retorna: tupla (B, G, R) com valores 0–255.
    """
    # Cria imagem 1×1 em HSV com saturação e valor máximos
    pixel_hsv = np.uint8([[[hue, 255, 255]]])
    # Converte para BGR
    pixel_bgr = cv2.cvtColor(pixel_hsv, cv2.COLOR_HSV2BGR)
    b, g, r = int(pixel_bgr[0, 0, 0]), int(pixel_bgr[0, 0, 1]), int(pixel_bgr[0, 0, 2])
    return (b, g, r)


# ─────────────────────────────────────────────────────────────
# FLUXO PRINCIPAL
# ─────────────────────────────────────────────────────────────
def main():
    NOME_JANELA = "HighGUI — Mouse + TrackBar"
    NOME_TB     = "Cor do Texto (Matiz HSV)"

    print("=" * 55)
    print("  HIGHGUI — Mouse + TrackBar")
    print("=" * 55)
    print("  Mova o mouse sobre a janela para ver (x, y).")
    print("  Ajuste o TrackBar para mudar a cor do texto.")
    print("  Pressione qualquer tecla para sair.\n")

    # ── 1. ABRE A WEBCAM ──────────────────────────────────────
    cap = cv2.VideoCapture(0)          # índice 0 = primeira câmera
    if not cap.isOpened():
        print("  ❌ Não foi possível abrir a webcam.")
        print("     Verifique se ela está conectada e não está em uso.")
        return

    largura  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ✅ Webcam aberta — {largura}×{altura} px\n")

    # ── 2. CRIA A JANELA ──────────────────────────────────────
    # WINDOW_NORMAL  → permite redimensionar arrastando a borda
    # WINDOW_AUTOSIZE → tamanho fixo igual à imagem
    cv2.namedWindow(NOME_JANELA, cv2.WINDOW_AUTOSIZE)

    # ── 3. REGISTRA CALLBACK DO MOUSE ────────────────────────
    # A partir daqui, on_mouse() é chamado automaticamente
    # pelo HighGUI a cada evento de mouse dentro da janela.
    cv2.setMouseCallback(NOME_JANELA, on_mouse)

    # ── 4. CRIA O TRACKBAR ────────────────────────────────────
    # createTrackbar(nome_barra, janela, valor_inicial, maximo, callback)
    # O matiz HSV no OpenCV vai de 0 a 179.
    # Callback = None → não precisamos de função; lemos no loop.
    VALOR_INICIAL_HUE = 60   # começa em verde
    cv2.createTrackbar(NOME_TB, NOME_JANELA, VALOR_INICIAL_HUE, 179, lambda v: None)

    # ── 5. LOOP PRINCIPAL ─────────────────────────────────────
    while True:
        # ── PRIMEIRO: verifica se a janela ainda existe ────────
        # Isso PRECISA vir antes de qualquer cv2.getTrackbarPos.
        # Se o usuário fechou a janela pelo botão X, a janela é
        # destruída internamente e qualquer leitura de TrackBar
        # nela levantaria cv2.error: NULL window.
        try:
            if cv2.getWindowProperty(NOME_JANELA, cv2.WND_PROP_VISIBLE) < 1:
                print("\n  Janela fechada — encerrando.")
                break
        except cv2.error:
            break

        ret, frame = cap.read()        # captura um frame da câmera
        if not ret:
            print("  ⚠️  Falha ao capturar frame. Encerrando...")
            break

        # Espelha horizontalmente (comportamento natural de webcam)
        frame = cv2.flip(frame, 1)

        # ── Lê o TrackBar para obter a cor atual ─────────────
        # Agora é seguro: sabemos que a janela ainda existe.
        try:
            hue = cv2.getTrackbarPos(NOME_TB, NOME_JANELA)
        except cv2.error:
            break
        cor_bgr = hsv_hue_para_bgr(hue)

        # ── Texto das coordenadas ─────────────────────────────
        texto_coords = f"x={mouse_x}  y={mouse_y}"

        # Sombra (preto) para melhorar legibilidade em qualquer fundo
        FONTE      = cv2.FONT_HERSHEY_SIMPLEX
        ESCALA     = 0.8
        ESPESSURA  = 2
        POS_COORDS = (20, 45)      # canto superior esquerdo
        POS_HUE    = (20, 80)      # linha abaixo

        # Camada de sombra (deslocada 2px) em preto
        cv2.putText(frame, texto_coords,
                    (POS_COORDS[0]+2, POS_COORDS[1]+2),
                    FONTE, ESCALA, (0, 0, 0), ESPESSURA + 1)
        # Texto principal com a cor do TrackBar
        cv2.putText(frame, texto_coords,
                    POS_COORDS, FONTE, ESCALA, cor_bgr, ESPESSURA)

        # ── Texto indicativo do matiz atual ───────────────────
        texto_hue = f"Matiz (H): {hue}"
        cv2.putText(frame, texto_hue,
                    (POS_HUE[0]+2, POS_HUE[1]+2),
                    FONTE, 0.65, (0, 0, 0), ESPESSURA + 1)
        cv2.putText(frame, texto_hue,
                    POS_HUE, FONTE, 0.65, cor_bgr, ESPESSURA)

        # ── Marcador visual na posição do mouse ───────────────
        # Círculo vazio (sem preenchimento) na posição do cursor
        if 0 <= mouse_x < largura and 0 <= mouse_y < altura:
            cv2.circle(frame, (mouse_x, mouse_y), 10, cor_bgr, 2)
            cv2.drawMarker(frame, (mouse_x, mouse_y), cor_bgr,
                           cv2.MARKER_CROSS, 20, 2)

        # ── Exibe o frame ─────────────────────────────────────
        # ATENÇÃO: waitKey é OBRIGATÓRIO para o HighGUI atualizar
        # a janela e processar eventos (mouse, teclado, trackbar).
        cv2.imshow(NOME_JANELA, frame)

        # waitKey(1) → espera 1ms; retorna ≥0 se tecla pressionada
        if cv2.waitKey(1) >= 0:
            print("\n  Tecla pressionada — encerrando.")
            break

    # ── 6. LIMPEZA ────────────────────────────────────────────
    cap.release()           # libera a webcam
    cv2.destroyAllWindows() # fecha todas as janelas do HighGUI
    print("  ✅ Recursos liberados. Até logo!\n")


if __name__ == "__main__":
    main()
