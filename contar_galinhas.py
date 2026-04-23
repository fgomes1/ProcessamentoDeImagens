"""
=============================================================
  CONTAGEM DE GALINHAS EM REGIÃO DE INTERESSE (ROI)
  Exercício de Visão Computacional / Processamento de Imagens

  Pipeline passo-a-passo:
    1. Carregar imagem de entrada
    2. Selecionar ROI (retângulo vermelho) — manual ou automática
    3. Pré-processamento: conversão para escala de cinza + suavização
    4. Binarização adaptativa / Otsu
    5. Operações morfológicas (abertura + fechamento leve)
    6. Separação por Watershed (dist. transform + marcadores)
    7. Detecção e filtragem de contornos
    8. Contagem e visualização dos resultados
    9. Salvar imagem com cada etapa do pipeline

  Uso:
    python contar_galinhas.py <caminho_da_imagem>
    python contar_galinhas.py              (usa webcam/imagem padrão)
=============================================================
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')          # backend sem janela GUI para salvar
import matplotlib.pyplot as plt
import os
import sys


# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
# Área mínima e máxima (em pixels) para considerar um contorno
# como sendo uma galinha. Ajuste conforme a resolução da imagem.
AREA_MIN_GALINHA = 300
AREA_MAX_GALINHA = 15000

# Cores para visualização (BGR)
COR_CONTORNO   = (0, 255, 0)     # verde — contorno aceito
COR_REJEITADO  = (0, 0, 255)     # vermelho — contorno rejeitado
COR_ROI        = (0, 0, 255)     # vermelho — retângulo ROI
COR_NUMERO     = (255, 255, 0)   # ciano — número sobre a galinha
COR_TEXTO_INFO = (230, 230, 230) # cinza claro — textos informativos


# ─────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────────────────────

def mostrar_barra(titulo: str) -> None:
    """Imprime um cabeçalho formatado no terminal."""
    largura = 60
    print("\n" + "═" * largura)
    print(f"  {titulo}")
    print("═" * largura)


def carregar_imagem(caminho: str) -> np.ndarray:
    """
    Carrega a imagem do disco.
    Levanta FileNotFoundError se o arquivo não existir
    ou ValueError se o cv2 não conseguir decodificar.
    """
    if not os.path.isfile(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: '{caminho}'")
    imagem = cv2.imread(caminho)
    if imagem is None:
        raise ValueError(
            f"cv2.imread não conseguiu ler '{caminho}'.\n"
            "Verifique se é um formato suportado (jpg, png, bmp, tiff, etc.)."
        )
    return imagem


def detectar_retangulo_vermelho(imagem: np.ndarray) -> tuple:
    """
    Tenta detectar automaticamente um retângulo vermelho na imagem.
    
    Converte para HSV e filtra pixels vermelhos (H ≈ 0° ou H ≈ 170°–180°).
    Se encontrar contornos retangulares vermelhos, retorna o boundingRect
    do maior deles.
    
    Retorna: (x, y, w, h) ou None se não encontrar.
    """
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # O vermelho no HSV ocupa duas faixas de Hue:
    #   H ∈ [0, 10] e H ∈ [170, 180]
    # Saturação e Value altos para pegar vermelho vivo (do retângulo desenhado)
    mascara1 = cv2.inRange(hsv, (0,   100, 100), (10,  255, 255))
    mascara2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    mascara_vermelho = cv2.bitwise_or(mascara1, mascara2)

    # Dilata para conectar pixels próximos do retângulo
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mascara_vermelho = cv2.dilate(mascara_vermelho, kernel, iterations=2)

    contornos, _ = cv2.findContours(
        mascara_vermelho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return None

    # Pega o maior contorno vermelho
    maior = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(maior)

    # Filtra contornos muito pequenos (ruído)
    if area < 500:
        return None

    return cv2.boundingRect(maior)


def selecionar_roi_manual(imagem: np.ndarray) -> tuple:
    """
    Abre uma janela para o usuário selecionar a ROI manualmente
    arrastando um retângulo com o mouse.
    """
    print("  📐 Selecione a região de interesse (ROI) com o mouse.")
    print("     Arraste para desenhar o retângulo e pressione ENTER/ESPAÇO.")
    print("     Pressione C para cancelar.\n")

    roi = cv2.selectROI("Selecione a ROI", imagem, showCrosshair=True)
    cv2.destroyWindow("Selecione a ROI")

    if roi == (0, 0, 0, 0):
        return None
    return roi


def preprocessar(roi_img: np.ndarray, ksize_gauss: int = 7) -> dict:
    """
    Pipeline de pré-processamento da ROI:
      1. Conversão BGR → Escala de cinza
      2. Equalização de histograma (CLAHE) para melhorar contraste
      3. Suavização Gaussiana para reduzir ruído
    
    Retorna dict com imagens intermediárias para visualização.
    """
    etapas = {}

    # ── Passo 1: Escala de cinza ──────────────────────────────
    cinza = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    etapas["1_cinza"] = cinza.copy()
    print(f"    → Convertido para escala de cinza: {cinza.shape}")

    # ── Passo 2: CLAHE — equalização adaptativa de histograma ─
    # CLAHE divide a imagem em blocos e equaliza cada um
    # individualmente. Melhora contraste local sem estourar
    # regiões já claras (melhor que equalizeHist global).
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalizado = clahe.apply(cinza)
    etapas["2_clahe"] = equalizado.copy()
    print(f"    → CLAHE aplicado (clipLimit=3.0, tiles=8×8)")

    # ── Passo 3: Suavização Gaussiana ─────────────────────────
    # Remove ruído de alta frequência preservando bordas maiores.
    # ksize deve ser ímpar.
    ksize = ksize_gauss if ksize_gauss % 2 == 1 else ksize_gauss + 1
    suavizado = cv2.GaussianBlur(equalizado, (ksize, ksize), sigmaX=0)
    etapas["3_suavizado"] = suavizado.copy()
    print(f"    → GaussianBlur aplicado (kernel={ksize}×{ksize})")

    return etapas


def binarizar(imagem_cinza: np.ndarray) -> dict:
    """
    Aplica dois métodos de binarização e retorna ambos:
      - Otsu: limiar global automático (bom para bimodais)
      - Adaptativo: limiar local (bom para iluminação variável)
    
    A imagem de câmera de segurança tem iluminação irregular,
    então o método adaptativo tende a funcionar melhor.
    """
    etapas = {}

    # ── Otsu ──────────────────────────────────────────────────
    limiar_otsu, bin_otsu = cv2.threshold(
        imagem_cinza, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    etapas["4a_otsu"] = bin_otsu.copy()
    print(f"    → Binarização Otsu: limiar automático = {int(limiar_otsu)}")

    # ── Adaptativo (Gaussiano) ────────────────────────────────
    # blockSize grande para capturar variações globais de iluminação
    # C = constante subtraída da média local
    bin_adapt = cv2.adaptiveThreshold(
        imagem_cinza,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=8,
    )
    etapas["4b_adaptativo"] = bin_adapt.copy()
    print(f"    → Binarização Adaptativa: blockSize=31, C=8")

    # Usamos Otsu como padrão (geralmente melhor para este cenário
    # onde queremos separar galinhas claras do fundo escuro)
    etapas["4_binaria"] = bin_otsu.copy()

    return etapas


def operacoes_morfologicas(binaria: np.ndarray) -> dict:
    """
    Aplica operações morfológicas para limpar a imagem binária:

      1. ABERTURA (erosion → dilation): remove pequenos ruídos
         brancos isolados (sal), preservando objetos maiores.
      
      2. FECHAMENTO LEVE (dilation → erosion): preenche pequenos
         buracos DENTRO de cada galinha, sem fundir galinhas vizinhas.
         IMPORTANTE: kernel e iterações devem ser pequenos para não
         juntar objetos separados!
    """
    etapas = {}

    # Elemento estruturante elíptico — mais suave que retangular
    kernel_pequeno = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Limpa o chão moderadamente
    kernel_fechar  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ── Passo 5a: Abertura — remover ruído ────────────────────
    aberta = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_pequeno, iterations=1)
    etapas["5a_abertura"] = aberta.copy()
    print(f"    → Abertura morfológica (kernel 5x5, 1 iter.) — equilíbrio de ruído")

    # ── Passo 5b: Fechamento LEVE — preencher buracos internos ─
    # Kernel 3×3 com 1 iteração: fecha buracos de 1-2 px dentro
    # de cada galinha sem fundir galinhas vizinhas que se tocam.
    fechada = cv2.morphologyEx(aberta, cv2.MORPH_CLOSE, kernel_fechar, iterations=1)
    etapas["5b_fechamento"] = fechada.copy()
    print(f"    → Fechamento morfológico LEVE (kernel 3×3, 1 iter.)")

    etapas["5_morfologia"] = fechada.copy()

    return etapas


def separar_por_watershed(binaria: np.ndarray, roi_img: np.ndarray) -> dict:
    """
    Usa a Transformada de Distância + Watershed para separar
    objetos (galinhas) que se tocam ou se sobrepõem.

    Conceito:
      1. Distance Transform: calcula a distância de cada pixel
         branco até a borda preta mais próxima. O centro de cada
         galinha terá valor alto (longe das bordas).
      2. Thresholding da distância: os picos (centros) viram
         marcadores "sure foreground" (certeza de ser galinha).
      3. Watershed: a partir dos marcadores, "inunda" a imagem
         como bacias hidrográficas, criando linhas divisórias
         onde duas "águas" se encontram.
    """
    etapas = {}

    # ── Passo 6a: Transformada de Distância ───────────────────
    # DIST_L2 = distância euclidiana
    # O resultado é uma imagem float onde cada pixel tem o valor
    # da distância até o pixel 0 (preto) mais próximo.
    dist = cv2.distanceTransform(binaria, cv2.DIST_L2, 5)

    # Normalizar para visualização [0, 255]
    dist_viz = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    etapas["6a_distancia"] = dist_viz.copy()
    print(f"    → Transformada de Distância calculada (DIST_L2)")
    print(f"      Distância máxima: {dist.max():.1f} px")

    # ── Passo 6b: Encontrar os picos da distância (Máximos Locais) ─
    # Em vez de um limiar global que apaga galinhas pequenas (como 50%),
    # nós usamos um filtro para achar o "cume de cada montanha".
    # Criamos uma janela onde cada pixel testa se ele é o pico da vizinhança.
    kernel_max = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    local_max = cv2.dilate(dist, kernel_max)
    
    # Marcamos como 255 se for o topo do morro E a espessura for > 4.5px
    sure_fg = np.zeros_like(dist, dtype=np.uint8)
    sure_fg[(dist == local_max) & (dist > 4.5)] = 255
    
    etapas["6b_picos"] = sure_fg.copy()
    print(f"    → Picos identificados usando Máximos Locais (raio > 4.5px)")

    # ── Passo 6c: "Sure background" — dilatação da binária ────
    # Expandir a região branca para ter certeza que cobre tudo
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(binaria, kernel, iterations=2)

    # ── Passo 6d: Região desconhecida ─────────────────────────
    # unknown = sure_bg - sure_fg
    # Essas são as regiões entre os centros e o fundo — é onde
    # o Watershed vai decidir a qual galinha pertence.
    unknown = cv2.subtract(sure_bg, sure_fg)

    # ── Passo 6e: Rotular marcadores ──────────────────────────
    # connectedComponents atribui um rótulo inteiro a cada
    # componente conectado nos picos (sure_fg).
    num_labels, markers = cv2.connectedComponents(sure_fg)
    print(f"    → {num_labels - 1} marcador(es) encontrado(s) (centros de galinhas)")

    # Incrementar todos os rótulos em 1, para que o fundo
    # fique como 1 (e não 0, que no Watershed significa "desconhecido")
    markers = markers + 1

    # Marcar as regiões desconhecidas como 0
    markers[unknown == 255] = 0

    # ── Passo 6f: Watershed ───────────────────────────────────
    # O Watershed precisa de uma imagem BGR (3 canais)
    markers_ws = cv2.watershed(roi_img, markers)
    # markers_ws == -1 nas linhas divisórias (bordas entre galinhas)
    # markers_ws == 1 no fundo
    # markers_ws >= 2 em cada galinha

    # Visualizar as linhas do watershed sobre a imagem
    ws_viz = roi_img.copy()
    ws_viz[markers_ws == -1] = [0, 255, 255]  # linhas amarelas

    # Criar visualização colorida dos rótulos para debug
    # Cada rótulo >= 2 recebe uma cor única
    separada_viz = np.zeros((*binaria.shape, 3), dtype=np.uint8)
    cores_rotulos = []
    for label_id in range(2, num_labels + 1):
        # Gerar cor única para cada rótulo usando HSV
        hue = int(180 * (label_id - 2) / max(num_labels - 1, 1))
        cor_hsv = np.uint8([[[hue, 200, 255]]])
        cor_bgr = cv2.cvtColor(cor_hsv, cv2.COLOR_HSV2BGR)[0][0]
        cores_rotulos.append(tuple(int(c) for c in cor_bgr))
        separada_viz[markers_ws == label_id] = cor_bgr

    etapas["6c_watershed"] = ws_viz
    etapas["6_separada_viz"] = separada_viz
    etapas["6_markers_ws"] = markers_ws   # os rótulos inteiros
    etapas["6_num_labels"] = num_labels   # total de rótulos

    num_objetos = num_labels - 1  # desconta o fundo
    print(f"    → Watershed concluído: {num_objetos} região(ões) separada(s)")

    return etapas


def detectar_e_contar_watershed(
    markers_ws: np.ndarray,
    num_labels: int,
    roi_img: np.ndarray,
    area_min: int = AREA_MIN_GALINHA,
    area_max: int = AREA_MAX_GALINHA,
) -> tuple:
    """
    Conta galinhas usando os RÓTULOS DO WATERSHED diretamente.

    Em vez de converter de volta para binário (o que reconecta
    regiões vizinhas!), iteramos sobre cada rótulo >= 2 do
    watershed individualmente:
      - Para cada rótulo, cria uma máscara individual
      - Encontra o contorno dessa máscara
      - Filtra por área
    
    Assim cada galinha separada pelo watershed é contada
    independentemente, mesmo que se toquem.

    Retorna:
      - imagem_resultado: ROI com contornos desenhados e numerados
      - contornos_aceitos: lista dos contornos considerados galinhas
      - contornos_rejeitados: lista dos contornos descartados
    """
    h, w = markers_ws.shape[:2]
    resultado = roi_img.copy()
    aceitos    = []
    rejeitados = []

    print(f"    → Analisando {num_labels - 1} regiões do Watershed...")

    # ── Iterar sobre cada rótulo do watershed ─────────────────
    # Rótulo 1 = fundo, rótulos >= 2 = objetos (galinhas)
    for label_id in range(2, num_labels + 1):
        # Criar máscara binária ISOLADA para este rótulo
        mascara = np.zeros((h, w), dtype=np.uint8)
        mascara[markers_ws == label_id] = 255

        # Encontrar contorno desta região específica
        cnts, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not cnts:
            continue

        # Pegar o maior contorno desta região
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area_min <= area <= area_max:
            aceitos.append(cnt)
        else:
            rejeitados.append(cnt)
            print(f"      Rótulo {label_id}: área = {int(area)} px² → REJEITADO")

    # ── Desenhar contornos aceitos (verde) com numeração ──────
    for i, cnt in enumerate(aceitos, start=1):
        cv2.drawContours(resultado, [cnt], -1, COR_CONTORNO, 2)

        # Número da galinha no centróide do contorno
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx, cy = bx + bw // 2, by + bh // 2

        # Fundo escuro atrás do número para legibilidade
        cv2.circle(resultado, (cx, cy), 14, (0, 0, 0), -1)
        cv2.circle(resultado, (cx, cy), 14, COR_NUMERO, 2)
        cv2.putText(
            resultado, str(i), (cx - 6, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_NUMERO, 2, cv2.LINE_AA
        )

    # Desenhar contornos rejeitados (vermelho, mais fino)
    cv2.drawContours(resultado, rejeitados, -1, COR_REJEITADO, 1)

    print(f"    → Contornos ACEITOS  (galinhas): {len(aceitos)}")
    print(f"    → Contornos rejeitados (ruído) : {len(rejeitados)}")
    print(f"      Filtro de área: {area_min} ≤ área ≤ {area_max} px²")

    return resultado, aceitos, rejeitados


def salvar_pipeline_visual(
    imagem_original: np.ndarray,
    roi_coords: tuple,
    etapas: dict,
    resultado: np.ndarray,
    contagem: int,
    pasta_saida: str,
    nome_base: str,
) -> str:
    """
    Cria uma figura matplotlib mostrando cada etapa do pipeline
    em uma grade, e salva como imagem PNG.
    """
    mostrar_barra("SALVANDO VISUALIZAÇÃO DO PIPELINE")

    # Selecionar as etapas principais para exibir
    nomes_etapas = [
        ("ROI Original",            "roi_original"),
        ("Escala de Cinza",         "1_cinza"),
        ("CLAHE (contraste)",       "2_clahe"),
        ("Suavizado (Gauss)",       "3_suavizado"),
        ("Binarização (Otsu)",      "4a_otsu"),
        ("Binar. Adaptativa",       "4b_adaptativo"),
        ("Abertura Morfol.",        "5a_abertura"),
        ("Fechamento Morfol.",      "5b_fechamento"),
        ("Dist. Transform",         "6a_distancia"),
        ("Picos (centros)",         "6b_picos"),
        ("Watershed (limites)",     "6c_watershed"),
        ("Resultado Final",         "resultado"),
    ]

    num_etapas = len(nomes_etapas)
    cols = 4
    rows = (num_etapas + cols - 1) // cols  # arredonda para cima

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(5 * cols, 4 * rows),
        facecolor="#1e1e2e",
    )
    axes = axes.flatten()

    # Extrair ROI da imagem original para exibir
    x, y, w, h = roi_coords
    roi_original = imagem_original[y:y+h, x:x+w].copy()
    cv2.rectangle(roi_original, (0, 0), (w-1, h-1), COR_ROI, 2)

    # Montar dicionário completo para iterar
    todas_etapas = {"roi_original": roi_original, "resultado": resultado}
    todas_etapas.update(etapas)

    for i, (titulo, chave) in enumerate(nomes_etapas):
        ax = axes[i]
        ax.set_facecolor("#13131f")

        if chave in todas_etapas:
            img = todas_etapas[chave]
            if len(img.shape) == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    color="white", fontsize=14, transform=ax.transAxes)

        ax.set_title(titulo, color="white", fontsize=10, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    # Esconder eixos extras
    for i in range(num_etapas, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"Pipeline de Contagem — {contagem} galinha(s) detectada(s)",
        color="white", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    # ── Salvar ────────────────────────────────────────────────
    caminho_saida = os.path.join(pasta_saida, f"{nome_base}_pipeline_contagem.png")
    fig.savefig(caminho_saida, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  📸 Pipeline visual salvo em:\n     {caminho_saida}\n")
    return caminho_saida


def salvar_resultado_final(
    imagem_original: np.ndarray,
    roi_coords: tuple,
    resultado_roi: np.ndarray,
    contagem: int,
    pasta_saida: str,
    nome_base: str,
) -> str:
    """
    Salva a imagem original com a ROI substituída pelo resultado
    (contornos e numeração) e informações de contagem.
    """
    x, y, w, h = roi_coords
    img_final = imagem_original.copy()

    # Substituir a ROI pelo resultado processado
    img_final[y:y+h, x:x+w] = resultado_roi

    # Desenhar retângulo da ROI na imagem completa
    cv2.rectangle(img_final, (x, y), (x+w, y+h), COR_ROI, 2)

    # Caixa de informação no topo
    altura_img = img_final.shape[0]
    largura_img = img_final.shape[1]

    texto = f"GALINHAS NA ROI: {contagem}"
    cv2.rectangle(img_final, (0, 0), (largura_img, 40), (15, 15, 15), -1)
    cv2.putText(
        img_final, texto, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COR_NUMERO, 2, cv2.LINE_AA
    )

    # Salvar
    caminho_saida = os.path.join(pasta_saida, f"{nome_base}_resultado_contagem.png")
    cv2.imwrite(caminho_saida, img_final)
    print(f"  📸 Resultado final salvo em:\n     {caminho_saida}\n")
    return caminho_saida


# ─────────────────────────────────────────────────────────────
# FLUXO PRINCIPAL
# ─────────────────────────────────────────────────────────────

def main() -> None:
    mostrar_barra("CONTAGEM DE GALINHAS — VISÃO COMPUTACIONAL")
    print("  Este script detecta e conta galinhas dentro de uma")
    print("  região de interesse (ROI) usando processamento de imagens.\n")

    # ── 1. ENTRADA: Carregar imagem ───────────────────────────
    mostrar_barra("ETAPA 1 — CARREGAR IMAGEM")

    if len(sys.argv) > 1:
        caminho = sys.argv[1]
    else:
        caminho = input("  Digite o caminho da imagem: ").strip().strip('"')

    try:
        imagem = carregar_imagem(caminho)
    except (FileNotFoundError, ValueError) as erro:
        print(f"\n  ❌ Erro: {erro}")
        return

    print(f"  ✅ Imagem carregada: {caminho}")
    print(f"     Dimensões: {imagem.shape[1]}×{imagem.shape[0]} px")
    print(f"     Canais   : {imagem.shape[2] if len(imagem.shape) == 3 else 1}\n")

    # Preparar pasta de saída
    pasta_saida = os.path.join(
        os.path.dirname(os.path.abspath(caminho)), "resultados"
    )
    os.makedirs(pasta_saida, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(caminho))[0]

    # ── 2. SELECIONAR ROI ─────────────────────────────────────
    mostrar_barra("ETAPA 2 — SELECIONAR REGIÃO DE INTERESSE (ROI)")
    print("  Tentando detectar retângulo vermelho automaticamente...")

    roi_coords = detectar_retangulo_vermelho(imagem)

    if roi_coords:
        x, y, w, h = roi_coords
        print(f"  ✅ Retângulo vermelho detectado automaticamente!")
        print(f"     Posição: ({x}, {y})")
        print(f"     Tamanho: {w}×{h} px\n")
    else:
        print("  ⚠️  Retângulo vermelho não encontrado.")
        print("     Abrindo seleção manual...\n")
        roi_coords = selecionar_roi_manual(imagem)

        if roi_coords is None or roi_coords == (0, 0, 0, 0):
            print("  ❌ Nenhuma ROI selecionada. Encerrando.\n")
            return

        x, y, w, h = roi_coords
        print(f"  ✅ ROI selecionada manualmente: ({x}, {y}) — {w}×{h} px\n")

    # Extrair região de interesse
    roi_img = imagem[y:y+h, x:x+w].copy()

    # ── 3. PRÉ-PROCESSAMENTO ─────────────────────────────────
    mostrar_barra("ETAPA 3 — PRÉ-PROCESSAMENTO")
    etapas_pre = preprocessar(roi_img, ksize_gauss=5)

    # ── 4. BINARIZAÇÃO ────────────────────────────────────────
    mostrar_barra("ETAPA 4 — BINARIZAÇÃO")
    etapas_bin = binarizar(etapas_pre["3_suavizado"])

    # ── 5. OPERAÇÕES MORFOLÓGICAS ─────────────────────────────
    mostrar_barra("ETAPA 5 — OPERAÇÕES MORFOLÓGICAS")
    etapas_morf = operacoes_morfologicas(etapas_bin["4_binaria"])

    # ── 6. SEPARAÇÃO POR WATERSHED ────────────────────────────
    mostrar_barra("ETAPA 6 — SEPARAÇÃO POR WATERSHED")
    etapas_ws = separar_por_watershed(etapas_morf["5_morfologia"], roi_img)

    # ── 7. DETECÇÃO E CONTAGEM ────────────────────────────────
    mostrar_barra("ETAPA 7 — DETECÇÃO E CONTAGEM DE CONTORNOS")

    # Calcular limites de área proporcionais ao tamanho da ROI
    area_roi = w * h
    area_min = max(int(area_roi * 0.015), 80)     # mín 1.5% da ROI
    area_max = max(int(area_roi * 0.45), 5000)   # máx 45% da ROI
    print(f"    → Limites de área ajustados à ROI ({w}×{h} = {area_roi} px²):")
    print(f"      Mínimo: {area_min} px²  |  Máximo: {area_max} px²\n")

    # Usar os rótulos do Watershed diretamente para contar
    # cada região separada individualmente
    resultado, aceitos, rejeitados = detectar_e_contar_watershed(
        etapas_ws["6_markers_ws"], etapas_ws["6_num_labels"],
        roi_img, area_min, area_max
    )

    contagem = len(aceitos)

    # ── 7. RESULTADOS ─────────────────────────────────────────
    mostrar_barra("RESULTADO FINAL")
    print(f"\n  🐔 GALINHAS ENCONTRADAS NA ROI: {contagem}\n")

    # Detalhes de cada galinha detectada
    if aceitos:
        print("  Detalhes por galinha:")
        print("  ┌─────┬──────────┬────────────────┐")
        print("  │  #  │  Área px²│  Centro (x, y) │")
        print("  ├─────┼──────────┼────────────────┤")
        for i, cnt in enumerate(aceitos, start=1):
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cx, cy = bx + bw // 2, by + bh // 2
            print(f"  │ {i:3d} │ {int(area):8d} │  ({cx:4d}, {cy:4d})  │")
        print("  └─────┴──────────┴────────────────┘\n")

    # ── 9. SALVAR RESULTADOS ──────────────────────────────────
    # Juntar todas as etapas intermediárias
    todas_etapas = {}
    todas_etapas.update(etapas_pre)
    todas_etapas.update(etapas_bin)
    todas_etapas.update(etapas_morf)
    todas_etapas.update(etapas_ws)

    # Pipeline visual (grade com todas as etapas)
    salvar_pipeline_visual(
        imagem, roi_coords, todas_etapas,
        resultado, contagem, pasta_saida, nome_base
    )

    # Resultado final (imagem original com contagem)
    salvar_resultado_final(
        imagem, roi_coords, resultado,
        contagem, pasta_saida, nome_base
    )

    # ── EXIBIR RESULTADO ──────────────────────────────────────
    print("  Deseja visualizar o resultado? (s/n): ", end="")
    resposta = input().strip().lower()

    if resposta in ("s", "sim", "y", "yes"):
        cv2.imshow("Resultado — Contagem de Galinhas", resultado)
        print("\n  Pressione qualquer tecla para fechar a janela...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    mostrar_barra("CONCLUÍDO")
    print("  ✅ Processamento finalizado!")
    print(f"  📁 Resultados salvos em: {pasta_saida}\n")


if __name__ == "__main__":
    main()
