"""
=============================================================
  PROCESSAMENTO DE IMAGENS - CONVERSÃO DE ESPAÇOS DE COR
  Professor de Visão Computacional
  Utiliza: OpenCV (cv2), matplotlib, numpy
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
# CONFIGURAÇÕES DOS ESPAÇOS DE COR
# ─────────────────────────────────────────────────────────────
ESPACOS_DE_COR = {
    "1": {
        "nome":    "HSV  (Matiz, Saturação, Valor)",
        "codigo":  cv2.COLOR_BGR2HSV,
        "canais":  ["H - Matiz (Hue)", "S - Saturação", "V - Valor (Brilho)"],
        "cmap":    ["hsv", "Greens", "gray"],
    },
    "2": {
        "nome":    "YCrCb (Luminância + Crominância)",
        "codigo":  cv2.COLOR_BGR2YCrCb,
        "canais":  ["Y  - Luminância", "Cr - Crominância Vermelha", "Cb - Crominância Azul"],
        "cmap":    ["gray", "RdBu", "RdBu_r"],
    },
    "3": {
        "nome":    "GRAY (Escala de Cinza)",
        "codigo":  cv2.COLOR_BGR2GRAY,
        "canais":  ["GRAY - Intensidade"],
        "cmap":    ["gray"],
    },
    "4": {
        "nome":    "Lab  (Luminosidade + a* + b*)",
        "codigo":  cv2.COLOR_BGR2Lab,
        "canais":  ["L  - Luminosidade", "a* - Verde-Vermelho", "b* - Azul-Amarelo"],
        "cmap":    ["gray", "RdYlGn_r", "RdYlBu"],
    },
    "5": {
        "nome":    "HLS  (Matiz, Luminosidade, Saturação)",
        "codigo":  cv2.COLOR_BGR2HLS,
        "canais":  ["H - Matiz (Hue)", "L - Luminosidade", "S - Saturação"],
        "cmap":    ["hsv", "gray", "Greens"],
    },
    "6": {
        "nome":    "XYZ  (CIE XYZ)",
        "codigo":  cv2.COLOR_BGR2XYZ,
        "canais":  ["X", "Y", "Z"],
        "cmap":    ["Purples", "Greens", "Blues"],
    },
    "7": {
        "nome":    "Luv  (CIE L*u*v*)",
        "codigo":  cv2.COLOR_BGR2Luv,
        "canais":  ["L* - Luminosidade", "u*", "v*"],
        "cmap":    ["gray", "coolwarm", "coolwarm"],
    },
}


# ─────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────────────────────

def mostrar_barra(titulo: str) -> None:
    """Imprime um cabeçalho formatado no terminal."""
    largura = 60
    print("\n" + "═" * largura)
    print(f"  {titulo}")
    print("═" * largura)


def criar_pasta_saida(caminho_imagem: str) -> str:
    """
    Cria uma pasta 'resultados' no mesmo diretório da imagem
    e retorna o caminho base para os arquivos de saída.
    """
    pasta = os.path.join(os.path.dirname(os.path.abspath(caminho_imagem)), "resultados")
    os.makedirs(pasta, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(caminho_imagem))[0]
    return pasta, nome_base


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


def exibir_menu() -> str:
    """Exibe o menu de espaços de cor e retorna a opção digitada."""
    mostrar_barra("MENU - ESCOLHA O ESPAÇO DE COR")
    for chave, info in ESPACOS_DE_COR.items():
        print(f"  [{chave}] {info['nome']}")
    print("  [0] Sair")
    print()
    return input("  Digite o número da opção desejada: ").strip()


def converter_imagem(imagem_bgr: np.ndarray, config: dict) -> np.ndarray:
    """Converte a imagem BGR para o espaço de cor escolhido."""
    return cv2.cvtColor(imagem_bgr, config["codigo"])


def separar_e_salvar_canais(
    imagem_convertida: np.ndarray,
    config: dict,
    pasta: str,
    nome_base: str,
    nome_espaco: str,
) -> list:
    """
    Separa os canais da imagem convertida com cv2.split,
    salva cada canal como imagem PNG e retorna a lista de canais.
    """
    mostrar_barra("SEPARAÇÃO DE CANAIS (cv2.split)")

    canais = cv2.split(imagem_convertida)
    nomes_canais = config["canais"]

    print(f"  → Número de canais separados: {len(canais)}\n")

    caminhos_salvos = []
    for i, (canal, nome_canal) in enumerate(zip(canais, nomes_canais)):
        nome_arquivo = f"{nome_base}_{nome_espaco}_canal_{i+1}.png"
        caminho_saida = os.path.join(pasta, nome_arquivo)

        cv2.imwrite(caminho_saida, canal)
        caminhos_salvos.append(caminho_saida)

        # Info no terminal
        print(f"  Canal {i+1}: {nome_canal}")
        print(f"    Dimensões : {canal.shape}")
        print(f"    Min/Max   : {canal.min()} / {canal.max()}")
        print(f"    Salvo em  : {caminho_saida}\n")

    return canais, caminhos_salvos


def calcular_e_salvar_histogramas(
    canais: list,
    config: dict,
    pasta: str,
    nome_base: str,
    nome_espaco: str,
) -> None:
    """
    Calcula o histograma de cada canal com cv2.calcHist,
    plota com matplotlib e salva os gráficos em disco.
    """
    mostrar_barra("HISTOGRAMAS (cv2.calcHist + matplotlib)")

    nomes_canais = config["canais"]
    cmaps        = config["cmap"]

    # ── figura com um subplot por canal ──────────────────────
    num_canais = len(canais)
    fig, eixos = plt.subplots(
        1, num_canais,
        figsize=(6 * num_canais, 4),
        facecolor="#1e1e2e",
    )

    # garante sempre iterável mesmo com 1 canal
    if num_canais == 1:
        eixos = [eixos]

    for i, (canal, nome_canal, cmap) in enumerate(zip(canais, nomes_canais, cmaps)):
        # ── cv2.calcHist ─────────────────────────────────────
        hist = cv2.calcHist(
            images   = [canal],
            channels = [0],
            mask     = None,
            histSize = [256],
            ranges   = [0, 256],
        )
        hist = hist.flatten()   # array 1-D com 256 valores

        # ── plot ─────────────────────────────────────────────
        ax = eixos[i]
        ax.set_facecolor("#13131f")
        x = np.arange(256)

        # gradiente de cor usando o colormap do espaço
        cores = plt.get_cmap(cmap)(np.linspace(0.2, 1.0, 256))
        ax.bar(x, hist, color=cores, width=1.0, alpha=0.9)

        ax.set_title(nome_canal, color="white", fontsize=10, pad=8)
        ax.set_xlabel("Intensidade do Pixel (0–255)", color="#aaaacc", fontsize=8)
        ax.set_ylabel("Frequência", color="#aaaacc", fontsize=8)
        ax.tick_params(colors="#888899")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        print(f"  Canal {i+1} ({nome_canal}) — histograma calculado")
        print(f"    Pixel mais frequente: intensidade {int(np.argmax(hist))} "
              f"({int(hist.max())} ocorrências)\n")

    titulo_fig = f"Histogramas — {nome_espaco} | {nome_base}"
    fig.suptitle(titulo_fig, color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    # ── salva a figura dos histogramas ───────────────────────
    nome_hist = f"{nome_base}_{nome_espaco}_histogramas.png"
    caminho_hist = os.path.join(pasta, nome_hist)
    fig.savefig(caminho_hist, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  Gráfico de histogramas salvo em:\n  {caminho_hist}\n")


def salvar_imagem_convertida(
    imagem_convertida: np.ndarray,
    pasta: str,
    nome_base: str,
    nome_espaco: str,
    config: dict,
) -> None:
    """
    Salva a imagem convertida no disco.
    Para GRAY (1 canal) salva diretamente.
    Para espaços multi-canal salva em BGRs equivalente e também
    a versão original convertida.
    """
    mostrar_barra("SALVANDO IMAGEM CONVERTIDA (cv2.imwrite)")

    nome_arquivo = f"{nome_base}_{nome_espaco}_convertida.png"
    caminho_saida = os.path.join(pasta, nome_arquivo)

    cv2.imwrite(caminho_saida, imagem_convertida)

    print(f"  Espaço de cor  : {config['nome']}")
    print(f"  Shape da imagem: {imagem_convertida.shape}")
    print(f"  Arquivo salvo  : {caminho_saida}\n")


def resumo_final(pasta: str) -> None:
    """Lista todos os arquivos gerados na pasta de resultados."""
    mostrar_barra("RESUMO — ARQUIVOS GERADOS")
    arquivos = sorted(os.listdir(pasta))
    for arq in arquivos:
        tamanho = os.path.getsize(os.path.join(pasta, arq))
        print(f"  📄 {arq}  ({tamanho/1024:.1f} KB)")
    print(f"\n  📁 Diretório de saída: {pasta}\n")


# ─────────────────────────────────────────────────────────────
# FLUXO PRINCIPAL
# ─────────────────────────────────────────────────────────────

def main() -> None:
    mostrar_barra("PROCESSAMENTO DE IMAGENS — VISÃO COMPUTACIONAL")
    print("  Este script demonstra conversão de espaços de cor,")
    print("  separação de canais e cálculo de histogramas com OpenCV.\n")

    # ── 1. ENTRADA: caminho da imagem ─────────────────────────
    while True:
        caminho = input("  Digite o caminho completo da imagem: ").strip().strip('"')
        try:
            imagem_bgr = carregar_imagem(caminho)
            print(f"\n  ✅ Imagem carregada com sucesso!")
            print(f"     Dimensões : {imagem_bgr.shape}  (altura × largura × canais)")
            print(f"     Tipo      : {imagem_bgr.dtype}\n")
            break
        except (FileNotFoundError, ValueError) as erro:
            print(f"\n  ❌ Erro: {erro}")
            print("  Tente novamente.\n")

    # ── 2. MENU: escolha do espaço de cor ─────────────────────
    while True:
        opcao = exibir_menu()
        if opcao == "0":
            print("\n  Encerrando o programa. Até logo!\n")
            sys.exit(0)
        if opcao in ESPACOS_DE_COR:
            config = ESPACOS_DE_COR[opcao]
            print(f"\n  ✅ Espaço selecionado: {config['nome']}\n")
            break
        print("  ⚠️  Opção inválida. Escolha um número do menu.\n")

    # ── 3. CONVERSÃO: cv2.cvtColor ────────────────────────────
    mostrar_barra("CONVERSÃO (cv2.cvtColor)")
    imagem_convertida = converter_imagem(imagem_bgr, config)

    # Nome curto do espaço (ex.: "HSV", "GRAY")
    nome_espaco = config["nome"].split()[0]
    print(f"  Convertendo BGR → {nome_espaco} ...")
    print(f"  Shape resultante: {imagem_convertida.shape}\n")

    # ── 4. CRIAR PASTA DE SAÍDA ───────────────────────────────
    pasta, nome_base = criar_pasta_saida(caminho)

    # ── 5. SALVAR IMAGEM CONVERTIDA ───────────────────────────
    salvar_imagem_convertida(imagem_convertida, pasta, nome_base, nome_espaco, config)

    # ── 6. SEPARAR CANAIS: cv2.split ──────────────────────────
    canais, _ = separar_e_salvar_canais(
        imagem_convertida, config, pasta, nome_base, nome_espaco
    )

    # ── 7. HISTOGRAMAS: cv2.calcHist + matplotlib ─────────────
    calcular_e_salvar_histogramas(
        canais, config, pasta, nome_base, nome_espaco
    )

    # ── 8. RESUMO FINAL ───────────────────────────────────────
    resumo_final(pasta)
    print("  Processamento concluído! Verifique a pasta 'resultados'.\n")


if __name__ == "__main__":
    main()
