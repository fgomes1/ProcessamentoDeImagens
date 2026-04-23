import cv2
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg') # Usar backend não-interativo
import matplotlib.pyplot as plt

def mostrar_barra(titulo: str) -> None:
    largura = 70
    print("\n" + "=" * largura)
    print(f"  {titulo}")
    print("=" * largura)

def gerar_imagem_binaria_e_passos(imagem: np.ndarray) -> tuple:
    """
    Gera uma imagem binária contendo todas as bolas e retorna 
    as etapas intermediárias para o passo a passo.
    """
    resultado = imagem.copy()
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # 1. Obter Máscaras Individuais por Cor (para lidar com iluminações diferentes)
    
    # Vermelho (duas faixas de matiz)
    mask_vermelho1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_vermelho2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    mask_vermelho = cv2.bitwise_or(mask_vermelho1, mask_vermelho2)

    # Azul
    mask_azul = cv2.inRange(hsv, np.array([100, 40, 50]), np.array([140, 255, 255]))

    # Branca
    mask_branca = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 90, 255]))

    # 2. Imagem Binária Final (Combinação de todas as bolas)
    # Requisito do exercício: "Crie uma imagem binária que apresente todas as bolas"
    binaria_todas = cv2.bitwise_or(mask_vermelho, mask_azul)
    binaria_todas = cv2.bitwise_or(binaria_todas, mask_branca)

    # Limpeza Morfológica (Remover ruídos e fechar buracos)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binaria_todas = cv2.morphologyEx(binaria_todas, cv2.MORPH_OPEN, kernel, iterations=2)
    binaria_todas = cv2.morphologyEx(binaria_todas, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Detectar Contornos na Binária para validação visual
    contornos, _ = cv2.findContours(binaria_todas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_bolinhas = 0
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area > 800:
            total_bolinhas += 1
            (x, y), raio = cv2.minEnclosingCircle(cnt)
            centro = (int(x), int(y))
            cv2.circle(resultado, centro, int(raio), (0, 255, 0), 3)

    etapas = {
        "mask_vermelho": mask_vermelho,
        "mask_azul": mask_azul,
        "mask_branca": mask_branca
    }

    return resultado, binaria_todas, total_bolinhas, etapas

def salvar_passo_a_passo(imagem_original, etapas, binaria_final, resultado_final, pasta_saida, nome_base):
    """
    Gera um plot com 6 imagens mostrando o pipeline visualmente.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    imagens = [
        ("1. Imagem Original", cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB), False),
        ("2. Máscara Vermelha (HSV)", etapas["mask_vermelho"], True),
        ("3. Máscara Azul (HSV)", etapas["mask_azul"], True),
        ("4. Máscara Branca (HSV)", etapas["mask_branca"], True),
        ("5. Imagem Binária (Requisito)", binaria_final, True),
        ("6. Resultado com Contornos", cv2.cvtColor(resultado_final, cv2.COLOR_BGR2RGB), False)
    ]

    for i, (titulo, img, is_gray) in enumerate(imagens):
        if is_gray:
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(img)
        axes[i].set_title(titulo, fontsize=14, pad=10)
        axes[i].axis('off')

    plt.tight_layout()
    caminho_figura = os.path.join(pasta_saida, f"{nome_base}_passo_a_passo.png")
    fig.savefig(caminho_figura, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return caminho_figura

def main() -> None:
    mostrar_barra("RESOLUÇÃO: IMAGEM BINÁRIA DE TODAS AS BOLAS E PASSO-A-PASSO")
    
    caminho = "bocha.JPG" if len(sys.argv) == 1 else sys.argv[1]

    if not os.path.exists(caminho):
        print(f"\n  [Erro]: Arquivo '{caminho}' não encontrado.")
        return

    imagem = cv2.imread(caminho)
    if imagem is None:
        print(f"\n  [Erro]: Não foi possível ler a imagem '{caminho}'.")
        return

    pasta_saida = "resultados"
    os.makedirs(pasta_saida, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(caminho))[0]

    # Processar
    resultado, binaria_final, total, etapas = gerar_imagem_binaria_e_passos(imagem)

    print(f"\n  [OK] Processamento concluído. Bolinhas encontradas: {total}")

    # Salvar Apenas a Imagem Binária (Atendendo ao requisito exato)
    caminho_binaria = os.path.join(pasta_saida, f"{nome_base}_imagem_binaria.png")
    cv2.imwrite(caminho_binaria, binaria_final)
    print(f"  [Arquivo] Imagem binária pura salva em:\n     -> {caminho_binaria}")

    # Salvar o Passo a Passo visual (Grid do Matplotlib)
    caminho_passos = salvar_passo_a_passo(imagem, etapas, binaria_final, resultado, pasta_saida, nome_base)
    print(f"  [Arquivo] Grade do passo-a-passo salva em:\n     -> {caminho_passos}")

    print("\n  Para fechar visualizações abertas no seu terminal, dê um ENTER na janela do script anterior, se houver.")

if __name__ == "__main__":
    main()
