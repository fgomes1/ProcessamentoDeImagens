# 🐔 Contagem de Galinhas em Região de Interesse (ROI)

## Objetivo

Dado uma imagem de câmera de segurança de um aviário (CAM04), **contar automaticamente quantas galinhas existem dentro de um retângulo vermelho** demarcado na imagem.

**Resultado final: 4 galinhas detectadas.**

![Resultado da contagem](resultados/galinha_resultado_contagem.png)

---

## Pipeline Completo — Passo a Passo

O script `contar_galinhas.py` implementa um pipeline de **9 etapas** de processamento de imagens. Cada etapa é explicada abaixo com sua fundamentação teórica.

![Pipeline completo com todas as etapas](resultados/galinha_pipeline_contagem.png)

---

### Etapa 1 — Carregar Imagem

```python
imagem = cv2.imread("galinha.png")
```

A imagem é carregada no formato **BGR** (Blue, Green, Red), que é o padrão do OpenCV. A imagem da CAM04 tem resolução de **352×240 pixels** e 3 canais de cor.

---

### Etapa 2 — Selecionar a Região de Interesse (ROI)

O script detecta **automaticamente** o retângulo vermelho na imagem usando filtragem por cor no espaço **HSV**:

```python
# Converte para HSV
hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

# O vermelho no HSV ocupa DUAS faixas de Hue:
#   H ∈ [0, 10]   → vermelho próximo de 0°
#   H ∈ [170, 180] → vermelho próximo de 360°
mascara1 = cv2.inRange(hsv, (0,   100, 100), (10,  255, 255))
mascara2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
mascara_vermelho = cv2.bitwise_or(mascara1, mascara2)
```

**Por que HSV?** No espaço HSV, a cor (Hue) é separada do brilho (Value) e saturação. Isso torna a detecção de cores muito mais robusta do que no BGR, onde vermelho depende de combinações complexas dos 3 canais.

**ROI detectada:** posição (174, 88), tamanho 84×87 pixels.

Se o retângulo vermelho não for encontrado, o script abre uma janela para seleção manual com `cv2.selectROI`.

---

### Etapa 3 — Pré-processamento

Três operações preparam a ROI para a binarização:

#### 3.1 — Conversão para Escala de Cinza

```python
cinza = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
```

Reduz de 3 canais (BGR) para 1 canal de intensidade. Simplifica todo o processamento subsequente.

#### 3.2 — CLAHE (Equalização Adaptativa de Histograma)

```python
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
equalizado = clahe.apply(cinza)
```

**O que faz:** Divide a imagem em blocos de 8×8 e equaliza o histograma de **cada bloco individualmente**. Diferente da equalização global (`equalizeHist`), o CLAHE melhora o contraste local sem estourar regiões já claras.

**Por que é importante:** A câmera de segurança tem iluminação irregular — o centro da imagem é mais claro que as bordas. O CLAHE corrige isso, revelando detalhes nas regiões escuras.

**Parâmetro `clipLimit=3.0`:** Limita a amplificação de contraste para evitar amplificar ruído demais.

#### 3.3 — Suavização Gaussiana

```python
suavizado = cv2.GaussianBlur(equalizado, (5, 5), sigmaX=0)
```

**O que faz:** Aplica uma média ponderada por uma função Gaussiana 2D nos pixels vizinhos. Cada pixel passa a ser a média dos seus vizinhos (com peso maior para os mais próximos).

**Por que é importante:** Remove ruído de alta frequência (granulação da câmera), que causaria falsos contornos na binarização. Kernel 5×5 é suficiente sem borrar demais os detalhes.

---

### Etapa 4 — Binarização

Converte a imagem em preto e branco (0 ou 255), separando as galinhas (brancas/claras) do fundo (escuro).

#### 4a — Método de Otsu (utilizado)

```python
limiar_otsu, binaria = cv2.threshold(
    suavizado, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
```

**O que faz:** Analisa o **histograma de intensidade** do frame inteiro e calcula automaticamente o limiar que **maximiza a variância entre as duas classes** (fundo e objeto). Funciona bem quando o histograma é bimodal (dois picos distintos).

**Limiar encontrado:** 152 → pixels ≥ 152 viram branco (galinha), pixels < 152 viram preto (fundo).

#### 4b — Método Adaptativo (alternativa)

```python
binaria_adapt = cv2.adaptiveThreshold(
    suavizado, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=31, C=8
)
```

**Diferença:** Em vez de um limiar global, calcula um limiar **diferente para cada região** da imagem (blocos de 31×31). Mais robusto com iluminação variável, mas pode gerar mais ruído.

> **Decisão de projeto:** Usamos Otsu neste caso porque as galinhas são significativamente mais claras que o fundo, criando um histograma bimodal bem definido.

---

### Etapa 5 — Operações Morfológicas

Limpam a imagem binária usando **elementos estruturantes** (pequenas máscaras de forma elíptica):

#### 5a — Abertura (Erosão → Dilatação)

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
aberta = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=2)
```

**O que faz:**
1. **Erosão:** Encolhe as regiões brancas → pixels brancos isolados (ruído) desaparecem
2. **Dilatação:** Expande de volta → objetos grandes retornam ao tamanho original

**Resultado:** Remove pequenos pontos brancos de ruído sem alterar significativamente as galinhas.

#### 5b — Fechamento LEVE (Dilatação → Erosão)

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fechada = cv2.morphologyEx(aberta, cv2.MORPH_CLOSE, kernel, iterations=1)
```

**O que faz:** Operação inversa da abertura — preenche pequenos **buracos pretos** dentro das regiões brancas (galinhas).

> ⚠️ **Cuidado com o fechamento!** Um kernel muito grande ou muitas iterações pode **fundir galinhas vizinhas** em um único blob. Na primeira versão do script, usamos kernel 9×9 com 3 iterações, que juntou todas as galinhas em 1 blob. Reduzimos para **3×3 com 1 iteração** para preservar a separação entre elas.

---

### Etapa 6 — Separação por Watershed

Esta é a etapa mais sofisticada. O problema é que galinhas que **se tocam ou se sobrepõem** aparecem como uma única região branca após a binarização. O Watershed resolve isso.

#### 6a — Transformada de Distância

```python
dist = cv2.distanceTransform(binaria, cv2.DIST_L2, 5)
```

**O que faz:** Para cada pixel branco, calcula a **distância euclidiana** até o pixel preto (borda) mais próximo.

**Intuição:** O **centro** de cada galinha terá valor alto (está longe de todas as bordas), enquanto as bordas e pontos de contato entre galinhas terão valores baixos.

```
  Galinha A    Galinha B
  ╭──────╮    ╭──────╮
  │  14  │    │  10  │    ← distância máxima no centro
  │      ├────┤      │    ← distância ≈ 0 no ponto de contato
  ╰──────╯    ╰──────╯
```

**Distância máxima encontrada:** 14.6 pixels (centro da galinha maior).

#### 6b — Identificação dos Picos (Centros)

```python
limiar_dist = 0.25 * dist.max()  # 25% do máximo
_, sure_fg = cv2.threshold(dist, limiar_dist, 255, cv2.THRESH_BINARY)
```

**O que faz:** Mantém apenas os pixels com distância ≥ 25% do máximo. Esses são os **centros prováveis** de cada galinha — regiões que estão bem longe de qualquer borda.

**Resultado:** 11 marcadores (centros) encontrados — alguns são ruído, que será filtrado depois pela área.

#### 6c — Rotulação dos Marcadores

```python
num_labels, markers = cv2.connectedComponents(sure_fg)
```

**O que faz:** Atribui um **número inteiro único** (rótulo) a cada componente conectado nos picos. Cada rótulo será a "semente" de uma galinha no Watershed.

#### 6d — Algoritmo Watershed

```python
markers_ws = cv2.watershed(roi_img, markers)
```

**O que faz:** Inspirado em hidrografia — imagine que cada marcador (centro de galinha) é uma nascente de rio. A "água" sobe gradualmente a partir de cada nascente. Onde as águas de **duas nascentes diferentes se encontram**, o algoritmo traça uma **linha divisória** (amarela na visualização).

```
  Marcador 1           Marcador 2
      ↓                    ↓
  ~~~~~ ↑ Água sobe ↑ ~~~~~
        │  DIVISÓRIA  │
  ██████│█████████████│██████
```

**Resultado:**
- `markers_ws == -1` → linhas divisórias (bordas entre galinhas)
- `markers_ws == 1` → fundo
- `markers_ws >= 2` → cada galinha com seu rótulo único

---

### Etapa 7 — Detecção e Contagem

Agora usamos os **rótulos do Watershed diretamente** para contar cada galinha:

```python
for label_id in range(2, num_labels + 1):
    # Criar máscara ISOLADA para este rótulo
    mascara = np.zeros((h, w), dtype=np.uint8)
    mascara[markers_ws == label_id] = 255
    
    # Encontrar contorno desta região específica
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, ...)
    
    # Filtrar por área
    area = cv2.contourArea(contorno)
    if area_min <= area <= area_max:
        aceitos.append(contorno)  # ← É uma galinha!
```

**Por que iterar sobre cada rótulo individualmente?**

> Na primeira tentativa, convertemos todos os rótulos de volta para uma **única imagem binária** (`markers > 1 → branco`). Isso **reconectava** as regiões vizinhas, perdendo toda a separação que o Watershed havia feito. O `findContours` achava apenas 1-2 contornos gigantes.
>
> A solução foi criar uma **máscara isolada por rótulo** e encontrar o contorno de cada uma separadamente.

**Filtro de área:**
- Mínimo: 109 px² (1.5% da ROI) → rejeita ruído
- Máximo: 5000 px² (45% da ROI) → rejeita fundo

---

### Etapa 8 — Resultado Final

As 4 galinhas detectadas com seus contornos e numeração:

| # | Área (px²) | Centro (x, y) |
|---|-----------|---------------|
| 1 | 373       | (63, 23)      |
| 2 | 191       | (12, 27)      |
| 3 | 1405      | (28, 54)      |
| 4 | 291       | (60, 55)      |

A galinha 3 é a maior (1405 px²), ocupando a região central-inferior da ROI.

---

### Etapa 9 — Salvar Resultados

Os resultados são salvos na pasta `resultados/`:
- **`galinha_pipeline_contagem.png`** — Grade com todas as etapas intermediárias
- **`galinha_resultado_contagem.png`** — Imagem original com os contornos e contagem

---

## Resumo das Funções OpenCV Utilizadas

| Função | Etapa | Propósito |
|--------|-------|-----------|
| `cv2.cvtColor` | 2, 3 | Conversão de espaço de cor (BGR→HSV, BGR→GRAY) |
| `cv2.inRange` | 2 | Filtragem de cor por faixa HSV |
| `cv2.createCLAHE` | 3 | Equalização adaptativa de histograma |
| `cv2.GaussianBlur` | 3 | Suavização por filtro Gaussiano |
| `cv2.threshold` + `THRESH_OTSU` | 4 | Binarização automática por Otsu |
| `cv2.adaptiveThreshold` | 4 | Binarização por limiar local |
| `cv2.morphologyEx` | 5 | Abertura e fechamento morfológicos |
| `cv2.distanceTransform` | 6 | Transformada de distância euclidiana |
| `cv2.connectedComponents` | 6 | Rotulação de componentes conectados |
| `cv2.watershed` | 6 | Segmentação por Watershed |
| `cv2.findContours` | 7 | Detecção de contornos |
| `cv2.contourArea` | 7 | Cálculo de área do contorno |
| `cv2.moments` | 7 | Cálculo do centróide |

---

## Como Executar

```bash
# Ativar o ambiente virtual
.\venv\Scripts\activate

# Executar com imagem específica
python contar_galinhas.py galinha.png

# Ou sem argumento (pedirá o caminho)
python contar_galinhas.py
```

## Dificuldades Encontradas e Soluções

### 1. Fechamento morfológico muito agressivo
- **Problema:** Kernel 9×9 com 3 iterações fundia todas as galinhas em 1 blob
- **Solução:** Reduzir para kernel 3×3 com 1 iteração

### 2. Conversão do Watershed de volta para binário
- **Problema:** `separada[markers > 1] = 255` reconectava regiões vizinhas
- **Solução:** Iterar sobre cada rótulo individualmente criando máscaras isoladas

### 3. Encoding Unicode no Windows
- **Problema:** Caracteres especiais (═, →, ✅) causavam `UnicodeEncodeError`
- **Solução:** Usar `$env:PYTHONIOENCODING='utf-8'` antes de executar
