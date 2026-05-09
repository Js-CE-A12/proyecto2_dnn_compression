# Guía Técnica del Proyecto — Compresión de DNN para Detección de Apnea del Sueño

Universidad Interamericana PR – Bayamón | Curso de Ciencia de Datos

---

## Tabla de Contenidos

1. [Visión General del Proyecto](#1-visión-general-del-proyecto)
2. [Background Teórico](#2-background-teórico)
   - 2.1 Señales EEG y apnea del sueño
   - 2.2 Redes Neuronales Convolucionales 1D (CNN 1D)
   - 2.3 Poda estructurada (Structured Pruning)
   - 2.4 Destilación del Conocimiento (Knowledge Distillation)
   - 2.5 Cuantización INT8 (Post-Training Quantization)
   - 2.6 Evaluación Cross-Dataset
3. [Métricas de Evaluación](#3-métricas-de-evaluación)
   - 3.1 Accuracy, Sensibilidad, Especificidad
   - 3.2 F1-Score
   - 3.3 AUC-ROC
   - 3.4 Cohen's Kappa y MCC
   - 3.5 Latencia, RAM y tamaño en disco
   - 3.6 Tabla de Ablación E0–E5
   - 3.7 Curva de Pareto
4. [Módulos del Código Fuente (`src/`)](#4-módulos-del-código-fuente-src)
5. [Scripts del Pipeline](#5-scripts-del-pipeline)
   - 5.1 `01_prepare_data.py`
   - 5.2 `03_isruc_preprocess.py`
   - 5.3 `02b_crossval.py`
   - 5.4 `03_pruning_kd.py`
   - 5.5 `04_quantize.py`
   - 5.6 `05_evaluate.py`
   - 5.7 `06_ablation.py`
   - 5.8 `07_pareto_plot.py`
   - 5.9 `generate_report_docx.py`

---

## 1. Visión General del Proyecto

El proyecto implementa un pipeline de **compresión de redes neuronales profundas (DNN)** aplicado a la detección automática de apnea del sueño usando señales EEG monocanal. El objetivo es reducir el tamaño y la latencia del modelo sin perder precisión clínicamente relevante, habilitando su despliegue en dispositivos portátiles de bajo recurso.

### Datasets utilizados

| Dataset | Rol | Sujetos | Frecuencia |
|---|---|---|---|
| UCDDB (St. Vincent's Hospital) | Entrenamiento + validación | 25 sujetos | 100 Hz (re-muestreado) |
| ISRUC-Sleep (Portugal) | Test cross-dataset exclusivo | 22 sujetos | 200 Hz → 100 Hz |

### Pipeline completo

```
UCDDB Raw EDF
    ↓ [01_prepare_data.py]
Épocas NPY (train/val/test)
    ↓ [02b_crossval.py]
Modelo baseline entrenado (sleep_baseline.pth)
    ↓ [03_pruning_kd.py]
Modelos podados + KD (p050_kd, p050_nokd, p070_kd, p070_nokd)
    ↓ [04_quantize.py]
Modelos ONNX INT8
    ↓ [05_evaluate.py]
metrics_eval.json
    ↓ [06_ablation.py + 07_pareto_plot.py]
Ablación E0–E5 + Curva de Pareto
    ↓ [generate_report_docx.py]
informe_final.docx
```

### Resultado principal

Los modelos comprimidos **mejoran** el AUC-ROC en la evaluación cross-dataset (ISRUC) respecto al baseline. E5 (Pruning 50% + KD + INT8) alcanza AUC = **0.6308** vs **0.6130** del baseline, con **3.9× aceleración** y **8.5× reducción de tamaño**.

---

## 2. Background Teórico

### 2.1 Señales EEG y Apnea del Sueño

El **electroencefalograma (EEG)** mide la actividad eléctrica del cerebro mediante electrodos colocados en el cuero cabelludo. En estudios de sueño, el canal estándar es **C3-A2**: electrodo activo en la región central izquierda (C3) referenciado al mastoides izquierdo (A2), según el sistema internacional 10-20.

**¿Por qué C3-A2?**
- Captura actividad cortical central relevante para las fases del sueño
- Está en la línea media-lateral, sensible a cambios globales de arousal
- Es el canal más documentado en la literatura de apnea automática

La **apnea del sueño** es una pausa en la respiración durante el sueño de ≥ 10 segundos. Tipos:
- **OA (Obstructive Apnea)**: obstrucción física de las vías aéreas
- **CA (Central Apnea)**: falta de esfuerzo respiratorio desde el cerebro
- **MA (Mixed Apnea)**: combinación de central y obstructiva
- **OH / MH (Hipopnea)**: reducción parcial (≥ 30%) del flujo aéreo

En el proyecto, cualquiera de estos eventos en una época de 30 segundos produce etiqueta `y = 1` (apnea).

**Frecuencia de muestreo**: El EEG se re-muestrea a **100 Hz** (del original variable de UCDDB y de 200 Hz de ISRUC). Esto da **3,000 muestras por época** de 30 segundos, suficiente para capturar ritmos EEG relevantes (delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–12 Hz).

**Normalización z-score por época**: Cada época se normaliza individualmente:
```
x_norm = (x - mean(x)) / (std(x) + 1e-8)
```
Esto elimina diferencias de amplitud absoluta entre sujetos y electrodos, estandarizando la distribución de cada ventana de 30 segundos.

---

### 2.2 Redes Neuronales Convolucionales 1D (CNN 1D)

Una **convolución 1D** aplica un filtro deslizante sobre una secuencia temporal. Para una entrada `x` de longitud `L` y un filtro `w` de tamaño `k`:

```
(x * w)[t] = sum_{i=0}^{k-1} x[t+i] · w[i]
```

Con `padding="same"`, la salida tiene la misma longitud que la entrada.

**¿Por qué CNN 1D para EEG?**
- Captura patrones locales en el tiempo sin requerir extracción manual de características (como STFT o wavelets)
- Los filtros aprenden automáticamente qué frecuencias son relevantes
- Invarianza traslacional: detecta un patrón apneico independientemente de cuándo ocurre en la época

#### Arquitectura SleepApneaCNN

```
Entrada: (B, 1, 3000)  — batch × canales × muestras
    ↓
Block 1: Conv1d(1→32, k=50) + BN + ReLU + MaxPool(8) + Dropout(0.5)
         → (B, 32, 375)    [kernel de 500ms captura ritmos lentos]
    ↓
Block 2: Conv1d(32→64, k=8) + BN + ReLU + MaxPool(4) + Dropout(0.5)
         → (B, 64, 93)     [kernel de 80ms, patrones locales]
    ↓
Block 3: Conv1d(64→128, k=8) + BN + ReLU + MaxPool(4) + Dropout(0.5)
         → (B, 128, 23)    [representación comprimida]
    ↓
Flatten: (B, 128×23) = (B, 2944)
    ↓
FC1: Linear(2944→64) + ReLU + Dropout(0.5)
    ↓
FC2: Linear(64→2) → logits
    ↓
Salida: (B, 2)  — clase 0=no apnea, clase 1=apnea
```

Componentes clave:
- **Batch Normalization (BN)**: normaliza las activaciones de cada mini-batch, acelerando el entrenamiento y reduciendo la sensibilidad al learning rate
- **ReLU**: `f(x) = max(0, x)`. Introduce no-linealidad sin saturación para valores positivos
- **MaxPool**: reduce la dimensión temporal tomando el valor máximo en una ventana. Proporciona invarianza traslacional local
- **Dropout**: desactiva aleatoriamente un porcentaje (50%) de neuronas durante el entrenamiento como regularización, previniendo sobreajuste

**Parámetros totales del baseline:**
- Block 1: 1×50×32 (pesos) + 32 (bias) + 4×32 (BN) = ~1,760
- Block 2: 32×8×64 + 64 + 4×64 = ~16,704
- Block 3: 64×8×128 + 128 + 4×128 = ~66,432
- FC1: 2944×64 + 64 = ~188,480
- **Total: 272,802 parámetros**

---

### 2.3 Poda Estructurada (Structured Pruning)

La poda (pruning) reduce el tamaño del modelo eliminando componentes redundantes. Existen dos tipos:

| Tipo | Qué elimina | Resultado | Hardware |
|---|---|---|---|
| **Poda no estructurada** | Pesos individuales (weights) | Matriz dispersa (sparse) | Requiere hardware especial |
| **Poda estructurada** | Filtros completos (canales) | Red densa más pequeña | Compatible con cualquier CPU/GPU |

Este proyecto usa **poda estructurada L1-norm de filtros Conv1d**.

#### Algoritmo de poda L1-norm

**Paso 1 — Calcular la norma L1 de cada filtro:**

Para una capa Conv1d con pesos `W ∈ R^(C_out × C_in × K)`:
```
||w_i||₁ = sum_{j,k} |W[i, j, k]|     para cada filtro i = 0..C_out-1
```

**Paso 2 — Ordenar y seleccionar:**

Con ratio de poda `p` (0.5 = 50%), se conservan los `(1-p) × C_out` filtros con mayor norma L1:
```python
n_keep = round(C_out * (1 - prune_ratio))
keep_indices = argsort(l1_norms, descending=True)[:n_keep]
```

**Fundamento**: Los filtros con norma L1 baja tienen pesos pequeños → contribuyen poco a la salida → son menos importantes. Es una heurística simple pero efectiva.

**Paso 3 — Construir red comprimida:**

Se crea una nueva red `PrunedSleepApneaCNN` con `C_out = n_keep` canales por capa. Los pesos del teacher se transfieren selectivamente:

```python
# Block 1: solo filas k1 del tensor de pesos
student.features[0].weight = teacher.features[0].weight[k1]

# Block 2: filas k2 Y columnas k1 (porque los inputs cambiaron)
student.features[5].weight = teacher.features[5].weight[k2][:, k1, :]
```

Esta transferencia inicializa el student con conocimiento del teacher antes de fine-tuning.

**Resultado de poda al 50%:**
```
Baseline:   conv1=32, conv2=64, conv3=128  → 272,802 params
p050:       conv1=16, conv2=32, conv3=64   → 116,018 params  (×2.35 menos)
p070:       conv1=10, conv2=19, conv3=38   →  64,127 params  (×4.25 menos)
```

---

### 2.4 Destilación del Conocimiento (Knowledge Distillation)

La **Knowledge Distillation (KD)** es una técnica donde un modelo pequeño (student) aprende imitando las salidas de un modelo grande (teacher), además de las etiquetas reales.

**Motivación**: Las etiquetas duras (0 o 1) solo dicen "apnea / no apnea". Las probabilidades del teacher contienen información más rica: qué tan seguro está el modelo y qué relaciones existen entre clases. Por ejemplo, si el teacher predice `[0.7, 0.3]`, comunica que hay algo de ambigüedad, lo que es más informativo que una etiqueta dura `[1, 0]`.

#### Función de pérdida KD (Hinton et al., 2015)

```
L_KD = α · L_CE(student_logits, y_hard)
     + (1-α) · T² · KL(softmax(teacher/T) || softmax(student/T))
```

Donde:
- **L_CE**: Cross-Entropy estándar con las etiquetas reales (0 o 1)
- **KL**: Divergencia de Kullback-Leibler, mide cuán diferentes son dos distribuciones de probabilidad
- **T (temperatura)**: Suaviza las probabilidades. Con T=4, `softmax(logits/T)` produce distribuciones más planas que con T=1, revelando "dark knowledge" (qué clases confunde el teacher)
- **α**: Balance entre aprender de las etiquetas (α grande) o del teacher (1-α grande)
- **T²**: Compensación matemática porque al dividir logits por T, los gradientes del KL se escalan por 1/T². Multiplicar por T² restaura la magnitud original

**Parámetros usados en el proyecto:**
- T = 4 (temperatura alta para soft targets más informativos)
- α = 0.5 (balance igual entre CE y KL)

**¿Por qué KD después de poda?**
El student podado comienza con pesos transferidos del teacher pero una arquitectura reducida. El fine-tuning con KD permite que el student imite el comportamiento completo del teacher (no solo sus predicciones binarias), recuperando capacidad predictiva con menos parámetros.

---

### 2.5 Cuantización INT8 (Post-Training Quantization)

La **cuantización** reduce la precisión numérica de los pesos y activaciones de FP32 (32 bits, ~7 decimales de precisión) a INT8 (8 bits entero, rango -128 a 127).

**Ventajas:**
- Modelo ~4× más pequeño en disco
- Operaciones aritméticas enteras son ~2–4× más rápidas en CPU
- Menor uso de memoria en caché

**Trade-off**: Reducción de precisión puede causar pequeñas caídas de AUC.

#### Cuantización Estática Post-Entrenamiento (PTQ)

El proyecto usa **Static PTQ** con formato **QDQ (Quantize-DeQuantize)** de ONNX Runtime:

**Paso 1 — Exportar a ONNX FP32:**
```python
torch.onnx.export(model, dummy_input, "model_fp32.onnx",
                  opset_version=17, do_constant_folding=True)
```
ONNX (Open Neural Network Exchange) es un formato estándar que permite ejecutar modelos en múltiples runtimes sin depender de PyTorch.

**Paso 2 — Calibración:**
Se pasan 200 muestras del val set por el modelo FP32 y se registran los rangos de activación (min, max) de cada capa. Esto permite calcular los factores de escala (scale) y punto cero (zero point) para mapear FP32 → INT8:

```
x_int8 = round(x_fp32 / scale) + zero_point
x_fp32 ≈ (x_int8 - zero_point) × scale
```

**Paso 3 — Insertar nodos QDQ:**
El modelo INT8 tiene nodos de Quantize (FP32→INT8) y DeQuantize (INT8→FP32) alrededor de cada operación. En CPU, las operaciones matemáticas ocurren en INT8.

**¿Por qué ONNX Runtime y no TFLite?**
TensorFlow ≥ 2.11 no tiene soporte oficial en Windows nativo. ONNX Runtime es cross-platform, compatible con Windows 11, y produce resultados equivalentes.

---

### 2.6 Evaluación Cross-Dataset

La **evaluación cross-dataset** es el estándar más riguroso de generalización: el modelo se entrena en un dataset y se evalúa en uno completamente diferente, sin ningún sujeto compartido.

**¿Por qué es importante?**
Un modelo puede memorizar características específicas de los sujetos de entrenamiento (sobreajuste) y no generalizar a nuevos pacientes. La evaluación intra-dataset puede inflarse si los sujetos de test son similares a los de train. Cross-dataset elimina este sesgo.

**En este proyecto:**
- **Entrenamiento**: UCDDB (25 sujetos, equipo PSG del St. Vincent's Hospital Dublin)
- **Test**: ISRUC-Sleep (22 sujetos, equipo PSG del Instituto de Telecomunicações Portugal)

Las diferencias entre datasets incluyen: equipo de grabación distinto, población diferente, protocolo de anotación diferente (eventos OA/CA/OH vs APNEA-O/HYP), tasa de apnea diferente (10.6% vs 21.2%).

**Hallazgo de este proyecto**: Los modelos comprimidos (podados) generalizan *mejor* a ISRUC que el baseline completo. Esto ocurre porque la poda actúa como **regularización implícita**: un modelo con menos parámetros tiene menos capacidad para memorizar patrones específicos de UCDDB y aprende características más transferibles.

---

## 3. Métricas de Evaluación

### 3.1 Accuracy, Sensibilidad y Especificidad

Para un clasificador binario, la **matriz de confusión** es:

```
                  Predicho: 0        Predicho: 1
Real: 0 (no apnea)   TN (True Neg)   FP (False Pos)
Real: 1 (apnea)      FN (False Neg)  TP (True Pos)
```

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
  → Proporción de predicciones correctas. **Problema**: con desbalance (80% clase 0), un modelo que siempre predice 0 tiene accuracy = 80% pero detecta 0 apneas.

- **Sensibilidad (Recall, TPR)** = TP / (TP + FN)
  → Proporción de apneas reales que el modelo detectó. Alta sensibilidad = pocos falsos negativos. **Crítica en clínica**: no detectar apneas es peligroso.

- **Especificidad (TNR)** = TN / (TN + FP)
  → Proporción de épocas sin apnea correctamente identificadas. Baja especificidad = muchas falsas alarmas.

**Trade-off**: Sensibilidad y especificidad están en tensión. Bajar el umbral de clasificación aumenta sensibilidad pero reduce especificidad. El AUC-ROC resume este trade-off.

### 3.2 F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Precision = TP / (TP + FP)
```

El F1 es la media armónica de precisión y recall. Es apropiado con desbalance de clases porque penaliza tanto los falsos positivos como los falsos negativos. Rango: 0 (peor) a 1 (mejor).

### 3.3 AUC-ROC

La **Curva ROC (Receiver Operating Characteristic)** grafica la Sensibilidad (TPR) vs. 1-Especificidad (FPR) para todos los umbrales de clasificación posibles (de 0 a 1).

```
      1 ┤        ●●●●●●●
        │    ●●●
Sens.   │  ●
(TPR)   │●
        │●
      0 ┼──────────────── 
        0               1
           1 - Especificidad (FPR)
```

El **AUC (Area Under the Curve)** es el área bajo esta curva:
- AUC = 1.0: clasificador perfecto
- AUC = 0.5: clasificador aleatorio (diagonal)
- AUC < 0.5: peor que aleatorio (invertir predicciones lo mejoraría)

**Ventaja sobre accuracy**: El AUC es independiente del umbral de clasificación y robusto al desbalance de clases. Por eso es la métrica principal de este proyecto.

En el proyecto, el AUC-ROC se calcula sobre las probabilidades de clase 1 (probabilidad de apnea):
```python
probs = softmax(logits)[:, 1]   # probabilidad de apnea
auc = roc_auc_score(y_true, probs)
```

### 3.4 Cohen's Kappa y MCC

**Cohen's Kappa (κ)**:
```
κ = (Po - Pe) / (1 - Pe)
```
Donde Po = accuracy observada, Pe = accuracy esperada por azar. κ = 0 significa rendimiento al nivel del azar; κ = 1 es perfecto. Es más riguroso que accuracy porque descuenta la concordancia por azar.

**MCC (Matthews Correlation Coefficient)**:
```
MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```
Rango de -1 a +1. Considera todos los cuadrantes de la matriz de confusión. Es especialmente útil con desbalance severo. MCC = 0 equivale a predicción aleatoria.

---

### 3.5 Latencia, RAM y Tamaño en Disco

**Latencia**: Tiempo de inferencia por muestra individual (batch=1) en CPU.

Protocolo de medición:
1. **Warm-up**: 50 inferencias (llenan el caché de la CPU, eliminan overhead de inicialización)
2. **Medición**: 500 inferencias con `time.perf_counter()`
3. Se reportan **media**, **std** y **percentil 95 (P95)**

El P95 es especialmente importante: en aplicaciones en tiempo real, el 5% de inferencias más lentas no deben superar un umbral crítico.

**RAM**: Uso de memoria del proceso (`psutil.Process().memory_info().rss`) antes y después de cargar el modelo. Mide el overhead de carga en MB.

**Tamaño en disco**: Tamaño del archivo `.onnx` en KB. Determina si el modelo cabe en la memoria flash de un dispositivo embebido.

---

### 3.6 Tabla de Ablación E0–E5

Un **estudio de ablación** desactiva sistemáticamente componentes del sistema para aislar la contribución individual de cada técnica. Es la forma estándar en deep learning de demostrar que cada parte del pipeline aporta valor.

| ID | Configuración | Qué técnica añade | Comparar con |
|---|---|---|---|
| E0 | Baseline FP32 | — (punto de partida) | — |
| E1 | Pruning 50% sin KD | Poda sola | E0 → mide efecto del pruning |
| E2 | Solo INT8 (baseline) | Cuantización sola | E0 → mide efecto de INT8 |
| E3 | Pruning 50% + INT8 | Pruning + INT8 sin KD | E1 → mide INT8 sobre modelo podado |
| E4 | Pruning 50% + KD | Pruning + KD sin INT8 | E1 → mide efecto del KD |
| E5 | Pruning + KD + INT8 | Pipeline completo | E4 → mide INT8 final |

Al comparar E4 vs E1 se ve el efecto *exclusivo* del KD. Al comparar E5 vs E4 se ve el efecto *exclusivo* de INT8 tras KD. Esto permite justificar cada componente por separado.

**ΔAUC** = AUC(modelo) - AUC(E0). Valores positivos indican mejora sobre el baseline.

---

### 3.7 Curva de Pareto

La **Curva de Pareto** (o **Frontera de Pareto**) visualiza el trade-off entre dos objetivos en conflicto. En este proyecto: **maximizar AUC** vs **minimizar latencia**.

Un modelo está en la **frontera de Pareto** si no existe ningún otro modelo que lo domine, es decir, ningún modelo con *mayor o igual AUC* **y** *menor o igual latencia* simultáneamente.

**Algoritmo de detección de dominancia:**
```python
def pareto_frontier(points):
    dominated = [False] * len(points)
    for i, (lat_i, auc_i) in enumerate(points):
        for j, (lat_j, auc_j) in enumerate(points):
            if i == j: continue
            # j domina a i si es mejor en ambos objetivos
            if lat_j <= lat_i and auc_j >= auc_i and (lat_j < lat_i or auc_j > auc_i):
                dominated[i] = True; break
    return [i for i, d in enumerate(dominated) if not d]
```

**Interpretación del gráfico:**
```
AUC ↑
     │  ●           ← E0 (Baseline FP32): alto AUC, lento
     │    ●  ●      ← modelos intermedios
     │       ●  ●   ← E5 ★ (óptimo): mayor AUC + más rápido
     └────────────→ Latencia ↑ (más lento)
```

Los modelos sobre la frontera (marcados con borde negro) son las opciones óptimas: elegir entre ellos es una decisión de diseño (¿priorizo AUC o velocidad?), pero cualquier modelo *fuera* de la frontera es objetivamente peor.

**E5 en la frontera de Pareto** significa que ningún otro modelo ofrece simultáneamente mayor AUC y menor latencia. Es la configuración más eficiente del pipeline.

---

## 4. Módulos del Código Fuente (`src/`)

### `src/models.py` — Arquitectura CNN

Define `SleepApneaCNN`, la CNN 1D baseline. Usa `nn.Sequential` para organizar las capas en dos bloques: `self.features` (capas convolucionales) y `self.classifier` (capas fully-connected).

El `forward` es simplemente:
```python
def forward(self, x):
    return self.classifier(self.features(x))
```

El índice de cada capa en `features` es importante porque `pruning.py` accede a ellas por índice fijo (`features[0]`, `features[5]`, `features[10]`).

---

### `src/pruning.py` — Poda Estructurada

Contiene tres componentes:

**1. `PrunedSleepApneaCNN`**: Misma arquitectura que `SleepApneaCNN` pero con número de canales variable, determinado por `arch_config`:
```python
arch_config = {"conv1_out": 16, "conv2_out": 32, "conv3_out": 64, "keep_indices": {...}}
```

**2. `compute_prune_masks(model, prune_ratio)`**: Calcula qué filtros conservar.
```python
l1 = conv.weight.detach().abs().sum(dim=(1, 2))  # norma L1 por filtro
keep = argsort(l1, descending=True)[:n_keep]      # los más grandes
```

**3. `build_pruned_model(teacher, masks)`**: Transfiere pesos selectivamente. El caso más delicado es la capa FC1 (`classifier[1]`):

Después del flatten, cada canal `ci` de `conv3` ocupa 23 posiciones consecutivas (porque el temporal después de los MaxPool es 23). Para seleccionar los canales `k3`:
```python
flat_idx = []
for ci in k3:
    flat_idx.extend(range(ci * 23, ci * 23 + 23))
student.classifier[1].weight = teacher.classifier[1].weight[:, flat_idx]
```

La función `_copy_bn` copia no solo los parámetros entrenables (`weight`, `bias`) del BatchNorm sino también las **running stats** (`running_mean`, `running_var`) que se acumulan durante el entrenamiento y son necesarias para inferencia correcta.

---

### `src/distillation.py` — Loss KD

Define `KDLoss`, una subclase de `nn.Module`. Implementa:

```python
ce  = F.cross_entropy(student_logits, labels, weight=ce_weight)
log_student  = F.log_softmax(student_logits / T, dim=1)
soft_teacher = F.softmax(teacher_logits / T, dim=1).detach()
kd  = F.kl_div(log_student, soft_teacher, reduction="batchmean") * T²
loss = α * ce + (1-α) * kd
```

El `.detach()` sobre `soft_teacher` es crítico: el teacher está congelado, sus gradientes no deben propagarse hacia atrás. `F.kl_div` espera inputs en escala log para el primer argumento, por eso se usa `log_softmax` para el student y `softmax` para el teacher.

---

## 5. Scripts del Pipeline

### 5.1 `01_prepare_data.py` — Preprocesamiento UCDDB

**Propósito**: Convertir los archivos EDF crudos de UCDDB en arrays NumPy listos para entrenamiento.

**Proceso por sujeto:**

```
1. Leer EDF (.rec) con MNE
   → Copia temporal a .edf porque MNE rechaza extensión .rec
   → Seleccionar canal C3-A2 (prueba varios nombres: "C3A2", "C3-A2", "EEG C3-A2")
   → Re-muestrear a 100 Hz si es necesario

2. Leer etiquetas de staging (_stage.txt)
   → Formato R&K: 0=Wake, 1=REM, 2=S1, 3=S2, 4=S3, 5=S4, 6=MT
   → Convertir a AASM: W=0, N1=1, N2=2, N3=3, REM=4, MT=-1 (descartar)

3. Segmentar señal en épocas de 30s
   → n_epochs = min(len_signal // 3000, len_staging_annotations)

4. Leer eventos de apnea (_respevt.txt)
   → Regex: HH:MM:SS + TIPO + DURACIÓN
   → Mapear cada evento a época(s) con solapamiento ≥ 1 segundo
   → Manejo de cruce de medianoche (eventos de 00:xx cuando grabación inicia en 23:xx)

5. Descartar épocas MT (staging == -1)

6. Z-score por época individual
```

**Partición por sujeto (no por época):**

Esta es una decisión de diseño crítica. Si se particionara por época, el mismo sujeto podría aparecer en train y test, generando **data leakage**: el modelo vería durante entrenamiento EEG del mismo cerebro que en test, inflando artificialmente las métricas.

```python
SPLIT = {
    "train": [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 25],
    "val":   [22, 23, 24],
    "test":  [26, 27, 28],
}
```

**Pesos de clase:**
Con 78.8% no-apnea y 21.2% apnea, el modelo sin corrección tendería a predecir siempre 0. La solución es usar `weighted cross-entropy`:
```python
w = compute_class_weight("balanced", classes=[0, 1], y=y_train)
# Resultado: peso_0 = 0.6245, peso_1 = 2.508
# La clase rara (apnea) pesa 4× más en la loss
```

**Salidas:**
- `data/processed/X_train.npy` → shape (15770, 1, 3000)
- `data/processed/y_apnea_train.npy` → shape (15770,)
- `data/processed/class_weights_apnea.json`
- `data/processed/split_subjects.json`

---

### 5.2 `03_isruc_preprocess.py` — Preprocesamiento ISRUC

**Propósito**: Preprocesar los sujetos de ISRUC-Sleep y guardarlos como archivos NPZ individuales para uso como test set cross-dataset.

**Diferencias respecto a UCDDB:**

| Aspecto | UCDDB | ISRUC |
|---|---|---|
| Frecuencia original | Variable (100-512 Hz) | 200 Hz fijo |
| Formato etiquetas apnea | `_respevt.txt` (timestamps + duración) | `.xlsx` (por época directamente) |
| Tipos de apnea | APNEA-O, APNEA-C, HYP-O... | OA, CA, OH, MA, MH |
| Etiquetas de staging | `_stage.txt` | Columna en .xlsx |

**Lectura de etiquetas desde Excel:**
El archivo `.xlsx` tiene columnas por época con una columna "Events". El script busca el encabezado para localizar la columna correcta y lee directamente el tipo de evento por época:
```python
if any(e in evt for e in {"OA", "CA", "OH", "MA", "MH"}):
    apnea[ep_idx] = 1
```

**Criterio de inclusión:**
Solo se procesan sujetos con tasa de apnea ≥ 2%. Los sujetos `{3, 4, 6, 11, 14, 25, 27, 29}` son excluidos porque tienen muy pocas épocas de apnea, lo que haría las métricas de AUC estadísticamente inestables.

**Salidas:**
- `data/processed/isruc_subjects/subj_NNN.npz` → contiene `X: (N, 1, 3000)` y `y_apnea: (N,)`

---

### 5.3 `02b_crossval.py` — Validación Cruzada 5-Fold

**Propósito**: Entrenar el baseline con validación cruzada por sujeto y obtener una estimación robusta del rendimiento, guardando el mejor modelo global.

**¿Por qué validación cruzada y no un split fijo?**

Con solo 25 sujetos, el rendimiento en un split fijo de 3 sujetos de test depende mucho de qué sujetos toquen en test (alta varianza). La validación cruzada promedia sobre 5 configuraciones distintas, dando una estimación más estable.

**Esquema de 5-fold por sujeto:**

```
Fold 1: Test=[2,3,5,6,7]   Val=[8,9,10]    Train=17 sujetos restantes
Fold 2: Test=[8,9,10,11,12] Val=[13,14,15]  Train=17 sujetos restantes
Fold 3: Test=[13,14,15,17,18] Val=[19,20,21] Train=17 sujetos restantes
Fold 4: Test=[19,20,21,22,23] Val=[24,25,26] Train=17 sujetos restantes
Fold 5: Test=[24,25,26,27,28] Val=[2,3,5]    Train=17 sujetos restantes
```

**Proceso por fold:**

```
1. Cargar datos EDF del sujeto en tiempo real (no precargados)
   → Re-procesar desde los EDF originales para cada fold
2. Entrenar SleepApneaCNN con Adam lr=1e-4, batch=32
   → Weighted cross-entropy (pesos calculados solo con train de ese fold)
   → Early stopping: paciencia=30 épocas de entrenamiento sin mejora en val AUC
   → ReduceLROnPlateau: reducir lr ×0.5 si val AUC no mejora en 10 épocas
3. Guardar mejor checkpoint del fold (mayor val AUC)
4. Evaluar en test set del fold con best checkpoint
```

**Early Stopping:**
Detiene el entrenamiento cuando el modelo deja de mejorar en validación, previniendo sobreajuste. La "paciencia" es cuántas épocas esperar antes de parar.

**ReduceLROnPlateau:**
Reduce el learning rate cuando el modelo se estanca. Un learning rate menor permite explorar con pasos más pequeños en el espacio de pesos, potencialmente encontrando mejores mínimos.

**Salidas:**
- `checkpoints/crossval_best.pth` → mejor modelo entre todos los folds
- `checkpoints/sleep_baseline.pth` → copia del mejor para usar como teacher
- `results/crossval_metrics.json` → métricas por fold + media + std

**Resultados obtenidos:**
```
AUC-ROC: 0.6902 ± 0.0403 (5-fold CV, test UCDDB)
```

---

### 5.4 `03_pruning_kd.py` — Pruning + Knowledge Distillation

**Propósito**: Generar las 4 variantes podadas (con y sin KD, al 50% y 70%) mediante fine-tuning post-poda. El test set es ISRUC (cross-dataset).

**Las 4 variantes:**

| Variante | Ratio poda | Knowledge Distillation |
|---|---|---|
| `p050_kd` | 50% | Sí (T=4, α=0.5) |
| `p050_nokd` | 50% | No (solo CE) |
| `p070_kd` | 70% | Sí |
| `p070_nokd` | 70% | No |

**Por qué las 4 variantes:**
El diseño factorial (2 ratios × 2 con/sin KD) permite aislar el efecto del KD independientemente del ratio de poda, y el efecto del ratio independientemente del KD. Esto es un diseño de experimento controlado.

**Proceso para cada variante:**

```
1. Calcular máscaras (una vez por ratio, reutilizadas)
   masks = compute_prune_masks(teacher, prune_ratio)

2. Construir student con pesos transferidos (fresh por variante)
   student, arch_config = build_pruned_model(teacher, masks)

3. Fine-tuning:
   - Adam lr=1e-4, 50 épocas, patience=20
   - Si use_kd: L = KDLoss(T=4, α=0.5)
   - Si no: L = CrossEntropy(weights=ce_weights)
   - Mejor checkpoint: mayor AUC en val (UCDDB)

4. Evaluar best checkpoint en ISRUC (cross-dataset test)

5. Guardar:
   checkpoints/{variante}/
     model.pth          — mejor checkpoint
     model_full.pth     — último checkpoint
     arch_config.json   — arquitectura comprimida
     pruning_masks.pkl  — índices conservados
     metrics.json       — historial de entrenamiento
```

**Importante**: Las máscaras se calculan **una sola vez** del teacher para cada ratio, y se reutilizan para KD y noKD del mismo ratio. Esto garantiza que p050_kd y p050_nokd tienen exactamente la misma arquitectura (mismos filtros conservados), y la diferencia es solo el método de fine-tuning.

**Carga del test ISRUC como TensorDataset:**
```python
x_isruc, y_isruc = load_isruc_test()
isruc_ds = TensorDataset(torch.from_numpy(x_isruc), torch.from_numpy(y_isruc))
test_loader = DataLoader(isruc_ds, batch_size=32, shuffle=False)
```

---

### 5.5 `04_quantize.py` — Cuantización INT8

**Propósito**: Cuantizar el modelo `p050_kd` a INT8 usando ONNX Runtime PTQ. Genera métricas de latencia y tamaño.

**Pipeline:**

```
1. Cargar p050_kd desde checkpoints/p050_kd/model.pth

2. Exportar a ONNX FP32 (opset 17)
   → dummy_input = torch.randn(1, 1, 3000)
   → dynamic_axes para batch variable
   → do_constant_folding=True: fusiona operaciones constantes para eficiencia

3. Calibración con EEGCalibrationReader
   → 200 muestras aleatorias del val set
   → Pasa las muestras por el modelo FP32 registrando rangos de activación

4. quantize_static con QDQ INT8
   → QuantFormat.QDQ: inserta nodos Q y DQ alrededor de cada op
   → per_channel=False: un solo factor de escala por tensor
   → activation_type=QInt8, weight_type=QInt8

5. Medir latencia (10 warmup + 100 runs para este script)

6. Evaluar en val y test UCDDB (nota: 05_evaluate.py evalúa en ISRUC)
```

**Nota sobre este script vs `05_evaluate.py`:**
Este script solo cuantiza `p050_kd` y evalúa en UCDDB. El script `05_evaluate.py` re-cuantiza *todos* los modelos durante la evaluación y evalúa en ISRUC. Esto puede parecer redundante, pero `04_quantize.py` es una fase intermedia de verificación rápida, mientras `05_evaluate.py` es la evaluación definitiva.

---

### 5.6 `05_evaluate.py` — Evaluación Completa

**Propósito**: Evaluar todos los modelos (baseline + 4 variantes) en FP32 e INT8 sobre el test cross-dataset ISRUC, con métricas de latencia, RAM y tamaño en disco.

**Modelos evaluados:**

| Modelo | Fuente |
|---|---|
| baseline | `checkpoints/sleep_baseline.pth` |
| p050_kd | `checkpoints/p050_kd/model.pth` |
| p050_nokd | `checkpoints/p050_nokd/model.pth` |
| p070_kd | `checkpoints/p070_kd/model.pth` |
| p070_nokd | `checkpoints/p070_nokd/model.pth` |

**Por cada modelo:**

```
1. Exportar a ONNX FP32 (si no existe ya)
2. Cuantizar a INT8 (si no existe ya)
   → Usa x_val (UCDDB) para calibración

3. RAM: medir antes/después de cargar sesión ONNX
   ram_before = psutil.Process().memory_info().rss / 1024 / 1024
   sess = ort.InferenceSession(path)
   ram_fp32 = ram_after - ram_before

4. Latencia: 50 warmup + 500 runs con batch=1
   → Medir con time.perf_counter() (resolución ~100ns en Windows)
   → Calcular media, std, P95

5. Métricas: evaluar en batch=64 sobre ISRUC completo
   → Softmax numérico estable: exp(out - max(out))
   → Calcular accuracy, F1, AUC-ROC

6. Guardar en results/metrics_eval.json
```

**Softmax numéricamente estable:**
```python
exp = np.exp(out - out.max(axis=1, keepdims=True))
probs = exp / exp.sum(axis=1, keepdims=True)
```
Restar el máximo evita overflow de `exp()` para logits muy grandes, sin cambiar el resultado final.

---

### 5.7 `06_ablation.py` — Estudio de Ablación E0–E5

**Propósito**: Construir la tabla de ablación comparando configuraciones del pipeline. Es un script de posprocesamiento que solo lee `metrics_eval.json` y reorganiza los datos.

**Definición de E0–E5:**

```python
rows = [
    ("E0", "Baseline",            "baseline",  "fp32"),
    ("E1", "Pruning 50% (no KD)", "p050_nokd", "fp32"),
    ("E2", "Quant INT8 sola",     "baseline",  "int8"),
    ("E3", "Pruning 50% + INT8",  "p050_nokd", "int8"),
    ("E4", "Pruning 50% + KD",    "p050_kd",   "fp32"),
    ("E5", "Pruning + KD + INT8", "p050_kd",   "int8"),
]
```

El ΔAUC se calcula siempre respecto a E0 (baseline FP32):
```python
delta_auc = auc(modelo) - auc("baseline", "fp32", "test")
```

El speedup es respecto a la latencia de E0:
```python
speedup = latencia_E0 / latencia_modelo
```

**Salida: `results/ablation_metrics.json`**

---

### 5.8 `07_pareto_plot.py` — Visualizaciones

**Propósito**: Generar dos figuras:
1. **Curva de Pareto** (`pareto.png`): scatter AUC vs latencia para todos los modelos
2. **Tabla visual de ablación** (`ablation_table.png`): tabla E0–E5 como imagen PNG

**Pareto plot:**

Todos los modelos FP32 e INT8 se grafican. Los puntos en la frontera de Pareto se marcan con borde negro y tamaño mayor. Se traza la línea discontinua conectando los puntos de Pareto ordenados por latencia.

El color codifica la arquitectura (baseline=azul, p050=verde, etc.) y el marcador codifica la precisión (círculo=FP32, cuadrado=INT8).

**Tabla de ablación como imagen:**
Usa `ax.table()` de Matplotlib para renderizar la tabla E0–E5 con colores alternos y resaltado verde para E5. Útil para incluir en presentaciones.

---

### 5.9 `generate_report_docx.py` — Generación del Informe

**Propósito**: Generar automáticamente `informe_final.docx` leyendo los JSONs de resultados, insertando las tablas y figuras, y componiendo el texto del informe.

**Estructura del documento generado:**

```
1. Portada
2. Resumen (Abstract) — con métricas reales del JSON
3. Introducción — contribuciones del proyecto
4. Trabajo Relacionado — tablas de papers
5. Datasets y Preprocesamiento — UCDDB + ISRUC
6. Métodos de Compresión — pruning, KD, cuantización, pipeline
7. Resultados Experimentales:
   - Baseline CV + cross-dataset
   - Ablación E0–E5
   - Eficiencia computacional
   - Curva de Pareto (imagen embebida)
8. Propuesta de Mejora — QAT + Channel Attention
9. Conclusiones
10. Referencias IEEE
11. Apéndice — hiperparámetros, comandos de reproducción, fold details
```

**Funciones helper:**

- `add_heading(doc, text, level)`: Títulos con alineación izquierda
- `add_para(doc, text, bold, italic, ...)`: Párrafos justificados
- `add_table(doc, headers, rows, ...)`: Tablas con header oscuro, filas alternas y fila destacada opcional
- `add_code(doc, code)`: Bloques de código en Courier New
- `set_cell_bg(cell, hex_color)`: Colorea celdas de tabla mediante XML directamente

Las métricas se leen en tiempo de ejecución desde los JSONs:
```python
with open(RESULTS_DIR / "metrics_eval.json") as f:
    ev = json.load(f)
# Uso:
f"{ev['p050_kd']['latency']['int8']['mean_ms']:.3f} ms"
```

Esto garantiza que el informe siempre refleja los resultados reales, no valores hardcodeados.

---

## Apéndice — Glosario Rápido

| Término | Definición |
|---|---|
| **EDF** | European Data Format — formato estándar para señales biomédicas |
| **MNE** | MNE-Python — biblioteca para análisis de señales EEG/MEG |
| **ONNX** | Open Neural Network Exchange — formato portátil de modelos |
| **PTQ** | Post-Training Quantization — cuantización sin reentrenar |
| **QAT** | Quantization-Aware Training — cuantización simulada durante entrenamiento |
| **QDQ** | Quantize-DeQuantize — nodos ONNX que encapsulan operaciones INT8 |
| **FP32** | Floating Point 32-bit — precisión estándar de PyTorch |
| **INT8** | Integer 8-bit — precisión reducida para inferencia eficiente |
| **R&K** | Rechtschaffen & Kales — sistema clásico de estadificación del sueño |
| **AASM** | American Academy of Sleep Medicine — sistema moderno de staging |
| **Adam** | Adaptive Moment Estimation — optimizador con tasa de aprendizaje adaptiva |
| **ReduceLROnPlateau** | Scheduler que reduce lr cuando la métrica se estanca |
| **Early Stopping** | Detener entrenamiento si val no mejora durante N épocas |
| **Batch size** | Número de muestras procesadas simultáneamente en cada paso de gradiente |
| **Epoch** | Una pasada completa sobre el dataset de entrenamiento (diferente de "época" de 30s) |
| **Logit** | Salida cruda de la red (antes de softmax) |
| **Softmax** | `exp(x_i) / sum(exp(x_j))` — convierte logits en probabilidades |
| **Cross-Entropy** | Loss para clasificación: `-sum(y_true * log(y_pred))` |
| **KL Divergence** | `sum(p * log(p/q))` — mide diferencia entre distribuciones p y q |
