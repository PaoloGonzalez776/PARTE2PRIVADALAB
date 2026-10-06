# my_llm — PaolosSoftware

Modelo de lenguaje general desde cero, decoder-only (RMSNorm + RoPE + SwiGLU),
entrenado progresivamente en etapas de 100M → 200M → 400M → 600M → 1B parámetros,
cada etapa heredando pesos de la anterior (ver `scripts/grow_model.py`).

## Estructura

```
configs/        Hiperparámetros de cada escala (100m.json ... 1b.json)
tokenizer/      Tokenizer BPE entrenado (vocab.json, merges.txt)
data/           raw -> cleaned -> deduplicated -> train/validation (.bin tokenizados)
model/          Arquitectura modular: embeddings, attention, mlp, transformer, gpt
training/       Loop de entrenamiento, optimizer, scheduler, checkpoints, distribuido
inference/      Generación de texto y chat en terminal
evaluation/     Perplejidad y benchmarks de opción múltiple
checkpoints/    Pesos guardados por step + symlink "latest"
scripts/        Wrappers de línea de comandos para cada paso del pipeline
tests/          pytest para model, tokenizer, attention, training
```

## Pipeline completo

```bash
pip install -r requirements.txt

# 1. Descargar datos crudos (Wikipedia es + OPUS-100)
python scripts/prepare_data.py --out_dir data/raw --max_per_dataset 500000

# 2. Entrenar tokenizer + tokenizar a .bin
python scripts/tokenize_data.py --raw_dir data/raw --tokenizer_dir tokenizer \
    --train_dir data/train --val_dir data/validation --vocab_size 32000

# 3. Entrenar (elige la escala: 100m, 200m, 400m, 600m, 1b)
python scripts/train.py --config 200m --data_dir data/train \
    --out_dir checkpoints --max_iters 20000

# 4. Evaluar
python scripts/evaluate.py --config 200m --checkpoint checkpoints/latest \
    --data_dir data/validation

# 5. Generar texto
python inference/generate.py --config 200m --checkpoint checkpoints/latest \
    --tokenizer_dir tokenizer --prompt "Había una vez"
```

## Crecimiento entre etapas

Cada etapa hereda los pesos de la anterior en vez de empezar random:

```bash
python scripts/grow_model.py --from_config 100m --from_checkpoint checkpoints_100m/latest \
    --to_config 200m --out checkpoints_200m/init.pt

python scripts/train.py --config 200m --init_from checkpoints_200m/init.pt \
    --data_dir data/train --out_dir checkpoints_200m --max_iters 20000
```

## GPU vs TPU

- GPU: `--attn_backend gpu` (usa flash-attention vía `scaled_dot_product_attention`)
- TPU: `--attn_backend tpu --dist_backend xla` (atención manual, sin kernel flash de CUDA)
- Multi-GPU: `--dist_backend ddp`
- Multi-chip TPU (ej. v5e-8): `--dist_backend xla`, lanzado con `xmp.spawn` (ver `training/distributed.py`)

## Aviso sobre escala vs. datos

Para la etapa 1B, la regla aproximada (Chinchilla) pide ~20 tokens de entrenamiento
por parámetro — eso son ~20 mil millones de tokens. Con solo Wikipedia+OPUS-100
(cientos de millones de tokens) el modelo de 1B va a sobreajustar rápido. Antes de
esa etapa, hace falta ampliar `data/raw/` con más fuentes.

## Tests

```bash
pytest tests/ -v
```
