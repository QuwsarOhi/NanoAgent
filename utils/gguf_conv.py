from pathlib import Path
import os

from mlx_lm.gguf import convert_to_gguf
from mlx_lm.utils import load_model
from mlx.utils import tree_flatten


# For converting MLX into huggingface inferencable
# https://huggingface.co/docs/transformers/v4.48.2/en/gguf#example-usage

# SIZE = "135M"
model_path = f'../weights/SmolLM2-135M-mlx-cdft-v10'
model, config = load_model(Path(model_path))
print("Model loaded")
weights = dict(tree_flatten(model.parameters()))
output_path = model_path.rstrip(os.path.sep) + '.gguf'


convert_to_gguf(model_path=model_path, weights=weights, config=config, output_file_path=output_path)
print("GGUF converstion complete", output_path)