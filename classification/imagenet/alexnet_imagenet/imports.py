
import os
import argparse
import math
import torch.nn.functional as F
import torch
from torch import optim
import torch.nn as nn

from PIL import Image
###################################################


def load_synset_mapping(synset_mapping_file):
    """Load synset mapping (WordNet ID → Class Index) from the given file."""
    mapping = {}
    with open(synset_mapping_file, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            wnid = line.strip().split(" ")[0]  # Extract only the WordNet ID
            mapping[wnid] = idx  # Assign index (0-999)

    # Debugging: Print first few mappings
    print(f"✅ Loaded {len(mapping)} synset mappings")
    print("🔍 First 5 mappings:", list(mapping.items())[:5])

    return mapping



def preprocess_image(image_path, preprocess):
    """Preprocess an image for the model."""
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)  # Add batch dimension


def predict(image_path, model, preprocess, synset_mapping):
    """Make a prediction on a single image."""
    try:
        input_tensor = preprocess_image(image_path, preprocess)
        with torch.no_grad():
            output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        top_class = probabilities.argmax().item()
        return synset_mapping[top_class]
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None


def load_ground_truth(val_solution_file):
    """Load ground truth labels for classification."""
    image_ids = []
    ground_truths = []
    with open(val_solution_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            image_ids.append(parts[0])  # ImageId
            ground_truths.append(parts[1].split()[0])  # SynsetID (first part of PredictionString)

    return image_ids, ground_truths



