import torch

def inspect_checkpoint(checkpoint_path):
    print(f"Inspecting {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Epoch in checkpoint: {checkpoint.get('epoch', 'N/A')}")
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    layers = set()
    for key in state_dict.keys():
        if 'transformer_encoder.layers' in key:
            parts = key.split('.')
            layers.add(parts[2]) # transformer_encoder.layers.X....
    
    print(f"Number of layers in transformer_encoder: {len(layers)}")
    print(f"Layer indices: {sorted([int(l) for l in layers])}")

if __name__ == "__main__":
    import os
    if os.path.exists('checkpoint.pth'):
        inspect_checkpoint('checkpoint.pth')
    if os.path.exists('best_model.pth'):
        inspect_checkpoint('best_model.pth')
    if os.path.exists('automata_transformer.pth'):
        inspect_checkpoint('automata_transformer.pth')
