from torch.nn import CrossEntropyLoss

vocab_size = 147  # TODO hwo to get it from dictionary
embed_size = 20
hidden_size = 30
num_directions = 2
num_layers = 2
batch_size = 1
epochs = 20
criterion = CrossEntropyLoss()
