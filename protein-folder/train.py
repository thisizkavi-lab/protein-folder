import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ProteinFolder
from data_loader import ProteinDataset, SEQ_LEN, PDB_IDS

# --- Configuration ---
BATCH_SIZE = 1
EPOCHS = 200    
LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using Device: {DEVICE}")

# --- Load Data ---
dataset = ProteinDataset(PDB_IDS, SEQ_LEN)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

if len(dataset) == 0:
    print("Error: No data loaded.")
    exit()

# --- Initialize Model ---
model = ProteinFolder(vocab_size=20).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
print("\nStarting training...")
losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for seq, target_map in train_loader:
        seq, target_map = seq.to(DEVICE), target_map.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(seq)
        
        loss = criterion(output, target_map)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

print("Training complete.")

# --- Evaluation on a sample ---
print("\nVisualizing prediction...")
model.eval()
with torch.no_grad():
    sample_seq, sample_target = dataset[0]
    sample_seq = sample_seq.unsqueeze(0).to(DEVICE)
    prediction = model(sample_seq)

# Plot
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Protein Folding Analysis', fontsize=16)

# 1. Sequence (Fixed the reshape error here)
# We reshape (60,) to (1, 60) so imshow can draw it as a line
axs[0].imshow(sample_seq.cpu().numpy()[0].reshape(1, -1), aspect='auto', cmap='viridis')
axs[0].set_title("Input: Amino Acid Sequence")
axs[0].set_yticks([])

# 2. Real Physics (Ground Truth)
axs[1].imshow(sample_target.numpy(), cmap='Blues')
axs[1].set_title("Ground Truth Contact Map")

# 3. Model Prediction
axs[2].imshow(prediction.cpu().numpy()[0], cmap='Blues')
axs[2].set_title("Prediction: Transformer Output")

plt.tight_layout()
plt.savefig('real_protein_result.png')
print("Result saved to 'real_protein_result.png'")
