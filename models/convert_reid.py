import torch
import torchreid
import torch.nn as nn

# 1. Definisci i percorsi
input_weights = "models\\model.osnet.pth.tar-10"  # Usa doppio backslash o slash normale per Windows
output_model = "osnet_x1_0_soccernet.pt"

print(f"🔄 Caricamento architettura OSNet x1.0...")
# Costruisci il modello
model = torchreid.models.build_model(
    name='osnet_x1_0', 
    num_classes=1000,  # Qui puoi lasciare 1000, tanto non caricheremo i pesi del classifier
    loss='softmax',
    pretrained=False
)

# 2. Carica i pesi
print(f"📥 Caricamento pesi da {input_weights}...")
checkpoint = torch.load(input_weights, map_location='cpu', weights_only=False)

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# --- CORREZIONE: Filtra i pesi del classifier prima di caricarli ---
new_state_dict = {}
for k, v in state_dict.items():
    # Rimuovi prefisso 'module.'
    name = k.replace("module.", "")
    
    # Se la chiave appartiene al classifier, LA SALTIAMO
    if "classifier" in name:
        continue
        
    new_state_dict[name] = v
# ------------------------------------------------------------------

print("Pesi filtrati (classifier rimosso). Caricamento in corso...")
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# 3. Esporta in TorchScript
print(f"💾 Esportazione in TorchScript: {output_model}...")
dummy_input = torch.randn(1, 3, 256, 128)
traced_script_module = torch.jit.trace(model, dummy_input)
traced_script_module.save(output_model)

print("✅ Fatto! File generato con successo.")