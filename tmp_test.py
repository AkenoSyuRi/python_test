import torchaudio.models
from torch import optim

torchaudio.models.hubert_pretrain_model()

optimizer = optim.Adam(params, lr=args.learn_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1, verbose=True)