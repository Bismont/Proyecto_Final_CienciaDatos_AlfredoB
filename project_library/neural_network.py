import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm.auto import tqdm


class SentimentLSTM(nn.Module):
    def __init__(
        self, vocab_size, domain_size,
        embed_dim=128, hidden_dim=128,
        domain_embed_dim=32, num_layers=2, dropout=0.3
    ):
        super().__init__()

        # --- embeddings ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.domain_emb = nn.Embedding(domain_size, domain_embed_dim)

        # --- LSTM bidireccional ---
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # --- capa final → 3 clases ---
        self.fc = nn.Linear(2 * hidden_dim + domain_embed_dim, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths, domain_ids):
        # x: (B, T)
        emb = self.embedding(x)  # (B, T, E)

        # empaquetar secuencia
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (h_n, _) = self.lstm(packed)

        # tomar últimas representaciones bidireccionales
        h_fwd = h_n[-2]      # (B, H)
        h_bwd = h_n[-1]      # (B, H)
        h_lstm = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2H)

        h_lstm = self.dropout(h_lstm)

        # --- domain embedding ---
        d_emb = self.domain_emb(domain_ids)  # (B, D_dom)

        # concatenar LSTM + dominio
        h = torch.cat([h_lstm, d_emb], dim=1)  # (B, 2H + D_dom)

        logits = self.fc(h)  # (B, 3)
        return logits

    def train_model(self, train_loader, val_loader=None, epochs=10, lr=2e-4, wd=1e-5,
                    patience=2, min_epochs=3):
        """
        train_loader : DataLoader de entrenamiento
        val_loader   : DataLoader de validación (si None, no hay early stopping)
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        print("Entrenando en:", next(self.parameters()).device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for ep in range(1, epochs + 1):

            # ============================
            # 1) ENTRENAMIENTO
            # ============================
            self.train()
            total_loss = 0
            total = 0
            correct = 0

            pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False)

            for x, lengths, domain_ids, y in pbar:
                x = x.to(device)
                lengths = lengths.to(device)
                domain_ids = domain_ids.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits = self(x, lengths, domain_ids)
                loss = criterion(logits, y)

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total += x.size(0)
                correct += (logits.argmax(1) == y).sum().item()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss = total_loss / total
            train_acc = correct / total

            # ============================
            # 2) VALIDACIÓN
            # ============================
            if val_loader is not None:
                self.eval()
                val_loss = 0
                val_total = 0
                val_correct = 0

                with torch.no_grad():
                    for x, lengths, domain_ids, y in val_loader:
                        x = x.to(device)
                        lengths = lengths.to(device)
                        domain_ids = domain_ids.to(device)
                        y = y.to(device)

                        logits = self(x, lengths, domain_ids)
                        loss = criterion(logits, y)

                        val_loss += loss.item() * x.size(0)
                        val_total += x.size(0)
                        val_correct += (logits.argmax(1) == y).sum().item()

                val_loss /= val_total
                val_acc = val_correct / val_total

                print(f"Epoch {ep:02d} | train_loss={train_loss:.4f} "
                      f"| train_acc={train_acc:.3f} "
                      f"| val_loss={val_loss:.4f} "
                      f"| val_acc={val_acc:.3f}")

                # ============================
                # 3) EARLY STOPPING
                # ============================
                if ep >= min_epochs:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = self.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f" Early stopping activado en epoch {ep}")
                        print(f" Recuperando mejor modelo (val_loss={best_val_loss:.4f})")
                        self.load_state_dict(best_state)
                        break

            else:
                # (si no hay validation loader)
                print(f"Epoch {ep:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.3f}")

        print("Entrenamiento completo.")



