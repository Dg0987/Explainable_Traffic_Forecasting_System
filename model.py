"""
LSTM + Transformer Hybrid Model for Traffic Speed Prediction
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import time
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import xgboost as xgb
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


class Time2Vec(nn.Module):
    """Time2Vec encoding for Transformer layers (replaces Sinusoidal)."""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super(Time2Vec, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.omega_linear = nn.Parameter(torch.randn(1))
        self.phi_linear = nn.Parameter(torch.randn(1))
        
        self.omega_periodic = nn.Parameter(torch.randn(d_model - 1))
        self.phi_periodic = nn.Parameter(torch.randn(d_model - 1))

    def forward(self, x):
        """x shape: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        
        tau = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)
        
        linear_term = self.omega_linear * tau + self.phi_linear
        periodic_term = torch.sin(self.omega_periodic * tau + self.phi_periodic)
        
        t2v = torch.cat([linear_term, periodic_term], dim=1).unsqueeze(0)
        
        x = x + t2v
        return self.dropout(x)


class TransformerEncoderLayerWithWeights(nn.Module):
    """
    Custom Transformer Encoder layer that captures and exposes
    multi-head self-attention weights for explainability.
    """

    def __init__(self, d_model, nhead, dim_feedforward=200, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # Storage for attention weights (set during forward)
        self.attn_weights = None

    def forward(self, src, need_weights=False):
        # Self-attention with optional weight capture
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            need_weights=need_weights,
            average_attn_weights=True
        )
        if need_weights:
            self.attn_weights = attn_weights  # (batch, seq_len, seq_len)

        # Add & norm
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feed-forward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class TrafficLSTM(nn.Module):
    """
    LSTM + Transformer Encoder Hybrid for Traffic Speed Prediction

    Architecture:
    - Input: 12 time steps x 4 features (speed, volume, hour, day_of_week)
    - LSTM layers: 2 layers with 50 hidden units (sequential feature extraction)
    - Positional Encoding: sinusoidal encoding on LSTM outputs
    - Transformer Encoder: 2 layers, 2-head self-attention (temporal refinement)
    - Output: 1 value (predicted speed for next hour)
    """

    def __init__(self, input_size=8, hidden_size=64, num_layers=2,
                 seq_length=12, n_heads=2, n_transformer_layers=2,
                 dim_feedforward=256, dropout=0.15):
        super(TrafficLSTM, self).__init__()

        assert hidden_size % n_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.n_transformer_layers = n_transformer_layers
        self.dim_feedforward = dim_feedforward

        # Stage 1: LSTM layers (sequential feature extraction)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Stage 2: Time2Vec Encoding
        self.pos_encoder = Time2Vec(
            d_model=hidden_size,
            max_len=seq_length + 10,
            dropout=dropout
        )

        # Stage 3: Transformer Encoder layers (with weight capture)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerWithWeights(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(n_transformer_layers)
        ])

        # Stage 4: Output projection
        self.fc = nn.Linear(hidden_size, 1)

        # Scaler for normalizing data
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def forward(self, x, need_weights=False):
        """
        Forward pass through LSTM + Transformer hybrid.

        Args:
            x: (batch, seq_length, input_size) tensor
            need_weights: If True, capture attention weights in transformer layers
        """
        # Stage 1: LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Stage 2: Positional encoding
        transformer_input = self.pos_encoder(lstm_out)

        # Stage 3: Transformer Encoder
        out = transformer_input
        for layer in self.transformer_layers:
            out = layer(out, need_weights=need_weights)

        # Stage 4: Last time step → FC
        last_output = out[:, -1, :]
        prediction = self.fc(last_output)

        return prediction

    def train_model(self, df, epochs=80, lr=0.002, batch_size=512, verbose=True, progress_callback=None):
        """
        Train the model on traffic data with train/validation split and OneCycleLR.

        Returns:
            Dictionary with training history and accuracy metrics
        """
        # Prepare features
        features = ['speed', 'volume', 'hour', 'day_of_week', 'temp', 'rain_1h', 'clouds_all', 'is_holiday']
        data = df[features].values.astype(np.float32)

        # Fit and transform scaler
        data_scaled = self.scaler.fit_transform(data)
        self.is_fitted = True

        # Create sequences
        X, y = self._create_sequences(data_scaled)

        if len(X) == 0:
            raise ValueError(f"Not enough data. Need at least {self.seq_length + 1} rows.")

        # Train/validation split (85/15)
        split = int(len(X) * 0.85)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # DataLoaders
        train_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )

        val_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False
        )

        # Setup training with OneCycleLR (converges much faster)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs,
            steps_per_epoch=steps_per_epoch, pct_start=0.3,
            anneal_strategy='cos', div_factor=10, final_div_factor=100
        )
        criterion = nn.HuberLoss(delta=1.0)  # More robust than MSE

        history = {'losses': [], 'val_losses': [], 'samples': len(X_train)}
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            # --- Train ---
            self.train()
            epoch_loss = 0.0
            num_batches = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            history['losses'].append(avg_loss)

            # --- Validate ---
            self.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            history['val_losses'].append(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Report progress with rich details + release GIL for Flask
            if progress_callback:
                progress_callback({
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'train_loss': round(avg_loss, 6),
                    'val_loss': round(avg_val_loss, 6),
                    'lr': optimizer.param_groups[0]['lr'],
                    'best_val_loss': round(best_val_loss, 6),
                    'patience': patience_counter,
                    'percent': int((epoch + 1) / epochs * 95)
                })
            time.sleep(0.01)  # Release GIL so Flask can respond to polls

            if verbose and (epoch + 1) % 10 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Val: {avg_val_loss:.6f}, LR: {lr_now:.1e}')

            # Early stopping
            if patience_counter >= 20:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Restore best model
        if best_state is not None:
            self.load_state_dict(best_state)

        # --- Compute accuracy metrics on validation set ---
        self.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = self(X_batch)
                all_preds.append(preds.numpy())
                all_true.append(y_batch.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_true = np.concatenate(all_true, axis=0)

        # Inverse-transform to get actual mph values
        dummy_pred = np.zeros((len(all_preds), self.input_size))
        dummy_pred[:, 0] = all_preds[:, 0]
        preds_mph = self.scaler.inverse_transform(dummy_pred)[:, 0]

        dummy_true = np.zeros((len(all_true), self.input_size))
        dummy_true[:, 0] = all_true[:, 0]
        true_mph = self.scaler.inverse_transform(dummy_true)[:, 0]

        mae = float(np.mean(np.abs(preds_mph - true_mph)))
        rmse = float(np.sqrt(np.mean((preds_mph - true_mph) ** 2)))

        # Accuracy: % of predictions within 5 mph of actual
        within_5 = float(np.mean(np.abs(preds_mph - true_mph) <= 5) * 100)
        # Accuracy: % within 3 mph
        within_3 = float(np.mean(np.abs(preds_mph - true_mph) <= 3) * 100)

        # R² score
        ss_res = np.sum((preds_mph - true_mph) ** 2)
        ss_tot = np.sum((true_mph - np.mean(true_mph)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))

        history['metrics'] = {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 4),
            'accuracy_within_5mph': round(within_5, 1),
            'accuracy_within_3mph': round(within_3, 1),
            'val_samples': len(X_val)
        }

        if verbose:
            print(f"\n--- Validation Metrics ---")
            print(f"  MAE:  {mae:.2f} mph")
            print(f"  RMSE: {rmse:.2f} mph")
            print(f"  R²:   {r2:.4f}")
            print(f"  Accuracy (within 3 mph): {within_3:.1f}%")
            print(f"  Accuracy (within 5 mph): {within_5:.1f}%")

        return history

    def fit(self, df, **kwargs):
        """Alias for train_model - trains the model on data"""
        return self.train_model(df, **kwargs)

    def predict(self, data):
        """
        Make prediction for next hour's traffic speed

        Args:
            data: numpy array of shape (seq_length, 4) or (1, seq_length, 4)
                  Features: [speed, volume, hour, day_of_week]

        Returns:
            Predicted speed in mph
        """
        self.eval()

        # Ensure data is the right shape
        if len(data.shape) == 2:
            data = data.reshape(1, -1, self.input_size)

        # Handle case where we have more or fewer time steps
        if data.shape[1] < self.seq_length:
            # Pad with the first value
            padding = np.tile(data[:, 0:1, :], (1, self.seq_length - data.shape[1], 1))
            data = np.concatenate([padding, data], axis=1)
        elif data.shape[1] > self.seq_length:
            # Take the last seq_length steps
            data = data[:, -self.seq_length:, :]

        # Scale the input
        data_flat = data.reshape(-1, self.input_size)
        data_scaled = self.scaler.transform(data_flat)
        data_scaled = data_scaled.reshape(1, self.seq_length, self.input_size)

        # Convert to tensor and predict
        X_tensor = torch.FloatTensor(data_scaled)

        with torch.no_grad():
            prediction_scaled = self(X_tensor).numpy()[0, 0]

        # Inverse transform to get actual speed
        dummy = np.zeros((1, self.input_size))
        dummy[0, 0] = prediction_scaled
        prediction = self.scaler.inverse_transform(dummy)[0, 0]

        return float(prediction)

    def get_attention_weights(self, data):
        """
        Get temporal attention weights showing which past time steps
        the model attends to most when making the prediction.

        Args:
            data: numpy array of shape (seq_length, 4) or (1, seq_length, 4)

        Returns:
            dict with temporal_attention (list of floats) and layer_weights
        """
        self.eval()

        if len(data.shape) == 2:
            data = data.reshape(1, -1, self.input_size)

        if data.shape[1] < self.seq_length:
            padding = np.tile(data[:, 0:1, :], (1, self.seq_length - data.shape[1], 1))
            data = np.concatenate([padding, data], axis=1)
        elif data.shape[1] > self.seq_length:
            data = data[:, -self.seq_length:, :]

        # Scale input
        data_flat = data.reshape(-1, self.input_size)
        data_scaled = self.scaler.transform(data_flat)
        data_scaled = data_scaled.reshape(1, self.seq_length, self.input_size)

        X_tensor = torch.FloatTensor(data_scaled)

        with torch.no_grad():
            _ = self(X_tensor, need_weights=True)

        # Extract attention weights from each transformer layer
        layer_weights = []
        for layer in self.transformer_layers:
            if layer.attn_weights is not None:
                layer_weights.append(layer.attn_weights[0].numpy())

        # Compute temporal attention: how the last time step attends to all positions
        if layer_weights:
            last_step_attentions = [w[-1, :] for w in layer_weights]
            avg_attention = np.mean(last_step_attentions, axis=0)
            avg_attention = avg_attention / (avg_attention.sum() + 1e-8)
            temporal_attention = avg_attention.tolist()
        else:
            temporal_attention = [1.0 / self.seq_length] * self.seq_length

        return {
            'temporal_attention': [round(w, 4) for w in temporal_attention],
            'layer_weights': [w.tolist() for w in layer_weights]
        }

    def _create_sequences(self, data):
        """Create sliding window sequences for training"""
        X, y = [], []

        for i in range(self.seq_length, len(data)):
            X.append(data[i-self.seq_length:i])
            y.append([data[i, 0]])  # Predict speed (first column)

        return np.array(X), np.array(y)

    def save(self, filepath):
        """Save model weights and scaler"""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'seq_length': self.seq_length,
            'n_heads': self.n_heads,
            'n_transformer_layers': self.n_transformer_layers,
            'dim_feedforward': self.dim_feedforward,
            'model_version': 'lstm_transformer_v1'
        }, filepath)

        scaler_path = filepath.replace('.pth', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model weights and scaler"""
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

        if 'model_version' not in checkpoint:
            print("Warning: Loading legacy LSTM checkpoint. Retrain recommended.")
            current_state = self.state_dict()
            legacy_state = {k: v for k, v in checkpoint['model_state_dict'].items()
                           if k in current_state and current_state[k].shape == v.shape}
            current_state.update(legacy_state)
            self.load_state_dict(current_state)
        else:
            self.load_state_dict(checkpoint['model_state_dict'])

        scaler_path = filepath.replace('.pth', '_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_fitted = True

        self.eval()
        print(f"Model loaded from {filepath}")


class XGBoostPredictor:
    """XGBoost model for traffic speed prediction to serve as a baseline."""
    def __init__(self, seq_length=12, input_size=8):
        self.seq_length = seq_length
        self.input_size = input_size
        self.model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(self.seq_length, len(data)):
            window = data[i-self.seq_length:i].flatten()
            X.append(window)
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def train_model(self, df, **kwargs):
        features = ['speed', 'volume', 'hour', 'day_of_week', 'temp', 'rain_1h', 'clouds_all', 'is_holiday']
        data = df[features].values.astype(np.float32)
        
        data_scaled = self.scaler.fit_transform(data)
        self.is_fitted = True
        
        X, y = self._create_sequences(data_scaled)
        if len(X) == 0:
            raise ValueError(f"Not enough data. Need at least {self.seq_length + 1} rows.")
            
        split = int(len(X) * 0.85)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds_scaled = self.model.predict(X_val)
        
        dummy_pred = np.zeros((len(preds_scaled), self.input_size))
        dummy_pred[:, 0] = preds_scaled
        preds_mph = self.scaler.inverse_transform(dummy_pred)[:, 0]
        
        dummy_true = np.zeros((len(y_val), self.input_size))
        dummy_true[:, 0] = y_val
        true_mph = self.scaler.inverse_transform(dummy_true)[:, 0]
        
        mae = float(np.mean(np.abs(preds_mph - true_mph)))
        rmse = float(np.sqrt(np.mean((preds_mph - true_mph) ** 2)))
        ss_res = np.sum((preds_mph - true_mph) ** 2)
        ss_tot = np.sum((true_mph - np.mean(true_mph)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))
        
        return {
            'metrics': {
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'r2': round(r2, 4),
                'val_samples': len(X_val)
            },
            'samples': len(X_train),
            'losses': [float(rmse)]  # Mock loss for API compatibility
        }
        
    def predict(self, data):
        if len(data.shape) == 2:
            data = data.reshape(1, -1, self.input_size)
            
        if data.shape[1] < self.seq_length:
            padding = np.tile(data[:, 0:1, :], (1, self.seq_length - data.shape[1], 1))
            data = np.concatenate([padding, data], axis=1)
        elif data.shape[1] > self.seq_length:
            data = data[:, -self.seq_length:, :]
            
        data_flat = data.reshape(-1, self.input_size)
        data_scaled = self.scaler.transform(data_flat)
        data_window_flat = data_scaled.flatten().reshape(1, -1)
        
        prediction_scaled = self.model.predict(data_window_flat)[0]
        
        dummy = np.zeros((1, self.input_size))
        dummy[0, 0] = prediction_scaled
        prediction = self.scaler.inverse_transform(dummy)[0, 0]
        
        return float(prediction)
        
    def save(self, filepath):
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save_model(filepath)
        scaler_path = filepath.replace('.json', '_scaler.pkl').replace('.pth', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
    def load(self, filepath):
        self.model.load_model(filepath)
        scaler_path = filepath.replace('.json', '_scaler.pkl').replace('.pth', '_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_fitted = True


class TFTPredictor:
    """Temporal Fusion Transformer optimized for 2GB VRAM."""
    def __init__(self, seq_length=12, input_size=8):
        self.seq_length = seq_length
        self.input_size = input_size
        self.model = None
        self.dataset_parameters = None
        self.is_fitted = False
        
    def _array_to_df(self, data_array):
        if len(data_array.shape) == 3:
            data_array = data_array[0]
            
        df = pd.DataFrame(data_array, columns=['speed', 'volume', 'hour', 'day_of_week', 'temp', 'rain_1h', 'clouds_all', 'is_holiday'])
        df['time_idx'] = np.arange(len(df))
        df['group'] = '0'
        
        df['hour'] = df['hour'].astype(float)
        df['day_of_week'] = df['day_of_week'].astype(float)
        df['is_holiday'] = df['is_holiday'].astype(float)
        return df

    def train_model(self, df, **kwargs):
        features = ['speed', 'volume', 'hour', 'day_of_week', 'temp', 'rain_1h', 'clouds_all', 'is_holiday']
        data_array = df[features].values
        
        train_df = self._array_to_df(data_array)
        
        max_encoder_length = self.seq_length
        max_prediction_length = 1

        training_cutoff = train_df["time_idx"].max() - int(len(train_df) * 0.15)
        
        training = TimeSeriesDataSet(
            train_df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="speed",
            group_ids=["group"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["group"],
            static_reals=[],
            time_varying_known_categoricals=[],
            time_varying_known_reals=["hour", "day_of_week", "is_holiday"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=["speed", "volume", "temp", "rain_1h", "clouds_all"],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        validation = TimeSeriesDataSet.from_dataset(training, train_df, predict=True, stop_randomization=True)
        
        train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=32, num_workers=0)
        
        self.dataset_parameters = training.get_parameters()
        
        self.model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="auto",
            devices=1,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False
        )
        
        trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        self.is_fitted = True
        
        return {
            'metrics': {
                'message': 'TFT trained successfully'
            },
            'samples': len(train_df),
            'losses': [0.0]
        }

    def predict(self, data):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        if len(data.shape) == 2:
            pass
        elif len(data.shape) == 3:
            data = data[0]
            
        if data.shape[0] < self.seq_length:
            padding = np.tile(data[0:1, :], (self.seq_length - data.shape[0], 1))
            data = np.concatenate([padding, data], axis=0)
        elif data.shape[0] > self.seq_length:
            data = data[-self.seq_length:, :]
            
        df = self._array_to_df(data)
        
        future_row = df.iloc[-1:].copy()
        future_row['time_idx'] += 1
        df = pd.concat([df, future_row], ignore_index=True)
        
        dataset = TimeSeriesDataSet.from_parameters(self.dataset_parameters, df)
        dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        
        predictions = self.model.predict(dataloader)
        return float(predictions[0][0])
        
    def save(self, filepath):
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'dataset_parameters': self.dataset_parameters,
            'input_size': self.input_size,
            'seq_length': self.seq_length
        }, filepath)
        
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        self.dataset_parameters = checkpoint['dataset_parameters']
        
        df = pd.DataFrame(np.zeros((15, 8)), columns=['speed', 'volume', 'hour', 'day_of_week', 'temp', 'rain_1h', 'clouds_all', 'is_holiday'])
        df['time_idx'] = np.arange(15)
        df['group'] = '0'
        dummy_ds = TimeSeriesDataSet.from_parameters(self.dataset_parameters, df)
        
        self.model = TemporalFusionTransformer.from_dataset(
            dummy_ds,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.is_fitted = True

