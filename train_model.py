import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.regularizers import l2
import joblib
import json

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ---------------- 0. Configuration and Setup ----------------
class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.15
    VAL_SIZE = 0.1765  # Results in 70/15/15 split
    BATCH_SIZE = 64
    EPOCHS = 200
    PATIENCE = 15
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.3
    L2_REG = 1e-4

# Create directories
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------- 1. Data Loading and Exploration ----------------
def load_and_explore_data(filepath):
    """Load data and perform basic exploration"""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"\nTarget statistics:")
    print(df[['Cl', 'Cd']].describe())
    
    # Check for outliers using IQR method
    Q1 = df[['Cl', 'Cd']].quantile(0.25)
    Q3 = df[['Cl', 'Cd']].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[['Cl', 'Cd']] < (Q1 - 1.5 * IQR)) | 
                (df[['Cl', 'Cd']] > (Q3 + 1.5 * IQR))).sum()
    print(f"\nOutliers detected: Cl={outliers['Cl']}, Cd={outliers['Cd']}")
    
    return df

df = load_and_explore_data("original.csv")

# ---------------- 2. Feature Engineering ----------------
def create_features(df):
    """Create additional features from existing ones"""
    features = df[["AoA"] + [f"CST Coeff {i}" for i in range(1, 9)]].copy()
    
    # Add polynomial features for AoA
    features['AoA_squared'] = features['AoA'] ** 2
    features['AoA_cubed'] = features['AoA'] ** 3
    
    # Add interaction terms (example: AoA with first few CST coefficients)
    features['AoA_CST1'] = features['AoA'] * features['CST Coeff 1']
    features['AoA_CST2'] = features['AoA'] * features['CST Coeff 2']
    
    return features.values

X = create_features(df)
y = df[["Cl", "Cd"]].values

print(f"Feature matrix shape: {X.shape}")

# ---------------- 3. Data Splitting with Stratification ----------------
# Create bins for stratified sampling based on Cl values
cl_bins = pd.cut(df['Cl'], bins=5, labels=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE,
    stratify=cl_bins
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=Config.VAL_SIZE, random_state=Config.RANDOM_STATE
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ---------------- 4. Advanced Preprocessing ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Optional: Scale targets for better training stability
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled = y_scaler.transform(y_val)
y_test_scaled = y_scaler.transform(y_test)

# ---------------- 5. Enhanced Model Architecture ----------------
def build_advanced_model(input_dim, output_dim=2):
    """Build an advanced neural network with regularization"""
    model = models.Sequential([
        layers.Dense(512, activation="relu", input_shape=(input_dim,),
                    kernel_regularizer=l2(Config.L2_REG)),
        layers.BatchNormalization(),
        layers.Dropout(Config.DROPOUT_RATE),
        
        layers.Dense(256, activation="relu",
                    kernel_regularizer=l2(Config.L2_REG)),
        layers.BatchNormalization(),
        layers.Dropout(Config.DROPOUT_RATE),
        
        layers.Dense(128, activation="relu",
                    kernel_regularizer=l2(Config.L2_REG)),
        layers.BatchNormalization(),
        layers.Dropout(Config.DROPOUT_RATE/2),
        
        layers.Dense(64, activation="relu",
                    kernel_regularizer=l2(Config.L2_REG)),
        layers.BatchNormalization(),
        layers.Dropout(Config.DROPOUT_RATE/2),
        
        layers.Dense(32, activation="relu"),
        layers.Dense(output_dim, activation="linear")
    ])
    return model

model = build_advanced_model(X_train_scaled.shape[1])

# Custom optimizer with learning rate scheduling
optimizer = optimizers.Adam(learning_rate=Config.LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae", "mse"]
)

print(f"Model parameters: {model.count_params():,}")

# ---------------- 6. Advanced Training with Callbacks ----------------
callbacks_list = [
    callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=Config.PATIENCE, 
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        "models/best_model.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# Train model
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1
)

# ---------------- 7. Comprehensive Evaluation ----------------
# Load best model and make predictions
model.load_weights("models/best_model.h5")
y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

# Calculate metrics
rmse_cl = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
rmse_cd = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
mae_cl = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
mae_cd = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
r2_cl = r2_score(y_test[:, 0], y_pred[:, 0])
r2_cd = r2_score(y_test[:, 1], y_pred[:, 1])

# Overall metrics
rmse_overall = np.sqrt(mean_squared_error(y_test, y_pred))
mae_overall = mean_absolute_error(y_test, y_pred)
r2_overall = r2_score(y_test, y_pred)

print("="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Overall -> RMSE: {rmse_overall:.6f}, MAE: {mae_overall:.6f}, R²: {r2_overall:.6f}")
print(f"Cl      -> RMSE: {rmse_cl:.6f}, MAE: {mae_cl:.6f}, R²: {r2_cl:.6f}")
print(f"Cd      -> RMSE: {rmse_cd:.6f}, MAE: {mae_cd:.6f}, R²: {r2_cd:.6f}")

# Save metrics
metrics = {
    'overall': {'rmse': float(rmse_overall), 'mae': float(mae_overall), 'r2': float(r2_overall)},
    'cl': {'rmse': float(rmse_cl), 'mae': float(mae_cl), 'r2': float(r2_cl)},
    'cd': {'rmse': float(rmse_cd), 'mae': float(mae_cd), 'r2': float(r2_cd)}
}

with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ---------------- 8. Enhanced Visualizations ----------------
# Set up the plot style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# 1. Training curves with more detail
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss curves
axes[0,0].plot(history.history["loss"], label="Training Loss", linewidth=2)
axes[0,0].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
axes[0,0].set_xlabel("Epochs")
axes[0,0].set_ylabel("MSE Loss")
axes[0,0].legend()
axes[0,0].set_title("Training and Validation Loss")
axes[0,0].grid(True, alpha=0.3)

# MAE curves
axes[0,1].plot(history.history["mae"], label="Training MAE", linewidth=2)
axes[0,1].plot(history.history["val_mae"], label="Validation MAE", linewidth=2)
axes[0,1].set_xlabel("Epochs")
axes[0,1].set_ylabel("Mean Absolute Error")
axes[0,1].legend()
axes[0,1].set_title("Training and Validation MAE")
axes[0,1].grid(True, alpha=0.3)

# Learning rate (if available)
if hasattr(history.history, 'lr'):
    axes[1,0].plot(history.history["lr"], linewidth=2)
    axes[1,0].set_xlabel("Epochs")
    axes[1,0].set_ylabel("Learning Rate")
    axes[1,0].set_title("Learning Rate Schedule")
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)
else:
    axes[1,0].text(0.5, 0.5, 'Learning Rate\nHistory Not Available', 
                   ha='center', va='center', transform=axes[1,0].transAxes)

# Model architecture visualization (simplified)
layer_names = [layer.__class__.__name__ for layer in model.layers if 'Dense' in layer.__class__.__name__]
layer_sizes = [layer.units for layer in model.layers if hasattr(layer, 'units')]

axes[1,1].barh(range(len(layer_sizes)), layer_sizes, color='skyblue', alpha=0.7)
axes[1,1].set_yticks(range(len(layer_sizes)))
axes[1,1].set_yticklabels([f'Layer {i+1}\n({size})' for i, size in enumerate(layer_sizes)])
axes[1,1].set_xlabel("Number of Neurons")
axes[1,1].set_title("Model Architecture")
axes[1,1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("plots/training_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Prediction accuracy plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Cl predictions
axes[0].scatter(y_test[:,0], y_pred[:,0], alpha=0.6, s=20)
min_cl, max_cl = y_test[:,0].min(), y_test[:,0].max()
axes[0].plot([min_cl, max_cl], [min_cl, max_cl], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel("True Cl")
axes[0].set_ylabel("Predicted Cl")
axes[0].set_title(f"Cl Prediction (R² = {r2_cl:.4f})")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cd predictions
axes[1].scatter(y_test[:,1], y_pred[:,1], alpha=0.6, s=20, color="green")
min_cd, max_cd = y_test[:,1].min(), y_test[:,1].max()
axes[1].plot([min_cd, max_cd], [min_cd, max_cd], 'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel("True Cd")
axes[1].set_ylabel("Predicted Cd")
axes[1].set_title(f"Cd Prediction (R² = {r2_cd:.4f})")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/prediction_accuracy.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Comprehensive residual analysis
residuals = y_test - y_pred
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Residual histograms
axes[0,0].hist(residuals[:,0], bins=50, alpha=0.7, density=True, label="Cl Residuals")
axes[0,0].axvline(0, color='red', linestyle='--', alpha=0.8)
axes[0,0].set_xlabel("Residuals")
axes[0,0].set_ylabel("Density")
axes[0,0].set_title("Cl Residual Distribution")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].hist(residuals[:,1], bins=50, alpha=0.7, density=True, color="green", label="Cd Residuals")
axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.8)
axes[0,1].set_xlabel("Residuals")
axes[0,1].set_ylabel("Density")
axes[0,1].set_title("Cd Residual Distribution")
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Residuals vs predictions
axes[1,0].scatter(y_pred[:,0], residuals[:,0], alpha=0.6, s=20)
axes[1,0].axhline(0, color='red', linestyle='--', alpha=0.8)
axes[1,0].set_xlabel("Predicted Cl")
axes[1,0].set_ylabel("Residuals")
axes[1,0].set_title("Cl Residuals vs Predictions")
axes[1,0].grid(True, alpha=0.3)

axes[1,1].scatter(y_pred[:,1], residuals[:,1], alpha=0.6, s=20, color="green")
axes[1,1].axhline(0, color='red', linestyle='--', alpha=0.8)
axes[1,1].set_xlabel("Predicted Cd")
axes[1,1].set_ylabel("Residuals")
axes[1,1].set_title("Cd Residuals vs Predictions")
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/residual_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Error distribution by angle of attack (if AoA is first feature)
aoa_test = X_test[:, 0]  # Assuming AoA is the first feature
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Bin AoA values and calculate mean errors
aoa_bins = np.linspace(aoa_test.min(), aoa_test.max(), 20)
bin_centers = (aoa_bins[:-1] + aoa_bins[1:]) / 2
cl_errors = []
cd_errors = []

for i in range(len(aoa_bins)-1):
    mask = (aoa_test >= aoa_bins[i]) & (aoa_test < aoa_bins[i+1])
    if mask.sum() > 0:
        cl_errors.append(np.abs(residuals[mask, 0]).mean())
        cd_errors.append(np.abs(residuals[mask, 1]).mean())
    else:
        cl_errors.append(0)
        cd_errors.append(0)

axes[0].plot(bin_centers, cl_errors, 'o-', linewidth=2, markersize=6)
axes[0].set_xlabel("Angle of Attack")
axes[0].set_ylabel("Mean Absolute Error")
axes[0].set_title("Cl Error vs Angle of Attack")
axes[0].grid(True, alpha=0.3)

axes[1].plot(bin_centers, cd_errors, 'o-', linewidth=2, markersize=6, color='green')
axes[1].set_xlabel("Angle of Attack")
axes[1].set_ylabel("Mean Absolute Error")
axes[1].set_title("Cd Error vs Angle of Attack")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/error_vs_aoa.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Enhanced visualizations saved to 'plots/' directory")

# ---------------- 9. Model and Preprocessor Saving ----------------
# Save the final model (unscaled version)
model_unscaled = build_advanced_model(X_train.shape[1])
model_unscaled.set_weights(model.get_weights())
model_unscaled.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])

model_unscaled.save("models/nn_model.h5")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(y_scaler, "models/y_scaler.pkl")

# Save training history
with open('models/training_history.json', 'w') as f:
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    json.dump(history_dict, f, indent=2)

# Save configuration
config_dict = {attr: getattr(Config, attr) for attr in dir(Config) if not attr.startswith('_')}
with open('models/config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

print("✅ Enhanced model saved as 'models/nn_model.h5'")
print("✅ Scalers saved as 'models/scaler.pkl' and 'models/y_scaler.pkl'")
print("✅ Training history and configuration saved")
print("✅ Evaluation metrics saved as 'models/metrics.json'")

# ---------------- 10. Model Summary and Feature Importance ----------------
print("\n" + "="*50)
print("MODEL SUMMARY")
print("="*50)
model.summary()

# Simple feature importance using permutation (basic implementation)
def calculate_feature_importance(model, X_test, y_test, scaler, y_scaler, feature_names):
    """Calculate feature importance using permutation method"""
    baseline_pred = model.predict(scaler.transform(X_test))
    baseline_pred = y_scaler.inverse_transform(baseline_pred)
    baseline_error = np.sqrt(mean_squared_error(y_test, baseline_pred))
    
    importances = []
    for i in range(X_test.shape[1]):
        X_permuted = X_test.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        
        perm_pred = model.predict(scaler.transform(X_permuted))
        perm_pred = y_scaler.inverse_transform(perm_pred)
        perm_error = np.sqrt(mean_squared_error(y_test, perm_pred))
        
        importance = perm_error - baseline_error
        importances.append(importance)
    
    return np.array(importances)

# Define feature names
feature_names = ["AoA"] + [f"CST Coeff {i}" for i in range(1, 9)] + \
                ["AoA²", "AoA³", "AoA×CST1", "AoA×CST2"]

importances = calculate_feature_importance(model, X_test, y_test, scaler, y_scaler, feature_names)

# Plot feature importance
plt.figure(figsize=(12, 8))
sorted_idx = np.argsort(np.abs(importances))
pos = np.arange(sorted_idx.shape[0]) + 0.5

plt.barh(pos, importances[sorted_idx], align='center')
plt.yticks(pos, [feature_names[i] for i in sorted_idx])
plt.xlabel('Feature Importance (MSE Increase)')
plt.title('Feature Importance via Permutation')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Feature importance analysis completed")
print("\n🎯 All enhanced analyses completed successfully!")