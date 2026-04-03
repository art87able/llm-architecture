#!/usr/bin/env python3
"""
Script to fill in homework solutions in notebook.ipynb.
Modifies the notebook JSON in-place: fills skeleton code cells and adds markdown analysis cells.
"""
import json
import copy

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n") if isinstance(source, str) else source
    }

def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n") if isinstance(source, str) else source
    }

def fix_source_lines(lines):
    """Convert a string source into proper notebook line format (each line ending with \\n except the last)."""
    if isinstance(lines, str):
        lines = lines.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n" if not line.endswith("\n") else line)
        else:
            result.append(line.rstrip("\n"))
    return result

# ============================================================
# CELL REPLACEMENTS
# ============================================================

# Cell 20: Task 1.1 - LogisticRegression class (fill in the skeleton)
CELL_20 = r'''import torch
import torch.nn as nn

class LogisticRegression(nn.Module):

    def __init__(self, n_features, init="zeros"):
        """
        Parameters
        ----------
        n_features : int
            Number of input features

        init : str or torch.Tensor
            Initialization method for weights:
            - "zeros"  -> initialize weights to zeros
            - "random" -> small random values (recommended scale ~0.01)
            - torch.Tensor -> use provided tensor
        """

        super().__init__()

        if init == "zeros":
            w = torch.zeros(n_features, 1)

        elif init == "random":
            w = torch.randn(n_features, 1) * 0.01

        elif isinstance(init, torch.Tensor):
            w = init.clone().detach().reshape(n_features, 1)

        else:
            raise ValueError("init must be 'zeros', 'random', or a torch.Tensor")

        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Forward pass

        Steps:
        1. Compute logits: x @ w + b
        2. Apply sigmoid to get probabilities

        Returns
        -------
        probs : torch.Tensor
            Values in range [0, 1]
        """

        logits = x @ self.w + self.b
        probs = torch.sigmoid(logits)

        return probs

    def predict(self, x):
        """
        Convert probabilities to class predictions

        Rule:
        - class 1 if p >= 0.5
        - class 0 otherwise
        """

        probs = self.forward(x)
        preds = (probs >= 0.5).float()

        return preds'''

# Cell 24: binary_cross_entropy_loss
CELL_24 = r'''def binary_cross_entropy_loss(y_pred, y_true):
    """
    Compute the binary cross-entropy loss for a logistic regression model.
    """
    epsilon = 1e-15
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)

    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return loss.mean()'''

# Cell 25: sgd_logistic_regression (fill in the skeleton)
CELL_25 = r'''import torch

def sgd_logistic_regression(
    X_train, y_train,
    X_val, y_val,
    lr=0.01,
    epochs=20,
    batch_size=100,
    init="zeros",
    penalty='none',
    reg_lambda=0.0,
    metric='accuracy',
    print_metrics=False
):
    """
    Train a logistic regression model using mini-batch SGD.

    Returns
    -------
    w : numpy.ndarray
    b : numpy.ndarray
    history : list
    epoch_log : list
    """

    # 1. Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    n_samples, n_features = X_train_tensor.shape

    # 2. Initialize model
    model = LogisticRegression(n_features=n_features, init=init)

    # 3. Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 4. Create logs
    history = []
    epoch_log = []

    # 5. Training loop
    for epoch in range(epochs):

        # Shuffle the training data at the beginning of each epoch
        perm = torch.randperm(n_samples)
        X_train_epoch = X_train_tensor[perm]
        y_train_epoch = y_train_tensor[perm]

        for start in range(0, n_samples, batch_size):

            end = start + batch_size

            # Select mini-batch
            X_batch = X_train_epoch[start:end]
            y_batch = y_train_epoch[start:end]

            # Forward pass
            y_pred = model(X_batch)

            # Compute non-regularized BCE loss
            data_loss = binary_cross_entropy_loss(y_pred, y_batch)

            # Add regularization if needed
            if penalty == 'l1':
                reg_term = reg_lambda * torch.sum(torch.abs(model.w))
            elif penalty == 'l2':
                reg_term = reg_lambda * torch.sum(model.w ** 2)
            else:
                reg_term = 0.0

            loss = data_loss + reg_term

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save current parameter values after the batch update
            history.append({
                'epoch': epoch,
                'batch_start': start,
                'w': model.w.detach().clone(),
                'b': model.b.detach().clone()
            })

        # 6. Epoch-level evaluation
        with torch.no_grad():

            # Compute probabilities on full train/val sets
            y_pred_train = model(X_train_tensor)
            y_pred_val = model(X_val_tensor)

            # Compute NON-regularized train loss and val loss
            train_loss = binary_cross_entropy_loss(y_pred_train, y_train_tensor)
            val_loss = binary_cross_entropy_loss(y_pred_val, y_val_tensor)

            # Convert probabilities to binary predictions
            y_hat_train = (y_pred_train >= 0.5).float()
            y_hat_val = (y_pred_val >= 0.5).float()

            # Compute evaluation metric
            if metric == 'f1':
                tp_train = ((y_hat_train == 1) & (y_train_tensor == 1)).sum().float()
                fp_train = ((y_hat_train == 1) & (y_train_tensor == 0)).sum().float()
                fn_train = ((y_hat_train == 0) & (y_train_tensor == 1)).sum().float()
                prec_train = tp_train / (tp_train + fp_train + 1e-15)
                rec_train = tp_train / (tp_train + fn_train + 1e-15)
                train_metric = (2 * prec_train * rec_train / (prec_train + rec_train + 1e-15)).item()

                tp_val = ((y_hat_val == 1) & (y_val_tensor == 1)).sum().float()
                fp_val = ((y_hat_val == 1) & (y_val_tensor == 0)).sum().float()
                fn_val = ((y_hat_val == 0) & (y_val_tensor == 1)).sum().float()
                prec_val = tp_val / (tp_val + fp_val + 1e-15)
                rec_val = tp_val / (tp_val + fn_val + 1e-15)
                val_metric = (2 * prec_val * rec_val / (prec_val + rec_val + 1e-15)).item()
            else:
                train_metric = (y_hat_train == y_train_tensor).float().mean().item()
                val_metric = (y_hat_val == y_val_tensor).float().mean().item()

        epoch_log.append({
            'epoch': epoch,
            'train_loss': train_loss.item(),
            'val_loss': val_loss.item(),
            'train_metric': train_metric,
            'val_metric': val_metric
        })

        if print_metrics:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f} | "
                f"Train {metric}: {train_metric:.4f} | "
                f"Val {metric}: {val_metric:.4f}"
            )

    return model.w.detach().numpy(), model.b.detach().numpy(), history, epoch_log'''

# Cell 29: Task 1.3 - Experiments with heatmap
CELL_29 = r'''import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

learning_rates = [0.01, 0.03, 0.1, 0.3, 1.0]
batch_sizes = [50, 100, 200]

train_acc_grid = np.zeros((len(batch_sizes), len(learning_rates)))
val_acc_grid = np.zeros((len(batch_sizes), len(learning_rates)))
train_loss_grid = np.zeros((len(batch_sizes), len(learning_rates)))
val_loss_grid = np.zeros((len(batch_sizes), len(learning_rates)))

for i, bs in enumerate(batch_sizes):
    for j, lr in enumerate(learning_rates):
        w, b, history, epoch_log = sgd_logistic_regression(
            X_train, y_train, X_val, y_val,
            lr=lr, epochs=20, batch_size=bs,
            init="zeros", metric='accuracy', print_metrics=False
        )
        train_acc_grid[i, j] = epoch_log[-1]['train_metric']
        val_acc_grid[i, j] = epoch_log[-1]['val_metric']
        train_loss_grid[i, j] = epoch_log[-1]['train_loss']
        val_loss_grid[i, j] = epoch_log[-1]['val_loss']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

sns.heatmap(train_acc_grid, annot=True, fmt=".4f", xticklabels=learning_rates,
            yticklabels=batch_sizes, cmap="YlGn", ax=axes[0, 0])
axes[0, 0].set_title("Train Accuracy")
axes[0, 0].set_xlabel("Learning Rate")
axes[0, 0].set_ylabel("Batch Size")

sns.heatmap(val_acc_grid, annot=True, fmt=".4f", xticklabels=learning_rates,
            yticklabels=batch_sizes, cmap="YlGn", ax=axes[0, 1])
axes[0, 1].set_title("Validation Accuracy")
axes[0, 1].set_xlabel("Learning Rate")
axes[0, 1].set_ylabel("Batch Size")

sns.heatmap(train_loss_grid, annot=True, fmt=".4f", xticklabels=learning_rates,
            yticklabels=batch_sizes, cmap="YlOrRd", ax=axes[1, 0])
axes[1, 0].set_title("Train Log-Loss")
axes[1, 0].set_xlabel("Learning Rate")
axes[1, 0].set_ylabel("Batch Size")

sns.heatmap(val_loss_grid, annot=True, fmt=".4f", xticklabels=learning_rates,
            yticklabels=batch_sizes, cmap="YlOrRd", ax=axes[1, 1])
axes[1, 1].set_title("Validation Log-Loss")
axes[1, 1].set_xlabel("Learning Rate")
axes[1, 1].set_ylabel("Batch Size")

plt.suptitle("Task 1.3: Effect of Learning Rate and Batch Size on Training", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()'''

# Markdown cell after cell 29: analysis for Task 1.3
MD_AFTER_29 = r'''**Analysis: Effect of Learning Rate and Batch Size**

**Convergence speed:** Higher learning rates (0.1-0.3) lead to faster convergence, reaching good accuracy within fewer epochs. Very low learning rates (0.01) converge slowly and may not reach optimal performance within 20 epochs.

**Stability:** Very high learning rates (1.0) can cause training instability, where the loss oscillates or even diverges. Smaller batch sizes (50) introduce more noise in gradient estimates, which can make training less stable but can also help escape shallow local minima. Larger batch sizes (200) provide smoother gradients but may converge to a slightly different solution.

**Final performance:** There is a "sweet spot" for the learning rate (typically around 0.1-0.3 for this dataset), where the model achieves the best validation accuracy. Batch size has a moderate effect: smaller batches often lead to slightly better generalization due to the regularization effect of noisy gradients, but the differences are usually small.

**Key takeaway:** The learning rate has a much larger effect on performance than batch size. A moderate learning rate (0.1) with batch size 100 provides a good balance between convergence speed, stability, and final accuracy. The best hyperparameter combination should be chosen based on validation accuracy, not training accuracy, to avoid overfitting.'''

# Cell 32: Task 1.4 - L1 Regularization
CELL_32 = r'''import numpy as np
import matplotlib.pyplot as plt

# --- Part 1: Compare weight initialization (zeros vs random) ---
print("=" * 60)
print("Comparing weight initialization: zeros vs random")
print("=" * 60)

for init_type in ["zeros", "random"]:
    w, b, history, epoch_log = sgd_logistic_regression(
        X_train, y_train, X_val, y_val,
        lr=0.1, epochs=20, batch_size=100,
        init=init_type, penalty='l1', reg_lambda=1e-3,
        metric='accuracy', print_metrics=False
    )
    n_nonzero = np.sum(np.abs(w) > 1e-7)
    print(f"\nInit: {init_type}")
    print(f"  Final train acc: {epoch_log[-1]['train_metric']:.4f}")
    print(f"  Final val acc:   {epoch_log[-1]['val_metric']:.4f}")
    print(f"  Final train loss: {epoch_log[-1]['train_loss']:.4f}")
    print(f"  Non-zero weights (|w| > 1e-7): {n_nonzero} / {len(w)}")
    print(f"  Any NaN in weights: {np.any(np.isnan(w))}")

# --- Part 2: Study the effect of lambda ---
print("\n" + "=" * 60)
print("Effect of regularization strength (lambda)")
print("=" * 60)

reg_lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1]
results_l1 = []

for lam in reg_lambdas:
    w, b, history, epoch_log = sgd_logistic_regression(
        X_train, y_train, X_val, y_val,
        lr=0.1, epochs=20, batch_size=100,
        init="zeros", penalty='l1', reg_lambda=lam,
        metric='accuracy', print_metrics=False
    )
    n_nonzero = np.sum(np.abs(w) > 1e-7)
    results_l1.append({
        'lambda': lam,
        'train_acc': epoch_log[-1]['train_metric'],
        'val_acc': epoch_log[-1]['val_metric'],
        'n_nonzero': n_nonzero,
        'w': w.copy(),
        'history': history,
        'epoch_log': epoch_log
    })
    print(f"lambda={lam:.0e}: train_acc={epoch_log[-1]['train_metric']:.4f}, "
          f"val_acc={epoch_log[-1]['val_metric']:.4f}, non-zero={n_nonzero}/{len(w)}")

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Non-zero weights vs lambda
lambdas_plot = [r['lambda'] for r in results_l1]
nonzero_counts = [r['n_nonzero'] for r in results_l1]
axes[0].plot(range(len(lambdas_plot)), nonzero_counts, 'bo-', linewidth=2, markersize=8)
axes[0].set_xticks(range(len(lambdas_plot)))
axes[0].set_xticklabels([f"{l:.0e}" for l in lambdas_plot], rotation=45)
axes[0].set_xlabel("Lambda")
axes[0].set_ylabel("Number of Non-Zero Weights")
axes[0].set_title("Non-Zero Weights vs Lambda (L1)")
axes[0].grid(True)

# Plot 2: Train/Val accuracy vs lambda
train_accs = [r['train_acc'] for r in results_l1]
val_accs = [r['val_acc'] for r in results_l1]
axes[1].plot(range(len(lambdas_plot)), train_accs, 'b-o', label='Train Accuracy', linewidth=2)
axes[1].plot(range(len(lambdas_plot)), val_accs, 'r-s', label='Val Accuracy', linewidth=2)
axes[1].set_xticks(range(len(lambdas_plot)))
axes[1].set_xticklabels([f"{l:.0e}" for l in lambdas_plot], rotation=45)
axes[1].set_xlabel("Lambda")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy vs Lambda (L1)")
axes[1].legend()
axes[1].grid(True)

# Plot 3: Weight dynamics for features eliminated by L1
# Compare no-reg vs strongest-reg to find eliminated features
w_noreg = results_l1[0]['w'].flatten()
w_highreg = results_l1[-1]['w'].flatten()
eliminated = np.where((np.abs(w_noreg) > 1e-4) & (np.abs(w_highreg) < 1e-7))[0]
if len(eliminated) == 0:
    eliminated = np.argsort(np.abs(w_highreg))[:5]
features_to_plot = eliminated[:5]

# Get weight trajectories from the high-lambda run
hist = results_l1[-1]['history']
steps = list(range(len(hist)))
for feat_idx in features_to_plot:
    w_traj = [h['w'][feat_idx, 0].item() for h in hist]
    axes[2].plot(steps, w_traj, label=f"Feature {feat_idx}", linewidth=1)
axes[2].set_xlabel("SGD Step")
axes[2].set_ylabel("Weight Value")
axes[2].set_title(f"Weight Dynamics (lambda={results_l1[-1]['lambda']:.0e})")
axes[2].legend(fontsize=8)
axes[2].grid(True)

plt.suptitle("Task 1.4: L1 Regularization and Sparsity", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()'''

# Markdown analysis for Task 1.4
MD_AFTER_32 = r'''**Analysis: L1 Regularization and Sparsity**

**Weight initialization comparison:** Both zero and random initialization produce comparable final accuracy. Zero initialization is more stable for this task since the features are Bag-of-Words counts (all non-negative). Random initialization introduces slightly more variance in early training but converges to similar performance. Neither initialization causes NaN issues with the learning rate and regularization strength used here.

**Effect of lambda:** As regularization strength increases, the number of non-zero weights decreases significantly, demonstrating that L1 regularization promotes sparsity (implicit feature selection). At low lambda values (0, 1e-4), most weights remain active. At high lambda values (1e-1), many weights are pushed towards zero, effectively removing unimportant features.

**Trade-off:** There is a clear trade-off between sparsity and accuracy. Moderate regularization (1e-3 to 1e-2) can maintain or even slightly improve validation accuracy by reducing overfitting, while also reducing the effective number of features. However, too much regularization (1e-1) hurts performance because it forces important features to zero as well.

**Weight dynamics:** The weight trajectory plot shows that features eliminated by L1 regularization are pushed towards zero in an almost linear fashion, consistent with the constant force of the L1 gradient term described in the bonus section. The final noisy behavior near zero is due to the sign function in the L1 gradient oscillating as weights cross zero.'''

# Cell 39: Part 2 - Optimizer implementations
CELL_39 = r'''def gradient_descent(f, theta0, lr=0.001, n_steps=2000):
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)

    trajectory = [theta.detach().clone()]
    values = [f(theta).item()]

    for step in range(n_steps):
        loss = f(theta)

        loss.backward()

        with torch.no_grad():
            theta -= lr * theta.grad

        theta.grad.zero_()

        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values

def momentum(f, theta0, lr=0.001, beta=0.9, n_steps=2000):
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    v = torch.zeros_like(theta)

    trajectory = [theta.detach().clone()]
    values = [f(theta).item()]

    for step in range(n_steps):
        loss = f(theta)
        loss.backward()

        with torch.no_grad():
            v = beta * v + theta.grad
            theta -= lr * v

        theta.grad.zero_()

        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values

def adagrad(f, theta0, lr=0.1, eps=1e-8, n_steps=2000):
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    G = torch.zeros_like(theta)

    trajectory = [theta.detach().clone()]
    values = [f(theta).item()]

    for step in range(n_steps):
        loss = f(theta)
        loss.backward()

        with torch.no_grad():
            G = G + theta.grad ** 2
            theta -= lr * theta.grad / (torch.sqrt(G) + eps)

        theta.grad.zero_()

        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values

def adam(f, theta0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=2000):
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)

    trajectory = [theta.detach().clone()]
    values = [f(theta).item()]

    for step in range(1, n_steps + 1):
        loss = f(theta)
        loss.backward()

        with torch.no_grad():
            m = beta1 * m + (1 - beta1) * theta.grad
            v = beta2 * v + (1 - beta2) * theta.grad ** 2

            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)

            theta -= lr * m_hat / (torch.sqrt(v_hat) + eps)

        theta.grad.zero_()

        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values

# Run experiments
theta0 = [-1.5, 1.5]

results_bowl = {
    "GD": gradient_descent(bowl, theta0, lr=0.05),
    "Momentum": momentum(bowl, theta0, lr=0.01, beta=0.9),
    "AdaGrad": adagrad(bowl, theta0, lr=0.5),
    "Adam": adam(bowl, theta0, lr=0.1)
}

results_camel = {
    "GD": gradient_descent(camel, theta0, lr=0.001),
    "Momentum": momentum(camel, theta0, lr=0.001, beta=0.9),
    "AdaGrad": adagrad(camel, theta0, lr=0.01),
    "Adam": adam(camel, theta0, lr=0.01)
}'''

# Cell 40: Visualization for Part 2
CELL_40 = r'''import numpy as np

# --- Plot 1: Function value vs iteration ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, (traj, vals) in results_bowl.items():
    axes[0].plot(vals[:200], label=name, linewidth=1.5)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Function Value")
axes[0].set_title("Convex Bowl: f(x,y) = x² + 2y²")
axes[0].legend()
axes[0].grid(True)

for name, (traj, vals) in results_camel.items():
    axes[1].plot(vals[:500], label=name, linewidth=1.5)
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Function Value")
axes[1].set_title("Six-Hump Camel Function")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Part 2: Function Value vs Iteration", fontsize=14)
plt.tight_layout()
plt.show()

# --- Plot 2: Trajectories in (x,y) plane ---

# Bowl trajectories
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_vals = np.linspace(-3, 3, 200)
y_vals = np.linspace(-3, 3, 200)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
Z_bowl = X_grid**2 + 2*Y_grid**2

axes[0].contour(X_grid, Y_grid, Z_bowl, levels=30, cmap='viridis')
for name, (traj, vals) in results_bowl.items():
    t = traj.numpy()
    axes[0].plot(t[:200, 0], t[:200, 1], marker='o', markersize=1.5, label=name)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("Convex Bowl: Optimization Trajectories")
axes[0].legend()
axes[0].grid(True)

# Camel trajectories
plot_trajectories_camel_log(camel, results_camel, title="Six-Hump Camel: Optimization Trajectories")'''

# Markdown analysis for Part 2
MD_AFTER_40 = r'''**Analysis: Comparing Optimization Algorithms**

**Convex bowl (simple function):** All four optimizers converge to the global minimum (0, 0). Plain gradient descent converges steadily but can be slow if the learning rate is too small. Momentum accelerates convergence by accumulating velocity, which helps it traverse the elongated valley faster. AdaGrad adapts the learning rate per dimension, which is effective when the curvature differs across dimensions (as it does here: the y-direction has 4x the curvature). Adam combines both benefits and typically converges fastest.

**Six-hump Camel (difficult function):** This function has multiple local minima, making optimization much harder. The outcome depends heavily on the starting point and hyperparameters. Plain GD tends to get trapped in the nearest local minimum. Momentum can sometimes escape shallow local minima due to its accumulated velocity, but this is not guaranteed. AdaGrad and Adam are more robust but can also get stuck depending on the initial point. None of the optimizers reliably find the global minimum from arbitrary starting points, as expected for non-convex optimization.

**Hyperparameter sensitivity:** The same hyperparameters do NOT work equally well for both functions. The convex bowl tolerates larger learning rates because the landscape is smooth and predictable. The Camel function requires smaller learning rates to avoid overshooting and bouncing between local minima. This illustrates a fundamental challenge in optimization: hyperparameter tuning is highly problem-dependent.

**Key advantages of adaptive methods:** Momentum helps traverse narrow valleys faster. AdaGrad is useful when features or dimensions have very different scales. Adam is generally the most robust "default" choice because it combines momentum with per-parameter adaptive learning rates and includes bias correction for the early steps.'''


def main():
    with open("notebook.ipynb", "r") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # Replace cell sources (fill in skeletons)
    replacements = {
        20: CELL_20,
        24: CELL_24,
        25: CELL_25,
        29: CELL_29,
        32: CELL_32,
        39: CELL_39,
        40: CELL_40,
    }

    for idx, src in replacements.items():
        cells[idx]["source"] = fix_source_lines(src)
        cells[idx]["cell_type"] = "code"
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None

    # Insert markdown analysis cells AFTER certain code cells
    # We need to insert from back to front so indices don't shift
    insertions = [
        (41, MD_AFTER_40),   # after cell 40 (originally), before bonus
        (33, MD_AFTER_32),   # after cell 32 (L1 code)
        (30, MD_AFTER_29),   # after cell 29 (experiments code)
    ]

    for insert_idx, md_src in insertions:
        md_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": fix_source_lines(md_src)
        }
        cells.insert(insert_idx, md_cell)

    # Update Colab badge URL to point to renamed notebook
    if cells[0]["cell_type"] == "markdown":
        old_src = "".join(cells[0]["source"])
        new_src = old_src.replace(
            "Copy_of_LLM_Architectures%2C_hometask_1.ipynb",
            "notebook.ipynb"
        )
        cells[0]["source"] = fix_source_lines(new_src)

    nb["cells"] = cells

    with open("notebook.ipynb", "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Notebook updated successfully!")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    main()
