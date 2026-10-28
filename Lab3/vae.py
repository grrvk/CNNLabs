import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from torchvision import datasets


class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(VAE, self).__init__()
        self.enc = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        # encoder outputs 2*D numbers: D means + D log-variances (log keeps sigma positive)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(),
                                 nn.Linear(512, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x.view(-1, 784))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.dec(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        # reparameterization trick: z = mu + sigma * eps, so gradients flow into both mu and sigma
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar):
    mse = F.mse_loss(recon, x, reduction='sum')
    # KL(N(mu, sigma^2) || N(0, 1)), closed form, summed over latent dims
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (mse + kl) / x.shape[0], mse.item() / x.shape[0], kl.item() / x.shape[0]


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def batches(n, batch_size, shuffle=False):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    return idx.split(batch_size)


def train_vae(vae, x_train, epochs, batch_size, lr):
    opt = optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    for epoch in range(1, epochs + 1):
        total, total_mse, total_kl = 0, 0, 0
        for idx in batches(len(x_train), batch_size, shuffle=True):
            x = x_train[idx]
            recon, mu, logvar = vae(x)
            loss, mse, kl = vae_loss(recon, x, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            total_mse += mse * len(idx)
            total_kl += kl * len(idx)
        n = len(x_train)
        print('[VAE epoch %d/%d]: loss %.2f (mse %.2f, kl %.2f)' % (epoch, epochs, total / n, total_mse / n, total_kl / n))


@torch.no_grad()
def reconstruct(vae, x, batch_size=1000):
    # Decoder(Encoder(x)) with the sampling step, same as in training
    vae.eval()
    return torch.cat([vae(x[idx])[0] for idx in batches(len(x), batch_size)])


@torch.no_grad()
def encode_mu(vae, x, batch_size=1000):
    vae.eval()
    return torch.cat([vae.encode(x[idx])[0] for idx in batches(len(x), batch_size)])


def train_classifier(clf, x_train, y_train, epochs, batch_size, lr, name):
    opt = optim.Adam(clf.parameters(), lr=lr)
    clf.train()
    for epoch in range(1, epochs + 1):
        total = 0
        for idx in batches(len(x_train), batch_size, shuffle=True):
            loss = F.cross_entropy(clf(x_train[idx]), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print('[%s classifier epoch %d/%d]: loss %.4f' % (name, epoch, epochs, total / len(x_train)))


@torch.no_grad()
def f1_scores(clf, x, y, batch_size=1000):
    # y is a numpy array of true labels; returns per-class F1 and their mean (macro F1)
    clf.eval()
    pred = torch.cat([clf(x[idx]).argmax(1) for idx in batches(len(x), batch_size)]).cpu().numpy()
    per_class = f1_score(y, pred, average=None, labels=range(10))
    return dict(macro_f1=float(per_class.mean()), per_class_f1=per_class.tolist())


def save_grid(rows, labels, path, n=10):
    fig, axes = plt.subplots(len(rows), n, figsize=(n, len(rows) * 1.1))
    for r, (imgs, label) in enumerate(zip(rows, labels)):
        for c in range(n):
            axes[r, c].imshow(imgs[c, 0].cpu(), cmap='gray', vmin=0, vmax=1)
            axes[r, c].axis('off')
        axes[r, 0].set_title(label, fontsize=9, loc='left')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def plot_f1_bars(ax, series, title):
    # grouped bars: one group per digit, one bar per entry of series {label: per-class F1}
    width = 0.8 / len(series)
    for i, (label, per_class) in enumerate(series.items()):
        ax.bar(np.arange(10) - 0.4 + width * (i + 0.5), per_class, width, label=label)
    lowest = min(min(v) for v in series.values())
    ax.set_ylim(max(0, lowest - 0.05), 1)
    ax.set_xticks(range(10))
    ax.set_xlabel('digit')
    ax.set_ylabel('F1')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--vae_epochs', type=int, default=30)
    parser.add_argument('--clf_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument('--tsne_points', type=int, default=2000)  # test images per t-SNE panel
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='./results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # MNIST fits in memory, so keep it as plain tensors in [0, 1] on the device
    train_ds = datasets.MNIST(root='./mnist_data/', train=True, download=True)
    test_ds = datasets.MNIST(root='./mnist_data/', train=False, download=True)
    x_train = (train_ds.data.float() / 255).unsqueeze(1).to(device)
    y_train = train_ds.targets.to(device)
    x_test = (test_ds.data.float() / 255).unsqueeze(1).to(device)
    y_test = test_ds.targets.numpy()

    # ---------------- Part 1: train the VAE, original vs reconstructed ----------------
    vae = VAE(args.latent_dim).to(device)
    train_vae(vae, x_train, args.vae_epochs, args.batch_size, args.lr)
    torch.save(vae.state_dict(), f'{args.out_dir}/vae.pth')

    x_test_rec = reconstruct(vae, x_test)
    test_mse = F.mse_loss(x_test_rec, x_test).item()
    print('test per-pixel MSE: %.4f' % test_mse)
    save_grid([x_test[:10], x_test_rec[:10], x_test[10:20], x_test_rec[10:20]],
              ['original', 'reconstructed', 'original', 'reconstructed'], f'{args.out_dir}/part1_reconstructions.png')

    # latent interpolation: walk in a straight line between the mean codes of two classes and decode
    mu_test = encode_mu(vae, x_test)
    centers = torch.stack([mu_test[torch.from_numpy(y_test == c).to(device)].mean(0) for c in range(10)])
    pairs = [(1, 7), (4, 9), (3, 8), (1, 8), (0, 6)]
    t = torch.linspace(0, 1, 10, device=device).unsqueeze(1)
    with torch.no_grad():
        paths = [vae.decode((1 - t) * centers[a] + t * centers[b]) for a, b in pairs]
    save_grid(paths, [f'{a} -> {b}' for a, b in pairs], f'{args.out_dir}/part1_interpolation.png')

    # ---------------- Part 2: baseline vs reconstructed-data classifier ----------------
    x_train_rec = reconstruct(vae, x_train)

    clf_base = Classifier().to(device)
    train_classifier(clf_base, x_train, y_train, args.clf_epochs, args.batch_size, args.lr, 'baseline')
    clf_rec = Classifier().to(device)
    train_classifier(clf_rec, x_train_rec, y_train, args.clf_epochs, args.batch_size, args.lr, 'reconstructed')
    classifiers = [('baseline', clf_base), ('reconstructed', clf_rec)]

    results = dict(args=vars(args), test_mse=test_mse, part2={}, part3=[])
    results['part2'] = {name: f1_scores(clf, x_test, y_test) for name, clf in classifiers}
    print('Part 2 (clean original test set) macro F1: ' +
          ', '.join('%s %.4f' % (name, r['macro_f1']) for name, r in results['part2'].items()))

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_f1_bars(ax, {name: r['per_class_f1'] for name, r in results['part2'].items()},
                 'Per-class F1 on the clean original test set')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{args.out_dir}/part2_f1.png', dpi=120)
    plt.close(fig)

    # ---------------- Part 3: noisy test set, with and without VAE denoising ----------------
    noisy_rows, noisy_labels = [x_test[:10]], ['original']
    noisy_sets = {}
    for sigma in args.sigmas:
        x_noisy = (x_test + sigma * torch.randn_like(x_test)).clamp(0, 1)
        x_denoised = reconstruct(vae, x_noisy)
        noisy_sets[sigma] = x_noisy
        row = dict(sigma=sigma)
        for name, clf in classifiers:
            row[f'{name}_noisy'] = f1_scores(clf, x_noisy, y_test)
            row[f'{name}_denoised'] = f1_scores(clf, x_denoised, y_test)
        results['part3'].append(row)
        print('Part 3 sigma %.1f macro F1: ' % sigma +
              ', '.join('%s %.4f' % (k, v['macro_f1']) for k, v in row.items() if k != 'sigma'))
        noisy_rows += [x_noisy[:10], x_denoised[:10]]
        noisy_labels += [f'noisy sigma={sigma}', f'VAE-denoised sigma={sigma}']

    save_grid(noisy_rows, noisy_labels, f'{args.out_dir}/part3_denoising.png')

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, color in [('baseline', 'tab:blue'), ('reconstructed', 'tab:orange')]:
        ax.plot(args.sigmas, [r[f'{name}_noisy']['macro_f1'] for r in results['part3']], '--o', color=color,
                label=f'{name} clf, noisy')
        ax.plot(args.sigmas, [r[f'{name}_denoised']['macro_f1'] for r in results['part3']], '-o', color=color,
                label=f'{name} clf, VAE-denoised')
    ax.set_xlabel('noise sigma')
    ax.set_ylabel('macro F1')
    ax.set_xticks(args.sigmas)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{args.out_dir}/part3_macro_f1.png', dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(len(args.sigmas), 1, figsize=(9, 3.5 * len(args.sigmas)))
    for ax, row in zip(np.atleast_1d(axes), results['part3']):
        plot_f1_bars(ax, {k.replace('_', ', '): v['per_class_f1'] for k, v in row.items() if k != 'sigma'},
                     f'Per-class F1, noise sigma={row["sigma"]}')
    np.atleast_1d(axes)[0].legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f'{args.out_dir}/part3_f1_per_class.png', dpi=120)
    plt.close(fig)

    # ---------------- Latent space: t-SNE of mu, clean and noisy test images in one embedding ----------------
    # one joint t-SNE so all panels share the same 2D map and noisy codes can be compared with clean ones
    n = args.tsne_points
    sets = [('clean', x_test[:n])] + [(f'noisy sigma={s}', noisy_sets[s][:n]) for s in args.sigmas]
    mu_all = torch.cat([encode_mu(vae, x) for _, x in sets]).cpu().numpy()
    emb = TSNE(n_components=2, init='pca', random_state=args.seed).fit_transform(mu_all).reshape(len(sets), n, 2)

    fig, axes = plt.subplots(1, len(sets), figsize=(4.5 * len(sets), 4.5))
    for i, (ax, (label, _)) in enumerate(zip(axes, sets)):
        if i > 0:
            ax.scatter(emb[0, :, 0], emb[0, :, 1], s=2, c='lightgrey')
        points = ax.scatter(emb[i, :, 0], emb[i, :, 1], s=2, c=y_test[:n], cmap='tab10', vmin=-0.5, vmax=9.5)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(points, ax=axes, ticks=range(10), label='true digit')
    plt.savefig(f'{args.out_dir}/latent_tsne.png', dpi=120)
    plt.close(fig)

    with open(f'{args.out_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
