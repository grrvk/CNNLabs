import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from einops.layers.torch import Rearrange

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


def G_train(G, D, batch_size, z_dim, criterion, G_optimizer):
    # =======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(batch_size, z_dim).to(device)
    y = torch.ones(batch_size, 1).to(device)

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

def D_train(x, G, D, z_dim, criterion, D_optimizer):

    D.zero_grad()
    batch_size = x.shape[0]
    # train discriminator on real

    x_real, y_real = x, torch.ones(batch_size, 1, 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    # train discriminator on fake
    z = torch.randn(batch_size, z_dim).to(device)

    with torch.no_grad():
        x_fake, y_fake = G(z), torch.zeros(batch_size, 1).to(device)

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


# ADDED: collapse metrics, computed on a fixed noise batch so they are comparable between runs
@torch.no_grad()
def gen_metrics(G, z_eval, real_diversity):
    images = G(z_eval)
    # share of pixels that are NOT pure black/white
    mid_frac = ((images > -0.9) & (images < 0.9)).float().mean().item()
    # mean pairwise distance between generated images / same for real images
    div_ratio = torch.pdist(images).mean().item() / real_diversity
    return images, mid_frac, div_ratio


if __name__ == '__main__':
    # ADDED: the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_g', type=float, default=0.0002)
    parser.add_argument('--lr_d', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=None)  # stop after this many steps (batch_size=1 -> 60k steps/epoch)
    parser.add_argument('--eval_every', type=int, default=None)  # steps between log lines / sample grids, default = 1 epoch
    parser.add_argument('--classes', type=int, nargs='+', default=None)  # train only on these digits
    parser.add_argument('--class_fractions', type=str, default='')  # skew, e.g. "0:0.05,1:1" keeps 5% of the zeros
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='./results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    batch_size = args.batch_size

    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5)),
        Rearrange('c h w -> c (h w)')
    ])

    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
    # BEFORE: test_dataset / test_loader were created but never used
    # FIX: removed

    # ADDED: unbalance the dataset - keep a fraction of every class (0 for digits outside --classes)
    fractions = {c: 1.0 for c in range(10)}
    fractions.update({int(k): float(v) for k, v in (i.split(':') for i in args.class_fractions.split(',') if i)})
    if args.classes:
        fractions = {c: (fractions[c] if c in args.classes else 0.0) for c in range(10)}
    keep = torch.rand(len(train_dataset)) < torch.tensor([fractions[int(t)] for t in train_dataset.targets])
    class_counts = torch.bincount(train_dataset.targets[keep], minlength=10).tolist()
    print('samples per digit:', class_counts)
    train_dataset = torch.utils.data.Subset(train_dataset, torch.nonzero(keep).squeeze(1).tolist())

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    z_dim = 64
    mnist_dim = 28 * 28

    G = Generator(g_input_dim=z_dim, g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    # loss
    criterion = nn.BCELoss()

    # optimizer
    # BEFORE: lr = 0.0002 for both
    # FIX: separate learning rates for G and D
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr_g)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr_d)

    # ADDED: fixed noise + real reference for the metrics, history of everything for plots
    z_eval = torch.randn(256, z_dim, generator=torch.Generator().manual_seed(1234)).to(device)
    real_ref = torch.stack([train_dataset[i][0][0] for i in range(256)]).to(device)
    real_diversity = torch.pdist(real_ref).mean().item()
    real_mid_frac = ((real_ref > -0.9) & (real_ref < 0.9)).float().mean().item()
    os.makedirs(args.out_dir, exist_ok=True)
    history = dict(args=vars(args), class_counts=class_counts, real_mid_frac=real_mid_frac, step_d=[], step_g=[], log=[])

    eval_every = args.eval_every or len(train_loader)
    n_epoch = args.n_epoch
    step = 0
    D_losses, G_losses = [], []
    for epoch in range(1, n_epoch + 1):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)

            D_losses.append(D_train(x, G, D, z_dim, criterion, D_optimizer))
            G_losses.append(G_train(G, D, batch_size, z_dim, criterion, G_optimizer))
            step += 1

            # BEFORE: loss printed once per epoch, a grid saved every 10 epochs from fresh random noise
            # FIX: every eval_every steps log losses + collapse metrics and save a grid from the FIXED noise
            if step % eval_every == 0:
                images, mid_frac, div_ratio = gen_metrics(G, z_eval, real_diversity)
                row = dict(step=step, epoch=step / len(train_loader), loss_d=sum(D_losses) / len(D_losses),
                           loss_g=sum(G_losses) / len(G_losses), mid_frac=mid_frac, div_ratio=div_ratio)
                history['log'].append(row)
                history['step_d'] += D_losses
                history['step_g'] += G_losses
                D_losses, G_losses = [], []
                print('[step %d | epoch %.2f/%d]: loss_d: %.3f, loss_g: %.3f, mid_frac: %.3f (real %.3f), div_ratio: %.3f' % (
                    step, row['epoch'], n_epoch, row['loss_d'], row['loss_g'], mid_frac, real_mid_frac, div_ratio))

                grid = make_grid(images[:64].view(-1, 1, 28, 28), nrow=8, normalize=True, value_range=(-1, 1))
                save_image(grid, f'{args.out_dir}/gen_step{step}.png')
                torch.save(G.state_dict(), f'{args.out_dir}/G.pth')
                torch.save(D.state_dict(), f'{args.out_dir}/D.pth')
                with open(f'{args.out_dir}/history.json', 'w') as f:
                    json.dump(history, f)

            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break