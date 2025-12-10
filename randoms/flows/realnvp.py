# Import required packages
import torch
import numpy as np
import normflows as nf
from tqdm import tqdm
from torch.utils.data import DataLoader


class RealNVP:
    def __init__(
        self, 
        n_features:int,
        n_flows:int=64,
        layer_width:int=32,
        n_layers:int=2,
        act_norm:bool=True,
        random_state:int=42,
        epochs:int=100
    ):
        torch.manual_seed(random_state)
        self.n_features = n_features
        self.n_flows = n_flows
        self.layer_width = layer_width
        self.epochs = epochs
        self.n_layers = n_layers
        self.act_norm = act_norm
        # Define Gaussian base distribution
        self.base = nf.distributions.base.DiagGaussian(self.n_features, trainable=False)
        self.init_model()
    
    def init_model(self):
        self.flows = []
        for i in range(self.n_flows):
            param_map = nf.nets.MLP([
                self.n_features // 2,
                *[self.layer_width for _ in range(self.n_layers)],
                self.n_features
            ], init_zeros=True)
            self.flows += [nf.flows.AffineCouplingBlock(param_map)]
            self.flows += [nf.flows.Permute(self.n_features, mode='swap')]
            if self.act_norm:
                self.flows += [nf.flows.ActNorm(self.n_features)]
        self._model = nf.NormalizingFlow(q0=self.base, flows=self.flows).double()
        # Move model on GPU if available
        enable_cuda = True
        device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
        self._model = self._model.to(device)
    
    def train(self, data, batch_size=64):
        train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        # Train model
        self.loss_hist = np.array([])
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        self._model.train()
        for it in tqdm(range(self.epochs)):
            for i, batch in enumerate(train_loader):
                # Clear gradients
                nf.utils.clear_grad(self._model)
                # Get training samples
                x = batch.to(device).double()
                # Compute loss
                loss = self._model.forward_kld(x)
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                    # Log loss
                    self.loss_hist = np.append(self.loss_hist, loss.to('cpu').data.numpy())
                
    
    def predict(data):
        self._model.eval()
        train_loader = DataLoader(data, batch_size=data.shape[0], shuffle=True)
        for i, batch in enumerate(train_loader):
            x = batch.to(device).float()
            z = self._model(x)
        return z