class SupervCausalLSTMMemory(nn.Module):
    def __init__(self, inp_size, out_size):
        super().__init__()
        self.out_size = out_size
        
        self.w = torch.randn(inp_size, out_size)
        self.causal = torch.zeros(1, inp_size, out_size)
        
        self.deep = 1
        self.outt_1 = torch.zeros(1, out_size)
        
        self.lstm = nn.LSTM(out_size, out_size, self.deep)
        self.init_hid()
        self.optim = torch.optim.AdamW(self.lstm.parameters(), lr=3e-2)
        self.loss_lstm = nn.MSELoss()
        
    
    def init_hid(self):
        self.h = torch.zeros(self.deep, self.out_size)
        self.c = torch.zeros(self.deep, self.out_size)
        
        self.h1 = torch.zeros(self.deep, self.out_size)
        self.c1 = torch.zeros(self.deep, self.out_size)
    
    def loss_fn(self, a, b):
        loss = a - b
        return loss
    
    def forward(self, sdr, target=None, learning=True, lr=3e-2):
        
        ########## FORWARD
        # sdr : (1, inp_size)
        

        if learning:
            out = target # (1, out_size)
            causality = sdr.T @ out # (inp_size, out_size)
            self.causal = torch.cat((self.causal, causality.unsqueeze(0)), axis=0)    
            causal = torch.mean(self.causal, 0)
            
            loss = self.loss_fn(causal, self.w)
            
            self.w = self.w + (loss * lr)
            
        else:
            out = torch.special.erf(sdr @ self.w) # (1, out_size)
        
        ########## PREDICTION

        if learning:
            pred_1, (self.h1, self.c1) = self.lstm(self.outt_1, (self.h1, self.c1))
            self.h1, self.c1 = self.h1.detach(), self.c1.detach()

            loss = self.loss_lstm(pred_1, out)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()
        
        self.outt_1 = out
        
        pred, (self.h, self.c) = self.lstm(out, (self.h, self.c))
        
        return out, pred

class CausalLSTMMemory(nn.Module):
    def __init__(self, inp_size, out_size):
        super().__init__()
        self.out_size = out_size
        
        self.w = torch.randn(inp_size, out_size)
        self.causal = torch.zeros(1, inp_size, out_size)
        
        self.deep = 1
        self.outt_1 = torch.zeros(1, out_size)
        
        self.lstm = nn.LSTM(out_size, out_size, self.deep)
        self.init_hid()
        self.optim = torch.optim.AdamW(self.lstm.parameters(), lr=0.1)
        self.loss_lstm = nn.MSELoss()
        
    
    def init_hid(self):
        self.h = torch.zeros(self.deep, self.out_size)
        self.c = torch.zeros(self.deep, self.out_size)
        
        self.h1 = torch.zeros(self.deep, self.out_size)
        self.c1 = torch.zeros(self.deep, self.out_size)
    
    def loss_fn(self, a, b):
        loss = a - b
        return loss
    
    def forward(self, sdr, learning=True, lr=3e-2):
        
        ########## FORWARD
        # sdr : (1, inp_size)
        out = torch.special.erf(sdr @ self.w) # (1, out_size)

        if learning:
            causality = sdr.T @ out # (inp_size, out_size)
            self.causal = torch.cat((self.causal, causality.unsqueeze(0)), axis=0)    
            causal = torch.mean(self.causal, 0)
            
            loss = self.loss_fn(causal, self.w)
            
            self.w = self.w + (loss * lr)
        
        
        ########## PREDICTION

        if learning:
            pred_1, (self.h1, self.c1) = self.lstm(self.outt_1, (self.h1, self.c1))
            self.h1, self.c1 = self.h1.detach(), self.c1.detach()

            loss = self.loss_lstm(pred_1, out)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()
        
        self.outt_1 = out
        
        pred, (self.h, self.c) = self.lstm(out, (self.h, self.c))
        
        return out, pred

class BatchedCausalLayer(nn.Module):
    def __init__(self, inp_size, out_size, device='cpu'):
        super().__init__()
        
        self.w = torch.randn(inp_size, out_size).to(device)
        self.causal = torch.zeros(1, inp_size, out_size).to(device)
        
        self.wp = torch.randn(out_size, out_size).to(device)
        self.outt_1 = None
        self.causal_pred = torch.zeros(1, out_size, out_size).to(device)
        
        self.inertie = 1
    
    def forward(self, sdr, learning=True, lr=3e-2):
        # sdr : (batch_size, inp_size)
        
        ####### Forward
        out = torch.special.erf(torch.mm(sdr, self.w)) # (batch_size, out_size)
        if torch.mean(torch.where(out > 0, 1, 0).float()) < 0.2:
            print("no activation")
            for x in range(out.shape[0]):
                for y in range(out.shape[1]):
                    out[x, y] = 1 if out[x, y] < 0 else out[x, y]
            
        if learning:
            sdr = sdr.unsqueeze(-1) # (b, inp, 1)
            out_unsq = out.unsqueeze(1)  # (b, 1, inp)

            causality = torch.special.erf(torch.bmm(sdr, out_unsq)) # (b, inp, out)
            self.causal = torch.cat((self.causal[-int(self.inertie):, :, :], causality), axis=0).detach() # (50, inp, out)

            causal = torch.mean(self.causal, 0) # (inp, out)

            loss = self.loss_fn(causal, self.w)
            self.w = self.w + (loss*lr)

            
        
        
        ####### Prediction 
        pred = torch.special.erf(torch.mm(out, self.wp)) # (batch_size, out_size)
        
        if learning:
            pred = pred.unsqueeze(-1) # (b, out, 1)
            outt_1 = self.outt_1.unsqueeze(1) if self.outt_1 is not None else torch.transpose(pred, 1, 2)  # (b, 1, out)
            causality = torch.special.erf(torch.bmm(pred, outt_1)) # (b, out, out)
            
            self.causal_pred = torch.cat((self.causal_pred[-int(self.inertie):, :, :], causality), axis=0).detach() # (50, out, out)
            causal = torch.mean(self.causal_pred, 0) # (out, out)
            
            loss = self.loss_fn(causal, self.wp)
            self.wp = self.wp + (loss * lr)
        
        self.inertie = self.inertie*1.1 if self.inertie < 50 else self.inertie
        self.outt_1 = out
        return out, pred
    
    def loss_fn(self, a, b):
        loss = a - b
        return loss
