import copy
import torch
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from trainers.promptfl import PromptFL
from Dassl.dassl.metrics import compute_accuracy

class PromptFLFedAMP(PromptFL):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.u = None

    def build_model(self):
        super().build_model()

    def forward_backward(self, batch, global_weight=None, fedprox=False, mu=0.5):

        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PROMPTFL.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
                if self.u is not None:
                    lam = self.cfg.TRAINER.PROMPTFLFEDAMP.LAMBDA
                    loss += lam * torch.norm(self.model.state_dict()["prompt_learner.ctx"] - self.u, p=2)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            if self.u is not None:
                lam = self.cfg.TRAINER.PROMPTFLFEDAMP.LAMBDA
                loss += lam * torch.norm(self.model.state_dict()["prompt_learner.ctx"] - self.u, p=2)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def fed_upload_model(self, idx):
        super().fed_upload_model(idx)
        self.local_info[idx].pop("u", None)

    def fed_aggregate_model(self, idx_users):
        global_list = []
        sigma = self.cfg.TRAINER.PROMPTFLFEDAMP.SIGMA
        for idx in idx_users:
            global_list.append(self.local_info[idx]["prompt_learner.ctx"])

        def attention_func(dist):
            return 1 