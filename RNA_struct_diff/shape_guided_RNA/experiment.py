import torch
from diffusion_utils.loss import elbo_bpd
from diffusion_utils.utils import add_parent_path
import math
add_parent_path(level=2)
from diffusion_utils.experiment import DiffusionExperiment
from diffusion_utils.experiment import add_exp_args as add_exp_args_parent
import torch.distributed as dist
from diffusion_utils.utils import is_main_process
import os
import shutil
def add_exp_args(parser):
    add_exp_args_parent(parser)
    parser.add_argument('--clip_value', type=float, default=None)
    parser.add_argument('--clip_norm', type=float, default=None)


class Experiment(DiffusionExperiment):

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0

        is_parallel =  isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
        device = None if is_parallel else self.args.device
        if hasattr(self.train_loader, "set_epoch"):
            self.train_loader.set_epoch(epoch)

        for batch in self.train_loader:
            # Unpack the batch based on structure
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    x, guidance = batch
                    lengths = None
                elif len(batch) == 3:
                    x, lengths, guidance = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")
            else:
                x = batch
                lengths = None
                guidance = None

            if device is not None:
                x = x.to(device)
                if guidance is not None:
                    guidance = guidance.to(device)
            self.optimizer.zero_grad()
            loss = elbo_bpd(self.model, x.to(self.args.device),guidance.to(self.args.device) if guidance is not None else None)
            loss.backward()
            if self.args.clip_value: torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_value)
            if self.args.clip_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            self.optimizer.step()
            if self.scheduler_iter: self.scheduler_iter.step()
            pad_token = 4
            valid_positions =(x != pad_token).sum().item()
            #slightly missleading calculation here. calculates per element in batch
            #however the batchsize/ elementsize per item can be different.
            #loss_sum += loss.detach().cpu().item() * len(x)
            #loss_count += len(x)
            #loss_sum += loss.item() * valid_positions
            loss_sum += loss.detach().cpu().item()*len(x)
            loss_count += len(x)
            #loss_count += valid_positions


            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'bpd': loss_sum/loss_count}

    def eval_fn(self, epoch):
        self.model.eval()

        #remove train val. Takes too long
        '''
        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            log_lines = []
            for batch in self.train_loader:
                # Unpack the batch based on structure
                if isinstance(batch, (tuple, list)):
                    if len(batch) == 2:
                        x, guidance = batch
                        lengths = None
                    elif len(batch) == 3:
                        x, lengths, guidance = batch
                    else:
                        raise ValueError(f"Unexpected batch size: {len(batch)}")
                else:
                    x = batch
                    lengths = None
                    guidance = None
                loss = elbo_bpd(self.model, x.to(self.args.device))
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('Train evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
            print('')
        '''
        loss_sum = 0.0
        loss_count = 0
        log_lines = []
        with torch.no_grad():

            for batch in self.eval_loader:
                # Unpack the batch based on structure
                if isinstance(batch, (tuple, list)):
                    if len(batch) == 2:
                        x, guidance = batch
                        lengths = None
                    elif len(batch) == 3:
                        x, lengths, guidance = batch
                    else:
                        raise ValueError(f"Unexpected batch size: {len(batch)}")
                else:
                    x = batch
                    #print(x[0])
                    lengths = None
                    guidance = None
                #hard coded
                pad_token=4
                valid_positions = (x != pad_token).sum().item()
                #slight error in the original : loss is calulated per element, not per
                loss = elbo_bpd(self.model, x.to(self.args.device),guidance.to(self.args.device) if guidance is not None else None)
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                #loss_sum += loss.item() * valid_positions
                #loss_count += valid_positions
                if not dist.is_available() or not dist.is_initialized() or is_main_process():
                    print(f"\r[Eval] Epoch {epoch + 1} | Processed tokens{loss_count} | Running BPD: {loss_sum / loss_count:.4f}",
                          end='', flush=True)
        if dist.is_available() and dist.is_initialized():
            loss_sum_tensor = torch.tensor([loss_sum], device=self.args.device)
            count_tensor = torch.tensor([loss_count], device=self.args.device)
            dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            loss_sum = loss_sum_tensor.item()
            loss_count = count_tensor.item()

        mean_bpd = loss_sum / loss_count
        if not dist.is_available() or not dist.is_initialized() or is_main_process():
            print(f"\n[Eval] Epoch {epoch + 1}/{self.args.epochs} | Final BPD: {mean_bpd:.4f}")
        return {'bpd': mean_bpd}
        #return {'bpd': loss_sum/loss_count}


    ###### early stopping trying
    def run(self):
        if getattr(self.args, "resume", False):
            self.resume()

        patience = getattr(self.args, "es_patience", None)  # e.g. 8; None => ES disabled
        min_delta = getattr(self.args, "es_min_delta", 0.0)

        best = math.inf  # monitoring a "lower is better" metric
        bad = 0
        stop = False
        ddp = dist.is_available() and dist.is_initialized()

        for epoch in range(self.current_epoch, self.args.epochs):
            # ---- Train ----
            train_dict = self.train_fn(epoch)
            self.log_train_metrics(train_dict)

            # ---- Eval ----
            if (epoch + 1) % self.eval_every == 0:
                eval_dict = self.eval_fn(epoch)
                self.log_eval_metrics(eval_dict)
                self.eval_epochs.append(epoch)

                # ---- Tiny ES block ----
                if patience is not None:
                    cur = float(eval_dict["bpd"])  # change to "val_loss" if that's what you return
                    if is_main_process():
                        if cur < best - min_delta:
                            best, bad = cur, 0
                            self.checkpoint_save(name="best.pt")  # rank-0 only inside
                        else:
                            bad += 1
                            stop = bad >= patience

                    if ddp:
                        buf = torch.tensor([best, float(bad), 1.0 if stop else 0.0], device=self.args.device)
                        dist.broadcast(buf, src=0)
                        best = float(buf[0]);
                        bad = int(buf[1]);
                        stop = bool(int(buf[2]))
            else:
                eval_dict = None

            # ---- Log ----
            self.save_metrics()
            self.log_fn(epoch, train_dict, eval_dict)

            # ---- Periodic latest checkpoint for resume ----
            self.current_epoch += 1
            if (epoch + 1) % self.check_every == 0:
                self.checkpoint_save(name="checkpoint.pt")

            if stop:
                break

        # ---- Only if ES actually stopped training, make checkpoint.pt = best ----
        if stop and is_main_process():
            src = os.path.join(self.check_path, "best.pt")
            dst = os.path.join(self.check_path, "checkpoint.pt")
            if os.path.exists(src):
                tmp = dst + ".tmp"
                shutil.copyfile(src, tmp)  # write complete file first
                os.replace(tmp, dst)  # atomic replace
        if ddp:
            dist.barrier()
