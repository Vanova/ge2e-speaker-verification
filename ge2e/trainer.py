#!/usr/bin/env python

# wujian@2018

import os
import sys
import time
import logging
import random
from collections import defaultdict
from utils import load_json
from nnet import Nnet
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        file=False):
    """
    Get python logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = get_logger(__name__)


def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class Reporter(object):
    """
    A simple progress reporter
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []
        self.timer = SimpleTimer()

    def add(self, loss):
        self.loss.append(loss)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            self.logger.info("Processed {:d} batches"
                             "(loss = {:+.2f})...".format(N, avg))

    def report(self, details=False):
        N = len(self.loss)
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
        return {
            "loss": sum(self.loss) / N,
            "batches": N,
            "cost": self.timer.elapsed()
        }


class GE2ELoss(nn.Module):
    def __init__(self):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(th.tensor(10.0))
        self.b = nn.Parameter(th.tensor(-5.0))

    def forward(self, e, N, M):
        """
        e: N x M x D, after L2 norm
        N: number of spks
        M: number of utts
        """
        # N x D
        c = th.mean(e, dim=1)
        s = th.sum(e, dim=1)
        # NM * D
        e = e.view(N * M, -1)
        # compute similarity matrix: NM * N
        sim = th.mm(e, th.transpose(c, 0, 1))
        # fix similarity matrix: eq (8), (9)
        for j in range(N):
            for i in range(M):
                cj = (s[j] - e[j * M + i]) / (M - 1)
                sim[j * M + i][j] = th.dot(cj, e[j * M + i])
        # eq (5)
        sim = self.w * sim + self.b
        # build label N*M
        ref = th.zeros(N * M, dtype=th.int64, device=e.device)
        for r, s in enumerate(range(0, N * M, M)):
            ref[s:s + M] = r
        # ce loss
        loss = F.cross_entropy(sim, ref)
        return loss


class GE2ETrainer(object):
    """
    Train speaker embedding model using GE2E loss
    """

    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 optimizer="sgd",
                 gpuid=None,
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=1000,
                 resume=None,
                 no_impr=6):

        if not th.cuda.is_available():
            gpuid = -1  # (-1, )
            device = th.device('cpu')
        else:
            device = th.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid
        self.device = device
        print(self.gpuid)

        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr

        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.device)
            # load ge2e
            ge2e_loss = GE2ELoss()
            ge2e_loss.load_state_dict(cpt["ge2e_state_dict"])
            self.ge2e = ge2e_loss.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
        else:
            self.nnet = nnet.to(self.device)
            ge2e_loss = GE2ELoss()
            self.ge2e = ge2e_loss.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0 ** 6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "ge2e_state_dict": self.ge2e.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        params = [{
            "params": self.nnet.parameters()
        }, {
            "params": self.ge2e.parameters()
        }]
        opt = supported_optimizer[optimizer](params, **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs):
        """
        Compute ge2e loss
        """
        N, M = egs["N"], egs["M"]
        # NM x D
        if self.gpuid == -1:
            embed = th.nn.parallel.data_parallel(
                self.nnet, egs["feats"], output_device=-1)
        else:
            embed = th.nn.parallel.data_parallel(
                self.nnet, egs["feats"], device_ids=self.gpuid)
        if embed.size(0) != N * M:
            raise RuntimeError(
                "Seems something wrong with egs, dimention check failed({:d} vs {:d})"
                    .format(embed.size(0), M * N))
        embed = embed.view(N, M, -1)
        loss = self.ge2e(embed, N, M)
        return loss

    def train(self, data_loader):
        self.logger.info("Set train mode...")
        self.nnet.train()
        reporter = Reporter(self.logger, period=self.logging_period)

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(egs)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()

            reporter.add(loss.item())
        return reporter.report()

    def eval(self, data_loader):
        self.logger.info("Set eval mode...")
        self.nnet.eval()
        reporter = Reporter(self.logger, period=self.logging_period)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)
                loss = self.compute_loss(egs)
                reporter.add(loss.item())
        return reporter.report(details=True)

    def test_metrics(self):
        pass

    def run(self, train_loader, dev_loader, num_epochs=50):
        # avoid alloc memory from gpu0
        # with th.cuda.device(self.gpuid[0]): # TODO does not work with CPU
        stats = dict()
        # check if save is OK
        self.save_checkpoint(best=False)
        # TODO initial loss
        # cv = self.eval(dev_loader)
        cv = {'loss': 100}
        best_loss = cv["loss"]
        self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
            self.cur_epoch, best_loss))
        no_impr = 0
        # make sure not inf
        self.scheduler.best = best_loss
        while self.cur_epoch < num_epochs:
            print('Epoche: %d' % self.cur_epoch)
            self.cur_epoch += 1
            cur_lr = self.optimizer.param_groups[0]["lr"]
            stats[
                "title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                cur_lr, self.cur_epoch)
            tr = self.train(train_loader)
            stats["tr"] = "train = {:+.4f}({:.2f}m/{:d})".format(
                tr["loss"], tr["cost"], tr["batches"])
            cv = self.eval(dev_loader)
            stats["cv"] = "dev = {:+.4f}({:.2f}m/{:d})".format(
                cv["loss"], cv["cost"], cv["batches"])

            stats["scheduler"] = ""
            if cv["loss"] > best_loss:
                no_impr += 1
                stats["scheduler"] = "| no impr, best = {:.4f}".format(
                    self.scheduler.best)
            else:
                best_loss = cv["loss"]
                no_impr = 0
                self.save_checkpoint(best=True)
            self.logger.info(
                "{title} {tr} | {cv} {scheduler}".format(**stats))
            # schedule here
            self.scheduler.step(cv["loss"])
            # flush scheduler info
            sys.stdout.flush()
            # save last checkpoint
            self.save_checkpoint(best=False)
            if no_impr == self.no_impr:
                self.logger.info(
                    "Stop training cause no impr for {:d} epochs".format(
                        no_impr))
                break
        self.logger.info("Training for {:d}/{:d} epoches done!".format(
            self.cur_epoch, num_epochs))


class GE2EValidator(object):

    def __init__(self, cpt_dir, gpuid, params):
        self.params = params
        # chunk size when inference
        loader_conf = load_json(cpt_dir, "loader.json")
        loader_conf['N'] = params['N']
        loader_conf['M'] = params['M']
        self.chunk_size = sum(loader_conf["chunk_size"]) // 2
        logger.info("Using chunk size {:d}".format(self.chunk_size))

        # GPU or CPU
        self.device = "cuda:{}".format(gpuid) if gpuid >= 0 else "cpu"
        print('Device: %s' % self.device)

        # load nnet
        nnet = self._load_nnet(cpt_dir)
        self.nnet = nnet.to(self.device)

    def _load_nnet(self, cpt_dir):
        # nnet config
        nnet_conf = load_json(cpt_dir, "mdl.json")
        nnet = Nnet(**nnet_conf)
        cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
        cpt = th.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        nnet.eval()
        return nnet

    def test(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.params['N'], shuffle=True,
                                 num_workers=self.params['num_workers'], drop_last=True)
        avg_EER = 0
        for e in range(self.params['epochs']):
            batch_avg_EER = 0
            for batch_id, mel_db_batch in enumerate(test_loader):
                assert self.params['M'] % 2 == 0
                # batch dim = [N, M, wnd, dim] !!! split by M, i.e. number of utterances
                enrollment_batch, verification_batch = th.split(mel_db_batch, int(mel_db_batch.size(1) / 2), dim=1)

                enrollment_batch = th.reshape(enrollment_batch, (
                    self.params['N'] * self.params['M'] // 2, enrollment_batch.size(2), enrollment_batch.size(3)))
                verification_batch = th.reshape(verification_batch, (
                    self.params['N'] * self.params['M'] // 2, verification_batch.size(2), verification_batch.size(3)))

                perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
                unperm = list(perm)
                for i, j in enumerate(perm):
                    unperm[j] = i

                verification_batch = verification_batch[perm]
                enrollment_embeddings = self.nnet(enrollment_batch)
                verification_embeddings = self.nnet(verification_batch)
                verification_embeddings = verification_embeddings[unperm]

                enrollment_embeddings = th.reshape(enrollment_embeddings,
                                                   (self.params['N'], self.params['M'] // 2, enrollment_embeddings.size(1)))
                verification_embeddings = th.reshape(verification_embeddings,
                                                     (self.params['N'], self.params['M'] // 2, verification_embeddings.size(1)))

                enrollment_centroids = self._get_centroids(enrollment_embeddings)

                sim_matrix = self._get_cossim(verification_embeddings, enrollment_centroids)

                # calculating EER
                diff = 1
                EER = 0
                EER_thresh = 0
                EER_FAR = 0
                EER_FRR = 0

                for thres in [0.01 * i + 0.5 for i in range(50)]:
                    sim_matrix_thresh = sim_matrix > thres

                    FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                                range(int(self.params['N']))])
                           / (self.params['N'] - 1.0) / (float(self.params['M'] / 2)) / self.params['N'])

                    FRR = (sum(
                        [self.params['M'] / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in range(int(self.params['N']))])
                           / (float(self.params['M'] / 2)) / self.params['N'])

                    # Save threshold when FAR = FRR (=EER)
                    if diff > abs(FAR - FRR):
                        diff = abs(FAR - FRR)
                        EER = (FAR + FRR) / 2
                        EER_thresh = thres
                        EER_FAR = FAR
                        EER_FRR = FRR
                batch_avg_EER += EER
                print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
            avg_EER += batch_avg_EER / (batch_id + 1)
            print("\n EER {0} epochs: {1:.4f}".format(e, avg_EER / (e + 1)))
        avg_EER = avg_EER / self.params['epochs']
        print("\n EER across {0} epochs: {1:.4f}".format(self.params['epochs'], avg_EER))

    def _get_centroids(self, embeddings):
        centroids = []
        for speaker in embeddings:
            centroid = 0
            for utterance in speaker:
                centroid = centroid + utterance
            centroid = centroid / len(speaker)
            centroids.append(centroid)
        centroids = th.stack(centroids)
        return centroids

    def _get_centroid(self, embeddings, speaker_num, utterance_num):
        centroid = 0
        for utterance_id, utterance in enumerate(embeddings[speaker_num]):
            if utterance_id == utterance_num:
                continue
            centroid = centroid + utterance
        centroid = centroid / (len(embeddings[speaker_num]) - 1)
        return centroid

    def _get_cossim(self, embeddings, centroids):
        # Calculates cosine similarity matrix. Requires (N, M, feature) input
        cossim = th.zeros(embeddings.size(0), embeddings.size(1), centroids.size(0))
        for speaker_num, speaker in enumerate(embeddings):
            for utterance_num, utterance in enumerate(speaker):
                for centroid_num, centroid in enumerate(centroids):
                    if speaker_num == centroid_num:
                        centroid = self._get_centroid(embeddings, speaker_num, utterance_num)
                    output = F.cosine_similarity(utterance, centroid, dim=0) + 1e-6
                    cossim[speaker_num][utterance_num][centroid_num] = output
        return cossim
