# wujian@2018

import random
import os
import os.path as op
import numpy as np
import torch as th
from torch.utils.data import Dataset
from kaldi_python_io import Reader, ScriptReader


class SpeakerSampler(object):
    """
    Remember to filter speakers which utterance number lower than M
    """

    def __init__(self, data_dir):
        depends = [op.join(data_dir, x) for x in ["feats.scp", "spk2utt"]]
        for depend in depends:
            if not op.exists(depend):
                raise RuntimeError("Missing {}!".format(depend))
        self.reader = ScriptReader(depends[0])
        self.spk2utt = Reader(depends[1], num_tokens=-1)

    def sample(self, N=64, M=10, chunk_size=(140, 180)):
        """
        N: number of spks
        M: number of utts
        """
        spks = random.sample(self.spk2utt.index_keys, N)
        chunks = []
        eg = dict()
        eg["N"] = N
        eg["M"] = M
        C = random.randint(*chunk_size)
        for spk in spks:
            utt_sets = self.spk2utt[spk]
            if len(utt_sets) < M:
                raise RuntimeError(
                    "Speaker {} can not got enough utterance with M = {:d}".
                        format(spk, M))
            samp_utts = random.sample(utt_sets, M)
            for uttid in samp_utts:
                utt = self.reader[uttid]
                pad = C - utt.shape[0]
                if pad < 0:  # random chunk of spectrogram
                    start = random.randint(0, -pad)
                    chunks.append(utt[start:start + C])
                else:
                    chunk = np.pad(utt, ((pad, 0), (0, 0)), "edge")
                    chunks.append(chunk)
        eg["feats"] = th.from_numpy(np.stack(chunks))
        return eg


class SpeakerLoader(object):
    def __init__(self,
                 data_dir,
                 N=64,
                 M=10,
                 chunk_size=(140, 180),
                 num_steps=10000):
        self.sampler = SpeakerSampler(data_dir)
        self.N, self.M, self.C = N, M, chunk_size
        self.num_steps = num_steps

    def _sample(self):
        return self.sampler.sample(self.N, self.M, self.C)

    def __iter__(self):
        for _ in range(self.num_steps):
            yield self._sample()


class SRESpeakerDataset(Dataset):

    def __init__(self, data_dir, params, shuffle=True, utter_start=0):

        # data path
        self.path = data_dir
        self.utter_num = params['M']

        depends = [os.path.join(self.path, x) for x in ['feats.scp', 'spk2utt']]
        for depend in depends:
            if not os.path.exists(depend):
                raise RuntimeError('Missing file {}!'.format(depend))

        self.shuffle = shuffle
        self.utter_start = utter_start

        ##############
        self.wnd_size = 160 #(140, 180)
        self.feat_reader = ScriptReader(depends[0])
        self.spk2utt = Reader(depends[1], num_tokens=-1)
        self.speakers = self.spk2utt.index_keys

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        """
        Sample one speaker with M utterances
        index: Integer, batch index
        """
        if self.shuffle:
            tmp_speaker = random.sample(self.speakers, 1)[0]  # select random speaker
        else:
            tmp_speaker = self.speakers[idx][0]

        return self._generate_data(tmp_speaker)

    def _generate_data(self, tmp_id):
        # load utterance spectrogram of selected speaker
        utt_sets = self.spk2utt[tmp_id]

        if len(utt_sets) < self.utter_num:
            raise RuntimeError('Speaker {} can not got enough utterance with M = {:d}'.
                               format(tmp_id, self.utter_num))

        # utterances of a speaker [batch(M), n_mels, frames]
        if self.shuffle:
            # select M utterances per speaker
            utter_ids = random.sample(utt_sets, self.utter_num)
        else:
            utter_ids = utt_sets[:, self.utter_num]

        chunks = []
        for uttid in utter_ids:
            utt = self.feat_reader[uttid]
            pad = utt.shape[0] - self.wnd_size
            if pad > 0:  # random chunk of spectrogram
                start = random.randint(0, pad)
                chunks.append(utt[start:start + self.wnd_size])
            else:
                chunk = np.pad(utt, ((-pad, 0), (0, 0)), 'edge')
                chunks.append(chunk)

        # utterance = utterance[:, :, :160]  # TODO implement variable length batch size
        # dimensions [batch, frames, n_mels]
        utterance = np.stack(chunks)
        utterance = th.tensor(utterance)
        return utterance
