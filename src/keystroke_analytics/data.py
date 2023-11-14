import math
import random
import subprocess
from abc import ABC, abstractmethod
from os.path import exists

import torch

torch.set_printoptions(threshold=10_000)


class DataLoader(ABC):
    @abstractmethod
    def __init__(
        self,
        directory,
        manifest,
        max_buffer,
        run_length,
        evaluation,
        sample_length=True,
        use_lengths=True,
        reserved_users=False,
    ):
        # Open dataset files
        self.directory = directory
        with open(directory + "/" + manifest) as infile:
            raw = [
                line.split("\t")
                for line in infile.read().split("\n")
                if len(line)
            ]
            if not evaluation:
                for line in raw:
                    self.shuffle(line[0])
            self.files = {
                line[0]: {
                    "weight": int(line[1]),
                    "fd": open(directory + "/" + line[0])
                    if exists(directory + "/" + line[0])
                    else None,
                    "line_buffer": [],
                    "eof": False,
                }
                for line in raw
            }

        if reserved_users:
            self.files["reserved"] = {
                "weight": 0,
                "fd": open(directory + "/" + "reserved"),
                "line_buffer": [],
                "eof": False,
            }

        # Setup buffer size
        self.max_buffer = max_buffer // len(self.files.keys())
        self.run_length = run_length
        self.eval = evaluation
        self.epoch = 0
        self.sl = sample_length
        self.ul = use_lengths

    def shuffle(self, filename=None):
        if filename is None:
            filenames = self.files
        else:
            filenames = [filename]
        for filename in filenames:
            path = self.directory + "/" + filename
            # output_stream = subprocess.Popen('./terashuf < ' + path + ' > ' + path + "_shuffled", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # output = output_stream.communicate()[0]
            # os.system('mv ' + path + "_shuffled" + ' ' + path)
            # print('./terashuf < ' + path + ' > ' + path + "_shuffled")
            # print('mv ' + path + "_shuffled" + ' ' + path)
            # output_stream = subprocess.Popen('./terashuf < ' + path + ' > ' + path + "_shuffled && mv " + path + '_shuffled ' + path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output_stream = subprocess.Popen(
                "shuf "
                + path
                + " > "
                + path
                + "_shuffled && mv "
                + path
                + "_shuffled "
                + path,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # print("shuffled " + filename)
            output = output_stream.communicate()[0]

    @abstractmethod
    def get_batch(self, batch_size=64, others=False, device="cpu"):
        pass

    def parse_lengths(self, item):
        if not self.sl:
            return [int(t) for t in item]
        else:
            sol_raw = [tuple(int(tt) for tt in t.split(":")) for t in item]
            return [random.randint(*t) for t in sol_raw]

    @abstractmethod
    def sample(self, others=False):
        pass

    @abstractmethod
    def refresh(self):
        # Close all files
        for file in self.files:
            if not self.files[file]["eof"]:
                self.files[file]["fd"].close()

        # Shuffle
        if not self.eval:
            self.shuffle()

        # Reopen all files
        for file in self.files:
            self.files[file]["eof"] = False
            self.files[file]["fd"] = open(self.directory + "/" + file)
            self.files[file]["line_buffer"] = []

    def transform_vector(self, vec):
        # Return padded vector with mask
        # Expected format : dim 1 -> times, dim 2 -> sizes
        size = self.run_length
        vec = vec.t()
        M, _ = vec.shape
        if M > size:
            vec = vec[:size]
            M = size
        elif M < size:
            vec = torch.cat((vec, torch.zeros((size - M, 2))), dim=0)
        vec = torch.cat((torch.zeros(1, 2), vec), dim=0)
        mask = torch.cat(
            (torch.zeros(M + 1), torch.ones(size - M)), dim=0
        ).bool()
        if not self.ul:
            vec = vec[:, 0].unsqueeze(1)
        # return torch.ones_like(vec), mask, M+1
        return vec, mask, M + 1

    def process_ip(self, ip):
        coords = [float(t) for t in ip.split(",")]
        bts = torch.zeros((32))
        for i, v in enumerate(coords):
            bts[2 * i] = math.cos(v)
            bts[2 * i + 1] = math.sin(v)
        return bts.float()


class KeystrokeDataLoader(DataLoader):
    @abstractmethod
    def __init__(
        self,
        directory,
        manifest,
        max_buffer,
        evaluation,
        run_length,
        include_others,
        sample_length=True,
        use_lengths=True,
        reserved_users=False,
    ):
        super().__init__(
            directory,
            manifest,
            max_buffer,
            run_length,
            evaluation,
            sample_length,
            use_lengths,
            reserved_users,
        )
        for cl in self.files:
            self.files[cl]["session_buffer"] = {
                "class": None,
                "st": 0,
                "buffer": [],
            }

        self.class_weights = {cl: self.files[cl]["weight"] for cl in self.files}
        self.choice_queue = []

        # Setup variables
        self.tuple_size = 1
        self.include_others = include_others

        # User encoding
        self.output_dim_map = {}
        self.sorted_users = sorted(self.class_weights.keys())
        self.inv_mapping = []
        for cl in self.sorted_users:
            self.sample_raw(cl, True)
            if not include_others and cl == "others":
                continue
            if cl == "reserved":
                continue
            self.output_dim_map[cl] = len(self.output_dim_map.keys())
            self.inv_mapping.append(cl)
        self.output_dim = len(self.output_dim_map.keys())

    def set_tuple_size(self, ts):
        if ts > 1 and not self.eval:
            print("Error --- Cannot train using tuples")
            exit(0)
        self.tuple_size = ts

    @abstractmethod
    def user_count(self):
        pass

    def user_from_dim(self, dim):
        return self.inv_mapping[dim]

    def user_from_tensor(self, t):
        return [
            self.inv_mapping[i]
            for i in torch.argmax(t, dim=1).data.cpu().numpy()
        ]

    @abstractmethod
    def reset_choice(self):
        pass

    def batch_count(self, batch_size):
        return len(self.choice_queue) // batch_size

    def refresh(self, cls=None):
        # print(f"Refreshing {cls}")
        if cls is None:
            cls = self.class_weights
        for cl in cls:
            if self.files[cl]["fd"] is None:
                continue
            if not self.files[cl]["eof"]:
                self.files[cl]["fd"].close()

            if not self.eval:
                self.shuffle(cl)

            self.files[cl]["eof"] = False
            self.files[cl]["fd"] = open(self.directory + "/" + cl)
            self.files[cl]["line_buffer"] = []
            self.files[cl]["session_buffer"] = {
                "class": None,
                "st": 0,
                "buffer": [],
            }

    def reset(self):
        self.epoch += 1
        self.refresh()
        self.reset_choice()
        for cl in self.class_weights:
            self.sample_raw(cl, True)

    def sample_raw(self, cl, continuous=False):
        file = self.files[cl]
        if file["eof"] or file["fd"] is None:
            return False
        already_reloaded = False
        if self.eval:
            while len(file["session_buffer"]["buffer"]) < self.tuple_size:
                if len(file["line_buffer"]) > 0:
                    raw = file["line_buffer"].pop().strip().split("\t")
                    real_class, timestamp = raw[0], float(raw[1])
                    try:
                        src_ip, dst_ip = self.process_ip(
                            raw[2]
                        ), self.process_ip(raw[3])
                    except:
                        src_ip, dst_ip = None, None
                    vecs = [None for _ in range(int(raw[4]))]
                    raw = raw[6:]
                    vlen = len(vecs)
                    cum_len = 0.0
                    pos = []
                    for i in range(vlen):
                        sot = [float(t) for t in raw[2 * i].split(",")]
                        sol = self.parse_lengths(raw[2 * i + 1].split(","))
                        vecs[i] = torch.tensor([sol, sot])
                        pos.append(cum_len)
                        cum_len += len(sot) / 512
                    file["session_buffer"] = {
                        "class": real_class,
                        "st": timestamp,
                        "buffer": vecs,
                        "src_ip": [src_ip for _ in range(vlen)],
                        "dst_ip": [dst_ip for _ in range(vlen)],
                        "pos": pos,
                    }
                else:
                    file["line_buffer"] = file["fd"].readlines(self.max_buffer)
                    if not file["line_buffer"] or not len(file["line_buffer"]):
                        ### Empty dataset -- Shuffle and reload
                        file["eof"] = True
                        file["fd"].close()
                        if already_reloaded:
                            print(
                                f"Warning - Class {cl} does not contain enough elements for tuple size {self.tuple_size}"
                            )
                            return False
                        if not continuous:
                            return False
                        self.refresh([cl])
                        already_reloaded = True
        else:
            while not len(file["session_buffer"]["buffer"]):
                raw = file["fd"].readlines(self.max_buffer)
                if not raw or not len(raw):
                    ### Empty dataset -- Shuffle and reload
                    file["eof"] = True
                    file["fd"].close()
                    if already_reloaded:
                        print(
                            f"Warning - Class {cl} does not contain enough elements for tuple size {self.tuple_size}"
                        )
                        return False
                    if not continuous:
                        return False
                    self.refresh([cl])
                    already_reloaded = True
                else:
                    parsed = [
                        item.strip().split("\t") for item in raw if len(item)
                    ]
                    try:
                        src_ips, dst_ips = [
                            self.process_ip(item[2]) for item in parsed
                        ], [self.process_ip(item[3]) for item in parsed]
                    except:
                        src_ips, dst_ips = [None for item in parsed], [
                            None for item in parsed
                        ]
                    pos = [float(item[-1]) for item in parsed]
                    sols = [
                        self.parse_lengths(item[5].split(","))
                        for item in parsed
                    ]
                    sots = [
                        [float(t) for t in item[4].split(",")]
                        for item in parsed
                    ]
                    vecs = [
                        torch.tensor([sols[i], sots[i]])
                        for i in range(len(sols))
                    ]
                    file["session_buffer"] = {
                        "class": parsed[0][0],
                        "st": 0.0,
                        "buffer": vecs,
                        "src_ip": src_ips,
                        "dst_ip": dst_ips,
                        "pos": pos,
                    }
        return True

    def get_item(self, cl, pop=True, non_pop_offset=0, ips=False):
        if not pop and self.eval:
            pop = True
        if ips and pop:
            return (
                self.files[cl]["session_buffer"]["buffer"].pop(0),
                self.files[cl]["session_buffer"]["src_ip"].pop(0),
                self.files[cl]["session_buffer"]["dst_ip"].pop(0),
                self.files[cl]["session_buffer"]["pos"].pop(0),
            )
        elif ips and not pop:
            return (
                self.files[cl]["session_buffer"]["buffer"][non_pop_offset],
                self.files[cl]["session_buffer"]["src_ip"][non_pop_offset],
                self.files[cl]["session_buffer"]["dst_ip"][non_pop_offset],
                self.files[cl]["session_buffer"]["pos"][non_pop_offset],
            )
        elif not ips and pop:
            return self.files[cl]["session_buffer"]["buffer"].pop(
                0
            ), self.files[cl]["session_buffer"]["pos"].pop(0)
        else:
            return (
                self.files[cl]["session_buffer"]["buffer"][non_pop_offset],
                self.files[cl]["session_buffer"]["pos"][non_pop_offset],
            )


class ClassificationKeystrokeDataLoader(KeystrokeDataLoader):
    def __init__(
        self,
        directory,
        max_buffer=100000000,
        evaluation=False,
        run_length=7,
        include_others=False,
        use_ips=False,
    ):
        super().__init__(
            directory,
            "classification_manifest",
            max_buffer,
            evaluation,
            run_length,
            include_others,
        )
        self.reset_choice()
        self.uip = use_ips

    def reset_choice(self):
        self.choice_queue = []
        for cl in self.class_weights:
            if self.files[cl]["fd"] is None:
                continue
            if cl == "others" and not self.include_others:
                continue
            if cl == "reserved":
                continue
            cnt = self.class_weights[cl] // self.tuple_size
            self.choice_queue += [cl for _ in range(cnt)]
        random.shuffle(self.choice_queue)

    def sample(self):
        while True:
            if not len(self.choice_queue):
                return False

            cl = self.choice_queue.pop(0)

            # Refresh buffers
            if self.sample_raw(cl, False):
                break

        file = self.files[cl]

        samples = []
        masks = []
        positions = []
        offsets = []
        src, dst = None, None
        for tidx in range(self.tuple_size):
            if self.uip:
                raw, src, dst, p = self.get_item(cl, True, ips=True)
                vec, mask, position = self.transform_vector(raw)
            else:
                raw, p = self.get_item(cl, True)
                vec, mask, position = self.transform_vector(raw)
            samples.append(vec)
            masks.append(mask)
            positions.append(position)
            offsets.append(p)

        if len(samples) == 1:
            ts = samples[0].unsqueeze(0)
            masks = masks[0].unsqueeze(0)
            positions = positions[0]
            offsets = offsets[0]
        else:
            ts = torch.stack(samples)
            masks = torch.stack(masks)
            offsets = torch.stack(offsets)

        lbl = torch.zeros(self.output_dim)
        lbl[self.output_dim_map[cl]] = 1.0

        return {
            "meta": {
                "positions": positions,
                "real_class": file["session_buffer"]["class"],
                "st": file["session_buffer"]["st"],
            },
            "data": ts,
            "masks": masks,
            "label": lbl,
            "src": src,
            "dst": dst,
            "offsets": offsets,
        }

    def get_batch(self, batch_size=64, device="cpu"):
        batch = []
        labels = []
        masks = []
        meta = []
        src_ips = []
        dst_ips = []
        offsets = []
        for b in range(batch_size):
            s = self.sample()
            if not s:
                return False, None, None, None, None, None, None, None, None
            else:
                batch.append(s["data"])
                labels.append(s["label"])
                meta.append(s["meta"])
                masks.append(s["masks"])
                src_ips.append(s["src"])
                dst_ips.append(s["dst"])
                offsets.append(s["offsets"])

        batch = torch.stack(batch).to(device)
        labels = torch.stack(labels).to(device)
        masks = torch.stack(masks).to(device)
        offsets = torch.tensor(offsets).to(device)
        if self.uip:
            src_ips = torch.stack(src_ips).to(device)
            dst_ips = torch.stack(dst_ips).to(device)
        else:
            src_ips, dst_ips = None, None

        # with open("test_new.log", "a") as outfile:
        #    for i in range(batch.size(0)):
        #        #dat = ','.join(str(t) for t in batch[i].cpu().numpy())
        #        dat = "data"
        #        lab = labels[i].argmax().item()
        #        outfile.write(f"{lab}\t{dat}\n")

        return True, batch, labels, masks, None, meta, offsets, src_ips, dst_ips

    def weights(self, device="cpu"):
        weights = [0 for _ in range(self.output_dim)]
        for cl in self.output_dim_map:
            weights[self.output_dim_map[cl]] = self.class_weights[cl]
        total_weight = sum(weights)
        weights = torch.tensor(
            [total_weight / w if w > 0 else 0 for w in weights]
        ).float()
        weights *= len(weights) / weights.sum().item()
        weights = weights.to(device)
        return weights

    def user_count(self):
        return (
            len([key for key in self.class_weights if key != "reserved"])
            if self.include_others
            else len(
                [
                    key
                    for key in self.class_weights
                    if key != "others" and key != "reserved"
                ]
            )
        )


class AuthenticationKeystrokeDataLoader(KeystrokeDataLoader):
    def __init__(
        self,
        directory,
        max_buffer=10000000,
        evaluation=False,
        run_length=7,
        sample_length=True,
        use_lengths=True,
        use_ips=False,
        reserved_users=False,
    ):
        super().__init__(
            directory,
            "authentication_manifest",
            max_buffer,
            evaluation,
            run_length,
            False,
            sample_length,
            use_lengths,
            reserved_users,
        )
        self.user_weights = {
            c: self.class_weights[c]
            for c in self.class_weights
            if c != "others" and c != "reserved"
        }
        self.ru = reserved_users
        self.reset_choice()
        self.uip = use_ips

    def reset_choice(self):
        if not self.eval:
            wpc = max(self.class_weights.values())
        else:
            wpc = max(self.class_weights.values()) // self.tuple_size
        self.choice_queue = []
        # Same class samples
        for cl in self.user_weights:
            if self.files[cl]["fd"] is None:
                continue
            self.choice_queue += [(cl, cl) for _ in range(wpc)]

        # Diffenrent class samples
        for cl in self.user_weights:
            if self.files[cl]["fd"] is None:
                continue
            if not self.ru:
                other_classes = random.choices(
                    [
                        c
                        for c in self.class_weights.keys()
                        if c != cl
                        and c != "reserved"
                        and self.files[cl]["fd"] is not None
                    ],
                    k=wpc,
                )
                self.choice_queue += [
                    (cl_data, cl) for cl_data in other_classes
                ]
            else:
                other_classes = random.choices(["reserved"], k=wpc)
                self.choice_queue += [
                    (cl_data, cl) for cl_data in other_classes
                ]

        if not self.eval:
            random.shuffle(self.choice_queue)

    def sample(self, others=False):
        while True:
            if not len(self.choice_queue):
                return False

            data_cl, label_cl = self.choice_queue.pop(0)
            pos = data_cl == label_cl

            if self.sample_raw(data_cl, True):
                break

        file = self.files[data_cl]

        samples = []
        masks = []
        positions = []
        offsets = []
        src, dst = None, None
        for tidx in range(self.tuple_size):
            if self.uip:
                raw, src, dst, p = self.get_item(
                    data_cl,
                    pop=pos or data_cl == "others" or data_cl == "reserved",
                    non_pop_offset=tidx,
                    ips=self.uip,
                )
                vec, mask, position = self.transform_vector(raw)
            else:
                raw, p = self.get_item(
                    data_cl,
                    pop=pos or data_cl == "others" or data_cl == "reserved",
                    non_pop_offset=tidx,
                )
                vec, mask, position = self.transform_vector(raw)
            samples.append(vec)
            masks.append(mask)
            positions.append(position)
            offsets.append(p)

        if len(samples) == 1:
            ts = samples[0].unsqueeze(0)
            masks = masks[0].unsqueeze(0)
            positions = positions[0]
            offsets = offsets[0]
        else:
            ts = torch.stack(samples)
            masks = torch.stack(masks)
            offsets = torch.stack(offsets)

        lbl = torch.tensor([0.0, 1.0]) if pos else torch.tensor([1.0, 0.0])

        user_lbl = torch.zeros(self.output_dim)
        user_lbl[self.output_dim_map[label_cl]] = 1.0

        # print(data_cl)
        # print(label_cl)
        # print ({"meta": {"real_class": data_cl, "st": file['session_buffer']['st']}, "data": ts, "masks": masks, "user": user_lbl, "label": lbl})
        return {
            "meta": {
                "positions": positions,
                "real_class": data_cl,
                "st": file["session_buffer"]["st"],
            },
            "data": ts,
            "masks": masks,
            "user": user_lbl,
            "label": lbl,
            "src": src,
            "dst": dst,
            "offsets": offsets,
        }

    def get_batch(self, batch_size=64, others=False, device="cpu"):
        batch = []
        labels = []
        masks = []
        meta = []
        users = []
        src_ips = []
        offsets = []
        dst_ips = []
        for b in range(batch_size):
            s = self.sample(others=others)
            if not s:
                return False, None, None, None, None, None, None, None, None
            else:
                batch.append(s["data"])
                labels.append(s["label"])
                meta.append(s["meta"])
                masks.append(s["masks"])
                users.append(s["user"])
                src_ips.append(s["src"])
                dst_ips.append(s["dst"])
                offsets.append(s["offsets"])

        batch = torch.stack(batch).to(device)
        labels = torch.stack(labels).to(device)
        masks = torch.stack(masks).to(device)
        users = torch.stack(users).to(device)
        offsets = torch.tensor(offsets).to(device)
        if self.uip:
            src_ips = torch.stack(src_ips).to(device)
            dst_ips = torch.stack(dst_ips).to(device)
        else:
            src_ips, dst_ips = None, None

        # print(batch)
        # print(labels)
        # print(users)

        return (
            True,
            batch,
            labels,
            masks,
            users,
            meta,
            offsets,
            src_ips,
            dst_ips,
        )

    def weights(self, device="cpu"):
        return torch.tensor([0.5, 0.5]).float().to(device)

    def user_count(self):
        return len(self.user_weights.keys())


class AlternateAuthenticationKeystrokeDataLoader(
    AuthenticationKeystrokeDataLoader
):
    def __init__(
        self,
        directory,
        max_buffer=10000000,
        evaluation=False,
        max_run_length=32,
        min_run_length=16,
        dropout=1.0,
        sample_length=True,
        use_lengths=True,
        use_ips=False,
        reserved_users=False,
    ):
        super().__init__(
            directory,
            max_buffer,
            evaluation,
            max_run_length,
            sample_length,
            use_lengths,
            use_ips,
            reserved_users,
        )
        self.min_run_length = min_run_length
        self.dropout = dropout

    def set_tuple_size(self, ts):
        print("Error --- Alternate data loaders do not use tuple sizes")
        exit(0)

    def get_batch(self, batch_size=64, others=False, device="cpu", strict=True):
        while True:
            (
                flag,
                batch,
                labels,
                masks,
                users,
                meta,
                offsets,
                src_ips,
                dst_ips,
            ) = super().get_batch(batch_size, others, device)

            dropout = self.dropout
            if (
                (self.eval and dropout < 0)
                or (not self.eval and dropout == 0.0)
                or (not flag)
            ):
                return (
                    flag,
                    batch,
                    labels,
                    masks,
                    users,
                    meta,
                    offsets,
                    src_ips,
                    dst_ips,
                )

            positions = [m["positions"] for m in meta]
            N, _, _, D = batch.shape
            # dropout -- remove end of vectors. Deterministic for eval, random for training
            if self.eval:
                trunc_len = self.run_length - math.floor(
                    dropout * (self.run_length - self.min_run_length)
                )
                mask = (
                    torch.cat(
                        (
                            torch.zeros(N, trunc_len + 1).bool(),
                            torch.ones(N, self.run_length - trunc_len).bool(),
                        ),
                        dim=1,
                    )
                    .to(device)
                    .unsqueeze(1)
                )
                if strict:
                    metamask = (
                        torch.logical_and(torch.logical_not(mask), masks)
                        .any(dim=2)
                        .squeeze()
                    )
                    if metamask.all():
                        continue
                    metamask = torch.logical_not(metamask)
                    masks |= mask
                    batch = batch[metamask]
                    masks = masks[metamask]
                    offsets = offsets[metamask]
                    if src_ips is not None:
                        src_ips = src_ips[metamask]
                        dst_ips = dst_ips[metamask]
                    users = users[metamask]
                    labels = labels[metamask]
                    meta = [
                        meta[i]
                        for i in range(metamask.size(0))
                        if metamask[i].item()
                    ]
                else:
                    masks |= mask
            elif False:
                max_trunc_len = self.run_length - math.floor(
                    dropout * (self.run_length - self.min_run_length)
                )
                indices = torch.randint(
                    self.run_length - max_trunc_len, self.run_length + 1, (N,)
                ).to(device)
                mask = torch.zeros(N, self.run_length + 1).to(device)
                mask[torch.arange(N), indices] = 1
                mask = mask[:, :-1]
                mask = torch.cat(
                    (
                        torch.zeros(N, 1).to(device),
                        torch.cumsum(mask, dim=1).bool().to(device),
                    ),
                    dim=1,
                )
                masks |= mask.unsqueeze(1).bool()
            else:
                max_trunc_len = self.run_length - math.floor(
                    dropout * (self.run_length - self.min_run_length)
                )
                new_masks = torch.zeros_like(masks).bool()
                for b in range(new_masks.size(0)):
                    trunc_len = random.randint(
                        max_trunc_len, self.run_length + 1
                    )
                    if positions[b] > trunc_len:
                        offset = random.randint(0, positions[b] - trunc_len + 1)
                    else:
                        offset = 0
                    new_masks[b, :, 1 : offset + 1] = True
                    new_masks[b, :, trunc_len + offset + 1 :] = True
                    offsets[b] += offset / 512
                masks |= new_masks

            return (
                True,
                batch,
                labels,
                masks,
                users,
                meta,
                offsets,
                src_ips,
                dst_ips,
            )

    def typenetbatch(
        self, batch_size=64, others=False, device="cpu", strict=True
    ):
        (
            flag,
            batch,
            labels,
            masks,
            users,
            meta,
            offsets,
            src_ips,
            dst_ips,
        ) = self.get_batch(batch_size, others, device, strict)
        if not flag:
            return flag, batch, labels, [], users, meta
        positions = []
        for i in range(masks.shape[0]):
            for j in range(masks[i].shape[1]):
                if not masks[i, 0, j].item():
                    start_idx = j
                    break
            for j in range(masks[i].shape[1] - 1, -1, -1):
                if not masks[i, 0, j].item():
                    end_idx = j
                    break
            # print(masks[i])
            # print(start_idx)
            # print(end_idx)
            batch[i, :, : end_idx - start_idx + 1] = batch[
                i, :, start_idx : end_idx + 1
            ]
            batch[i, :, end_idx - start_idx + 1 :] = 0.0
            positions.append(end_idx - start_idx)
        return flag, batch, labels, positions, users, meta

    def set_dropout(self, dp):
        self.dropout = dp


class AlternateClassificationKeystrokeDataLoader(
    ClassificationKeystrokeDataLoader
):
    def __init__(
        self,
        directory,
        max_buffer=10000000,
        evaluation=False,
        max_run_length=32,
        min_run_length=16,
        dropout=1.0,
        use_ips=False,
        include_others=False,
    ):
        super().__init__(
            directory,
            max_buffer,
            evaluation,
            max_run_length,
            include_others,
            use_ips,
        )
        self.min_run_length = min_run_length
        self.dropout = dropout

    def set_tuple_size(self, ts):
        print("Error --- Alternate data loaders do not use tuple sizes")
        exit(0)

    def get_batch(self, batch_size=64, device="cpu", strict=True):
        while True:
            (
                flag,
                batch,
                labels,
                masks,
                _,
                meta,
                offsets,
                src_ips,
                dst_ips,
            ) = super().get_batch(batch_size, device)

            dropout = self.dropout
            if (
                (self.eval and dropout < 0)
                or (not self.eval and dropout == 0.0)
                or (not flag)
            ):
                return (
                    flag,
                    batch,
                    labels,
                    masks,
                    None,
                    meta,
                    offsets,
                    src_ips,
                    dst_ips,
                )

            positions = [m["positions"] for m in meta]
            N, _, _, D = batch.shape
            # dropout -- remove end of vectors. Deterministic for eval, random for training
            if self.eval:
                trunc_len = self.run_length - math.floor(
                    dropout * (self.run_length - self.min_run_length)
                )
                mask = (
                    torch.cat(
                        (
                            torch.zeros(N, trunc_len + 1).bool(),
                            torch.ones(N, self.run_length - trunc_len).bool(),
                        ),
                        dim=1,
                    )
                    .to(device)
                    .unsqueeze(1)
                )
                if strict:
                    metamask = (
                        torch.logical_and(torch.logical_not(mask), masks)
                        .any(dim=2)
                        .squeeze()
                    )
                    if metamask.all():
                        continue
                    metamask = torch.logical_not(metamask)
                    masks |= mask
                    offsets = offsets[metamask]
                    if src_ips is not None:
                        src_ips = src_ips[metamask]
                        dst_ips = dst_ips[metamask]
                    batch = batch[metamask]
                    masks = masks[metamask]
                    labels = labels[metamask]
                    meta = [
                        meta[i]
                        for i in range(metamask.size(0))
                        if metamask[i].item()
                    ]
                else:
                    masks |= mask
            elif False:
                max_trunc_len = self.run_length - math.floor(
                    dropout * (self.run_length - self.min_run_length)
                )
                indices = torch.randint(
                    self.run_length - max_trunc_len, self.run_length + 1, (N,)
                ).to(device)
                mask = torch.zeros(N, self.run_length + 1).to(device)
                mask[torch.arange(N), indices] = 1
                mask = mask[:, :-1]
                mask = torch.cat(
                    (
                        torch.zeros(N, 1).to(device),
                        torch.cumsum(mask, dim=1).bool().to(device),
                    ),
                    dim=1,
                )
                masks |= mask.unsqueeze(1).bool()
            else:
                max_trunc_len = self.run_length - math.floor(
                    dropout * (self.run_length - self.min_run_length)
                )
                new_masks = torch.zeros_like(masks).bool()
                for b in range(new_masks.size(0)):
                    trunc_len = random.randint(
                        max_trunc_len, self.run_length + 1
                    )
                    if positions[b] > trunc_len:
                        offset = random.randint(0, positions[b] - trunc_len + 1)
                    else:
                        offset = 0
                    new_masks[b, :, 1 : offset + 1] = True
                    new_masks[b, :, trunc_len + offset + 1 :] = True
                    offsets[b] += offset / 512
                masks |= new_masks

            return (
                True,
                batch,
                labels,
                masks,
                None,
                meta,
                offsets,
                src_ips,
                dst_ips,
            )

    def set_dropout(self, dp):
        self.dropout = dp
