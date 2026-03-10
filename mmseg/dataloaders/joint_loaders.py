from itertools import cycle
import torch


class JointLoader:
    def __init__(self, dataloader1, dataloader2):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.sampler = dataloader1.sampler

    def __iter__(self):
        # self.iter_loader = iter(zip(cycle(self.loader_w_label), self.loader_wo_label))
        # print("hello world")
        self.iter_loader1 = iter(self.dataloader1)
        self.iter_loader2 = iter(self.dataloader2)
        return self

    def __next__(self):
        try:
            data1, data2 = next(self.iter_loader1), next(self.iter_loader2)
            for k in data1.keys():
                if isinstance(data1[k].data[0], list):
                    data1[k].data[0].extend(data2[k].data[0])
                elif isinstance(data1[k].data[0], torch.Tensor):
                    data1[k].data[0] = torch.cat([data1[k].data[0], data2[k].data[0]], dim=0)
                else:
                    raise NotImplementedError
            return data1
        except StopIteration:
            raise StopIteration
        # try:
        #     data_w_label, data_wo_label = next(self.iter_loader)
        #     for k in data_w_label.keys():
        #         if isinstance(data_w_label[k].data[0], list):
        #             data_w_label[k].data[0].extend(data_wo_label[k].data[0])
        #         elif isinstance(data_w_label[k].data[0], torch.Tensor):
        #             data_w_label[k].data[0] = torch.cat([data_w_label[k].data[0], data_wo_label[k].data[0]], dim=0)
        #         else:
        #             raise NotImplementedError
        #     return data_w_label
        # except StopIteration:
        #     raise StopIteration

        # try:
        #     data_w_label, data_wo_label = next(self.iter_loader)
        #     return data_w_label
        # except StopIteration:
        #     raise StopIteration

    def __len__(self):
        return len(self.dataloader1)