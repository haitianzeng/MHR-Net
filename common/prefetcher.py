import torch


class Prefetcher_test():
    def __init__(self, loader):
        self.loader = iter(loader)
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_inputs_3d, self.batch_inputs_2d, self.batch_inputs_scale = next(self.loader)
        except StopIteration:
            self.next_inputs_3d = None
            self.next_inputs_2d = None
            self.next_inputs_scale = None
            return

        if torch.cuda.is_available():
            with torch.cuda.stream(self.stream):
                self.next_inputs_3d = self.batch_inputs_3d.cuda(non_blocking=True)
                self.next_inputs_2d = self.batch_inputs_2d.cuda(non_blocking=True)
        else:
            self.next_inputs_3d = self.batch_inputs_3d
            self.next_inputs_2d = self.batch_inputs_2d
        self.next_inputs_scale = self.batch_inputs_scale

    def next(self):
        if torch.cuda.is_available():
            torch.cuda.current_stream()
        inputs_3d = self.next_inputs_3d
        inputs_2d = self.next_inputs_2d
        inputs_scale = self.next_inputs_scale

        self.preload()
        return inputs_3d, inputs_2d, inputs_scale


class Prefetcher_train():
    def __init__(self, loader):
        self.loader = iter(loader)
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        if torch.cuda.is_available():
            with torch.cuda.stream(self.stream):
                self.next_input = self.batch.cuda(non_blocking=True)
        else:
            self.next_input = self.batch

    def next(self):
        if torch.cuda.is_available():
            torch.cuda.current_stream()
        input = self.next_input

        self.preload()
        return input
