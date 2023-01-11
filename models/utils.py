import torch
import numpy as np
import os


class ReDirectSTD(object):
    """
    overwrites the sys.stdout or sys.stderr
    Args:
      fpath: file path
      console: one of ['stdout', 'stderr']
      immediately_visiable: False
    Usage example:
      ReDirectSTD('stdout.txt', 'stdout', False)
      ReDirectSTD('stderr.txt', 'stderr', False)
    """

    def __init__(self, fpath=None, console='stdout', immediately_visiable=False):
        import sys
        import os
        assert console in ['stdout', 'stderr']
        self.console = sys.stdout if console == "stdout" else sys.stderr
        self.file = fpath
        self.f = None
        self.immediately_visiable = immediately_visiable
        if fpath is not None:
            # Remove existing log file
            if os.path.exists(fpath):
                os.remove(fpath)
        if console == 'stdout':
            sys.stdout = self
        else:
            sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, **args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            if not os.path.exists(os.path.dirname(os.path.abspath(self.file))):
                os.makedirs(os.path.dirname(os.path.abspath(self.file)))
            if self.immediately_visiable:
                with open(self.file, 'a') as f:
                    f.write(msg)
            else:
                if self.f is None:
                    self.f = open(self.file, 'w')
                self.f.write(msg)

    def flush(self):
        self.console.flush()
        if self.f is not None:
            self.f.flush()
            import os
            os.fsync(self.f.fileno())

    def close(self):
        self.console.close()
        if self.f is not None:
            self.f.close()


def may_mkdir(fname):
    if not os.path.exists(os.path.dirname(os.path.abspath(fname))):
        os.makedirs(os.path.dirname(os.path.abspath(fname)))


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / (self.count + 1e-10)


def parse_param(param_str):
    """
    Parse param_str like 'a=1,b=2.5,c=Name' into dict
    """
    str_list = param_str.strip().split(',')
    params = {}
    for s in str_list:
        name, value = s.split('=')
        if value.find('.') > 0:
            value = float(value)
        elif value.isdigit():
            value = int(value)
        else:
            value = str(value)
        params[name] = value

    return params

