
import torch
from collections.abc import Iterable
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper


def ind_sel(target: torch.Tensor, ind: torch.Tensor, dim: int=1):
    """
    :param target: [... (can be k or 1), n > M, ...]
    :param ind: [... (k), M]
    :param dim: dim to apply index on
    :return: sel_target [... (k), M, ...]
    """
    assert len(ind.shape) > dim, "Index must have the target dim, but get dim: %d, ind shape: %s" % (dim, str(ind.shape))

    target = target.expand(*tuple([ind.shape[k] if target.shape[k] == 1 else -1 for k in range(dim)] + [-1, ] * (len(target.shape) - dim)))

    ind_pad = ind

    if len(target.shape) > dim + 1:
        for _ in range(len(target.shape) - (dim + 1)):
            ind_pad = ind_pad.unsqueeze(-1)
        ind_pad = ind_pad.expand(*(-1, ) * (dim + 1), *target.shape[(dim + 1)::])

    return torch.gather(target, dim=dim, index=ind_pad)


class RandomSamplingFoo():
    def __call__(self, total_len, num_sample, **kwargs):
        return torch.randperm(total_len)[:num_sample]


class RandomPatchSamplingFoo():
    def __init__(self, patch_size, tensor_size=(256, 256), unique=False, max_try=5):
        self.patch_size = patch_size
        self.unique = unique
        self.max_try = max_try
        self.tensor_size = tensor_size

    def __call__(self, total_len, num_sample, **kwargs):
        h, w = self.tensor_size
        patch_size = self.patch_size

        out = torch.zeros(0, dtype=torch.long)

        remain = num_sample
        for ee in range(self.max_try):
            num_patch = remain // (patch_size * patch_size)
            yy = torch.randint(patch_size // 2, h - 1 - patch_size // 2, (num_patch, 1, 1))
            xx = torch.randint(patch_size // 2, w - 1 - patch_size // 2, (num_patch, 1, 1))

            y_sliding = torch.arange(patch_size // 2 + 1 - patch_size, patch_size // 2 + 1).view(1, -1, 1)
            x_sliding = torch.arange(patch_size // 2 + 1 - patch_size, patch_size // 2 + 1).view(1, 1, -1)

            yy = yy + y_sliding
            xx = xx + x_sliding

            out_index = (yy * w + xx).view(-1).type(torch.long)
            if not self.unique:
                remain = num_sample - out_index.shape[0]
                return torch.cat((out_index, torch.randint(0, w * h - 1, (remain, ))))

            out = torch.unique(torch.cat((out, out_index)))
            remain = num_sample - out.shape[0]

            if remain < patch_size * patch_size:
                break
        if remain == 0:
            return out
        mask_unfilled = torch.ones((h, w), dtype=torch.float32).view(-1)
        mask_unfilled = mask_unfilled.scatter_(dim=0, index=out, value=0.0)
        return torch.cat((out, torch.multinomial(mask_unfilled, remain)))

# if __name__ == '__main__':
#     sampler = RandomPatchSamplingFoo(3, unique=True)
#     get = sampler(64, 20, torch.ones((10, 10)))
#     print(get // 8)
#     print(get % 8)

class LinearRate():
    def __init__(self, start_val, end_val, cutoff_epoch):
        self.start_val = start_val
        self.end_val = end_val
        self.cutoff_epoch = cutoff_epoch

    def __call__(self, current_epoch):
        return min(current_epoch / self.cutoff_epoch, 1) * (self.end_val - self.start_val) + self.start_val



class BalancedMaskSamplingFoo():
    def __init__(self, mask_sampling_rate=0.5, mask_name='mask', update_rate_policy=None, eps=1e-5):
        self.mask_sampling_rate = mask_sampling_rate
        self.mask_name = mask_name
        self.update_rate_policy = update_rate_policy
        self.eps = eps

    def update_rate(self, epoch, ):
        if self.update_rate_policy is not None:
            self.mask_sampling_rate = self.update_rate_policy(epoch)

    def __call__(self, total_len, num_sample, **kwargs):
        mask = kwargs.get(self.mask_name)
        mask = mask.type(torch.float32)
        n_sample_mask = min(int(num_sample * self.mask_sampling_rate), (mask > self.eps).sum().item())
        n_sample_bg = num_sample - n_sample_mask

        if n_sample_mask <= 10:
            return torch.multinomial(1 - mask, n_sample_bg)

        samples_mask = torch.multinomial(mask, n_sample_mask)
        samples_bg = torch.multinomial(1 - mask, n_sample_bg)

        return torch.cat((samples_mask, samples_bg), dim=-1)

class IndexSampler():
    def __init__(self, num_sample, sampleing_foo, sample_args, target_dims=None, remain_dims=None, tbar=False):
        if isinstance(sample_args, str):
            sample_args = (sample_args, )

        if target_dims is not None:
            if isinstance(target_dims, int):
                target_dims = (target_dims, )
            self.target_dims = tuple(target_dims)
            self.remain_dims = None
        else:
            if isinstance(remain_dims, int):
                remain_dims = (remain_dims, )
            self.remain_dims = tuple(remain_dims)
            self.target_dims = None

        self.num_sample = num_sample
        self.sample_args = tuple(sample_args)
        self.tbar = tbar

        self.sampleing_foo = sampleing_foo

        assert len(self.sample_args) > 0

    def __call__(self, func):
        """
        Function decorator for Batchifier.
        :param func: a callable function or object (e.g. torch.nn.functional)
        :return: the batchified function
        """
        def wrapper(*args, **kwargs):
            kwargs = dict(kwargs)

            total_len = -1

            recorded_shape = None
            save_idx = None
            for k in self.sample_args:
                get = kwargs[k]

                assert isinstance(get, torch.Tensor)

                if self.target_dims is not None:
                    this_dims = tuple([d if d >= 0 else len(get.shape) + d for d in self.target_dims])
                    other_dims = tuple([i for i in range(len(get.shape)) if i not in this_dims])
                else:
                    other_dims = tuple([d if d >= 0 else len(get.shape) + d for d in self.remain_dims])
                    this_dims = tuple([i for i in range(len(get.shape)) if i not in other_dims])

                to_shape = [get.shape[i] if i in other_dims else -1 for i in range(len(get.shape))]
                t_l = len(to_shape)

                for i in range(t_l - 1):
                    if to_shape[t_l - 1 - i] == -1 and to_shape[t_l - 2 - i] == -1:
                        del to_shape[t_l - 1 - i]

                assert to_shape.count(-1) == 1

                to_record_shape = get.shape[0:to_shape.index(-1) + len(this_dims)]
                if recorded_shape is None:
                    recorded_shape = tuple(to_record_shape)
                    save_idx = to_shape.index(-1)
                else:
                    assert recorded_shape == tuple(to_record_shape)

                kwargs[k] = get.view(*to_shape)

                total_len = kwargs[k].shape[to_shape.index(-1)]

            sampling_idx = self.sampleing_foo(total_len, self.num_sample, **kwargs)
            # sampling_idx = torch.randperm(total_len)[:self.num_sample]

            if to_shape.index(-1) > len(sampling_idx.shape) - 1:
                exec("sampling_idx = sampling_idx[%s]" % ('None, ' * (to_shape.index(-1) - len(sampling_idx.shape) + 1)))

            assert total_len >= 0, 'No batchify parameters found!'

            out = []
            
            this_kwargs = dict()
            for k in kwargs.keys():
                this_kwargs[k] = kwargs[k]
                if k in self.sample_args:
                    this_kwargs[k] = ind_sel(this_kwargs[k], dim=to_shape.index(-1), ind=sampling_idx)

            return func(*args, **this_kwargs)
            

        return wrapper

'''
    training_args =  ('ray_dirs_', 'mask', 'img_', 'zbuf_')
    total_epochs = conf.get_int('train.total_epochs', default=2000)
    batch_size = conf.get_int('train.batch_size', default=1024 * 16)
    batchifier = Batchifier(batch_size=batch_size, batch_args=training_args, target_dims=(0, 1))
    sampler = IndexSampler(num_sample=batch_size, sampleing_foo=sampling_foo, sample_args=training_args, target_dims=(0, 1))
    sampler(train_step)(ray_dirs_=ray_dir, ray_source_=ray_source, mask=mask, img_=image)
'''