import torch.distributed as dist
import math
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, DistributedSampler
import random
import torch
from Dataset.multi_dataset import multi_dataset

def make_batch(index_list, batch_size, drop_last):  
    if drop_last:
        batches = []
        whole_batch_num = len(index_list)//batch_size
        for _ in range(whole_batch_num):
            batches.append(index_list[batch_size*_:(batch_size*(_+1))])
    else:
        batches = []
        whole_batch_num = math.ceil(len(index_list)/batch_size)
        for _ in range(whole_batch_num):
            batches.append(index_list[batch_size*_:(batch_size*(_+1))])
    return batches   
    
def batch_generation(dataset,batch_size_2D, batch_size_3D,drop_last=False,shuffle = True, seed = 0):
    
    len_2D = len(dataset.data_whole_2D)
    len_3D = len(dataset.data_whole_3D)
    index_2D = list(range(len_2D))
    index_3D = list(range(len_2D,(len_2D+len_3D)))
    assert len(index_2D) + len(index_3D) == len(dataset.data_whole)
    
    if shuffle:   
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(seed)
        random.shuffle(index_2D)
        random.shuffle(index_3D)
        
    batch_2D = make_batch(index_2D, batch_size_2D, drop_last)
    batch_3D = make_batch(index_3D, batch_size_3D, drop_last)
        
    batch_chunk = batch_2D + batch_3D 
    return batch_chunk               
    
class My_DistributedBatchSampler(Sampler):
    """ Iterable wrapper that distributes data across multiple workers.

    Args:
        iterable (iterable)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within ``num_replicas``.

    Example:
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=0))
        [0, 2, 4, 6, 8]
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=1))
        [1, 3, 5, 7, 9]
    """

    def __init__(self, dataset, num_replicas=None, rank=None, batch_size_2D = 4, batch_size_3D = 1, drop_last = False, shuffle = True, seed: int = 0):
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.dataset = dataset
        self.batch_size_2D = batch_size_2D
        self.batch_size_3D = batch_size_3D
        self.seed = seed 
        self.epoch = 0
        
        if num_replicas is None or rank is None:  # pragma: no cover
            if not torch.distributed.is_initialized():
                raise RuntimeError('Requires `torch.distributed` to be initialized.')

            self.num_replicas = (
                torch.distributed.get_world_size() if num_replicas is None else num_replicas)
            self.rank = torch.distributed.get_rank() if rank is None else rank

        indices =  batch_generation(self.dataset,self.batch_size_2D,self.batch_size_3D,self.drop_last,self.shuffle)
        if self.rank >= self.num_replicas:
            raise IndexError('`rank` must be smaller than the `num_replicas`.')

        if self.drop_last and len(indices) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(indices) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(indices) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        
    def __iter__(self):
        indices =  batch_generation(self.dataset,self.batch_size_2D,self.batch_size_3D,self.drop_last,self.shuffle,self.seed + self.epoch)
        # print(indices)
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            random.shuffle(indices)
            
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        

# print(My_DistributedBatchSampler)
# Train_dataset = multi_dataset(text_tokenizer = '/mnt/petrelfs/share_data/zhangxiaoman/CODE/RadFM/src/Language_models/tokenizer')    

# DDP_sample_0 = list(My_DistributedBatchSampler(dataset= Train_dataset , num_replicas = 32, rank = 0,))
# DDP_sample_1 = list(My_DistributedBatchSampler(dataset= Train_dataset , num_replicas = 32, rank = 1,))

# for ii in DDP_sample_0:
#     print(ii)

# for ii in DDP_sample_1:
#     print(ii)
    
