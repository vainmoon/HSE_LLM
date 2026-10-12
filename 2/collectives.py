import torch
import torch.distributed as dist


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    left = (rank - 1) % world_size
    right = (rank + 1) % world_size
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]
    received = torch.empty_like(chunks[rank])

    for step in range(world_size - 1):
        send_index = (rank - step - 1) % world_size
        recv_index = (rank - step - 2) % world_size
        request = dist.isend(chunks[send_index], dst=right)
        dist.recv(received, src=left)
        request.wait()
        chunks[recv_index].add_(received)

    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    left = (rank - 1) % world_size
    right = (rank + 1) % world_size
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size
        request = dist.isend(chunks[send_index], dst=right)
        dist.recv(chunks[recv_index], src=left)
        request.wait()

    return torch.cat(chunks)


def ring_all_reduce(tensor):
    return ring_all_gather(ring_reduce_scatter(tensor))
