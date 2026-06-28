import torch


def optimize_memory():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def enable_gradient_checkpointing(model):
    from torch.utils.checkpoint import checkpoint_sequential
    return model


def profile_gpu_memory(model, crystal_graph, line_graph):
    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(activities=[ProfilerActivity.GPU]) as prof:
        with record_function("model_forward"):
            _ = model(crystal_graph, line_graph)

    print(prof.key_averages().table(
        sort_by="cuda_memory_usage", row_limit=10
    ))
