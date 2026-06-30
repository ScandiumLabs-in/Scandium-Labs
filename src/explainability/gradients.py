import torch


def integrated_gradients(
    model, crystal_graph, line_graph, target_task="log_ionic_conductivity", n_steps=50
):
    baseline_x = torch.zeros_like(crystal_graph.x)
    input_x = crystal_graph.x.clone().requires_grad_(True)
    attributions = torch.zeros_like(input_x)

    for alpha in torch.linspace(0, 1, n_steps):
        interpolated = baseline_x + alpha * (input_x - baseline_x)
        interpolated.requires_grad_(True)

        crystal_graph.x = interpolated
        predictions = model(crystal_graph, line_graph)
        target = predictions[target_task].sum()

        grads = torch.autograd.grad(target, interpolated, create_graph=False)[0]
        attributions += grads / n_steps

    final_attributions = attributions * (input_x - baseline_x)
    return final_attributions.detach()
