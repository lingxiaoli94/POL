import torch


def compute_jacobian(outputs, inputs, create_graph=True, retain_graph=True):
    # outputs: ...xD1, inputs: ...xD2
    # returns: ...xD1xD2
    J = torch.cat([
        torch.autograd.grad(
            outputs=outputs[..., d], inputs=inputs,
            create_graph=create_graph, retain_graph=retain_graph,
            grad_outputs=torch.ones(inputs.size()[:-1]).to(inputs)
        )[0].unsqueeze(-2)
        for d in range(outputs.shape[-1])
    ], -2)  # ...xD1xD2
    return J
