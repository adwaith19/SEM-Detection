from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1="path/to/real/image/folder",
    input2="path/to/fake/image/folder",
    cuda=False,
    fid=True,
)

print(metrics)
