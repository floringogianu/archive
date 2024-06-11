import gc
from collections import defaultdict
import torch
from IPython.lib.pretty import pretty


def wtf_mem(topk=None):
    """ Call it when playgrounds says OOMKilled.
    """
    obj_cnt = defaultdict(int)
    max_key_len = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or torch.is_storage(obj):
            try:
                shape = tuple(obj.size())
            except TypeError:
                shape = obj.size()

            key = f"[T]: {pretty(type(obj))}, {shape}"
            obj_cnt[key] += 1
        else:
            key = pretty(type(obj))
            obj_cnt[key] += 1
        max_key_len = max(max_key_len, len(key))

    sorted_cnt = sorted(obj_cnt.items(), key=lambda kv: kv[1], reverse=True)
    th_objects = {k: v for k, v in sorted_cnt if "[T]" in k}
    py_objects = {k: v for k, v in sorted_cnt if "[T]" not in k}

    header = "{:{width}} |    {:6}".format("Torch", "Count", width=max_key_len)
    sep = "-" * len(header)
    table = f"{sep}\n{header}\n{sep}\n"

    # print torch objects
    for i, (k, v) in enumerate(th_objects.items()):
        table += "{:{width}} |   {:6d}\n".format(k, v, width=max_key_len)
        if topk is not None and i == topk:
            table += f"... {len(th_objects) - i} tensors not displayed ...\n"
            break

    # print the other python objects
    header = "{:{width}} |    {:6}".format("Other", "Count", width=max_key_len)
    table += f"\n{sep}\n{header}\n{sep}\n"
    for i, (k, v) in enumerate(py_objects.items()):
        table += "{:{width}} |   {:6d}\n".format(k, v, width=max_key_len)
        if topk is not None and i == topk:
            table += f"... {len(py_objects) - i} py objects not displayed ...\n"
            break
    table += sep + "\n"
    table += "{:{width}} |   {:6d}\n".format(
        "Tensors Allocated", sum(th_objects.values()), width=max_key_len
    )
    table += "{:{width}} |   {:6d}\n".format(
        "Total Allocated", sum(obj_cnt.values()), width=max_key_len
    )
    table += sep
    print(table)


def main():
    a = torch.tensor(100)
    z = torch.tensor(100)
    b = torch.Storage(10)
    d = torch.Storage(20)
    c = torch.nn.Parameter(torch.rand(10))
    wtf_mem(topk=3)


if __name__ == "__main__":
    main()
