import numpy as np
import torch
import torch.nn.functional as F


def get_appnp_profile(data, hops=None, alpha=0.1, k_iterations=10):
    if hops is None:
        hops = [0, 2]

    x = data.x
    adj_t = data.adj_t
    deg = adj_t.sum(dim=1).to(torch.float).clamp(min=1)
    norm = deg.pow(-0.5).unsqueeze(1)

    smoothed_features = []
    h_initial = x
    if 0 in hops:
        smoothed_features.append(h_initial)

    h_current = h_initial
    for i in range(1, k_iterations + 1):
        h_prev_prop = h_current * norm
        h_prev_prop = adj_t.matmul(h_prev_prop)
        h_prev_prop = h_prev_prop * norm
        h_current = (1 - alpha) * h_prev_prop + alpha * h_initial

        if i in hops:
            smoothed_features.append(h_current)

    final_features = []
    seen_hops = set()
    for hop in hops:
        if hop in seen_hops:
            continue
        if hop == 0:
            final_features.append(smoothed_features[0])
        else:
            final_features.append(smoothed_features[hops.index(hop)])
        seen_hops.add(hop)
    return torch.cat(final_features, dim=1)


class LogReg(torch.nn.Module):
    def __init__(self, ft_in, nb_classes):
        super().__init__()
        self.fc = torch.nn.Linear(ft_in, nb_classes)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc.weight.data)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        return self.fc(x)


class AttentionalFiLMGenerator(torch.nn.Module):
    def __init__(self, task_emb_dim, node_feature_dim, output_dim, num_heads=3, rank=4):
        super().__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.rank = rank

        self.generator_A = torch.nn.Linear(task_emb_dim, num_heads * rank)
        self.generator_B = torch.nn.Parameter(torch.empty(rank, output_dim * 2))
        self.attention_net = torch.nn.Linear(node_feature_dim, num_heads)
        self.task_embedding = torch.nn.Parameter(torch.randn(1, task_emb_dim))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.generator_A.weight, a=np.sqrt(3))
        torch.nn.init.kaiming_uniform_(self.generator_B, a=np.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.attention_net.weight, a=np.sqrt(5))
        if self.generator_A.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.generator_A.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.generator_A.bias, -bound, bound)
        if self.attention_net.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.attention_net.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.attention_net.bias, -bound, bound)

    def forward(self, node_features):
        delta_A = self.generator_A(self.task_embedding).view(self.num_heads, self.rank)
        base_modulations = torch.matmul(delta_A, self.generator_B)
        attention_scores = self.attention_net(node_features)
        attention_weights = F.softmax(attention_scores, dim=1)
        final_modulations = torch.matmul(attention_weights, base_modulations)
        gamma, beta = torch.split(final_modulations, self.output_dim, dim=1)
        return gamma, beta


class TAAMNet(torch.nn.Module):
    def __init__(self, sgc_backbone, total_classes, cls_per_task, d_data, device, args):
        super().__init__()
        self.sgc_backbone = sgc_backbone
        self.total_classes = total_classes
        self.cls_per_task = cls_per_task
        self.d_data = d_data
        self.device = device
        self.args = args

        self.task_emb_dim = args.get("task_emb_dim", 64)
        self.num_heads = args.get("num_heads", 3)
        self.rank = args.get("rank", 4)
        self.hid_units = sgc_backbone.out_dim

        self.film_generators = torch.nn.ModuleList()
        self.layer_norm = torch.nn.LayerNorm(d_data)
        self.task_prototypes = []
        self.tasks = 0
        self.classifier = LogReg(self.hid_units, cls_per_task)

    def _create_new_adapter(self):
        return AttentionalFiLMGenerator(
            task_emb_dim=self.task_emb_dim,
            node_feature_dim=self.d_data,
            output_dim=self.d_data,
            num_heads=self.num_heads,
            rank=self.rank,
        ).to(self.device)

    def _get_prototype(self, data, node_ids):
        with torch.no_grad():
            enhanced_features = get_appnp_profile(
                data,
                hops=self.args.get("prototype_hops", [0, 2]),
                alpha=self.args.get("prototype_alpha", 0.1),
                k_iterations=self.args.get("prototype_steps", 10),
            )
            prototype = torch.mean(enhanced_features[node_ids], dim=0)
        return prototype

    def prepare_for_new_task(self, data, train_ids, n_new_classes):
        trainable_params = []

        new_adapter = self._create_new_adapter()
        self.film_generators.append(new_adapter)
        current_prototype = self._get_prototype(data, train_ids)
        trainable_params.extend(list(self.film_generators[-1].parameters()))

        if self.tasks == 0:
            if self.classifier.fc.out_features != n_new_classes:
                self.classifier = LogReg(self.hid_units, max(n_new_classes, 1)).to(self.device)
        elif n_new_classes > 0:
            prev_n_cls = self.classifier.fc.out_features
            new_n_cls = prev_n_cls + n_new_classes
            new_classifier = LogReg(self.hid_units, new_n_cls).to(self.device)
            with torch.no_grad():
                new_classifier.fc.weight.data[:prev_n_cls, :] = self.classifier.fc.weight.data
                new_classifier.fc.bias.data[:prev_n_cls] = self.classifier.fc.bias.data
            self.classifier = new_classifier

        self.task_prototypes.append(current_prototype.cpu())
        self.tasks += 1
        return trainable_params

    def infer_task_id(self, data, node_ids):
        test_prototype = self._get_prototype(data, node_ids)
        prototypes = torch.stack(self.task_prototypes).to(test_prototype.device)
        similarity_scores = F.cosine_similarity(test_prototype, prototypes, dim=1)
        inferred_id = torch.argmax(similarity_scores)
        return inferred_id.item()

    def forward(self, data, task_id):
        x = self.sgc_backbone.neighbor_agg(data.x, data.adj_t)
        adapter = self.film_generators[task_id]
        gamma, beta = adapter(x)
        x_modulated = gamma * self.layer_norm(x) + beta
        x_modulated = x_modulated + x
        x_processed = self.sgc_backbone.feat_trans(x_modulated)
        logits = self.classifier(x_processed)
        return logits


def _plain_accuracy(logits, labels, offset1=0, offset2=None):
    if offset2 is None:
        offset2 = logits.shape[1]

    task_logits = logits[:, offset1:offset2]
    pred = task_logits.detach().argmax(dim=1)
    adjusted = labels - offset1
    if len(adjusted) == 0:
        return 0.0
    correct = torch.sum(pred == adjusted)
    return correct.item() / len(adjusted)


def _plain_accuracy_selected(logits, labels, class_indices):
    if len(labels) == 0 or len(class_indices) == 0:
        return 0.0

    class_index_tensor = torch.tensor(class_indices, device=logits.device, dtype=torch.long)
    selected_logits = logits.index_select(1, class_index_tensor)
    remapped = torch.empty_like(labels)
    for local_idx, global_idx in enumerate(class_indices):
        remapped[labels == global_idx] = local_idx

    pred = selected_logits.detach().argmax(dim=1)
    correct = torch.sum(pred == remapped)
    return correct.item() / len(remapped)


class TAAM():
    def __init__(self, model, tasks, device, args, cls_per_task):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.device = device
        self.args = args
        self.cls_per_task = cls_per_task
        self.nb_classes = max(int(torch.max(task.y).item()) for task in tasks) + 1
        d_data = tasks[0].x.shape[1]
        self.net = TAAMNet(model, self.nb_classes, cls_per_task, d_data, device, args).to(device)
        self.opt = None
        self.class_to_idx = {}
        self.task_class_indices = []

    def _get_task_new_classes(self, task):
        if hasattr(task, "new_nodes_mask") and int(task.new_nodes_mask.sum()) > 0:
            return sorted(int(x) for x in torch.unique(task.y[task.new_nodes_mask.bool()]).tolist())
        if hasattr(task, "new_node_mask") and int(task.new_node_mask.sum()) > 0:
            return sorted(int(x) for x in torch.unique(task.y[task.new_node_mask.bool()]).tolist())
        return sorted(int(x) for x in torch.unique(task.y[task.train_mask]).tolist())

    def _encode_labels(self, labels):
        mapped = torch.empty_like(labels)
        for global_idx, mapped_idx in self.class_to_idx.items():
            mapped[labels == global_idx] = mapped_idx
        return mapped

    def prepare(self, task):
        current_task_classes = self._get_task_new_classes(task)
        newly_added_classes = [cls for cls in current_task_classes if cls not in self.class_to_idx]
        for cls in newly_added_classes:
            self.class_to_idx[cls] = len(self.class_to_idx)

        self.task_class_indices.append([self.class_to_idx[cls] for cls in current_task_classes])
        train_ids = task.train_mask.nonzero(as_tuple=True)[0]
        trainable_params = self.net.prepare_for_new_task(task, train_ids, len(newly_added_classes))
        self.opt = torch.optim.Adam(
            trainable_params,
            lr=self.args.get("lr", 0.005),
            weight_decay=self.args.get("weight_decay", 5e-4),
        )

    def observer(self, epoches, IL):
        tasks = self.tasks
        performance_matrix = torch.zeros(len(tasks) + 1, len(tasks) + 1)
        APs = []

        for k in range(len(tasks)):
            task = tasks[k].to(self.device, "x", "y", "adj_t", "train_mask", "test_mask")
            self.prepare(task)
            seen_cls = len(self.class_to_idx)

            for _ in range(epoches):
                self.net.train()
                self.net.sgc_backbone.eval()
                self.opt.zero_grad()

                logits = self.net(task, task_id=k)
                mapped_train_labels = self._encode_labels(task.y[task.train_mask])
                n_per_cls = [(mapped_train_labels == j).sum() for j in range(seen_cls)]
                loss_w = [1.0 / max(int(i), 1) for i in n_per_cls]
                loss_w = torch.tensor(loss_w, device=self.device)
                loss = F.cross_entropy(
                    logits[task.train_mask, :seen_cls],
                    mapped_train_labels,
                    weight=loss_w,
                )
                loss.backward()
                self.opt.step()

            accs = []
            AF = 0
            for k_ in range(k + 1):
                task_ = tasks[k_].to(self.device, "x", "y", "adj_t", "test_mask")
                test_ids = task_.test_mask.nonzero(as_tuple=True)[0]
                pred_task_id = self.net.infer_task_id(task_, test_ids)
                self.net.eval()
                with torch.no_grad():
                    output = self.net(task_, pred_task_id)
                mapped_test_labels = self._encode_labels(task_.y[task_.test_mask])

                if IL == "classIL":
                    acc = _plain_accuracy_selected(
                        output[task_.test_mask],
                        mapped_test_labels,
                        list(range(seen_cls)),
                    ) * 100
                else:
                    acc = _plain_accuracy_selected(
                        output[task_.test_mask],
                        mapped_test_labels,
                        self.task_class_indices[k_],
                    ) * 100

                accs.append(acc)
                task_.to("cpu")
                print(f"T{k_} {acc:.2f}", end="|", flush=True)
                performance_matrix[k, k_] = acc

            AP = sum(accs) / len(accs)
            APs.append(AP)
            print(f"AP: {AP:.2f}", end=", ", flush=True)
            for t in range(k):
                AF += performance_matrix[k, t] - performance_matrix[t, t]
            AF = AF / k if k != 0 else AF
            print(f"AF: {AF:.2f}", flush=True)
            task.to("cpu")

        return AP, np.mean(APs), AF
