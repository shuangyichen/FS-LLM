import os
import torch
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor


class ClientsAvgAggregator(Aggregator):
    """
    Implementation of vanilla FedAvg refer to 'Communication-efficient \
    learning of deep networks from decentralized data' [McMahan et al., 2017] \
    http://proceedings.mlr.press/v54/mcmahan17a.html
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config
        self.mode = False

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """
        # print("Aggregate!!!")

        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        avg_model = self._para_weighted_avg(models, recover_fun=recover_fun)

        return avg_model

    def update(self, model_parameters):
        """
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1):
        assert self.model is not None
        # print("Client Avg Aggregator: Saving model to {}".format(path))
        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        # torch.save(ckpt, path)
        path = path.split(".")[0]
        print("Client Avg Aggregator: Saving model to {}".format(path))
        self.model.model.save_pretrained(path)

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

    def extract_layer_number(self,string):
        parts = string.split('.')
        for i, part in enumerate(parts):
            if part == 'h' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    return None
        return None
        
    def multiply_corresponding_params(self,params):
        result = {}
        keys = sorted(params.keys())
        for i in range(0, len(keys), 2):
            
            key_a = keys[i]
            key_b = keys[i + 1]
            # print(key_a)
            # print(params[key_a].size())
            # print(key_b)
            # print(params[key_b].size())
            if self.extract_layer_number(key_a) == self.extract_layer_number(key_b):  # Check if the suffix (layer number) is the same
                result[self.extract_layer_number(key_a)] = torch.matmul(params[key_b],params[key_a])
            else:
                print(f"Unmatched keys: {key_a} and {key_b}")
        return result

    def _para_weighted_avg(self, models, recover_fun=None):
        """
        Calculates the weighted average of models.
        """
        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        sample_size, avg_model = models[0]
        if self.mode:
            Results = []
            # Multiply W = B*A
            # sample_size, avg_model = models[0]
            for i in range(len(models)):
                Results.append(self.multiply_corresponding_params(models[i][1]))
            # Avg of W
            # print("multiply first params")
            avg_w = Results[0]
            for key in avg_w:
                for i in range(len(models)):
                    local_sample_size, local_model = models[i]
                    if self.cfg.federate.ignore_weight:
                        weight = 1.0 / len(models)
                    elif self.cfg.federate.use_ss:
                        # When using secret sharing, what the server receives
                        # are sample_size * model_para
                        weight = 1.0
                    else:
                        weight = local_sample_size / training_set_size

                    if i == 0:
                        avg_w[key] = Results[i][key] * weight
                    else:
                        avg_w[key] += Results[i][key] * weight

            # Matrix decomposition
            for key in avg_w:
                U, S, V = torch.svd(avg_w[key])


                U_reduced = U[:, :8]
                S_reduced = torch.diag(S[:8])
                V_reduced = V[:, :8]

                B = torch.matmul(U_reduced, torch.sqrt(S_reduced))
                A = torch.matmul(torch.sqrt(S_reduced), V_reduced.T)


                key_a = 'base_model.model.transformer.h.'+str(key)+'.attn.c_attn.lora_A.default.weight'
                key_b = 'base_model.model.transformer.h.'+str(key)+'.attn.c_attn.lora_B.default.weight'
                avg_model[key_a] = A
                avg_model[key_b] = B


        else:

            for key in avg_model:
                for i in range(len(models)):
                    local_sample_size, local_model = models[i]
                    

                    if self.cfg.federate.ignore_weight:
                        weight = 1.0 / len(models)
                    elif self.cfg.federate.use_ss:
                        # When using secret sharing, what the server receives
                        # are sample_size * model_para
                        weight = 1.0
                    else:
                        weight = local_sample_size / training_set_size

                    if not self.cfg.federate.use_ss:
                        local_model[key] = param2tensor(local_model[key])
                    if i == 0:
                        avg_model[key] = local_model[key] * weight
                    else:
                        avg_model[key] += local_model[key] * weight

                if self.cfg.federate.use_ss and recover_fun:
                    avg_model[key] = recover_fun(avg_model[key])
                    # When using secret sharing, what the server receives are
                    # sample_size * model_para
                    avg_model[key] /= training_set_size
                    avg_model[key] = torch.FloatTensor(avg_model[key])

        return avg_model


class OnlineClientsAvgAggregator(ClientsAvgAggregator):
    """
    Implementation of online aggregation of FedAvg.
    """
    def __init__(self,
                 model=None,
                 device='cpu',
                 src_device='cpu',
                 config=None):
        super(OnlineClientsAvgAggregator, self).__init__(model, device, config)
        self.src_device = src_device

    def reset(self):
        """
        Reset the state of the model to its initial state
        """
        self.maintained = self.model.state_dict()
        for key in self.maintained:
            self.maintained[key].data = torch.zeros_like(
                self.maintained[key], device=self.src_device)
        self.cnt = 0

    def inc(self, content):
        """
        Increment the model weight by the given content.
        """
        if isinstance(content, tuple):
            sample_size, model_params = content
            for key in self.maintained:
                # if model_params[key].device != self.maintained[key].device:
                #    model_params[key].to(self.maintained[key].device)
                self.maintained[key] = (self.cnt * self.maintained[key] +
                                        sample_size * model_params[key]) / (
                                            self.cnt + sample_size)
            self.cnt += sample_size
        else:
            raise TypeError(
                "{} is not a tuple (sample_size, model_para)".format(content))

    def aggregate(self, agg_info):
        """
        Returns the aggregated value
        """
        return self.maintained
