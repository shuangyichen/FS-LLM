import os
import torch
import numpy as np
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor
from scipy.sparse import csr_matrix
def random_matrix(p,s,w, seed):
    np.random.seed(seed)
    counts = np.int(np.random.normal(loc=s*w*p*2, scale=np.sqrt(s * w * 2 * p * (1 - 2 * p)), size=(1)))
    np.random.seed(seed)
    rows = np.random.uniform(low=0,high=s,size=(counts)).astype(int)
    np.random.seed(seed)
    cols = np.random.uniform(low=0,high=w,size=(counts)).astype(int)
    vals = np.random.binomial(n=1,p=0.5,size=(counts))*2-1
    vals = vals.astype(np.float32)
    SparseTensor = csr_matrix((vals, (rows,cols)), shape=(s,w))
    return SparseTensor


def transform_dict_list(model_params):
    res = []
    seq = []
    for i,k in enumerate(model_params.keys()):
        #print("k",k)
        model_param = model_params[k].cpu().detach().numpy()
        #print(model_param.shape)
        res.append(np.reshape(model_param,(-1,1)))
        #seq.append('res[%d]' % i)
        #print(model_param.shape)
        #res.append(model_params[k].detach().numpy())
        if i==1:
            model_params_concat = np.concatenate((res[0],res[1]),axis=0)
        elif i>1:
            model_params_concat = np.concatenate((model_params_concat,res[i]),axis=0)
        #res.append(model_params[k].detach().numpy())
    return model_params_concat#np.concatenate(,axis=0)


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
        self.round = 0
        self.compress = True
        #compression rate
        # self.rate = args.compression_rate
        # self.samples = int(self.params_count / self.rate)
        # self.error = np.zeros((self.params_count,1))
        # self.alpha = args.compression_alpha
        # self.beta = 1 / self.alpha / (self.rate + 1 + 1 / self.alpha)

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

        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

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


        elif self.compress and self.round>0:
            print("Compressing ...")
            self.rate = 2.0
            self.params_count = 0
            # self.samples = int(self.params_count / self.rate)
            for k, v in avg_model.items():
                self.params_count += v.numel()
            self.samples = int(self.params_count / self.rate)
            if self.round==1:
                self.error = np.zeros((self.params_count,1))
            self.alpha = 0.5
            self.beta = 1 / self.alpha / (self.rate + 1 + 1 / self.alpha)
            
            
            global_update = None
            total_samples = sum(local_sample_size for local_sample_size, _ in models)
            
            compressed_updates = []
    
             # Client-side operation
            for local_sample_size, local_model in models:
                # Extract weights and compress
                # weights = local_model.cpu().state_dict()
                # weights_diff = {name: param.data - self.model[name].data for name, param in local_model.named_parameters()}
                
                weights_diff = {k: self.old_model[k] - local_model[k] for k in local_model}

        # Compress the weights difference
                weights_diff_flat = transform_dict_list(weights_diff)  # Adjust this function as needed
                weights_diff_flat = weights_diff_flat.reshape(-1, 1)
                error_compensated = weights_diff_flat + self.error
                # Compression using your scheme
                phi = random_matrix(self.alpha / 2 / self.samples, self.samples, self.params_count, seed = self.round)
                compressed = self.beta * phi.dot(weights_diff_flat)
                recov = phi.transpose().dot(compressed)
                self.error = error_compensated - recov
                
                # Collect compressed updates along with their sample sizes
                compressed_updates.append((local_sample_size, compressed))
    
            # Server-side operation: Decompress and average updates
            for local_sample_size, compressed in compressed_updates:
                # Decompress
                phi = random_matrix(self.alpha / 2 / self.samples, self.samples, self.params_count, seed = self.round)
                decompressed = phi.T.dot(compressed)
                
                # Weighted average preparation by scaling with the sample size
                if global_update is None:
                    global_update = np.zeros_like(decompressed)
                
                global_update += (local_sample_size / total_samples) * decompressed

            idx = 0
            device = next(iter(self.old_model.values())).device
            for k, v in avg_model.items():
                shape = v.shape
                count = v.numel()
                # Update avg_model with reshaped global_update segment
                avg_model[k] = torch.from_numpy(global_update[idx:idx + count].reshape(shape)).to(device) 
                idx += count

            cur_model = {k: self.old_model[k] - avg_model[k] for k in avg_model}
            avg_model = {k: cur_model[k] for k in avg_model}
        
        else:
            print("FedAVG ...")
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
        
        self.round += 1
        self.old_model = avg_model
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
