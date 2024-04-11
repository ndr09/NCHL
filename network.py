import numpy as np
import torch
import torch.nn as nn
import time
from scipy.stats import kstest

class NN(nn.Module):
    def __init__(self, nodes: list, grad=False, init=None, device="cpu", wopt=True):
        super().__init__()
        self.device = torch.device(device)
        self.nodes = torch.tensor(nodes).to(self.device)
        self.nweights = sum([self.nodes[i] * self.nodes[i + 1] for i in
                             range(len(self.nodes) - 1)])  # nodes[0]*nodes[1]+nodes[1]*nodes[2]+nodes[2]*nodes[3]

        self.networks = []
        self.activations = []
        self.grad = grad
        for i in range(len(nodes) - 1):
            self.networks.append(nn.Linear(nodes[i], nodes[i + 1], bias=False))

        if wopt:
            self.networks = nn.ParameterList(self.networks)

        for l in self.networks:
            if init == 'xa_uni':
                torch.nn.init.xavier_uniform(l.weight.data, 0.3)
            elif init == 'sparse':
                torch.nn.init.sparse_(l.weight.data, 0.8)
            elif init == 'uni':
                torch.nn.init.uniform_(l.weight.data, -0.1, 0.1)
            elif init == 'normal':
                torch.nn.init.normal_(l.weight.data, 0, 0.024)
            elif init == 'ka_uni':
                torch.nn.init.kaiming_uniform_(l.weight.data, 3)
            elif init == 'uni_big':
                torch.nn.init.uniform_(l.weight.data, -1, 1)
            elif init == 'xa_uni_big':
                torch.nn.init.xavier_uniform(l.weight.data)
            elif init == 'zero':
                torch.nn.init.zeros_(l.weight.data)
        self.float()

    def forward(self, inputs):

        self.activations = []
        x = inputs.to(self.device)
        self.activations.append(torch.clone(x).to(self.device))
        # print(x)
        c = 0
        for l in self.networks:
            x = l(x)
            # print(x, l.weight.data)
            x = torch.tanh(x)

            c += 1
            self.activations.append(torch.clone(x))

        return x

    def get_weights(self):
        tmp = []
        for l in self.networks:
            tmp.append(l.weight)
        return tmp

    def set_weights(self, weights):
        if type(weights) == list and type(weights[0]) == torch.Tensor:
            for i in range(len(self.networks)):
                self.networks[i].weight.data = weights[i]
        elif len(weights) == self.nweights:
            tmp = self.get_weights()
            start = 0
            i = 0
            for l in tmp:
                size = l.size()[0] * l.size()[1] + start
                params = torch.tensor(weights[start:size], requires_grad=self.grad)
                start = size
                rsh = torch.reshape(params, (l.size()[0], l.size()[1]))
                self.networks[i].weight.data = rsh
                i += 1


class HNN(NN):
    def __init__(self, nodes: list, eta: float, hrules=None, grad=False, init=None):
        super(HNN, self).__init__(nodes, grad=grad, init=init)

        self.hrules = []
        self.eta = eta
        start = 0
        if hrules is not None:
            self.set_hrules(hrules)

    def set_hrules(self, hrules: list):
        assert len(hrules) == self.nweights * 4, "needed " + str(
            self.nweights * 4) + " received " + str(len(hrules))
        start = 0
        for l in self.get_weights():
            size = l.size()[0] * l.size()[1] * 4 + start
            params = torch.tensor(hrules[start:size])
            self.hrules.append(torch.reshape(params, (l.size()[0], l.size()[1], 4)))
            start = size

    def set_etas(self, etas: list):
        start = 0
        self.eta = []
        for l in self.get_weights():
            size = l.size()[0] * l.size()[1] + start
            params = torch.tensor(etas[start:size])
            self.eta.append(torch.reshape(params, (l.size()[0], l.size()[1])))
            start = size

    def update_weights(self):

        weights = self.get_weights()
        for i in range(len(weights)):
            l = weights[i]
            activations_i = self.activations[i].to(self.device)
            activations_i1 = torch.reshape(self.activations[i + 1].to(self.device),
                                           (self.activations[i + 1].size()[0], 1))
            hrule_i = self.hrules[i].to(self.device)
            # la size dovra essere l1, l
            pre = hrule_i[:, :, 0] * activations_i
            post = hrule_i[:, :, 1] * activations_i1
            C_i = activations_i * hrule_i[:, :, 2]
            C_j = activations_i1 * hrule_i[:, :, 2]
            C = C_i * C_j
            D = hrule_i[:, :, 3]
            dw = pre + post + C + D
            weights[i] += self.eta[i] * dw

        self.set_weights(weights)


class NHNN(NN):
    def __init__(self, nodes: list, hrules=None, grad=False, device="cpu", init=None):
        super(NHNN, self).__init__(nodes, grad=grad, device=device, init=init,wopt=False)

        self.hrules = []
        self.nparams = sum(self.nodes) * 5 - self.nodes[0] - self.nodes[-1]
        if hrules is not None:
            self.set_hrules(hrules)
        # self.hrules = torch.nn.ParameterList(self.hrules)
        self.float()

    def forward(self, inputs):

        self.activations = []
        x0 = inputs
        self.activations.append(torch.clone(x0))
        # print(x)
        c = 0

        for l in self.networks:
            x1 = l(x0)
            x1 = torch.tanh(x1)
            self.activations.append(torch.clone(x1))
            self.update_weights_layer(c, x0, x1)
            x0 = x1
            c += 1

        return x1


    def reset_weights(self):
        self.set_weights(torch.zeros(self.nweights).detach().numpy().tolist())

    def set_hrules2(self, hrules):
        a = []
        b = []
        c = []
        d = []
        e = []

        start = 0
        a.append(torch.tensor(hrules[start:start+self.nodes[0]]))
        start += self.nodes[0]

        b.append(torch.zeros(self.nodes[0]))

        c.append(torch.tensor(hrules[start:start + self.nodes[0]]))
        start += self.nodes[0]

        d.append(torch.tensor(hrules[start:start + self.nodes[0]]))
        start += self.nodes[0]

        e.append(torch.tensor(hrules[start:start + self.nodes[0]]))
        start += self.nodes[0]

        # print(self.nodes.tolist()[1,-1])
        for l in self.nodes[1:-1]:
            a.append(torch.tensor(hrules[start:start + self.nodes[l]]))
            start += self.nodes[l]

            b.append(torch.tensor(hrules[start:start + self.nodes[l]]))
            start += self.nodes[l]

            c.append(torch.tensor(hrules[start:start + self.nodes[l]]))
            start += self.nodes[l]

            d.append(torch.tensor(hrules[start:start + self.nodes[l]]))
            start += self.nodes[l]

            e.append(torch.tensor(hrules[start:start + self.nodes[l]]))
            start += self.nodes[l]


        a.append(torch.zeros(self.nodes[-1]))

        b.append(torch.tensor(hrules[start:start + self.nodes[-1]]))
        start += self.nodes[-1]

        c.append(torch.tensor(hrules[start:start + self.nodes[-1]]))
        start += self.nodes[-1]

        d.append(torch.tensor(hrules[start:start + self.nodes[-1]]))
        start += self.nodes[-1]

        e.append(torch.tensor(hrules[start:start + self.nodes[-1]]))
        start += self.nodes[-1]

        self.a = torch.nn.ParameterList(a)
        self.b = torch.nn.ParameterList(b)
        self.c = torch.nn.ParameterList(c)
        self.d = torch.nn.ParameterList(d)
        self.e = torch.nn.ParameterList(e)



    def set_hrules(self, hrules: list):
        assert len(hrules) == sum(self.nodes) * 5 - self.nodes[0] - self.nodes[-1], "needed " + str(
            sum(self.nodes) * 5 - self.nodes[0] - self.nodes[-1]) + " received " + str(len(hrules))
        start = 0

        size = self.nodes[0] * 4 + start
        tmp = np.reshape(hrules[start:size], (self.nodes[0], 4))
        tmp1 = np.zeros((self.nodes[0], 5))
        for i in range(self.nodes[0]):
            tmp1[i] = np.insert(tmp[i], 1, 0.)

        params = torch.tensor(tmp1)
        self.hrules.append(params)

        for l in self.nodes[1:-1]:
            size = l * 5 + start
            params = torch.tensor(hrules[start:size])
            self.hrules.append(torch.reshape(params, (l, 5)))

            start = size

        size = self.nodes[-1] * 4 + start
        params = torch.tensor(hrules[start:size])
        tmp = torch.reshape(params, (self.nodes[-1], 4))
        tmp1 = torch.tensor([[0.] for i in range(self.nodes[-1])])
        self.hrules.append(torch.hstack((tmp1, tmp)))

        self.hrules = torch.nn.ParameterList(self.hrules)

    def set_weights_layer(self, weights, i):

        self.networks[i].weight.data = weights#torch.nn.Parameter(weights)


    def update_weights_layer(self, i, activations_i, activations_i1):
        weights = self.get_weights()
        l = weights[i]

        # hrule_i = self.hrules[i]
        # hrule_i1 = self.hrules[i + 1]
        # print(self.a[i] * activations_i)
        pre_i = torch.reshape(self.a[i] * activations_i, (1, activations_i.size()[0]))
        print("a_i", activations_i)
        print("A*a_i",pre_i)
        print("A",self.a[i])
        print("aa ", self.a[i]*activations_i)
        # exit(0)
        # print("ABCD",hrule_i1)


        pre_i = pre_i.repeat((activations_i1.size()[0], 1))

        post_j = torch.reshape(self.b[i+1] * activations_i1, (activations_i1.size()[0], 1))
        post_j = post_j.repeat((1, activations_i.size()[0]))

        c_i = torch.reshape(self.c[i] * activations_i, (1, activations_i.size()[0]))
        c_j = torch.reshape(self.c[i+1] * activations_i1, (activations_i1.size()[0], 1))
        # print(self.d)
        d_i = torch.reshape(self.d[i], (1, activations_i.size()[0]))
        d_j = torch.reshape(self.d[i+1], (activations_i1.size()[0], 1))

        dw = pre_i + post_j + c_i * c_j + d_i * d_j

        pre_eta = self.e[i].repeat(activations_i1.size()[0], 1)
        post_eta = torch.reshape(self.e[i+1], (activations_i1.size()[0], 1)).repeat((1, activations_i.size()[0]))
        nl = l+ (pre_eta + post_eta) / 2 * dw
        print("\\\\\\\\",nl)
        print("=========", self.a[1])
        # exit(0)

        self.set_weights_layer(nl, i)
        exit(1)

    def update_weights_layer2(self, i, activations_i, activations_i1):
        weights = self.get_weights()
        l = weights[i]

        hrule_i = self.hrules[i]
        hrule_i1 = self.hrules[i + 1]
        print(hrule_i[:, 0] * activations_i)
        pre_i = torch.reshape(hrule_i[:, 0] * activations_i, (1, activations_i.size()[0]))
        print(1,pre_i)
        print(2,hrule_i[:, 0])
        print(3,hrule_i)
        exit(0)

        pre_i = pre_i.repeat((activations_i1.size()[0], 1))

        post_j = torch.reshape(hrule_i1[:, 1] * activations_i1, (activations_i1.size()[0], 1))
        post_j = post_j.repeat((1, activations_i.size()[0]))

        c_i = torch.reshape(torch.where(hrule_i[:, 2] == 1., 1., hrule_i[:, 2] * activations_i),
                            (1, activations_i.size()[0]))
        c_j = torch.reshape(torch.where(hrule_i1[:, 2] == 1., 1., hrule_i1[:, 2] * activations_i1),
                            (activations_i1.size()[0], 1))
        d_i = torch.reshape(hrule_i[:, 3], (1, activations_i.size()[0]))
        d_j = torch.reshape(hrule_i1[:, 3], (activations_i1.size()[0], 1))

        dw = pre_i + post_j + c_i * c_j + d_i * d_j

        pre_eta = hrule_i[:, 4].repeat(activations_i1.size()[0], 1)
        post_eta = torch.reshape(hrule_i1[:, 4], (activations_i1.size()[0], 1)).repeat((1, activations_i.size()[0]))
        nl = l+ (pre_eta + post_eta) / 2 * dw
        self.set_weights_layer(nl, i)


class WLNHNN(NN):
    def __init__(self, nodes, window, eta=0.1):
        super().__init__(nodes)

        self.hrules = [[[0 for _ in range(window + 5)] for i in range(node)] for node in nodes]
        self.window = window
        self.eta = eta
        self.nparams = (sum(nodes) * 5) - nodes[0] - nodes[-1]

    def activate(self, inputs):
        self.activations[0] = [np.tanh(x) for x in inputs]
        for i in range(len(inputs)):
            self.hrules[0][i][5] = self.hrules[0][i][6]
            self.hrules[0][i][6] = self.activations[0][i]

        for l in range(1, len(self.nodes)):
            self.activations[l] = [0. for _ in range(self.nodes[l])]
            for o in range(self.nodes[l]):
                sum = 0  # self.weights[i - 1][j][0]
                for i in range(self.nodes[l - 1]):
                    dw = 0
                    for w in range(self.window):
                        dw += self.cdw(l, i, o, 5 + w)

                    sum += (dw) * self.activations[l - 1][i]
                self.activations[l][o] = np.tanh(sum)
                for w in range(6, 5 + self.window):
                    self.hrules[l][o][w - 1] = self.hrules[l][o][w]

                self.hrules[l][o][-1] = self.activations[l][o]
        return np.array(self.activations[-1])

    def cdw(self, l, i, o, t):
        dw = (
                self.hrules[l - 1][i][2] * self.hrules[l][o][2] * self.hrules[l - 1][i][t] *
                self.hrules[l][o][t] +  # both
                self.hrules[l][o][1] * self.hrules[l][o][t] +  # post
                self.hrules[l - 1][i][0] * self.hrules[l - 1][i][t] +  # pre
                self.hrules[l][o][3] * self.hrules[l - 1][i][3])
        eta = 0.5 * (self.hrules[l][o][4] + self.hrules[l - 1][i][4])
        return eta * dw

    def set_hrules(self, hrules):
        c = 0
        for layer in range(len(self.nodes)):
            for node in range(self.nodes[layer]):
                if layer == 0:  # input
                    self.hrules[layer][node][0] = hrules[c]
                    self.hrules[layer][node][1] = 0  # hrules[c + 1]
                    self.hrules[layer][node][2] = hrules[c + 1]  # hrules[c + 1]
                    self.hrules[layer][node][3] = hrules[c + 2]  # hrules[c + 1]
                    self.hrules[layer][node][4] = hrules[c + 3]
                    c += 4

                elif layer == (len(self.nodes) - 1):  # output
                    self.hrules[layer][node][0] = 0
                    self.hrules[layer][node][1] = hrules[c]  # hrules[c + 1]
                    self.hrules[layer][node][2] = hrules[c + 1]  # hrules[c + 1]
                    self.hrules[layer][node][3] = hrules[c + 2]  # hrules[c + 1]
                    self.hrules[layer][node][4] = hrules[c + 3]
                    c += 4

                else:
                    self.hrules[layer][node][0] = hrules[c]
                    self.hrules[layer][node][1] = hrules[c + 1]  # hrules[c + 1]
                    self.hrules[layer][node][2] = hrules[c + 2]  # hrules[c + 1]
                    self.hrules[layer][node][3] = hrules[c + 3]  # hrules[c + 1]
                    self.hrules[layer][node][4] = hrules[c + 4]
                    c += 5
