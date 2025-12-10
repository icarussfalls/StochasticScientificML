import torch.nn as nn
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)



class NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param, regularization_exp, retrain_seed):
        super(NeuralNet, self).__init__()

        # number of input dimension n
        self.input_dimension = input_dimension
        # number of output dimension m
        self.output_dimension = output_dimension
        # number of neurons per layers
        self.neurons = neurons
        # number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # activation function
        self.activation = nn.Tanh()
        self.regularization_param = regularization_param
        # regularization exp
        self.regularization_exp = regularization_exp

        # defining the layers
        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed
        # random seed for weight initialization
        self.init_xavier()
        # init xavier will be defined below

    def forward(self, x):
        # the forward function performs the set of affine and non_linear transformation defining the network
        x = self.activation(self.input_layer(x))
        for k,l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0)
            
        self.apply(init_weights)

    def regularization(self):
        reg_loss  = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss
    

# now lets define the model
def fit_model(model, training_set, num_epochs, optimizer, p, verbose=True):
    history = list()

    # loop over epochs
    for epoch in range(num_epochs):
        if verbose:
            print("##########", num_epochs, '##########')

        running_loss = list([0])

        for j, (x_train_, u_train_) in enumerate(training_set):
            # here u is y
            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                u_pred_ = model(x_train_)
                # item 1 below
                loss = torch.mean((u_pred_.reshape(-1,) - u_train_.reshape(-1, )) ** p) + model.regularization()
                # item 2 below
                loss.backward()
                # compute average training loss over batches for the current epoch
                running_loss += loss.item()
                return loss
            
            # item 3 below
            optimizer.step(closure=closure)

        if verbose:
            print('Loss: ', (running_loss[0] / len(training_set)))
            history.append(running_loss[0])

    return history


class Legendre(nn.Module):
    """ Univariate Legendre Polynomial """
    def __init__(self, Polydegree):
        super(Legendre, self).__init__()
        self.degree = Polydegree

    def legendre(self, x, degree):
        x = x.reshape(-1, 1)
        list_poly = list()
        zeroth_pol = torch.ones(x.size(0), 1)
        list_poly.append(zeroth_pol)

        if degree > 0:
            first_pol = x
            list_poly.append(first_pol)
            ith_m_pol = torch.clone(zeroth_pol)

            for ii in range(1, degree):
                ith_p_pol = ((2 * ii + 1) * x * ith_pol - ii * ith_m_pol) / (ii + 1)
                list_poly.append(ith_p_pol)
                ith_m_pol = torch.clone(ith_pol)
                ith_pol = torch.clone(ith_p_pol)
        list_poly = torch.cat(list)

    def forward(self, x):
        eval_poly = self.legendre(x, self.degree)
        return eval_poly

class MultiVariatePoly(nn.Module):
    def __init__(self, dim, order):
        self.order = order
        self.dim = dim
        self.polys = Legendre(order)
        self.num = (order + 1) ** dim
        self.linear = torch.nn.linear(self.num, 1)

    def forward(self, x):
        poly_eval = list()
        leg_eval = torch.cat([self.polys(x[:, i]).reshape(1, x.shape(0), self.order + 1) for i in range(self.dim)])
        for i in range(x.shape[0]):
            poly_eval.append(torch.torch.cartesian_prod(*leg_eval[:, i, :]).prod(dim=1).view(1, -1))
        poly_eval = torch.cat(poly_eval)
        return self.linear(poly_eval)
        
