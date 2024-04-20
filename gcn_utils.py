# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InstanceGCN(nn.Module):

    def __init__(self, trigger_dim, entity_dim, num_layers=2):
        super(InstanceGCN, self).__init__()
        self.num_layers = num_layers
        self.trigger_dim = trigger_dim
        self.entity_dim = entity_dim

        self.unary_dim = self.trigger_dim

        # gcn layer
        self.T_T = nn.ModuleList()
        self.T_E = nn.ModuleList()

        self.E_T = nn.ModuleList()
        self.E_E = nn.ModuleList()

        for _ in range(self.num_layers):
            self.T_T.append(nn.Linear(self.unary_dim, self.unary_dim))
            self.T_E.append(nn.Linear(self.unary_dim, self.unary_dim))

            self.E_T.append(nn.Linear(self.unary_dim, self.unary_dim))
            self.E_E.append(nn.Linear(self.unary_dim, self.unary_dim))

        # forget gates

        self.f_t = nn.Sequential(
            nn.Linear(self.trigger_dim * 2, self.trigger_dim),
            nn.Sigmoid()
        )

        self.f_e = nn.Sequential(
            nn.Linear(self.entity_dim * 2, self.entity_dim),
            nn.Sigmoid()
        )

    def forward(self, A, T, E, t_n, e_n):
        '''
        T.shape = [bs, t_n, d]
        E.shape = [bs, e_n, d]
        n = t_n + e_n
        '''
        n_1 = t_n
        n_2 = t_n + e_n

        A_t = A[:, :n_1, :]  # [bs, t_n, n]
        A_e = A[:, n_1:n_2, :]  # [bs, e_n, n]

        for l in range(self.num_layers):
            new_T = F.relu(self.T_T[l](A_t[:, :, :n_1].bmm(T)) + \
                           self.T_E[l](A_t[:, :, n_1: n_2].bmm(E)))

            new_E = F.relu(self.E_T[l](A_e[:, :, :n_1].bmm(T)) + \
                           self.E_E[l](A_e[:, :, n_1: n_2].bmm(E)))

            forget_T = self.f_t(torch.cat([new_T, T], dim=2))  # [bs, t_n, d]
            T = forget_T * new_T + (1 - forget_T) * T

            forget_E = self.f_e(torch.cat([new_E, E], dim=2))  # [bs, e_n, d]
            E = forget_E * new_E + (1 - forget_E) * E

        return T, E
