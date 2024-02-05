import torch.nn as nn
import torch

class MSCLoss(nn.Module):
    def __init__(self, config_data):
        super(MSCLoss, self).__init__()
        self.m = config_data['m']
        self.mu = config_data['mu'] # mu in number
        self.eps = 1e-9
        self.k = config_data['k'] # k for knn
        self.n_per_domain = 32
        self.n_get_unk_1 = config_data['unk_n_1'] #4
        self.n_get_unk_2 = config_data['unk_n_2'] #2
        self.unk_label = config_data['unk_label_num']

    def __get_sim_matrix(self, out_src, out_tar):
        matrix = torch.cdist(out_src, out_tar)
        matrix = matrix + 1.0
        matrix = 1.0/matrix
        return matrix

    def __target_pseudolabels(self, sim_matrix, src_labels):
        ind = torch.sort(sim_matrix, descending=True, dim=0).indices
        ind_split = torch.split(ind, 1, dim=1)
        ind_split = [id.squeeze() for id in ind_split]
        vr_src = src_labels.unsqueeze(-1).repeat(1, self.n_per_domain)
        label_list = []
        for i in range(0, self.n_per_domain):
            _row = ind_split[i].long()
            _col = (torch.ones(self.n_per_domain) * i).long()
            _val = vr_src[_row, _col]
            top_n_val = _val[[j for j in range(0, self.k)]]
            label_list.append(top_n_val)

        all_top_labels = torch.stack(label_list, dim=1)
        assigned_target_labels = torch.mode(all_top_labels, dim=0).values
        return assigned_target_labels

    def forward(self, src_features, src_labels, tgt_features, device):

        sim_matrix = self.__get_sim_matrix(src_features, tgt_features)  #sim_matrix
        flat_src_labels = src_labels.squeeze()

        assigned_tgt_labels = self.__target_pseudolabels(sim_matrix, src_labels)  #pseudo_tgt_labels

        # select unk target
        # v1 sum
        # sum_line_matrix = torch.sum(sim_matrix, dim=0)
        # sum_ind = torch.sort(sum_line_matrix, descending=False, dim=0).indices
        # select_sum_ind = sum_ind[[jj for jj in range(0, self.n_get_unk)]]
        # # for jj in range(0, self.n_get_unk):
        # #     assigned_tgt_labels[select_sum_ind[jj]] = self.unk_label

        # v2 max+sum
        std_list = []
        select_sum_ind = []
        line_max = torch.max(sim_matrix,0)[0]
        idx_line_max = torch.sort(line_max, descending=False, dim=0).indices
        cut_idx_max = idx_line_max[:self.n_get_unk_1]
        for ii in range(0, self.n_get_unk_1):
            std_list.append(sim_matrix[:, cut_idx_max[ii]].sum())
        idx_std = torch.sort(torch.tensor(std_list), descending=False).indices
        cut_idx_std = idx_std[:self.n_get_unk_2]
        for jj in range(0, self.n_get_unk_2):
            select_sum_ind.append(idx_line_max[cut_idx_std[jj]])
        select_sum_ind = torch.tensor(select_sum_ind).to(device)

        assigned_tgt_labels[select_sum_ind[[jj for jj in range(0, self.n_get_unk_2)]]] = int(self.unk_label)

        selected_unk_sim_matrix_list = []
        for jj in range(0, self.n_get_unk_2):
            selected_unk_sim_matrix_list.append(sim_matrix[select_sum_ind[jj]])
        selected_unk_sim_matrix = torch.stack(selected_unk_sim_matrix_list, dim=1)

        picked_unk_sim_matrix = sim_matrix.clone().detach()
        picked_assigned_tgt_labels = assigned_tgt_labels.float().detach()
        for jj in range(0, self.n_get_unk_2):
            picked_unk_sim_matrix[:, select_sum_ind[jj]] = float('nan')
            picked_assigned_tgt_labels[select_sum_ind[jj]] = float('nan')

        picked_unk_sim_matrix = picked_unk_sim_matrix[~torch.isnan(picked_unk_sim_matrix)]
        picked_unk_sim_matrix = picked_unk_sim_matrix.reshape(self.n_per_domain, -1)

        picked_assigned_tgt_labels = picked_assigned_tgt_labels[~torch.isnan(picked_assigned_tgt_labels)]
        picked_assigned_tgt_labels = picked_assigned_tgt_labels.int()



        simratio_score = [] #sim-ratio (knn conf measure) for all tgt

        sim_matrix_split = torch.split(picked_unk_sim_matrix, 1, dim=1)
        sim_matrix_split = [_id.squeeze() for _id in sim_matrix_split]

        r = self.m
        for i in range(0, self.n_per_domain - self.n_get_unk_2): #nln: nearest like neighbours, nun: nearest unlike neighbours
            t_label = picked_assigned_tgt_labels[i]
            nln_mask = (flat_src_labels == t_label)
            nln_sim_all = sim_matrix_split[i][nln_mask]
            if nln_sim_all.shape[0] < r:
                nln_sim_r = torch.narrow(torch.sort(nln_sim_all, descending=True)[0], 0, 0, nln_sim_all.shape[0])
            else:
                nln_sim_r = torch.narrow(torch.sort(nln_sim_all, descending=True)[0], 0, 0, r)

            nun_mask = ~(flat_src_labels == t_label)
            nun_sim_all = sim_matrix_split[i][nun_mask]
            if nun_sim_all.shape[0] < r:
                nun_sim_r = torch.narrow(torch.sort(nun_sim_all, descending=True)[0], 0, 0, nun_sim_all.shape[0])
            else:
                nun_sim_r = torch.narrow(torch.sort(nun_sim_all, descending=True)[0], 0, 0, r)

            conf_score = (1.0*torch.mean(nln_sim_r)/torch.mean(nun_sim_r)).item() #sim ratio : confidence score
            simratio_score.append(conf_score)

        sort_ranking_score, ind_tgt = torch.sort(torch.tensor(simratio_score), descending=True)
        top_n_tgt_ind = torch.narrow(ind_tgt, 0, 0, self.mu) #take top mu confident tgt labels

        filtered_sim_matrix_list = []

        for idx in top_n_tgt_ind:
            filtered_sim_matrix_list.append(sim_matrix_split[idx])

        filtered_sim_matrix = torch.stack(filtered_sim_matrix_list, dim=1) # filtered_sim_matrix
        filtered_tgt_labels = picked_assigned_tgt_labels[top_n_tgt_ind]

        final_filtered_sim_matrix = torch.cat((filtered_sim_matrix, selected_unk_sim_matrix), 1)
        unk_label = []
        for jj in range(0, self.n_get_unk_2):
            unk_label.append(int(self.unk_label))
        unk_label = torch.tensor(unk_label).to(device)
        final_filtered_tgt_labels = torch.cat((filtered_tgt_labels, unk_label), 0)


        loss = self.calc_loss(final_filtered_sim_matrix, src_labels, final_filtered_tgt_labels,device)
        sum_line_matrix = torch.sum(sim_matrix, dim=0)
        return loss, select_sum_ind, assigned_tgt_labels, sum_line_matrix

    def __build_mask(self, vr_src, hr_tgt, n_tgt, same=True):
        if same:
            mask = (vr_src == hr_tgt).float()
        else:
            mask = (~(vr_src == hr_tgt)).float()

        sum_row = (torch.sum(mask, dim=1)) > 0.

        to_remove = [torch.ones(n_tgt) if (_s) else torch.zeros(n_tgt) for _s in sum_row]
        to_remove = torch.stack(to_remove, dim=0)
        return  to_remove

    def calc_loss(self, filtered_sim_matrix, src_labels, filtered_tgt_labels,device):
        n_src = src_labels.shape[0]
        n_tgt = filtered_tgt_labels.shape[0]
        
        vr_src = src_labels.unsqueeze(-1).repeat(1, n_tgt)
        hr_tgt = filtered_tgt_labels.unsqueeze(-2).repeat(n_src, 1)
        
        mask_sim = (vr_src == hr_tgt).float()

        same = self.__build_mask(vr_src, hr_tgt, n_tgt, same=True)
        diff = self.__build_mask(vr_src, hr_tgt, n_tgt, same=False)

        same, diff, mask_sim = same.to(device), diff.to(device), mask_sim.to(device)
        
        final_mask = (same.bool() & diff.bool())

        filtered_sim_matrix, final_mask = filtered_sim_matrix.to(device), final_mask.to(device)
        
        filtered_sim_matrix[~final_mask] = float('-inf')
        unpurged_exp_scores = torch.softmax(filtered_sim_matrix, dim=1)
        
        exp_scores = unpurged_exp_scores[~torch.isnan(unpurged_exp_scores.sum(dim=1))]
        sim_labels = mask_sim[~torch.isnan(unpurged_exp_scores.sum(dim=1))]

        contrastive_matrix = torch.sum(exp_scores * sim_labels, dim=1)
        loss = -1 * torch.mean(torch.log(contrastive_matrix))
        return loss
