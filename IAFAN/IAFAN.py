import os
from data import *
from utilities import *
from networks import *
# import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import criterion_factory as cf
import sys
import torch
import scipy.stats
import mmd
# import ori_mmd
import argparse
SEED = 3407#42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

sys.path.append('/root/autodl-tmp/IAFAN')

parser = argparse.ArgumentParser(description='IAFAN')
# dataset parameters
parser.add_argument('-root', default='/root/autodl-tmp/OpensetData/', help='root path of dataset')
parser.add_argument('-s', '--source', default='UCM', help='source domain')
parser.add_argument('-t', '--target', default='AID', help='target domain')

parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
parser.add_argument('--tag', type=str, default='/U2A', help='record tag name')
parser.add_argument('--mmd_coeff', type=float, default=0.2)
parser.add_argument('--ce_ep_coeff', type=float, default=0.7)
parser.add_argument('--select_unk_num_2', type=int, default=2)
parser.add_argument('--select_unk_num_1', type=int, default=9)
parser.add_argument('--msc_coeff', type=float, default=0.4)

# model parameters
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 32)')
parser.add_argument('-i', '--iters', default=10000, type=int, help='Number of iterations per epoch')
parser.add_argument('-p', '--test_freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 200)')

parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                    help="When phase is 'test', only test the model."
                            "When phase is 'analysis', only analysis the model.")
args = parser.parse_args()
device = torch.device("cuda:{}".format(args.gpu_id))
    
def skip(data, label, is_train):
    return False

share_num = 9
all_num = 12
def transform1(data, label, is_train):
    label = one_hot(share_num + 1, label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label

def transform2(data, label, is_train):
    if label in range(share_num):
        label = one_hot(share_num + 1, label)
    else:
        label = one_hot(share_num + 1, share_num)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label

def transform3(data, label, is_train):
    label = one_hot(all_num, label)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = np.transpose(data, [2, 0, 1])
    data = np.asarray(data, np.float32) / 255.0
    return data, label


batch_size = args.batch_size
domain_train = args.source
domain_test = args.target
store_name = domain_train[0] + '_' + domain_test[0]
select_unk_num_1 = args.select_unk_num_1
select_unk_num_2 = args.select_unk_num_2
mmd_coeff = args.mmd_coeff
ce_ep_coeff = args.ce_ep_coeff
msc_coeff = args.msc_coeff

train_record = args.tag
log = Logger(store_name + train_record, clear=True)
print('domain_train', domain_train)
print('domain_test', domain_test)
out_file = open(osp.join(store_name, train_record[1:]+'_'+str(mmd_coeff)+'mmd_'+str(ce_ep_coeff)+'ceep_'+str(msc_coeff)+'msc_'+str(select_unk_num_2)+'_'+str(select_unk_num_1)+'unk.txt'), "w")
model_save_path = store_name + train_record

ds = FileListDataset(args.root + domain_train + '_shared_list.txt', \
                    '/home/username/data/office/', transform=transform1, skip_pred=skip, is_train=True, imsize=256)
source_train = CustomDataLoader(ds, batch_size=batch_size, num_threads=1)

ds1 = FileListDataset(args.root + domain_test + '_list.txt', \
                    '/home/username/data/office/', transform=transform2, skip_pred=skip, is_train=True, imsize=256)
target_train = CustomDataLoader(ds1, batch_size=batch_size, num_threads=1)

ds2 = FileListDataset(args.root + domain_test + '_list.txt', \
                    '/home/username/data/office/', transform=transform3, skip_pred=skip, is_train=False, imsize=256)
target_test = CustomDataLoader(ds2, batch_size=batch_size, num_threads=1)

discriminator = LargeAdversarialNetwork(256).to(device)
feature_extractor_nofix = ResNetFc(device, model_name='resnet50',
                                model_path='/root/autodl-tmp/pretrain/resnet50.pth')
cls_down = CLS(feature_extractor_nofix.output_num(), share_num + 1, bottle_neck_dim=256)
net_down = nn.Sequential(feature_extractor_nofix, cls_down).to(device)

scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_discriminator = OptimWithSheduler(
    optim.SGD(discriminator.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9, nesterov=True), scheduler)
optimizer_feature_extractor_nofix = OptimWithSheduler(
    optim.SGD(feature_extractor_nofix.parameters(), lr=1e-4, weight_decay=1e-3, momentum=0.9, nesterov=True), scheduler)
optimizer_cls_down = OptimWithSheduler(
    optim.SGD(cls_down.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9, nesterov=True), scheduler)

kk = 0
best_ALL, best_os, best_os_star, best_os_unk, best_ALL_star, best_ALL_unk, best_iter = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
best_h_score = 0.0
while kk < args.iters:
    for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(
            zip(source_train.generator(), target_train.generator())):

        im_source = Variable(torch.from_numpy(im_source)).to(device)
        label_source = Variable(torch.from_numpy(label_source)).to(device)
        im_target = Variable(torch.from_numpy(im_target)).to(device)

        # ---------------------------------------------------------------------------------------------------------------------------
        fs1, feature_source, __, cls_predict_s = net_down.forward(im_source)
        ft1, feature_target, __, cls_predict_t = net_down.forward(im_target)
        cls_loss = CrossEntropyLoss(label_source, cls_predict_s)
        msc_config = {
            'k': 6,
            'm': 2,
            'mu': 19,
            'unk_n_1': select_unk_num_1,#4
            'unk_n_2': select_unk_num_2,#2
            'unk_label_num': share_num#label# 9
        }
        msc_module = cf.MSCLoss(msc_config)
        msc_module = msc_module.to(device)
        src_label = torch.topk(label_source, 1)[1].squeeze(1)
        msc_loss, add_unk_idx, MSC_tgt_labels, adv_weight = msc_module(feature_source, src_label, feature_target, device)
        mix_unk = add_unk_idx
        all_unk_num = select_unk_num_2
        feature_otherep = torch.index_select(ft1, 0, mix_unk.view(all_unk_num))
        feature_unk, feature_target_unkonwn, _, predict_prob_otherep = cls_down.forward(feature_otherep)
        ce_ep = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((all_unk_num, share_num)), np.ones((all_unk_num, 1))), axis=-1).astype('float32'))).to(device), predict_prob_otherep)

        label_s = torch.argmax(label_source, -1)
        mmd_loss = mmd.lmmd(device, share_num+1, feature_source, feature_target, label_s, cls_predict_t)

        with OptimizerManager([optimizer_cls_down, optimizer_feature_extractor_nofix, optimizer_discriminator]):
            loss = cls_loss + mmd_coeff * mmd_loss + ce_ep_coeff*ce_ep + msc_coeff * msc_loss
            loss.backward()

        kk += 1
        log.step += 1
        if log.step % 10 == 1:
            counter_down = AccuracyCounter()
            counter_down.addOntBatch(variable_to_numpy(cls_predict_s), variable_to_numpy(label_source))
            acc_train = Variable(
                torch.from_numpy(np.asarray([counter_down.reportAccuracy()], dtype=np.float32))).to(device)
            track_scalars(log, ['cls_loss', 'acc_train', 'mmd_loss', 'msc_loss'], globals())


        if log.step % args.test_freq == 0:
            with TrainingModeManager([feature_extractor_nofix, cls_down], train=False) \
                    as mgr, Accumulator(['predict_prob', 'predict_index', 'label']) as accumulator:
                for (i, (im, label)) in enumerate(target_test.generator()):
                    im = Variable(torch.from_numpy(im), volatile=True).to(device)
                    label = Variable(torch.from_numpy(label), volatile=True).to(device)

                    ft1, feature_target, __, predict_prob = net_down.forward(im)

                    predict_prob, label = [variable_to_numpy(x) for x in (predict_prob, label)]
                    label = np.argmax(label, axis=-1).reshape(-1, 1)
                    predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
                    accumulator.updateData(globals())

            for x in accumulator.keys():
                globals()[x] = accumulator[x]

            y_true = label.flatten()
            y_pred = predict_index.flatten()
            m = extended_confusion_matrix(y_true, y_pred, true_labels=list(range(share_num)) + list(range(share_num, all_num)),
                                        pred_labels=list(range(share_num+1)))

            cm = m
            acc_ALL_star = sum([cm[i][i] for i in range(share_num)]) / np.sum(cm[:share_num,:])#share_class OA without unk class
            acc_ALL = (sum([cm[i][i] for i in range(share_num)]) + sum(cm[share_num:,-1])) / np.sum(cm)
            acc_ALL_unk = sum(cm[share_num:,-1]) / np.sum(cm[share_num:,:])

            cm = cm.astype(np.float32) / np.sum(cm, axis=1, keepdims=True)
            acc_os_star = sum([cm[i][i] for i in range(share_num)]) / share_num
            acc_os = (acc_os_star * share_num + acc_ALL_unk) / (share_num+1)

            h_score = 2 * acc_os_star * acc_ALL_unk / (acc_os_star + acc_ALL_unk)
            per_class = []
            for i in range(share_num):
                per_class.append(round(cm[i][i]*100, 2))
            per_class.append(round(acc_ALL_unk*100, 2))
            log_str = "iter {:05d}, os: {:.2f}, os*: {:.2f}, ALL: {:.2f}, ALL*: {:.2f}, ALL_unk: {:.2f}, h_score: {:.2f}, per_class:{} \n"\
                .format(log.step, acc_os*100, acc_os_star*100, acc_ALL*100, acc_ALL_star*100, acc_ALL_unk*100, h_score*100, per_class)
            print(log_str)

            temp_model = nn.Sequential(net_down)
            if acc_ALL >= best_ALL:
                best_h_score = h_score
                best_class = per_class
                best_os, best_os_star, best_ALL, best_ALL_star, best_ALL_unk =  acc_os, acc_os_star, acc_ALL, acc_ALL_star, acc_ALL_unk
                best_model = temp_model
                torch.save(best_model.state_dict(), osp.join(model_save_path, "best_model.pth.tar"))
                print("best model is saved in iteration", log.step)
                best_iter = log.step
                best_confusion_matrix = cm
            best_res = "Topi {:05d}, os: {:.2f}, os*: {:.2f}, ALL: {:.2f}, ALL*: {:.2f}, ALL_unk: {:.2f}, h_score: {:.2f}, per_class:{}\n"\
                .format(best_iter, best_os*100, best_os_star*100, best_ALL*100, best_ALL_star*100, best_ALL_unk*100, best_h_score*100, best_class)
            print(best_res)
            
            out_file.write(log_str)
            if log.step == args.iters:
                out_file.write(best_res)
            out_file.flush()
    print(kk)