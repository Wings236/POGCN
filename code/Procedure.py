import world
import numpy as np
import torch
import utils
from utils import timer
import multiprocessing


CORES = multiprocessing.cpu_count() // 2

def BPR_train_original(dataset, recommend_model, loss_class):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    with timer(name="Sample"):
        S = utils.PO_Sample(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    with timer(name='train'):
        for batch_users,batch_pos,batch_neg in utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size']):
            cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
            aver_loss += cri
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()    # 只是采样的时间，训练的时间其实没有记录
    timer.zero()
    return f"loss:{aver_loss:.4f} {time_info}"

    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}


def VALID(dataset, Recmodel, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    validDict_list = dataset.bh_validDict
    bh_name = dataset.behavior_list
    testdataset_num = len(validDict_list)
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = [{'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))} for _ in range(testdataset_num)]
    for idx, validDict in enumerate(validDict_list):
        with torch.no_grad():
            users = list(validDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            total_batch = len(users) // u_batch_size + 1

            allPos = dataset.allPos
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                batch_Pos = [allPos[u] for u in batch_users]
                groundTrue = [validDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = Recmodel.getUsersRating(batch_users_gpu)
                #rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(batch_Pos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10) # mask，变成-1024
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results[idx]['recall'] += result['recall']
            results[idx]['precision'] += result['precision']
            results[idx]['ndcg'] += result['ndcg']
        results[idx]['recall'] /= float(len(users))
        results[idx]['precision'] /= float(len(users))
        results[idx]['ndcg'] /= float(len(users))
        if multicore == 1:
            pool.close()
        print(bh_name[idx], results[idx])
    return results       

      
def Test(dataset, Recmodel, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    testDict_list = dataset.bh_testDict
    bh_name = dataset.behavior_list
    testdataset_num = len(testDict_list)
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = [{'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))} for _ in range(testdataset_num)]
    for idx, testDict in enumerate(testDict_list):
        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            total_batch = len(users) // u_batch_size + 1

            allPos = dataset.allPos
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                batch_Pos = [allPos[u] for u in batch_users]
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = Recmodel.getUsersRating(batch_users_gpu)
                #rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(batch_Pos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10) # mask，变成-1024
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results[idx]['recall'] += result['recall']
            results[idx]['precision'] += result['precision']
            results[idx]['ndcg'] += result['ndcg']
        results[idx]['recall'] /= float(len(users))
        results[idx]['precision'] /= float(len(users))
        results[idx]['ndcg'] /= float(len(users))
        if multicore == 1:
            pool.close()
        print(bh_name[idx], results[idx])
    return results