import world
import utils
from world import cprint
from dataloader import Loader
import time
import model
import Procedure
import torch
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
print(">>GPUID:", world.args.gpu_id)
# ==============================

MODELS = {
    'BPR': model.MFBPR,
    'POGCN':model.POGCN
}

dataset = Loader(root_path="../data/"+world.dataset, behavior_list=world.behaviors)
Recmodel = MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

# no earyly stopping
record_best = 0
for epoch in range(world.TRAIN_epochs):
    if epoch % 5 == 0:
        with torch.no_grad():
            Recmodel.pre_score()
        test_start = time.time()
        cprint("[VALID]")
        res = Procedure.VALID(dataset, Recmodel, world.config['multicore'])
        target_recall = res[world.config['out_behavior']]["recall"]
        cprint("[TEST]")
        res = Procedure.Test(dataset, Recmodel, world.config['multicore'])
        
        if target_recall > record_best:
            best_epoch = epoch + 1
            best_res = res
            record_best = target_recall
        print(f'test time:{time.time()-test_start:.4f}s')
    output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr)
    print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
    # torch.save(Recmodel.state_dict(), weight_file)
print("[best]")
print(best_epoch, best_res)