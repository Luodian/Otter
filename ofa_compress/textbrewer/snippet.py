import torch
from functools import partial
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig

# We omit the initialization of models, optimizer, and dataloader. 
teacher_model : torch.nn.Module = ...
student_model : torch.nn.Module = ...
dataloader : torch.utils.data.DataLoader = ...
optimizer : torch.optim.Optimizer = ...
scheduler : torch.optim.lr_scheduler = ...

def simple_adaptor(batch, model_outputs):
    # We assume that the first element of model_outputs 
    # is the logits before softmax
    return {'logits': model_outputs[0]}  

train_config = TrainingConfig()
distill_config = DistillationConfig()
distiller = GeneralDistiller(
    train_config=train_config, distill_config = distill_config,
    model_T = teacher_model, model_S = student_model, 
    adaptor_T = simple_adaptor, adaptor_S = simple_adaptor)

distiller.train(optimizer, scheduler, 
    dataloader, num_epochs, callback=None)





def predict(model, eval_dataset, step, args): 
  raise NotImplementedError
# fill other arguments
my_callback = partial(predict, eval_dataset=my_eval_dataset, args=args) 
train_config = TrainingConfig()

# 自定义的预测与评估函数
def predict(model, eval_dataset, step, args): 
  '''
  eval_dataset: 验证集
  args: 评估中需要的其他参数
  '''
  raise NotImplementedError

 # 填充多余的参数
my_callback = partial(predict, eval_dataset=my_eval_dataset, args=args) 
distillator.train(..., callback = my_callback)
