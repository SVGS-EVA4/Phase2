import torch
def save_model(epoch , model, optimizer, best_loss,best_acc,path ):

  print(f'Test Accuracy Improved! Saving Model to {path}')
  torch.save({
            'epoch': epoch,
            'state_dict': model,
            'best_test_loss': best_loss,
            'best_test_acc': best_acc,
            'optimizer' : optimizer,
        }, path)

def save_model_cpu(model,path):
	model.to('cpu')
	model.eval()
	traced_model = torch.jit.trace(model,torch.randn(1,3,244,244)) 
	traced_model.save(path)
	model.to('cuda')
	print(f'Saved Model to {path}')
