import torch
def evaluate_classwise_accuracy(model, device, classes, test_loader):
    class_correct = list(0. for i in range(4))
    class_total = list(0. for i in range(4))
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            batch = c.size()
            if len(batch)==0:
              break
            for i in range(batch[0]):
            	label = labels[i]
            	class_correct[label] += c[i].item()
            	class_total[label] += 1

    for i in range(4):
        print(f'Accuracy of {classes[i]} : {100 * class_correct[i] / class_total[i]} , Correctly Classified: { class_correct[i] }, Misclassified: { class_total[i] - class_correct[i] }' )
