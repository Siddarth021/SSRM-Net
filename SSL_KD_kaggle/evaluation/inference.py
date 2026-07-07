import torch

def run_inference(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, age_gender, _ in dataloader:
            inputs = inputs.to(device)
            age_gender = age_gender.to(device)
            outputs = model(inputs, age_gender)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())
            all_labels.append(labels)

    return torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)
