import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_loss import FeatureLoss

class DistillTrainer(object):
    def __init__(self, student_model, teacher_model1, teacher_model2, optimizer, criterion, kd_loss_fn, device, alpha=0.2, temp=8.0, beta=0.1, feature_loss_fn=None):
        self.student_model = student_model
        self.teacher_model1 = teacher_model1
        self.teacher_model2 = teacher_model2
        self.optimizer = optimizer
        self.criterion = criterion
        self.kd_loss_fn = kd_loss_fn
        self.feature_loss_fn = feature_loss_fn if feature_loss_fn is not None else FeatureLoss()
        self.device = device
        self.alpha = alpha
        self.temp = temp
        self.beta = beta

    def train_epoch(self, dataloader, selected_leads=None, lead_mapping=None):
        self.student_model.train()
        self.teacher_model1.eval()
        self.teacher_model2.eval()

        epoch_loss = 0.0
        distill_loss_accum = 0.0

        for batch_idx, (inputs, labels, age_gender, _) in enumerate(dataloader):
            if selected_leads is not None and lead_mapping is not None:
                lead_pos = [lead_mapping[lead] for lead in selected_leads]
                student_inputs = inputs[:, lead_pos, :]
            else:
                student_inputs = inputs

            teacher_inputs = inputs.to(self.device)
            student_inputs = student_inputs.to(self.device)
            labels = labels.to(self.device)
            age_gender = age_gender.to(self.device)

            with torch.no_grad():
                teacher1_out = self.teacher_model1(teacher_inputs, age_gender)
                teacher2_out = self.teacher_model2(student_inputs, age_gender)

            self.optimizer.zero_grad()
            student_out = self.student_model(student_inputs, age_gender)

            # Extract logits (fallback to raw output if not a dict)
            t1_logits = teacher1_out.get('logits') if isinstance(teacher1_out, dict) else teacher1_out
            t2_logits = teacher2_out.get('logits') if isinstance(teacher2_out, dict) else teacher2_out
            s_logits = student_out.get('logits') if isinstance(student_out, dict) else student_out

            student_loss = self.criterion(s_logits, labels)
            
            # Compute KD Loss if teachers provide logits
            if t1_logits is not None and t2_logits is not None:
                kd_loss_1 = self.kd_loss_fn(s_logits, t1_logits)
                kd_loss_2 = self.kd_loss_fn(s_logits, t2_logits)
                kd_total = 0.7 * kd_loss_1 + 0.3 * kd_loss_2
            else:
                kd_total = torch.tensor(0.0).to(self.device)

            # Compute Feature Loss
            # Align student's 'projected' features with teacher's 'embedding' features
            t1_features = [teacher1_out['embedding']] if isinstance(teacher1_out, dict) and 'embedding' in teacher1_out else []
            t2_features = [teacher2_out['embedding']] if isinstance(teacher2_out, dict) and 'embedding' in teacher2_out else []
            s_features = [student_out['projected']] if isinstance(student_out, dict) and 'projected' in student_out else []

            f_loss_1 = self.feature_loss_fn(s_features, t1_features) if s_features and t1_features else torch.tensor(0.0).to(self.device)
            f_loss_2 = self.feature_loss_fn(s_features, t2_features) if s_features and t2_features else torch.tensor(0.0).to(self.device)
            feat_loss = 0.7 * f_loss_1 + 0.3 * f_loss_2

            # Combine all losses
            loss = self.alpha * student_loss + (1.0 - self.alpha) * kd_total + self.beta * feat_loss
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            distill_loss_accum += (kd_total.item() + feat_loss.item()) * inputs.size(0)

        return epoch_loss / len(dataloader.dataset), distill_loss_accum / len(dataloader.dataset)
