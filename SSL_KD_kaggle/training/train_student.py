import torch
import torch.nn as nn
import torch.optim as optim
from SSL_KD.models.student.student_model import CustomResnet
from SSL_KD.models.teacher.teacher_model import resnet18
from SSL_KD.distillation.kd_loss import KDLoss
from SSL_KD.distillation.distill_trainer import DistillTrainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Distilling Student using device: {device}")

    # For lightweight student model, we might consider reduced channels
    student = CustomResnet(input_channel=6, num_classes=24).to(device)
    teacher1 = resnet18(in_channel=12, out_channel=24).to(device)
    teacher2 = resnet18(in_channel=6, out_channel=24).to(device)

    optimizer = optim.Adam(student.parameters(), lr=0.003, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    kd_loss_fn = KDLoss(temperature=8.0)

    trainer = DistillTrainer(
        student_model=student,
        teacher_model1=teacher1,
        teacher_model2=teacher2,
        optimizer=optimizer,
        criterion=criterion,
        kd_loss_fn=kd_loss_fn,
        device=device,
        alpha=0.2,
        temp=8.0
    )

    print("Student Distillation Trainer Initialized successfully.")

if __name__ == "__main__":
    main()
