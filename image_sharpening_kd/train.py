
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from utils.dataset import CustomDataset

teacher = TeacherModel().eval()
student = StudentModel()

criterion = nn.MSELoss()
optimizer = optim.Adam(student.parameters(), lr=1e-4)
train_loader = DataLoader(CustomDataset("data/lr", "data/raw"), batch_size=4, shuffle=True)

for epoch in range(5):
    for i, (lr, hr) in enumerate(train_loader):
        with torch.no_grad():
            teacher_out = teacher(lr)
        student_out = student(lr)
        loss = criterion(student_out, teacher_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

torch.save(student.state_dict(), "student_model.pth")
