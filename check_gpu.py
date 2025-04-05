import torch
import torchvision.ops as ops

# Tạo các bounding boxes và scores trên GPU
boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float).cuda()
scores = torch.tensor([0.9, 0.8], dtype=torch.float).cuda()

# Áp dụng Non-Maximum Suppression với ngưỡng IoU 0.5
selected = ops.nms(boxes, scores, 0.5)
print("Selected indices:", selected)
