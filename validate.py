import lightning as L
from configs.config import CFG
from data.dataset import CityscapesDataModule
from models.lit_module import Module

# 1. Khởi tạo môi trường Lightning Trainer ở chế độ đánh giá
trainer = L.Trainer(
    accelerator="auto", 
    devices="auto",
    logger=L.loggers.TensorBoardLogger("logs/validation"),
)

# 2. Tải mô hình từ file checkpoint (.ckpt) lưu trong quá trình train
# (Thay thế đường dẫn bằng file checkpoint tốt nhất của bạn)
best_model_path = "checkpoints/0.5pruned_EfficientVitL2_city.ckpt" 

# Load lại trạng thái mô hình hoàn chỉnh kèm cấu hình CFG
model = Module.load_from_checkpoint(checkpoint_path=best_model_path, cfg=CFG)

# 3. Khởi tạo DataModule dữ liệu
dm = CityscapesDataModule(CFG)

# 4. Gọi lệnh Validate tự động để xuất bảng báo cáo mIoU chi tiết
trainer.validate(model, datamodule=dm)