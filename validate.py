import lightning as L
from configs.config import CFG
from data.dataset import CityscapesDataModule, ADE20KDataModule
from models.lit_module import Module

# 1. Khởi tạo môi trường Lightning Trainer ở chế độ đánh giá
def validate_model(cfg=CFG,model_path=None):
    trainer = L.Trainer(
        accelerator="auto", 
        devices="auto",
        logger=L.loggers.TensorBoardLogger("logs/validation"),
    )

    # 2. Tải mô hình từ file checkpoint (.ckpt) lưu trong quá trình train
    # (Thay thế đường dẫn bằng file checkpoint tốt nhất của bạn)
    best_model_path = model_path
    if best_model_path is None:
        
        raise ValueError("Please provide the path to the best model checkpoint (.ckpt) for validation.")

    # Load lại trạng thái mô hình hoàn chỉnh kèm cấu hình CFG
    model = Module.load_from_checkpoint(checkpoint_path=best_model_path, cfg=cfg)

    # 3. Khởi tạo DataModule dữ liệu
    dataset_name = getattr(cfg, "dataset_name", None).lower()
    if dataset_name == "cityscapes":
        dm = CityscapesDataModule(cfg)
    elif dataset_name == "ade20k":
        dm = ADE20KDataModule(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset_name}")

    # 4. Gọi lệnh Validate tự động để xuất bảng báo cáo mIoU chi tiết
    trainer.validate(model, datamodule=dm)