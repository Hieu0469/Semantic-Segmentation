import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

# Import các cấu hình và module trong project của bạn
from src.config import CFG
from src.module import Module, CityscapesDataModule
from src.prune import prune_model

# Thử import thư viện segmentation_models_pytorch và efficientvit
try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

try:
    from efficientvit.seg_model_zoo import create_efficientvit_seg_model
except ImportError:
    create_efficientvit_seg_model = None


def get_model_from_cfg(cfg):
    """
    Hàm kiểm tra và khởi tạo model linh hoạt.
    Hỗ trợ truyền trực tiếp một PyTorch Model Object hoặc khởi tạo tự động qua String config.
    """
    # TRƯỜNG HỢP 1: Nếu người dùng truyền trực tiếp một Model Object vào CFG.model
    if hasattr(cfg, 'load_path') and cfg.load_path is not None:
        print(f"\n>>> [INFO] Phát hiện load_path trong CFG: {cfg.load_path}. Tiến hành load model từ file .pt.")
        model = torch.load(cfg.load_path, map_location="cpu",weight_only=False)
        if getattr(cfg, 'use_pruning', False):
            print(f"\n>>> [INFO] Áp dụng pruning với tỉ lệ {cfg.pruning_ratio} sau khi load model.")
            model = prune_model(model, pruning_ratio=cfg.pruning_ratio)
        return model
    if hasattr(cfg, 'model') and isinstance(cfg.model, torch.nn.Module):
        print("\n>>> [INFO] Phát hiện PyTorch Model Object trong CFG.model. Tiến hành sử dụng trực tiếp để train.")
        model = cfg.model
        if getattr(cfg, 'use_pruning', False):
            print(f"\n>>> [INFO] Áp dụng pruning với tỉ lệ {cfg.pruning_ratio} trước khi train.")
            model = prune_model(model, pruning_ratio=cfg.pruning_ratio)
        return model
    

    # TRƯỜNG HỢP 2: Khởi tạo tự động dựa trên cấu hình chuỗi text trong CFG
    model_type = getattr(cfg, 'model_type', 'efficientvit').lower()
    
    if model_type == 'efficientvit':
        if create_efficientvit_seg_model is None:
            raise ImportError("Không tìm thấy thư viện efficientvit. Hãy cài đặt trước khi chạy.")
        
        # Nếu cấu hình yêu cầu sử dụng cắt tỉa (pruning)
        model_name = f'efficientvit-seg-{cfg.efficientvit_variant}-{cfg.dataset}'
        model = create_efficientvit_seg_model(name=model_name, pretrained=cfg.pretrained,weight_url=cfg.pretrained_url, num_classes=cfg.num_classes)
        if getattr(cfg, 'use_pruning', False):
            print(f"\n>>> [INFO] Áp dụng pruning với tỉ lệ {cfg.pruning_ratio} trước khi train.")
            model = prune_model(model, pruning_ratio=cfg.pruning_ratio)
        return model
            
    elif model_type == 'smp':
        if smp is None:
            raise ImportError("Không tìm thấy thư viện segmentation_models_pytorch. Hãy chạy: pip install segmentation-models-pytorch")
            
        smp_architecture = getattr(cfg, 'smp_architecture', 'Unet').lower()
        encoder_name = getattr(cfg, 'smp_encoder', 'resnet34')
        print(f"\n>>> [INFO] Khởi tạo model từ SMP: Kiến trúc [{cfg.smp_architecture}] với Backbone [{encoder_name}]")
        
        # Map các kiến trúc phổ biến của thư viện SMP
        smp_mapping = {
            'unet': smp.Unet,
            'unetplusplus': smp.UnetPlusPlus,
            'manet': smp.MAnet,
            'linknet': smp.Linknet,
            'fpn': smp.FPN,
            'pspnet': smp.PSPNet,
            'deeplabv3': smp.DeepLabV3,
            'deeplabv3plus': smp.DeepLabV3Plus,
            'pan': smp.PAN
        }
        
        if smp_architecture in smp_mapping:
            model = smp_mapping[smp_architecture](
                encoder_name=encoder_name,
                encoder_weights="imagenet" if getattr(cfg, 'pretrained', True) else None,
                in_channels=3,
                classes=cfg.num_classes
            )
        else:
            raise ValueError(f"Kiến trúc '{cfg.smp_architecture}' không hỗ trợ trong script hiện tại. Chọn trong: {list(smp_mapping.keys())}")
        
        if getattr(cfg, 'use_pruning', False):
            print(f"\n>>> [INFO] Áp dụng pruning với tỉ lệ {cfg.pruning_ratio} trước khi train.")
            model = prune_model(model, pruning_ratio=cfg.pruning_ratio)
        return model
            
    else:
        raise ValueError(f"model_type '{model_type}' không hợp lệ. Hãy chọn 'efficientvit' hoặc 'smp'.")


def run(cfg=CFG):
    # Đảm bảo tính nhất quán dữ liệu ngẫu nhiên
    L.seed_everything(42, workers=True)
    
    # Khởi tạo model dựa trên chiến lược kiểm tra CFG
    model_object = get_model_from_cfg(cfg)
    
    # Gán object chuẩn vào cfg.model để PyTorch Lightning Module (models/lit_module.py) lấy sử dụng
    cfg.model = model_object
    
    # Khởi tạo pipeline dữ liệu và Lightning Module bao bọc quanh model
    dm = CityscapesDataModule(cfg)
    model = Module(cfg)
    
    # Thiết lập nơi lưu log và checkpoint tự động
    logger = TensorBoardLogger(save_dir=cfg.log_dir, name=cfg.model_name)
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.ckpt_dir, 
        filename=f"{cfg.model_name}-",
        monitor="val/mIoU", 
        mode="max", 
        save_top_k=1
    )
    early_stop_cb = EarlyStopping(monitor="val/mIoU", patience=15, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Cấu hình bộ Trainer chính của PyTorch Lightning
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs, 
        accelerator="auto", 
        devices="auto",
        precision="16-mixed", 
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        log_every_n_steps=10, 
        val_check_interval=1.0,
        gradient_clip_val=1.0, 
        check_val_every_n_epoch=1,
    )
    
    # Bắt đầu quá trình huấn luyện/fine-tuning
    trainer.fit(model, datamodule=dm)
    print(f"\n[SUCCESS] Huấn luyện hoàn tất! Checkpoint tốt nhất lưu tại: {checkpoint_cb.best_model_path}")
    
    # Lưu file model .pt dạng object đầy đủ độc lập
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    final_path = os.path.join(cfg.ckpt_dir, f"{cfg.model_name}_final.pt")
    torch.save(model.model, final_path)
    print(f"[INFO] Đã lưu file trọng số cuối cùng (.pt) tại: {final_path}")


if __name__ == "__main__":
    run()
