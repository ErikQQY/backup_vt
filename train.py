import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import json
from pathlib import Path
from DeepUtils.MiscFunctions import *
import logging,random,wandb
# import torchsummary

from DeepUtils.models import build_model_from_cfg
from DeepUtils.optim import build_optimizer_from_cfg
from DeepUtils.dataset import build_dataloader_from_cfg
from DeepUtils.scheduler import build_scheduler_from_cfg

GLOBAL_WANDB_PROJECT_NAME="DeepVortexExtraction"
initLogging()



#! TODO (s) of PyFLowVis
#! todo: DEFINE THE cnn MODEL:   NET0: VORETXNET, NET1: VORETXsegNET, NET3: RESNET , net4 vortexViz
#! todo: what is the label? ->second stage the segmentation of as steady as possible (asap) field.
#! todo: visualize the model's output: classification of vortex bondary+ coreline  in 3d space (2d space+ 1D time)
#! todo:test the model's with RFC, bossineq, helix, spriral motion
#? - [ ]  TODO: draw pathline?
#? - [ ]  TODO: train vector field for killing +regular+ Di term 
#? - [ ]  TODO: curvature gradient
#? - [ ]  TODO: implement hanger of imgui widget

    



def validate(model, val_data_loader, device, log_segmentation_metrics=False,
             return_metrics=False, log_prefix='Validation'):
    was_training = model.training
    model.eval()
    val_loss = sample_count = tp = fp = fn = pixel_count = 0
    try:
        with torch.no_grad():
            for data, label in val_data_loader:
                if isinstance(data, (list, tuple)):
                    data = tuple(item.to(device) for item in data)
                else:
                    data = data.to(device)
                label = label.to(device)
                prediction = model(data)
                loss = model.get_loss(prediction, label)
                val_loss += loss.item() * label.shape[0]
                sample_count += label.shape[0]
                if log_segmentation_metrics or return_metrics:
                    if prediction.shape != label.shape:
                        raise ValueError('Segmentation prediction and label shapes must match.')
                    pred_positive, target_positive = prediction >= 0.5, label >= 0.5
                    tp += int((pred_positive & target_positive).sum().item())
                    fp += int((pred_positive & ~target_positive).sum().item())
                    fn += int((~pred_positive & target_positive).sum().item())
                    pixel_count += label.numel()
        if sample_count == 0:
            raise ValueError('Validation data loader is empty.')
        metrics = {'val_loss': val_loss / sample_count}
        if log_segmentation_metrics or return_metrics:
            metrics.update({
                'val_iou': tp / max(tp + fp + fn, 1),
                'val_dice': 2 * tp / max(2 * tp + fp + fn, 1),
                'recall': tp / max(tp + fn, 1),
                'target_foreground': (tp + fn) / pixel_count,
                'predicted_foreground': (tp + fp) / pixel_count,
            })
        if log_segmentation_metrics:
            logging.info(
                '%s pixels: IoU=%.6f, Dice=%.6f, recall=%.6f, '
                'target_foreground=%.6f, predicted_foreground=%.6f, BCE=%.6f.',
                log_prefix, metrics['val_iou'], metrics['val_dice'], metrics['recall'],
                metrics['target_foreground'], metrics['predicted_foreground'], metrics['val_loss'],
            )
        return metrics if return_metrics else metrics['val_loss']
    finally:
        model.train(was_training)



#MAKE THIS FUNCIION INDENPENT OF THE MODEL
def train_model(model, data_loader, validation_data_loader, optimizer,scheduler,config):
    # Initialize training parameters from args
    epochs,device,valiate_every,run_Name,lossName=config['epochs'],config['device'],config['val_freq'],config['run_name'],config["model"]["criterion_args"]["NAME"]
    total_iterations,oom_time,best_val_loss=0,0,float('inf')
    model.to(device)
    sched_on_epoch=getattr(config,"sched_on_epoch",True)
    accumulation_steps=getattr(config,'gradient_accumulate', 1) 
    if accumulation_steps < 1 or len(data_loader) == 0:
        raise ValueError('Training requires a nonempty loader and gradient_accumulate >= 1.')
    max_grad_norm = getattr(config,'gradient_norm_clip', 1.0)   # Maximum norm for gradient clipping
    early_stopping=bool(getattr(config, 'early_stopping', False))
    early_stopping_patience=int(getattr(config, 'early_stopping_patience', 10))
    early_stopping_min_delta=float(getattr(config, 'early_stopping_min_delta', 0.0))
    validations_without_improvement=0
    stop_training=False
    is_mvunet = config.get('dataset', {}).get('NAME') == 'SteadyVelocityGridSegmentationMVUnet'
    monitor = config.get('checkpoint_monitor', 'val_loss')
    if monitor not in ('val_loss', 'val_iou', 'val_dice'):
        raise ValueError('checkpoint_monitor must be val_loss, val_iou or val_dice.')
    maximize_monitor = monitor != 'val_loss'
    best_monitor = -float('inf') if maximize_monitor else float('inf')
    best_checkpoint_loss = float('inf')
    metric_scheduler = str(config.get('scheduler', '')).lower() == 'plateau'
    scheduler_metric = config.get('eval_metric', 'val_loss')
    if metric_scheduler and (not sched_on_epoch or validation_data_loader is None):
        raise ValueError('Plateau scheduling requires epoch scheduling and validation data.')
    if metric_scheduler and scheduler_metric not in ('val_loss', 'val_iou', 'val_dice'):
        raise ValueError('Plateau eval_metric must be val_loss, val_iou or val_dice.')
    training_eval_loader = None
    diagnostic_samples = int(config.get('train_eval_samples', 0))
    if is_mvunet and diagnostic_samples > 0:
        # Shallow-copy the preloaded dataset, replacing only stochastic transforms.
        from DeepUtils.dataset.transforms import build_transforms_from_cfg
        clean_dataset = copy.copy(data_loader.dataset)
        clean_dataset.transform = build_transforms_from_cfg('val', config.datatransforms)
        clean_dataset.augment_d4 = False
        rng = np.random.default_rng(int(config.get('random_seed', 0)))
        indices = rng.choice(len(clean_dataset), min(diagnostic_samples, len(clean_dataset)), replace=False)
        training_eval_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(clean_dataset, indices.tolist()),
            batch_size=data_loader.batch_size, shuffle=False, num_workers=0,
        )

    def save_training_state(epoch, metrics, filename):
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'validation': metrics,
            'checkpoint_monitor': monitor,
        }, folder_name=os.path.join(config['save_model_path'], run_Name),
            checkpoint_name=filename)

    if early_stopping and validation_data_loader is None:
        raise ValueError("early_stopping requires a validation data loader.")
    if early_stopping and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive.")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0
        running_accumulate_loss=0
        try:
            for batch_idx, (data, label) in enumerate(data_loader):
                if isinstance(data, list):
                    # Unpack the tuple
                    vectorFieldImage, pathlines = data
                    # Move each element to the device
                    vectorFieldImage = vectorFieldImage.to(device)
                    pathlines = pathlines.to(device)
                    label = label.to(device)
                    # Repack into a tuple if needed
                    data = (vectorFieldImage, pathlines)
                else:
                    # If data is not a tuple, directly move to the device
                    data = data.to(device)
                    label = label.to(device)    
                    
                predictition= model(data)                
                loss=model.get_loss(predictition,label)
                # penalty =1/ (0.1+torch.mean((predictition - 0.5) ** 2 ))
                # loss += 0.01 * penalty 
                # Normalize a final incomplete accumulation group by its actual size.
                group_start = (batch_idx // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(data_loader) - group_start)
                loss = loss / group_size
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(data_loader):
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)                    
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None and not sched_on_epoch:
                        scheduler.step(epoch)         

                 
        
                epoch_loss += loss.item()* group_size
                running_accumulate_loss+=loss.item()* group_size
                total_iterations += 1
                if batch_idx % config['print_freq'] == 0:
                    lr=optimizer.param_groups[0]["lr"]
                    print_freq=config['print_freq'] if config['print_freq'] >0 and batch_idx>1 else 1
                    running_accumulate_loss/=float(print_freq)
                    logging.info(f'Epoch {epoch+1}, iter {batch_idx+1}, total_iterations: {total_iterations}. {lossName}: {running_accumulate_loss},lr:{lr}.')                    
                    if config['wandb']:
                        wandb.log({"train_loss": running_accumulate_loss,  "total_iterations": total_iterations,"lr":lr})
                    running_accumulate_loss=0
                               
            if scheduler is not None and sched_on_epoch and not metric_scheduler:
                scheduler.step(epoch + 1)
                        
            epoch_loss /= len(data_loader)
            
            # Validate the model
            if validation_data_loader is not None:
                if (epoch+1) % valiate_every == 0 or epoch==0:
                    need_metrics = is_mvunet or maximize_monitor or (
                        metric_scheduler and scheduler_metric != 'val_loss')
                    validation = validate(
                        model, validation_data_loader, device,
                        log_segmentation_metrics=need_metrics,
                        return_metrics=need_metrics,
                    )
                    metrics = validation if need_metrics else {'val_loss': validation}
                    val_loss = metrics['val_loss']
                    if training_eval_loader is not None:
                        validate(model, training_eval_loader, device,
                                 log_segmentation_metrics=True, log_prefix='Train(eval)')
                    logging.info(f'Epoch {epoch+1}, epoch_Loss: {epoch_loss}, Val Loss: {val_loss}')
                    if config['wandb']:
                        wandb.log({"epoch": epoch+1, "epoch_Loss": epoch_loss, **metrics})
                    if metric_scheduler and scheduler is not None:
                        old_lr = optimizer.param_groups[0]['lr']
                        scheduler.step(epoch + 1, metric=metrics[scheduler_metric])
                        new_lr = optimizer.param_groups[0]['lr']
                        if old_lr != new_lr:
                            logging.info('Validation scheduler: %s=%.6f, lr %g -> %g.',
                                         scheduler_metric, metrics[scheduler_metric], old_lr, new_lr)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if config['save_model'] and maximize_monitor:
                            save_training_state(epoch, metrics, 'best_loss_checkpoint.pth.tar')
                    current_monitor = metrics[monitor]
                    improved = (
                        current_monitor > best_monitor + early_stopping_min_delta
                        if maximize_monitor else
                        current_monitor < best_monitor - early_stopping_min_delta
                    )
                    # Identical masks can have different calibration early in training.
                    if maximize_monitor and current_monitor == best_monitor:
                        improved = val_loss < best_checkpoint_loss - early_stopping_min_delta
                    if improved:
                        best_monitor = current_monitor
                        best_checkpoint_loss = val_loss
                        validations_without_improvement = 0
                        logging.info('Best %s=%.6f at epoch %d.', monitor, best_monitor, epoch + 1)
                        if config['save_model']:
                            save_training_state(epoch, metrics, 'best_checkpoint.pth.tar')
                    else:
                        validations_without_improvement += 1
                        if (early_stopping and
                                validations_without_improvement >= early_stopping_patience):
                            logging.info(
                                "Early stopping at epoch %d after %d validation "
                                "checks without a %s improvement greater than %g.",
                                epoch + 1,
                                validations_without_improvement,
                                monitor,
                                early_stopping_min_delta,
                            )
                            stop_training = True
                    if config['save_model'] and config.get('save_last', False):
                        save_training_state(epoch, metrics, 'last_checkpoint.pth.tar')
                    
            #periodically save model
            if  config['save_model'] and config["save_freq"]>0 and epoch % config["save_freq"]== 0 and epoch>0 :
                saving_path= os.path.join(config['save_model_path'],run_Name) 
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, folder_name=saving_path, checkpoint_name= f'epoch_{epoch+ 1}.pth.tar')
                
            else:
                logging.info(f'Epoch {epoch+1}, epoch_Loss: {epoch_loss}')
                if config['wandb']:
                    wandb.log({"epoch": epoch+1,  "epoch_Loss": epoch_loss})

            if stop_training:
                break

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                oom_time += 1
                logging.info("WARNING: ran out of memory,times: {}".format(oom_time))
                if oom_time > 3:
                    logging.info("WARNING: ran out of memory 3 times, stopping training")
                    break
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logging.error(str(exception))
                raise exception
            
    return best_val_loss




def train_pipeline():
    print("torch.cuda.is_available():",torch.cuda.is_available())  # Should return True
    print("torch.version.cuda:",torch.version.cuda)         # Should print the CUDA version PyTorch was built with
    phase = 'setup'
    try:
        cfg=argParseAndPrepareConfig()
        cfg["gitInfo"]=get_git_commit_id()
        run_Name,runTags=runNameTagGenerator(cfg)
        cfg['run_name']=run_Name
        logging.info(f"run name: {run_Name}, run tags: {runTags}")
        #get dataset realted config paramters,like PathlineGroupsCount, minV,maxV, etc.
        if cfg.dataset.get("NAME") == "UnsteadyVastisPathlineSeg":
            readDataSetRelatedConfig(cfg)
        elif cfg.dataset.get("NAME") == "SteadyVelocityGridSegmentationMVUnet":
            prepare_mvunet_config(cfg)
        # Resolve the evaluator before preloading/training, so deployment errors
        # cannot first appear after a long run. MVUNet has no dependency on test.py.
        if cfg.dataset.get('NAME') == 'SteadyVelocityGridSegmentationMVUnet':
            from test_mvunet import evaluate_trained_model as final_evaluator
        else:
            from test import test_model as final_evaluator
        logging.info(
            'batch_size=%d, gradient_accumulate=%d, effective_batch_size=%d.',
            cfg.batch_size, cfg.gradient_accumulate,
            cfg.batch_size * cfg.gradient_accumulate,
        )
        model = build_model_from_cfg(cfg.model)

        # V2's added attention paths are zero-gated, so a V1 checkpoint is an
        # exact functional warm start.  Keep this opt-in through the existing
        # --model_path argument; other encoders retain their previous behavior.
        warm_start_path = cfg.get("model_path")
        if warm_start_path is not None:
            encoder = getattr(model, "encoder", model)
            if not hasattr(encoder, "load_v1_checkpoint"):
                raise ValueError(
                    "--model_path warm-start is only supported by encoders "
                    "that implement load_v1_checkpoint().")
            incompatible = encoder.load_v1_checkpoint(warm_start_path)
            logging.info(
                "Warm-started V2 from V1 checkpoint %s; initialized %d "
                "new physics-attention tensors.",
                warm_start_path,
                len(incompatible.missing_keys),
            )
        model.to(cfg['device'])
    
        # if "unsteadyFieldTimeStep" not in rootInfo:            
        #     torchsummary.summary(model, (2, rootInfo["Xdim"], rootInfo["Ydim"]))
        # else:
        train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                                cfg.dataset,
                                                cfg.dataloader,
                                                datatransforms_cfg=cfg.datatransforms,
                                                split='train'                                             
                                                )
        print(f"length of training dataset: {len(train_loader.dataset)}")

        if cfg["val_freq"]>0:
            val_loader = build_dataloader_from_cfg(cfg.batch_size,
                                                    cfg.dataset,
                                                    cfg.dataloader,
                                                    datatransforms_cfg=cfg.datatransforms,
                                                    split='val'                                             
                                                    )
            print(f"length of val dataset: {len(val_loader.dataset)}")
        else:
            val_loader=None

        if (cfg.dataset.get('NAME') == 'SteadyVelocityGridSegmentationMVUnet'
                and int(cfg.get('dataset_audit_samples', 0)) > 0):
            from DeepUtils.dataset.mvunet_audit import audit_mvunet_splits
            report = audit_mvunet_splits(
                train_loader.dataset, val_loader.dataset if val_loader is not None else None,
                sample_count=int(cfg.dataset_audit_samples), seed=int(cfg.random_seed),
            )
            audit_dir = Path(cfg.save_model_path) / run_Name
            audit_dir.mkdir(parents=True, exist_ok=True)
            (audit_dir / 'dataset_audit.json').write_text(json.dumps(report, indent=2) + '\n')
            logging.info('MVUNet dataset audit: %s', audit_dir / 'dataset_audit.json')

      
       # optimizer & scheduler
        optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
        scheduler = build_scheduler_from_cfg(cfg, optimizer) if "scheduler" in cfg else None

        # Initialize wandb
        if cfg['wandb']:
            run=wandb.init(project=GLOBAL_WANDB_PROJECT_NAME,
                        name=run_Name,
                        tags=runTags,
                        config=cfg)
            arti_code=wandb.Artifact("code", type="code")
            arti_code=CollectFiles4Backup(config=cfg,arti_code=arti_code)
            wandb.log_artifact(arti_code) 
        print_args(cfg)
        if cfg['save_model'] and cfg.dataset.get('NAME') == 'SteadyVelocityGridSegmentationMVUnet':
            run_dir = Path(cfg.save_model_path) / run_Name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'resolved_config.yaml').write_text(yaml.safe_dump(cfg.dict(), sort_keys=False))
        
        # Training
        phase = 'training'
        best_val_loss=train_model(model, train_loader, val_loader, optimizer,scheduler,cfg)

        # Evaluate the checkpoint selected by the configured validation metric.
        if cfg['save_model'] and val_loader is not None:
            best_checkpoint_path = os.path.join(
                cfg['save_model_path'], run_Name,
                "best_checkpoint.pth.tar",
            )
            if os.path.exists(best_checkpoint_path):
                best_checkpoint = torch.load(
                    best_checkpoint_path, map_location=cfg['device'])
                model.load_state_dict(best_checkpoint['state_dict'])
                logging.info(
                    "Reloaded best validation checkpoint from %s.",
                    best_checkpoint_path,
                )
        #final test 
        phase = 'final evaluation'
        test_result:dict=final_evaluator(model,cfg)

        if cfg['wandb'] and  isinstance(test_result,dict):            
            wandb.summary.update({"best_val_loss": best_val_loss})
            for key, value in test_result.items():
                wandb.summary.update({key: value})
            wandb.finish()

    except Exception:
        logging.exception('Pipeline failed during %s.', phase)
        raise
   











if __name__ == '__main__':
    train_pipeline()
    
