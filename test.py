import torch,random
import numpy as np
import os
import datetime
import inspect
from DeepUtils.dataset import build_dataloader_from_cfg
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from DeepUtils.dataset import UnsteadyVastisDataset
from FLowUtils.vortexCriteria import *
from DeepUtils.MiscFunctions import argParseAndPrepareConfig,readDataSetRelatedConfig,prepare_mvunet_config
from DeepUtils.models import build_model_from_cfg
from DeepUtils.dataset.data_utils import read_binary_file, save_segmentation_as_png


class TestReconstructSteadyField(object):
    def __init__(self, device, data_dir,**kwargs):
        self.device=device
        bs=1
        #deformingZerofieldData
        self.dataset= UnsteadyVastisDataset(data_dir,"test",None)

    def __call__(self,model,test_data_loader) -> None:
        from FLowUtils.LicRenderer import LicRenderingUnsteadyCpp
        # from FLowUtils.GlyphRenderer import glyphsRenderUnsteadyField
        device=self.device
        minV=-3.699822425842285
        maxV= 3.9069676399230959
        model.eval()
        total_error=0
        for sample in range(len( self.dataset)):
            vectorFieldImage, label=self.dataset[sample]
            UnsteadyField=  UnsteadyVectorField2D(16,16,5,[-2,-2],[2,2],tmin=0,tmax=0.7853981633974483)
            UnsteadyField.field=vectorFieldImage.transpose(0,-1).cpu().numpy()
            #feed network
            vectorFieldImage=(vectorFieldImage-minV)/(maxV-minV)
            vectorFieldImage = vectorFieldImage.unsqueeze(0).to(device)
            predictition= model(vectorFieldImage)
            predictition=predictition[0].cpu().numpy()
            name=self.dataset.getSampleName(sample)
            abc,abc_dot=predictition[0:3],predictition[3:6]
            recRes=referenceFrameReconstruct(abc,abc_dot,UnsteadyField)
            LicRenderingUnsteadyCpp(recRes,128,timeStepSKip=2,saveFolder="./testOutput",saveName=f"lic__{name}_rec")
            print(f"reconstruct task {name}, predicts {predictition}, vs label ={label}")
            diff= (recRes.field-UnsteadyField.field)*(recRes.field-UnsteadyField.field)
            print(f"reconstructdiff max={diff.max()}, min={diff.min()}, mean={diff.mean()}")
            total_error+=diff.mean()
        print(f"reconstruct  totaldiff ={total_error}")
        return None    
    


def segmentationCriteria(pred, gt):
    """
    Computes precision, recall, F1 score, and Intersection over Union (IoU) for segmentation.
    Parameters:
        pred (numpy.ndarray): Predicted segmentation mask, shape [batch_size, width, height, 2].
        gt (numpy.ndarray): Ground truth segmentation mask, shape [batch_size, width, height, 2].
    Returns:
        np.array( [TP,FP,FN, precision, recall, F1, IoU],dtype=np.float32) 
    """
    # Extract the binary segmentation mask (second channel)
    pred_mask = pred[..., 0]  # shape [batch_size, width, height]
    gt_mask = gt[..., 0]      # shape [batch_size, width, height]
    
    # Flatten the masks to compute metrics for the entire batch
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    total_samples=gt_flat.shape[0]

    # True positives, False positives, False negatives
    TP = np.sum((pred_flat > 0.5) & (gt_flat == 1))  # True Positive
    FP = np.sum((pred_flat > 0.5) & (gt_flat == 0))  # False Positive
    FN = np.sum((pred_flat < 0.5) & (gt_flat == 1))  # False Negative

    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Intersection over Union (IoU): TP / (TP + FP + FN)
    IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    TP=TP/float(total_samples)
    FP=FP/float(total_samples)
    FN=FN/float(total_samples)
    return np.array( [TP,FP,FN, precision, recall, F1, IoU],dtype=np.float32) 





class TestSegmentation(object):
    """ TestSegmentation  is the default test task for Segmentation tasks """
    def __init__(self, device,config,samples=10,**kwargs):
          self.device=device
          self.samples=samples
          self.runName=config["run_name"]


    def __call__(self, model,test_data_loader):
        device=self.device
        segError=0.0
        meta_=test_data_loader.dataset.dastasetMetaInfo
        test_loss=0
     
        for batch_idx, (data, label) in enumerate(test_data_loader):
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
            test_loss += loss.item()                
            segError_=segmentationCriteria(predictition.cpu().numpy(),label.cpu().numpy())
            segError+=segError_

        segError /= len(test_data_loader)
        TP,FP,FN, precision, recall, F1, IoU=segError[0],segError[1],segError[2],segError[3],segError[4],segError[5],segError[6]
        print(f"TP,FP,FN={TP},{FP},{FN}")
        print(f"precision, recall, F1, IoU={precision},{recall},{F1},{IoU}")
        
        result = {
            "precision": precision,
            "F1": F1,
            "IoU": IoU
        }
        return result


class TestPathlineSegmentation(object):
    """Evaluate every predicted pathline with global micro statistics.

    ``TestSegmentation`` predates the pathline model and indexes ``[..., 0]``;
    for a ``[B, K]`` PathlineTransformer output that evaluates only one line
    per sample.  This task preserves the legacy evaluator while providing the
    complete TP/FP/FN/TN protocol needed for V1/V2 comparisons.
    """

    def __init__(self, device, config, threshold=0.5, **kwargs):
        self.device = device
        self.threshold = float(threshold)
        self.runName = config["run_name"]

    @staticmethod
    def _safe_ratio(numerator, denominator):
        return (
            float(numerator) / float(denominator)
            if denominator else 0.0
        )

    def __call__(self, model, test_data_loader):
        tp = fp = fn = tn = 0
        weighted_loss = 0.0
        sample_count = 0

        for data, label in test_data_loader:
            if isinstance(data, list):
                vector_field_image, pathlines = data
                data = (
                    vector_field_image.to(self.device),
                    pathlines.to(self.device),
                )
            else:
                data = data.to(self.device)
            label = label.to(self.device)

            prediction = model(data)
            if prediction.shape != label.shape:
                raise ValueError(
                    "Binary segmentation expects prediction and label "
                    f"to have the same shape, got "
                    f"{tuple(prediction.shape)} and {tuple(label.shape)}.")
            loss = model.get_loss(prediction, label)
            batch_size = label.shape[0]
            weighted_loss += float(loss.item()) * batch_size
            sample_count += batch_size

            predicted_positive = prediction >= self.threshold
            target_positive = label >= 0.5
            tp += int((predicted_positive & target_positive).sum().item())
            fp += int((predicted_positive & ~target_positive).sum().item())
            fn += int((~predicted_positive & target_positive).sum().item())
            tn += int((~predicted_positive & ~target_positive).sum().item())

        total = tp + fp + fn + tn
        accuracy = self._safe_ratio(tp + tn, total)
        precision = self._safe_ratio(tp, tp + fp)
        recall = self._safe_ratio(tp, tp + fn)
        specificity = self._safe_ratio(tn, tn + fp)
        f1 = self._safe_ratio(
            2.0 * precision * recall, precision + recall)
        iou = self._safe_ratio(tp, tp + fp + fn)
        balanced_accuracy = 0.5 * (recall + specificity)
        test_loss = self._safe_ratio(weighted_loss, sample_count)

        print(f"TP,FP,FN,TN={tp},{fp},{fn},{tn}")
        print(
            "accuracy, balanced_accuracy, precision, recall, F1, IoU="
            f"{accuracy},{balanced_accuracy},{precision},{recall},{f1},{iou}"
        )
        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "IoU": iou,
            "test_loss": test_loss,
        }


class TestGridSegmentation(TestPathlineSegmentation):
    """Evaluate all pixels in [B,Y,X] masks using global confusion counts."""


class TestVectorFieldSegmentation(object):
    """ TestSegmentation  is the default test task for Segmentation tasks """
    def __init__(self, device,config,samples=10,**kwargs):
          self.device=device
          self.samples=samples
          self.runName=config["run_name"]
          self.config=config


    def __call__(self, model,test_data_loader):
        device=self.device
        segError=0.0
        meta_=test_data_loader.dataset.dastasetMetaInfo
        Xdim,Ydim=meta_["Xdim"],meta_["Ydim"]
        dm_min,dm_max=meta_["domainMinBoundary"],meta_["domainMaxBoundary"]     
        # grid_dx,grid_dy=(dm_max[0]-dm_min[0])/float( Xdim-1),(dm_max[1]-dm_min[1])/float( Ydim-1)
        # minV,maxV=getattr(self.config,"minV",-1.0),getattr(self.config,"maxV",1.0)
        out_folder=f"./testOutput/{self.runName}"
        className=str(__class__)
        print(f"{className} save to {out_folder}")
        # #random select  samples to visualize
        for i in range(self.samples):
            sample=random.randint(0,len(test_data_loader.dataset)-1)
            data, label=test_data_loader.dataset[sample]
            if isinstance(data, list) or isinstance(data, tuple) :
                # Unpack the tuple
                vectorFieldImage, pathlines = data
                # Move each element to the device
                batch_vectorFieldImage = vectorFieldImage.unsqueeze(0).to(device)
                pathlines = pathlines.unsqueeze(0).to(device)
                # Repack into a tuple if needed
                data = (batch_vectorFieldImage, pathlines)
            else:
                # If data is not a tuple, directly move to the device
                data = data.unsqueeze(0).to(device)
              

            predictition= model(data)
            predictition=predictition[0].cpu().numpy()
            label=label.cpu().numpy()
            name=test_data_loader.dataset.getSampleName(sample)
            save_segmentation_as_png(predictition,f"{out_folder}/{name}_pred.png",upSample=10.0)
            save_segmentation_as_png(label,f"{out_folder}/{name}_gt.png",upSample=10.0)
            # rawVectorField=data.squeeze().transpose(0,-1).cpu().numpy()
            # rawVectorField=rawVectorField*(maxV-minV)+minV
            # qCriterion=computeQcriterion(rawVectorField,grid_dx,grid_dy)
            # ivd=computeIVD(rawVectorField,grid_dx,grid_dy)
            # saveCriteriaPicture(qCriterion,f"{out_folder}/{name}_q_cri.png",upSample=10.0)
            # saveCriteriaPicture(ivd,f"{out_folder}/{name}_ivd.png",upSample=10.0)





def pathlineSegToFieldSeg(Pathlines,PathlineSeg,Xdim,Ydim,DominMin,DominMax):
    L,K,C=Pathlines.shape
    # Extract seeding points from Pathlines
    seeding_points = Pathlines[ 0,:, :2]  # Assuming first two dimensions are x, y coordinates
    dx,dy=(DominMax[0]-DominMin[0])/float(Xdim-1),(DominMax[1]-DominMin[1])/float(Ydim-1)
    # Compute the segmentation for each grid point
    grid_segmentation = np.zeros((Ydim, Xdim), dtype=np.float32)
    for (idx) in range(K):
        this_pathline_seeding_pos=seeding_points[idx]
        corresponding_discrete_gridx=int((this_pathline_seeding_pos[0]-DominMin[0])/(dx))
        corresponding_discrete_gridy=int((this_pathline_seeding_pos[1]-DominMin[1])/(dy))
        corresponding_discrete_gridx = max(0, min(corresponding_discrete_gridx, Xdim-1))
        corresponding_discrete_gridy = max(0, min(corresponding_discrete_gridy, Ydim-1))
        grid_segmentation[corresponding_discrete_gridy,corresponding_discrete_gridx]+= PathlineSeg[idx]
        
    return grid_segmentation


def _build_unsteady_field_compatible(
    Xdim, Ydim, Tdim, domain_min, domain_max):
    """Construct UnsteadyVectorField2D across the 2-D/3-D boundary APIs.

    Repository revisions disagree on whether spatial bounds are ``[x,y]``
    or spatiotemporal bounds are ``[x,y,t]``.  Select the format from the
    installed class signature instead of forcing one revision's contract.
    """
    boundary_dimensions = []
    try:
        parameters = inspect.signature(
            UnsteadyVectorField2D.__init__).parameters
        default_min = parameters["domainMinBoundary"].default
        default_dimension = len(default_min)
        if default_dimension in (2, 3):
            boundary_dimensions.append(default_dimension)
    except (KeyError, TypeError):
        pass

    for dimension in (3, 2):
        if dimension not in boundary_dimensions:
            boundary_dimensions.append(dimension)

    spatial_min = [float(domain_min[0]), float(domain_min[1])]
    spatial_max = [float(domain_max[0]), float(domain_max[1])]
    time_min = float(domain_min[2]) if len(domain_min) > 2 else 0.0
    time_max = float(domain_max[2]) if len(domain_max) > 2 else time_min
    errors = []
    for dimension in boundary_dimensions:
        compatible_min = list(spatial_min)
        compatible_max = list(spatial_max)
        if dimension == 3:
            compatible_min.append(time_min)
            compatible_max.append(time_max)
        try:
            return UnsteadyVectorField2D(
                Xdim, Ydim, Tdim, compatible_min, compatible_max)
        except (AssertionError, IndexError, TypeError) as error:
            errors.append(f"{dimension}D bounds: {error}")

    raise RuntimeError(
        "Could not construct UnsteadyVectorField2D with either [x,y,t] "
        f"or [x,y] boundaries: {'; '.join(errors)}")


class TestPathlineSeg(object):
    def __init__(self, device, config,samples=30, **kwargs):
          self.device=device
          self.samples=samples
          self.runName=config["run_name"]
          self.data_dir=config["dataset"]["data_dir"]
          self.outputPathlineLength=config["outputPathlineLength"]
          self.outputPathlinesCountK=config["outputPathlinesCountK"]        
          self.pathlineFeatures=config["PathlineFeature"]
          self.config=config
               
    
    def __call__(self, model,test_data_loader) :
        # LIC depends on optional platform-specific C++ plugins.  Import it
        # only for this visualization task so ordinary training/evaluation is
        # portable to platforms without those plugins.
        from FLowUtils.LicRenderer import LicRenderingPathlineSegmentation

        device=self.device
        # first random select  samples to visualize
        out_folder=f"./testOutput/{self.runName}"
        className=str(__class__)
        print(f"{className} save to {out_folder}")
        DomainMin=test_data_loader.dataset.dastasetMetaInfo["domainMinBoundary"]
        DomainMax=test_data_loader.dataset.dastasetMetaInfo["domainMaxBoundary"]

        sample_count = min(self.samples, len(test_data_loader.dataset))
        for i in range(sample_count):
            # sample=random.randint(0,len(test_data_loader.dataset)-1)
            sample=i
            data, label=test_data_loader.dataset[sample]
            assert(isinstance(data, list) or isinstance(data, tuple) ) 
            # Unpack the tuple
            vectorFieldImage, pathlines = data
            # Repack into a tuple if needed
            data = (None, pathlines.unsqueeze(0).to(device))
            predictition= model(data)
            predictition=predictition[0].cpu().numpy()
            label=label.cpu().numpy()
            name=os.path.basename(
                test_data_loader.dataset.getSampleName(sample))
            if name.endswith("meta.json"):
                name=name[:-len("meta.json")]
            
            # Dataset field layout is [C,Y,X,T].  Current generated velocity
            # binaries contain only t0, so T may legitimately be one.
            if vectorFieldImage.ndim != 4 or vectorFieldImage.shape[0] < 2:
                raise ValueError(
                    f"TestPathlineSeg requires a [C,Y,X,T] velocity field "
                    f"with u/v channels; sample {name} has shape "
                    f"{tuple(vectorFieldImage.shape)}."
                )
            Ydim=int(vectorFieldImage.shape[1])
            Xdim=int(vectorFieldImage.shape[2])
            Tdim=int(vectorFieldImage.shape[3])
            if Xdim < 2 or Ydim < 2:
                raise ValueError(
                    f"TestPathlineSeg cannot project onto a {Xdim}x{Ydim} "
                    f"field for sample {name}. Check that its velocity .bin "
                    "is available and has a supported layout."
                )
            UnsteadyField=_build_unsteady_field_compatible(
                Xdim,Ydim,Tdim,DomainMin,DomainMax)
            UnsteadyField.field=(
                vectorFieldImage.permute(3,1,2,0).cpu().numpy())
            pathlines_np=pathlines.cpu().numpy()
            label_seg=pathlineSegToFieldSeg(pathlines_np,label,Xdim,Ydim,DomainMin,DomainMax)
            predictition_seg=pathlineSegToFieldSeg(pathlines_np,predictition,Xdim,Ydim,DomainMin,DomainMax)
            LicRenderingPathlineSegmentation(UnsteadyField,predictition_seg,4.0,saveFolder=out_folder,saveName=f"{name}__pred")
            LicRenderingPathlineSegmentation(UnsteadyField,label_seg,4.0,saveFolder=out_folder,saveName=f"{name}__gt")
                    
                    
                    
        
        
            

   
      

def test_model(model,cfg):
    device = cfg['device']
    test_data_loader = build_dataloader_from_cfg(cfg.batch_size,
                                        cfg.dataset,
                                        cfg.dataloader,
                                        datatransforms_cfg=cfg.datatransforms,
                                        split='test'                                             
                                        )
    print(f"length of test dataset: {len(test_data_loader.dataset)}")
    model.to(cfg['device'])
    model.eval()
    #building test tasks
    test_cfg=cfg['test_tasks']
    kwagrs=test_cfg["kwargs"] if "kwargs" in test_cfg else {}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    cfg['run_name']=getattr(cfg,'run_name',f"default/{timestamp}")
    
    test_tasks=[]
    for  cfg_task_name in test_cfg['tasks']:
        task_init_fn=eval(cfg_task_name)
        t=task_init_fn(device=device,config=cfg,**kwagrs)
        test_tasks.append(t)
        
    model.eval()
    retValues={}
    with torch.no_grad():
        for t in test_tasks:
            key=str(t.__class__.__name__)
            print(f"run test task: [{key}]")
            retValues[key]=t(model,test_data_loader)
    if "TestLoss" in retValues:
        retLoss=retValues["TestLoss"]
    elif "TestClassification" in retValues:
        retLoss=retValues["TestClassification"]
    elif "TestPathlineSegmentation" in retValues:
        retLoss=retValues["TestPathlineSegmentation"]
    elif "TestGridSegmentation" in retValues:
        retLoss=retValues["TestGridSegmentation"]
    elif "TestSegmentation" in retValues:
        retLoss=retValues["TestSegmentation"]
    else :
        retLoss=None
    return retLoss

    

def test_pipeline():
    cfg=argParseAndPrepareConfig()
    model_path=cfg['model_path']
    if cfg.dataset.get('NAME') == 'SteadyVelocityGridSegmentationMVUnet':
        prepare_mvunet_config(cfg)
    else:
        readDataSetRelatedConfig(cfg)

    requested_device = torch.device(cfg.get('device', 'cpu'))
    if requested_device.type == 'cuda' and not torch.cuda.is_available():
        print(
            "CUDA was requested but is not available; falling back to CPU. "
            "Use --device cpu to select it explicitly."
        )
        requested_device = torch.device('cpu')
    elif requested_device.type == 'mps' and not (
        hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    ):
        print(
            "MPS was requested but is not available; falling back to CPU. "
            "Use --device cpu to select it explicitly."
        )
        requested_device = torch.device('cpu')
    cfg['device'] = requested_device

    if model_path is not None and os.path.exists(model_path):
        checkpoint=torch.load(model_path, map_location=requested_device)
        model = build_model_from_cfg(cfg.model)

        model.load_state_dict(checkpoint['state_dict'])
        model.to(cfg['device'])
        test_model(model,cfg)
  


# test_model_path="outputModels\\TobiasVortexBoundaryUnet\\bs_256_ep_100_lr_0.0001_20240924_170417_seed_4462\\best_checkpoint.pth.tar"    
# test_model_path="outputModels\\bs_100_ep_200_lr_0.0001_20240926_123336_seed_3097\\best_checkpoint.pth.tar"    
if __name__ == '__main__':
    test_pipeline()
