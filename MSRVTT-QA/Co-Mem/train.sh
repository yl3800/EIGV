python train_val.py -v='grid-resnet-resnext-1'  -app_feat=resnet -mot_feat=resnext -gpu=0 &&
python train_val.py -v='grid-resnet-3dres152-2'  -app_feat=resnet -mot_feat=3dres152 -gpu=0 &&
python train_val.py -v='grid-res152-resnext-3'  -app_feat=res152 -mot_feat=resnext -gpu=0 &&
python train_val.py -v='grid-res152-3dres152-4'  -app_feat=res152 -mot_feat=3dres152 -gpu=0