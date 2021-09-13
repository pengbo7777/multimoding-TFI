CUDA_VISIBLE_DEVICES=1 python test.py --model SKNet_dilation --dataset buyan_cdx_cls --test-batch-size 128
CUDA_VISIBLE_DEVICES=1 python test.py --model SKNet_dilation --dataset buyan_changxian_close_focusing --batch-size 64 --test-batch-size 128 --lr 0.002 --epochs 500 --weight-decay 1e-4 --boarddir SKNet_dilation5_1_close
CUDA_VISIBLE_DEVICES=1 python train.py --model SKNet_dilation --batch-size 64 --test-batch-size 64 --lr 0.01 --epochs 500 --weight-decay 1e-4 --boarddir SKNet_dilation_2_1_cdx_cls_SKNet_dilation --dataset buyan_cdx_cls
CUDA_VISIBLE_DEVICES=0,1 python train.py --model SKNet_dilation --batch-size 32 --test-batch-size 32 --lr 0.01 --epochs 500 --weight-decay 1e-4 --boarddir SKNet_dilation_2_1_cdx_cls_SKNet_dilation_pre --dataset buyan_cdx_cls --pretrain=/home/pengbo/workspace/ResNeSt_DEP/runs/buyan_cdx_cls/SKNet_dilation/SKNet_dilation_2_1_cdx_cls_/model_best.pth
Deep_texture_transformer
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model ResNeSt_DEP --batch-size 64 --test-batch-size 64 --lr 0.001 --epochs 500 --weight-decay 1e-4 --boarddir ResNeSt_DEP --dataset buyan_cxf_dtd
CUDA_VISIBLE_DEVICES=0 python train.py --model ResNest_DEP --batch-size 128 --test-batch-size 128 --lr 0.001 --epochs 500 --weight-decay 1e-4 --boarddir ResNeSt_DEP_8 --dataset buyan_cxf_dtd


python train_dist.py --dataset minc --model deepten_resnet50_minc --lr-scheduler cos --epochs 120 --checkname resnet50_check --lr 0.025 --batch-size 64
