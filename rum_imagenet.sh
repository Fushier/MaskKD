python train_student.py --path-t ./save/models/ResNet34_vanilla/resnet34_transformed.pth \
--batch_size 256 --epochs 100 --dataset imagenet --gpu_id 0 --dist-url tcp://127.0.0.1:23333 \
--print-freq 500 --num_workers 8 --distill mask --model_s ResNet18 -r 1 -a 1 -b 0.02 --trial 1 \
--multiprocessing-distributed --learning_rate 0.1 --lr_decay_epochs 30,60 --weight_decay 1e-4 --dali gpu