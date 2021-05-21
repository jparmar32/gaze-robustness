train_set = cxr_p
test_set = cxr_p


parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--min_lr", type=float, default=0, help="Minimum learning rate")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0001, help="L2 Weight")
    parser.add_argument("--save_model", action='store_true', help="Whether to save the best model found or not") ##will be false unles flag is specified
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Whether to use a given saved model")
    parser.add_argument("--seed", type=int, default=0, help="Seed to use in dataloader")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    parser.add_argument("--train_set", type=str, choices=['cxr_a','cxr_p'], required=True, help="Set to train on")
    parser.add_argument("--test_set", type=str, choices=['cxr_a','cxr_p', 'mimic_cxr', 'chexpert', 'chestxray8'], required=True, help="Test set to evaluate on")
    parser.add_argument("--ood_shift", type=str, choices=['age','hospital_age', 'age', None], default=None, help="Distribution shift to experiment with")

    parser.add_argument("--save_dir", type=str, default="/mnt/gaze_robustness_results/resnet_only", help="Save Dir")


for seed in 0 1 2 3 4 5 6 7 8 9 
do
    python ./train.py \
    --epochs 25 \
    --min_lr 0 \
    --lr 0.0001 \
    --wd 0.0001 \
    --save_model \ 
    #--checkpoint_dir  
    --seeed $seed \
    --batch_size 32 \
    --train_set $train_set \
    --test_set $test_set \
    #--ood_shift \
    --save_dir /mnt/gaze_robustness_results/resnet_only \
 
done