# set -x
# setup_env gpu
# pip install scikit-image
# echo "start training"


python main.py --model_name "DeblurUnet" \
        --mode "train" \
        --model_save_dir '/data/juicefs_hz_cv_v2/11145199/deblur/results/' \
        --data_dir '/data/juicefs_hz_cv_v3/11145199/datas/pix2pixGen' 
        # --train_count "256" 
