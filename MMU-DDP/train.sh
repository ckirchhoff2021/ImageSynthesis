# set -x
# setup_env gpu
# pip install scikit-image
# echo "start training"

# pip install albumentations
# pip install opencv-python==4.5.5

python main.py --model_name "GradientDeblurGAN" \
        --mode "train" \
        --model_save_dir '/data/juicefs_hz_cv_v2/11145199/deblur/finetune/0207' \
        --resume '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0119/GradientDeblurGAN/weights/model_150.pkl' \
        --data_dir '/data/juicefs_hz_cv_v3/11145199/datas/pix2pixGen'
        # --train_count "512" 
        # --data_dir '/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/GoPro' \
        # --model_save_dir '/data/juicefs_hz_cv_v2/11145199/deblur/results/FFT' \
        # --model_save_dir '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANDDP' \
        # --model_save_dir '/data/juicefs_hz_cv_v2/11145199/deblur/results/GAN' \
        # --has_bbox \
