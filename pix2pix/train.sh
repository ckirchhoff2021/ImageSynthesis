# set -x
# setup_env gpu
# pip install scikit-image
# echo "start training"
python trains.py --model_name "Pix2Pix" \
        --mode "train" \
        --model_save_dir '/data/juicefs_hz_cv_v2/11145199/deblur/results/'  \
        --result_dir '/data/juicefs_hz_cv_v2/11145199/deblur/results/' 
        # --has_bbox \
