work_dir='../TOV_mmdetection_cache/work_dir/coco/' && CUDA_VISIBLE_DEVICES=0 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  \
--cfg-options evaluation.save_result_file=${work_dir}'_1200_latest_result.json'