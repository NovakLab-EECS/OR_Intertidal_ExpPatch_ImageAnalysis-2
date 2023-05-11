export MU=(0 0)
export S=(0.25 0.25)  # sigma
export SR=0.25 # size_range
export VERSION=1
export CORNER=""
export T="val"
#export T="train"
PYTHONPATH=. python huicv/coarse_utils/noise_data_mask_utils.py "generate_noisept_dataset" \
    "data/coco/annotations/${T}_annotations.json" \
    "data/coco/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_instances_${T}2017_coarse.json" \
    --rand_type 'center_gaussian' --range_gaussian_sigma "(${MU[0]},${MU[1]})" --range_gaussian_sigma "(${S[0]},${S[1]})" \
    --size_range "${SR}"