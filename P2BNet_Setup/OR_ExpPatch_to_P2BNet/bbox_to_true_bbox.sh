export VERSION=1
export MU=(0 0)
export S=(0.25 0.25)  # sigma
export SR=0.25
export CORNER=""
export WH=(64 64)
export T="val"
PYTHONPATH=. python huicv/coarse_utils/noise_data_utils.py "generate_pseudo_bbox_for_point" \
    "data/coco/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_instances_${T}2017_coarse.json"  \
    "data/coco/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_instances_${T}2017_coarse_with_gt.json"  \
    --pseudo_w ${WH[0]} --pseudo_h ${WH[1]}