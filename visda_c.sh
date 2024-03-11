DATASET="visda" # imagenet_c domainnet126 officehome
METHOD="difo"          # shot nrc plue difo

echo DATASET: $DATASET
echo METHOD: $METHOD


# source_free_adaptation() {
# if [ "$DATASET" == "office-home" ];then
s_list=(0)
t_list=(1)
# fi
for s in ${s_list[*]}; do
    for t in ${t_list[*]}; do
    if [[ $s = $t ]]
        then
        continue
    fi
        CUDA_VISIBLE_DEVICES=1 ~/anaconda3/envs/sfa_susu/bin/python image_target_of_oh_vs.py --cfg "cfgs/${DATASET}/${METHOD}.yaml" \
            SETTING.S "$s" SETTING.T "$t" &
        wait
    done
done 

