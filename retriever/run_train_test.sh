if [ "$#" -ge 1 ]; then
    mode="$1"
else
    mode="test"
fi

if [ "$#" -ge 2 ]; then
    gpu="$2"
else
    gpu=0
fi

# default
if [ "$#" -ge 3 ]; then
    par_id="$3"
else
    par_id="0"
fi

# arxiv, multi_news, multi_x_science_sum, wcep, wikisum
if [ "$#" -ge 4 ]; then
    dataset_name="$4" 
else
    dataset_name="multi_news"
fi

par_id="${dataset_name}_${par_id}"

output_file="primer-${mode}_${par_id}.out"

host_name=$(hostname)
current_time=$(date)
model_path="./run_saves/rank_${mode}_${par_id}/"
echo "$model_path on $host_name at $current_time. " > $output_file


export CUDA_LAUNCH_BLOCKING=1


resume_ckpt="./run_saves/save_path/checkpoints/save_model.ckpt"


CUDA_VISIBLE_DEVICES=${gpu} nohup python rank_primer.py --mode ${mode} --primer_path allenai/PRIMERA --loss_type bpr\
            --model_path ${model_path} --beam_size 5 --batch_size 1 --resume_ckpt ${resume_ckpt} \
            --strategy auto --dataset_name ${dataset_name} >> $output_file 2>&1 &

