# train test few_shot
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


# "google/pegasus-large", 
# "facebook/bart-large", "allenai/led-large-16384",
# "primer"
if [ "$#" -ge 5 ]; then
    model_name="$5"
else
    model_name='primer'
fi

model_name_par_id=$(basename "$model_name")

par_id="${dataset_name}_${model_name_par_id}_${par_id}"


host_name=$(hostname)
current_time=$(date)



# steps to add:
## 1. add it on function __getitem__ of SummarizationDataset at dataloader.py 
## 2. add join_method to primer_summarizer_module.py
# join_method:
## concat_start_wdoc_global (baseline used)
## original
## original_ranking_filtering
## truncate_last_ranking_filtering
if [ $mode = "test" ]; then
    resume_ckpt="./run_saves/saved_model.ckpt"
    model_path="./run_saves/save_path_${mode}_${par_id}/"
    output_file="primer_${mode}_${par_id}.out"
    echo "$model_path on $host_name at $current_time. " > $output_file

    CUDA_VISIBLE_DEVICES=${gpu} nohup python primer_hf_main_modify.py --mode ${mode} \
                --model_path ${model_path} --beam_size 5 --batch_size 16 --strategy auto \
                --model_name ${model_name} --join_method truncate_last_ranking_filtering \
                --dataset_name ${dataset_name} --filter_score 0.0 --max_length_tgt 1024 \
                --resume_ckpt ${resume_ckpt} >> $output_file 2>&1 &
elif [ $mode = "train" ]; then
    model_path="./run_saves/save_path_${mode}_${par_id}/"
    output_file="primer-${mode}_${par_id}.out"
    echo "$model_path on $host_name at $current_time. " > $output_file

    if [ $model_name = "primer" ]; then
        CUDA_VISIBLE_DEVICES=${gpu} nohup python primer_hf_main_modify.py --mode ${mode} \
                    --model_path ${model_path} --beam_size 5 --batch_size 16 --strategy auto \
                    --model_name ${model_name} --join_method truncate_last_ranking_filtering \
                    --dataset_name ${dataset_name} --filter_score 0.0 --max_length_tgt 1024 >> $output_file 2>&1 &
    else
        CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${gpu} nohup python compared_model_main_modify.py --mode ${mode} \
                    --model_path ${model_path} --beam_size 5 --batch_size 2 --strategy auto \
                    --model_name ${model_name} --join_method truncate_last_ranking_filtering \
                    --dataset_name ${dataset_name} --filter_score 0.0 --no_doc_sep True >> $output_file 2>&1 &
    fi

elif [ $mode = "few_shot" ]; then
    model_path="./run_saves/save_path_${mode}_${par_id}/"
    output_file="primer-${mode}_${par_id}.out"
    echo "$model_path on $host_name at $current_time. " > $output_file

    if [ $model_name = "primer" ]; then
        CUDA_VISIBLE_DEVICES=${gpu} nohup python primer_hf_main_modify.py --mode train \
                    --model_path ${model_path} --beam_size 5 --batch_size 8 --strategy auto \
                    --model_name ${model_name} --join_method truncate_last_ranking_filtering \
                    --dataset_name ${dataset_name} --filter_score 0.0 --ratio_train_data 0.01 >> $output_file 2>&1 &
    else
        CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${gpu} nohup python compared_model_main_modify.py --mode train \
                    --model_path ${model_path} --beam_size 5 --batch_size 4 --strategy auto \
                    --model_name ${model_name} --join_method truncate_last_ranking_filtering \
                    --dataset_name ${dataset_name} --filter_score 0.0 --no_doc_sep True --ratio_train_data 0.01 \
                    >> $output_file 2>&1 &
    fi
else
    echo "wrong mode ${mode} inputed. "
fi

