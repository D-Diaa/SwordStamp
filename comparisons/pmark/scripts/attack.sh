samples=64
median="hd"
embedder="path_to_embedder"
model="path_to_opt1.3b"
attack_name="Doc-P(GPT)"
ratio=0.05 # ignored in Doc-P and Doc-T
msig=4
device=0
start=0
end=500
data_name="c4"
log=${data_name}"_logs"

cd ../MarkLLM
CUDA_VISIBLE_DEVICES=$device python attack_pmark.py --start ${start} --end ${end} --sub_dir ${embedder}_${median}${samples}_${msig} --attack_name ${attack_name} --ratio ${ratio} --prompt_attack 1  --log_dir ../PMark/${log}/${model}_log &
wait
cd ../PMark
CUDA_VISIBLE_DEVICES=$device python detect.py --num_samples ${samples} --log_dir ${log}/${model}_log --model_name ../models/${model}  --tokenizer_name ../models/${model} --sub_dir ${embedder}_${median}${samples}_${msig}/attack/${attack_name}${ratio}-1  --pivot rand --start ${start} --end ${end} --median_method ${median}  --embedder_name ../models/$embedder --msig ${msig} &
wait

echo "Attacking finished!"