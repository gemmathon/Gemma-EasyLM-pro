export TPU_NAME='tpu-16'
export TPU_USER='yunjiyeong0106'
export ZONE='us-central2-b'

echo "[local] Killing TPU"
gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME \
--zone $ZONE --worker=all --command "sudo fuser -k /dev/accel0"

echo "[local] Removing TPU Lock"
gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME \
--zone $ZONE --worker=all --command "sudo rm -f /tmp/libtpu_lockfile"

echo "[local] Removing screens"
gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME \
--zone $ZONE --worker=all --command "killall screen"
gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME \
--zone $ZONE --worker=all --command "rm screenlog.0"

echo "[local] Git pull"
gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME --zone $ZONE --worker=all --command \
"cd Gemma-EasyLM-pro && git fetch origin && \
git reset --hard origin/main && rm /home/$TPU_USER/Gemma-EasyLM-pro/train.sh"

echo "[local] Set runner.sh"
# Log per 128 * 50 steps, matching the gradient accumulation steps = Real 1 step
gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME --zone $ZONE --worker=all --command "
cat > /home/$TPU_USER/Gemma-EasyLM-pro/runner.sh << 'EOF'
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

python -m EasyLM.models.gemma.gemma_train \
--load_checkpoint=flax_params::/home/$TPU_USER/flax_model.msgpack \
--mesh_dim=1,-1,4 \
--dtype=bf16 \
--total_steps=80889 \
--log_freq=1 \
--save_model_freq=80889 \
--save_milestone_freq=80889 \
--train_dataset.type='huggingface' \
--train_dataset.text_processor.fields='text' \
--train_dataset.huggingface_dataset.path='gemmathon/merged-pb-kw-nw' \
--train_dataset.huggingface_dataset.name='default' \
--train_dataset.huggingface_dataset.seq_length=4 \
--train_dataset.huggingface_dataset.batch_size=2 \
--train_dataset.huggingface_dataset.streaming=True \
--optimizer.accumulate_gradient_steps=32 \
--optimizer.type=adamw \
--optimizer.adamw_optimizer.weight_decay=0.1 \
--optimizer.adamw_optimizer.lr=0.0002 \
--optimizer.adamw_optimizer.end_lr=0.0002 \
--optimizer.adamw_optimizer.lr_warmup_steps=240 \
--checkpointer.save_optimizer_state=True \
--checkpointer.float_dtype=bf16 \
--logger.online=True \
--logger.output_dir=gs://gemma-pro/gemma-checkpoint \
--logger.project='gemma-pro'
EOF
chmod +x /home/$TPU_USER/Gemma-EasyLM-pro/runner.sh"
#8192 
echo "[local] RUN!!!"

gcloud compute tpus tpu-vm ssh $TPU_USER@$TPU_NAME --zone us-central2-b --worker=all --command \
"screen -L -d -m bash -i -c 'export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=107374182400; \
cd Gemma-EasyLM-pro; /home/$TPU_USER/Gemma-EasyLM-pro/runner.sh'"
