import pprint
from functools import partial

from tqdm import tqdm, trange
import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState

from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG,
    JaxDistributedConfig,
    next_rng,
    match_partition_rules,
    cross_entropy_loss_and_accuracy,
    global_norm,
    get_float_dtype_by_name,
    set_random_seed,
    average_metrics,
    get_weight_decay_mask,
    make_shard_and_gather_fns,
    with_sharding_constraint,
)
from EasyLM.models.gemma.gemma_model import FlaxGemmaForCausalLMModule
from EasyLM.models.gemma.configuration_gemma import GemmaConfig

from transformers import AutoTokenizer

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim="1,-1,1",
    dtype="bf16",
    total_steps=10000,
    update_gemma_config="",
    load_checkpoint="",
    load_dataset_state="",
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    # tokenizer=GemmaConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    # gemma=GemmaConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
)


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    # tokenizer = GemmaConfig.get_tokenizer(FLAGS.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("gemmathon/gemma-2b-pro")
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != "":
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length

    # if FLAGS.load_gemma_config != "":
    #     gemma_config = GemmaConfig.load_config(FLAGS.load_gemma_config)
    # else:
    #     gemma_config = GemmaConfig(**FLAGS.gemma)
    gemma_config = GemmaConfig.from_pretrained("gemmathon/gemma-2b-pro")

    # if FLAGS.update_gemma_config != "":
    #     gemma_config.update(dict(eval(FLAGS.update_gemma_config)))

    # gemma_config.update(
    #     dict(
    #         bos_token_id=dataset.tokenizer.bos_token_id,
    #         eos_token_id=dataset.tokenizer.eos_token_id,
    #     )
    # )
    # if gemma_config.vocab_size < dataset.vocab_size:
    #     gemma_config.update(dict(vocab_size=dataset.vocab_size))
    def zero_init(params):
        return jax.tree_map(lambda x: jnp.zeros_like(x), params)
    def count_nonzero_parameters(params):
        """0이 아닌 파라미터의 개수를 카운트하는 함수"""
        return sum(jnp.sum(p != 0).astype(int) for p in jax.tree_util.tree_leaves(params))
    model = FlaxGemmaForCausalLMModule(
        gemma_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )
    # 모델 복제
    zero_initialized_model = FlaxGemmaForCausalLMModule(gemma_config, dtype=get_float_dtype_by_name(FLAGS.dtype))
    dummy_input = jnp.ones((1, gemma_config.max_position_embeddings), dtype=jnp.int32)
    rngs = {'params': jax.random.PRNGKey(0)}
    zero_initialized_params = zero_init(zero_initialized_model.init(rngs, dummy_input, attention_mask=dummy_input)['params'])
    
    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(GemmaConfig.get_weight_decay_exclusions()),
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(gemma_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))

        def loss_and_accuracy(params):
            logits = model.apply(
                params,
                batch["input_tokens"],
                deterministic=False,
                rngs=rng_generator(gemma_config.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(
                logits, batch["target_tokens"], batch["loss_masks"]
            )

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)

        # 파라미터를 freeze하는 로직 추가
        new_grads = {}
        for key, grad in grads.items():
            # "model.layers"로 시작하지 않거나 특정 레이어인 경우 업데이트
            if not key.startswith("model.layers") or "model.layers.6." in key or "model.layers.13." in key or "model.layers.20." in key:
                new_grads[key] = grad  # 해당 조건을 만족하는 경우 업데이트
            else:
                new_grads[key] = jnp.zeros_like(grad)  # 그 외는 freeze

        train_state = train_state.apply_gradients(grads=new_grads)
        # 업데이트 되고 있는 파라미터
        updated_zero_initialized_params = zero_initialized_params.copy(update=train_state.params)
        nonzero_params_count = count_nonzero_parameters(updated_zero_initialized_params)
        print(f"Number of nonzero parameters in the trained model: {nonzero_params_count}")
        
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info["learning_rate_schedule"](train_state.step),
            gradient_norm=global_norm(new_grads),
            param_norm=global_norm(train_state.params),
        )

        return train_state, rng_generator(), metrics


    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))
        logits = model.apply(
            train_state.params,
            batch["input_tokens"],
            deterministic=True,
            rngs=rng_generator(gemma_config.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch["target_tokens"], batch["loss_masks"]
        )
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    # print("train_state_shapes:", train_state_shapes)
    train_state_partition = match_partition_rules(
        GemmaConfig.get_partition_rules(), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer,
        logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn, in_shardings=PS(), out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params,),
        out_shardings=train_state_partition,
        donate_argnums=(0,),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            gemma_config=gemma_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    mesh = GemmaConfig.get_jax_mesh(FLAGS.mesh_dim)
    print("Setup Mesh with:", mesh.shape, mesh.size)
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != "":
            print("Loading checkpoint from", FLAGS.load_checkpoint)
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))
        print("Start training from step", start_step)

        # print("Test save_checkpoint")
        # if FLAGS.save_model_freq > 0:
        #     save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch
            )

            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if (
                FLAGS.save_milestone_freq > 0
                and (step + 1) % FLAGS.save_milestone_freq == 0
            ):
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    mlxu.run(main)
