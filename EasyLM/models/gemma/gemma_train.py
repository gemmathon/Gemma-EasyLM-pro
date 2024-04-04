import pprint
from functools import partial

from tqdm import tqdm, trange
import numpy as np
import mlxu
import optax

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
    from flax.core import FrozenDict,frozen_dict
    def create_mask(params,keys, label_fn):
        '''Recursively apply `label_fn` to the key-value pairs of a nested dict.'''
        def _map(params,mask,label_fn):
            for k in keys:
                if label_fn(k):
                    mask['weight'] = "adamw" 
                    mask['kernel'] = "adamw" 
                   # mask['weight'] = "adamw" 
                else:
                    if isinstance(params[k],FrozenDict):
                        mask['weight'] = {}
                        mask['kernel'] = {}
                        mask['embedding'] = {}
                        _map(params[k],mask[k],label_fn)
                    else:
                        mask['weight'] = 'zero'
                        mask['kernel'] = 'zero'
                        mask['embedding'] = 'zero'
            #return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                #       for k, v in nested_dict.items()}
        mask = {}
        _map(params, mask, label_fn)
        return frozen_dict.freeze(mask)
    
    model = FlaxGemmaForCausalLMModule(
        gemma_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(GemmaConfig.get_weight_decay_exclusions()),
    )
    # 해당 layer 제외하고 파라미터 모두 변경한 것 돌리기
    def freeze_mask_back(params, layer_list):
        for k in params['params']['model']['layers'].keys():
            if k not in layer_list:
                name = "default"
                #print(params['params']['model']['layers'][k])
                params['params']['model']['layers'][k]['input_layernorm']['weight'] = params['params']['model']['layers'][k]['input_layernorm'].pop(name)                    
                params['params']['model']['layers'][k]['mlp']['down_proj']['kernel'] = params['params']['model']['layers'][k]['mlp']['down_proj'].pop(name)
                params['params']['model']['layers'][k]['mlp']['gate_proj']['kernel'] = params['params']['model']['layers'][k]['mlp']['gate_proj'].pop(name)
                params['params']['model']['layers'][k]['mlp']['up_proj']['kernel'] = params['params']['model']['layers'][k]['mlp']['up_proj'].pop(name)
                params['params']['model']['layers'][k]['post_attention_layernorm']['weight'] = params['params']['model']['layers'][k]['post_attention_layernorm'].pop(name)
                params['params']['model']['layers'][k]['self_attn']['k_proj']['kernel'] = params['params']['model']['layers'][k]['self_attn']['k_proj'].pop(name)
                params['params']['model']['layers'][k]['self_attn']['o_proj']['kernel'] = params['params']['model']['layers'][k]['self_attn']['o_proj'].pop(name)
                params['params']['model']['layers'][k]['self_attn']['q_proj']['kernel'] = params['params']['model']['layers'][k]['self_attn']['q_proj'].pop(name)
                params['params']['model']['layers'][k]['self_attn']['v_proj']['kernel'] = params['params']['model']['layers'][k]['self_attn']['v_proj'].pop(name)

    def freeze_mask(params,layer_list):
        for k in params['params']['model']['layers'].keys():
            if k not in layer_list:
                name = "default"
                #print(params['params']['model']['layers'][k])
                params['params']['model']['layers'][k]['input_layernorm'][name] = params['params']['model']['layers'][k]['input_layernorm'].pop('weight')
                params['params']['model']['layers'][k]['mlp']['down_proj'][name] = params['params']['model']['layers'][k]['mlp']['down_proj'].pop('kernel')
                params['params']['model']['layers'][k]['mlp']['gate_proj'][name] = params['params']['model']['layers'][k]['mlp']['gate_proj'].pop('kernel')
                params['params']['model']['layers'][k]['mlp']['up_proj'][name] = params['params']['model']['layers'][k]['mlp']['up_proj'].pop('kernel')
                params['params']['model']['layers'][k]['post_attention_layernorm'][name] = params['params']['model']['layers'][k]['post_attention_layernorm'].pop('weight')
                params['params']['model']['layers'][k]['self_attn']['k_proj'][name] = params['params']['model']['layers'][k]['self_attn']['k_proj'].pop('kernel')
                params['params']['model']['layers'][k]['self_attn']['o_proj'][name] = params['params']['model']['layers'][k]['self_attn']['o_proj'].pop('kernel')
                params['params']['model']['layers'][k]['self_attn']['q_proj'][name] = params['params']['model']['layers'][k]['self_attn']['q_proj'].pop('kernel')
                params['params']['model']['layers'][k]['self_attn']['v_proj'][name] = params['params']['model']['layers'][k]['self_attn']['v_proj'].pop('kernel')


    def create_trainstate_from_params(params):
        # transformation
        # condition
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
    
    ############################################################
    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))
        
        optimizer = optax.multi_transform(
            {'adamw': optimizer, 'zero': optax.zero_grads()},
            create_mask(train_state.params['params']['model']['layers'],train_state.params['params']['model']['layers'].keys(), lambda s: s in ['6','13','20'])
        )
        print(train_state.params['params']['model']['layers'].keys())
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
        #print("grads",grads)

        # 특정 layer 제외 모두 값 default 처리
        #freeze_mask(train_state.params,['6','13','20'])
        #freeze_mask(grads,['6','13','20'])

        transforms = {
            'weight': optax.adamw(0.0002),
            'kernel': optax.adamw(0.0002),
            'embedding': optax.set_to_zero(),
            'default': optax.set_to_zero(),
        }

        #label_fn = map_nested_fn(lambda k, _: k)
        #tx = optax.multi_transform(transforms, label_fn)
        
        # 새로운 상태 초기화 및 업데이트 적용
        #state = optimizer.init(train_state.params)
        #updates, state = tx.update(grads, train_state.params)
        #new_params = optax.apply_updates(train_state.params, updates)
        #print(new_params,"new Params")

        # 원상 복구
        #freeze_mask_back(new_params,['6','13','20'])
        #freeze_mask_back(grads,['6','13','20'])
        #print(new_params,"re Params")
        #print(train_state,"train Params")

        #train_state = train_state.replace(params=new_params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info["learning_rate_schedule"](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics
    ############################################################
    

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
