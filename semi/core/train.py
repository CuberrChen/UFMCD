from semi.ssl import train_semi_base
from semi.ssl import train_UFMCD

ssl_algorithm={
        'BASE': train_semi_base,
        'UFMCD':train_UFMCD
}

def train_ssl(
            cfg,
            model,
            train_dataset,
            label_ratio,
            split_ids,
            ssl_method,
            semi_start_iter,
            val_dataset,
            optimizer,
            save_dir,
            iters,
            batch_size,
            resume_model,
            save_interval,
            log_iters,
            num_workers,
            use_vdl,
            losses,
            keep_checkpoint_max,
            test_config,
            fp16,
            profiler_options,
            to_static_training,case):
    ssl_algorithm[ssl_method](
            cfg,
            model,
            train_dataset,
            label_ratio,
            split_ids,
            semi_start_iter,
            val_dataset,
            optimizer,
            save_dir,
            iters,
            batch_size,
            resume_model,
            save_interval,
            log_iters,
            num_workers,
            use_vdl,
            losses,
            keep_checkpoint_max,
            test_config,
            fp16,
            profiler_options,
            to_static_training,case)



