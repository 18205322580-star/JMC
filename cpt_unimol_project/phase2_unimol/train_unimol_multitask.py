from pathlib import Path
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from unimol_tools import MolTrain
from unimol_tools.tasks.trainer import Trainer as UniMolTrainer, EarlyStopper
from unimol_tools.tasks import trainer as trainer_module
from unimol_tools.models.nnmodel import NNModel


_orig_trainer_init = UniMolTrainer.__init__


def _patched_trainer_init(self, *args, **kwargs):
    _orig_trainer_init(self, *args, **kwargs)
    if not hasattr(self, 'scaler'):
        self.scaler = None


UniMolTrainer.__init__ = _patched_trainer_init


_orig_init_optimizer_scheduler = UniMolTrainer._initialize_optimizer_scheduler


def _patched_init_optimizer_scheduler(self, model, train_dataloader):
    num_training_steps = len(train_dataloader) * self.max_epochs
    num_warmup_steps = int(num_training_steps * self.warmup_ratio)
    wd = float(getattr(self, 'weight_decay', 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, eps=1e-6, weight_decay=wd)
    scheduler = trainer_module.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )
    return optimizer, scheduler


UniMolTrainer._initialize_optimizer_scheduler = _patched_init_optimizer_scheduler


_orig_early_stop_choice = EarlyStopper.early_stop_choice


def _patched_early_stop_choice(self, model, epoch, loss, metric_score=None):
    if not isinstance(self.metrics_str, str) or self.metrics_str in ['loss', 'none', '']:
        return self._judge_early_stop_loss(loss, model, epoch)

    if not hasattr(self, 'max_score'):
        self.max_score = float('-inf')

    is_early_stop, self.min_loss, self.wait, self.max_score = self.metrics._early_stop_choice(
        self.wait,
        self.min_loss,
        metric_score,
        self.max_score,
        model,
        self.dump_dir,
        self.fold,
        self.patience,
        epoch,
    )
    self.is_early_stop = is_early_stop
    return self.is_early_stop


EarlyStopper.early_stop_choice = _patched_early_stop_choice


_orig_nn_init_model = NNModel._init_model


class DeepMLPHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, out_dim=2, dropout=0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _patched_nn_init_model(self, model_name, **params):
    model = _orig_nn_init_model(self, model_name, **params)
    if params.get('use_deep_head', False):
        head_dropout = float(params.get('head_dropout', 0.08))
        head_hidden_dim = int(params.get('head_hidden_dim', 256))
        input_dim = int(getattr(model.args, 'encoder_embed_dim', 512))
        out_dim = int(getattr(model, 'output_dim', 2))
        model.classification_head = DeepMLPHead(
            input_dim=input_dim,
            hidden_dim=head_hidden_dim,
            out_dim=out_dim,
            dropout=head_dropout,
        )

    custom_dropout = params.get('custom_dropout', None)
    if custom_dropout is not None:
        p = float(custom_dropout)
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = p
    return model


NNModel._init_model = _patched_nn_init_model

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'cpt_unimol_project' / 'data'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--early_stopping', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--dropout', type=float, default=0.08)
    parser.add_argument('--conformer_aug', type=int, default=1)
    parser.add_argument('--freeze_strategy', type=str, default='first_n', choices=['head_only', 'first_n', 'none'])
    parser.add_argument('--freeze_n_layers', type=int, default=8)
    parser.add_argument('--max_rows', type=int, default=0)
    parser.add_argument('--split', type=str, default='scaffold', choices=['scaffold', 'random'])
    parser.add_argument('--kfold', type=int, default=1)
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_cuda', type=int, default=-1, choices=[-1, 0, 1])
    parser.add_argument('--max_atoms', type=int, default=120)
    parser.add_argument('--run_name', type=str, default='artifacts_unimol_opt')
    args = parser.parse_args()

    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(int(args.num_threads))

    out_dir = ROOT / 'cpt_unimol_project' / 'phase2_unimol' / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    src = DATA_DIR / 'master_multitask.csv'
    if not src.exists():
        raise FileNotFoundError(f'{src} not found. Run phase1 first.')

    df = pd.read_csv(src, encoding='utf-8-sig')
    df = df.rename(columns={'smiles': 'SMILES'})[['SMILES', 'hepg2_pIC50', 'hct116_pIC50']]
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    # 构象增强关闭，极致加速
    # if args.conformer_aug > 1:
    #     df = pd.concat([df.copy() for _ in range(args.conformer_aug)], ignore_index=True)
    #     df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    if args.freeze_strategy == 'head_only':
        freeze_layers = 'embed_tokens,encoder,gbf,gbf_proj'
    elif args.freeze_strategy == 'first_n':
        layer_prefix = [f'encoder.layers.{i}' for i in range(max(0, int(args.freeze_n_layers)))]
        freeze_layers = ','.join(['embed_tokens', 'gbf', 'gbf_proj'] + layer_prefix)
    else:
        freeze_layers = None

    # 优先加载3D特征缓存
    import pickle
    feat_path = DATA_DIR / 'conformer_features.pkl'
    if feat_path.exists():
        with open(feat_path, 'rb') as f:
            features = pickle.load(f)
        # 构建分子到特征映射，仅保留有效3D特征
        feat_map = {
            str(item.get('smiles')): item
            for item in features
            if item.get('coords') is not None and item.get('atoms') is not None
        }
        coordinates_list = []
        atoms_list = []
        for smiles in df['SMILES']:
            item = feat_map.get(str(smiles))
            if item is not None:
                coordinates_list.append(item['coords'])
                atoms_list.append(item['atoms'])
            else:
                coordinates_list.append(None)
                atoms_list.append(None)

        valid_cached = sum(c is not None for c in coordinates_list)
        if valid_cached == 0:
            raise RuntimeError('conformer_features.pkl found but no valid cached 3D features are usable.')

        if valid_cached < len(df):
            # Strict cache-only mode: keep only rows with cached conformers to avoid on-the-fly generation.
            keep_idx = [i for i, c in enumerate(coordinates_list) if c is not None]
            dropped = len(df) - len(keep_idx)
            df = df.iloc[keep_idx].reset_index(drop=True)
            coordinates_list = [coordinates_list[i] for i in keep_idx]
            atoms_list = [atoms_list[i] for i in keep_idx]
            print(
                f'Cached 3D features incomplete ({valid_cached}/{len(keep_idx)+dropped}). '
                f'Using cache-only training and dropping {dropped} rows without cached conformers.'
            )
        else:
            print(f'Loaded cached 3D features for {valid_cached}/{len(df)} molecules. Skip conformer generation.')

        # Drop very large molecules that can trigger heavy distance-matrix allocations on CPU.
        if args.max_atoms and args.max_atoms > 0:
            keep_idx = [i for i, atoms in enumerate(atoms_list) if atoms is not None and len(atoms) <= int(args.max_atoms)]
            dropped_large = len(df) - len(keep_idx)
            if dropped_large > 0:
                df = df.iloc[keep_idx].reset_index(drop=True)
                coordinates_list = [coordinates_list[i] for i in keep_idx]
                atoms_list = [atoms_list[i] for i in keep_idx]
                print(f'Dropped {dropped_large} molecules with atom_count > {int(args.max_atoms)} for stability/speed.')

        data_dict = {
            'SMILES': df['SMILES'].tolist(),
            'hepg2_pIC50': df['hepg2_pIC50'].tolist(),
            'hct116_pIC50': df['hct116_pIC50'].tolist(),
            'atoms': atoms_list,
            'coordinates': coordinates_list,
        }
    else:
        data_dict = {
            'SMILES': df['SMILES'].tolist(),
            'hepg2_pIC50': df['hepg2_pIC50'].tolist(),
            'hct116_pIC50': df['hct116_pIC50'].tolist(),
        }
        print('No conformer_features.pkl found, will generate 3D conformers on the fly.')

    # 分层学习率
    def get_param_groups(model):
        backbone_params = []
        head_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'classification_head' in n:
                head_params.append(p)
            else:
                backbone_params.append(p)
        groups = []
        if backbone_params:
            groups.append({'params': backbone_params, 'lr': 2e-5})
        if head_params:
            groups.append({'params': head_params, 'lr': 5e-4})
        return groups

    auto_cuda = torch.cuda.is_available()
    if args.use_cuda == 1:
        use_cuda = True
    elif args.use_cuda == 0:
        use_cuda = False
    else:
        use_cuda = auto_cuda
    use_amp = bool(use_cuda)

    def _patched_nndataloader(
        feature_name=None,
        dataset=None,
        batch_size=None,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
        distributed=False,
        valid_mode=False,
    ):
        if distributed:
            sampler = trainer_module.DistributedSampler(dataset, shuffle=shuffle)
            g = trainer_module.get_ddp_generator()
        else:
            sampler = None
            g = None
        if valid_mode:
            g = None

        # CPU path: disable pin_memory and use a modest worker count to reduce data-loading overhead.
        if use_cuda:
            pin_memory = True
            num_workers = 0
        else:
            pin_memory = False
            num_workers = max(0, int(args.num_workers))

        loader_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'drop_last': drop_last,
            'pin_memory': pin_memory,
            'sampler': sampler,
            'generator': g,
            'num_workers': num_workers,
        }
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = True
        return trainer_module.TorchDataLoader(**loader_kwargs)

    trainer_module.NNDataLoader = _patched_nndataloader

    # 训练主入口
    trainer = MolTrain(
        task='multilabel_regression',
        data_type='molecule',
        model_name='unimolv1',
        model_size='84m',
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=2e-5,
        early_stopping=args.early_stopping,
        metrics='mae',
        split=args.split,
        kfold=max(1, int(args.kfold)),
        smiles_col='SMILES',
        target_cols=['hepg2_pIC50', 'hct116_pIC50'],
        save_path=str(out_dir),
        remove_hs=False,
        use_cuda=use_cuda,
        use_amp=use_amp,
        freeze_layers=freeze_layers,
        use_deep_head=True,
        head_dropout=args.dropout,
        head_hidden_dim=256,
        weight_decay=args.weight_decay,
    )
    # 替换优化器
    def patched_optimizer(self, model, train_dataloader):
        param_groups = get_param_groups(model)
        optimizer = torch.optim.AdamW(param_groups, eps=1e-6, weight_decay=args.weight_decay)
        num_training_steps = len(train_dataloader) * self.max_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        scheduler = trainer_module.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
        return optimizer, scheduler
    UniMolTrainer._initialize_optimizer_scheduler = patched_optimizer

    model = trainer.fit(data_dict)
    print('training_done', bool(model))
    print('saved_dir', out_dir)
    print('settings', {
        'use_cuda': use_cuda,
        'use_amp': use_amp,
        'torch_num_threads': torch.get_num_threads(),
        'num_workers': int(args.num_workers),
        'max_atoms': int(args.max_atoms),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'early_stopping': args.early_stopping,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'conformer_aug': args.conformer_aug,
        'freeze_strategy': args.freeze_strategy,
        'freeze_n_layers': args.freeze_n_layers,
        'split': args.split,
        'kfold': max(1, int(args.kfold)),
        'freeze_layers': freeze_layers,
        'n_rows_after_aug': len(df),
        'head_type': 'DeepMLPHead',
        'lr_backbone': 2e-5,
        'lr_head': 5e-4,
    })


if __name__ == '__main__':
    main()
