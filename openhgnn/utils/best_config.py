BEST_CONFIGS = {
    "node_classification": {
        'CompGCN': {
            'general': {
                'seed': 0, 'patience': 50, 'validation': True
            },
            'aifb': {
                'lr': 0.01, 'weight_decay': 0.0001, 'max_epoch': 100, 'in_dim': 32,
                'hidden_dim': 32, 'n_layers': 2, 'dropout': 0.2, 'comp_fn': 'sub', 'batch_size': 128,
                'mini_batch_flag': False, 'validation': True
                     },
            'mutag': {
                'lr': 0.01, 'weight_decay': 0.0001, 'max_epoch': 100, 'in_dim': 32,
                'hidden_dim': 32, 'n_layers': 2, 'dropout': 0.2, 'comp_fn': 'sub',
                'mini_batch_flag': False, 'validation': True
            },
            'bgs': {

            },
            'am': {
            },
        },
        'GTN': {
            'general': {'lr': 0.005, 'weight_decay': 0.001, 'hidden_dim': 64, 'max_epoch': 50, 'patience': 10,
                        'norm_emd_flag': True, 'mini_batch_flag': False},
            'acm4GTN': {
                'num_layers': 2, 'num_channels': 2, 'adaptive_lr_flag': True,
            },
            'imdb4GTN': {
                'num_layers': 2, 'num_channels': 2, 'adaptive_lr_flag': True,
            },
            'dblp4GTN': {
                'num_layers': 3, 'num_channels': 2, 'adaptive_lr_flag': False,
            },
        },
        'MHNF': {
            'general': {'lr': 0.005, 'weight_decay': 0.001, 'hidden_dim': 64, 'max_epoch': 50, 'patience': 10,
                        'norm_emd_flag': True, 'mini_batch_flag': False},
            'acm4GTN': {
                'num_layers': 2, 'num_channels': 2, 'adaptive_lr_flag': True,
            },
            'imdb4GTN': {
                'num_layers': 2, 'num_channels': 2, 'adaptive_lr_flag': True,
            },
            'dblp4GTN': {
                'num_layers': 3, 'num_channels': 2, 'adaptive_lr_flag': False,
            },
        },
        'HAN': {
            'general': {},
            'acm_han_raw': {
                'lr': 0.005, 'num_heads': [8], 'hidden_dim': 128, 'dropout': 0.6, 'weight_decay': 0.001,
                'num_epochs': 200, 'patience': 100
            },
            'acm4GTN': {
                'lr': 0.001, 'weight_decay': 0.00001, 'num_heads': [8, 8],
            },
            'imdb4GTN': {
                'lr': 0.001, 'weight_decay': 0.00001, 'num_heads': [2, 2, 2]
            },
            'HGBn-ACM': {
            },
            'HGBn-DBLP': {
            }

        },
        'NARS': {
            'general': {},
            'HGBn-ACM': {
                'lr': 0.001, 'num_heads': [8], 'num_hops': 2, 'hidden_dim': 64, 'dropout': 0.5, 'weight_decay': 0.001,
                'num_epochs': 200, 'patience': 100, 'metric': 'acc',
            },
            'HGBn-DBLP': {
                'lr': 0.001, 'num_heads': [8], 'num_hops': 2, 'hidden_dim': 64, 'dropout': 0.5, 'weight_decay': 0.001,
                'num_epochs': 200, 'patience': 100, 'metric': 'acc',
            },
            'HGBn-Freebase': {
                'lr': 0.001, 'num_heads': [8], 'num_hops': 2, 'hidden_dim': 64, 'dropout': 0.5, 'weight_decay': 0.001,
                'num_epochs': 200, 'patience': 100, 'metric': 'acc',
            },
            'HGBn-IMDB': {
                'lr': 0.001, 'num_heads': [8], 'num_hops': 2, 'hidden_dim': 64, 'dropout': 0.5, 'weight_decay': 0.001,
                'num_epochs': 200, 'patience': 100, 'metric': 'acc',
            },


        },
        'HetGNN': {
            'general': {'max_epoch': 500, 'patience': 20, 'mini_batch_flag': True},
            'academic4HetGNN': {
                'lr': 0.001, 'weight_decay': 0.00001, 'dim': 128, 'batch_size': 64, 'window_size': 5,
                'batches_per_epoch': 50, 'rw_length': 50, 'rw_walks': 10, 'rwr_prob': 0.5,
            }
        },
        'MAGNN': {
            'general': {'mini_batch_flag': False, 'max_epoch': 500, 'patience': 30},
            'imdb': {
                'lr': 0.005, 'weight_decay': 0.001, 'dropout': 0.5, 'hidden_dim': 64, 'encoder_type': 'RotateE',
                'inter_attn_feats': 128, 'num_heads': 8, 'num_layers': 2,
            },
        },
        'NSHE': {
            'general': {},
            'acm4SNHE': {'weight_decay': 0.001, 'num_e_neg': 1, 'num_ns_neg': 4,
                         'max_epoch': 500, 'patience': 10,
                         }
        },
        'RGCN': {
            'general': {
                'seed': 0, 'patience': 20, 'validation': True
            },
            'aifb': {
                'lr': 0.01, 'weight_decay': 0, 'max_epoch': 100, 'in_dim': 16,
                'hidden_dim': 16, 'n_bases': -1, 'n_layers': 2, 'batch_size': 126, 'dropout': 0,
                'mini_batch_flag': False, 'validation': False
                     },
            'mutag': {
                'lr': 0.01, 'weight_decay': 0.0005, 'max_epoch': 50, 'in_dim': 8,
                'hidden_dim': 16, 'n_bases': 30, 'n_layers': 2, 'batch_size': 50, 'fanout': 4, 'dropout': 0.05,
                'mini_batch_flag': True, 'validation': False
            },
            'bgs': {
                'lr': 0.005, 'weight_decay': 0.0005, 'max_epoch': 50, 'in_dim': 32,
                'hidden_dim': 16, 'n_bases': 40, 'n_layers': 3, 'batch_size': 126, 'fanout': 4, 'dropout': 0.1,
                'mini_batch_flag': False, 'validation': False
            },
            'am': {
                'lr': 0.01, 'weight_decay': 0.0005, 'max_epoch': 100, 'in_dim': 16,
                'hidden_dim': 16, 'n_bases': 40, 'n_layers': 2, 'batch_size': 126, 'fanout': 4, 'dropout': 0,
                'mini_batch_flag': True, 'validation': False
            },
            'ogbn-mag': {
                'lr': 0.01, 'weight_decay': 0.0005, 'max_epoch': 10,
                'hidden_dim': 16, 'n_bases': -1, 'n_layers': 2, 'batch_size': 126, 'fanout': 4, 'dropout': 0,
                'mini_batch_flag': True, 'validation': True
            },
        },
        'RSHN': {
            'general': {},
            'aifb': {
                'in_dim': 16, 'hidden_dim': 16, 'num_node_layer': 2,
                'num_edge_layer': 2, 'rw_len': 4, 'batch_size': 5000,
            },
            'mutag': {
                'hidden_dim': 16, 'num_node_layer': 2, 'num_edge_layer': 2, 'rw_len': 4, 'batch_size': 5000,
                'dropout': 0.2,
            },
            'bgs': {
                'hidden_dim': 16, 'num_node_layer': 2, 'num_edge_layer': 2, 'rw_len': 4, 'batch_size': 5000
            },
            'am': {'hidden_dim': 16, 'num_node_layer': 2, 'num_edge_layer': 2, 'rw_len': 4, 'batch_size': 5000
                   },
        }

    },
    "entity_classification": {
        "deepwalk": {
            "general": {},
        },
    },
    "link_prediction": {
        'general': {'max_epoch': 500, 'patience': 10, 'mini_batch_flag': True},
        'academic4HetGNN': {
            'lr': 0.01, 'weight_decay': 0.0001, 'dim': 128, 'batch_size': 64, 'window_size': 5,
            'batches_per_epoch': 50, 'rw_length': 50, 'rw_walks': 10, 'rwr_prob': 0.5,
        }
    }
}
