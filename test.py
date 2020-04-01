entities_base = 'data/project'
def get_torchbiggraph_config():
    config = dict(
        # I/O data
        entity_path=entities_base,
        edge_paths=[],
        checkpoint_path='model/project',
        # Graph structure
        entities={
            'user_id': {'num_partitions': 1},
        },
        relations=[{
            'name': 'follow',
            'lhs': 'user_id',
            'rhs': 'user_id',
            'operator': 'none',
        }],
        # Scoring model
        dimension=1024,
        global_emb=False,
        # Training
        num_epochs=8,
        lr=0.001,
        # Misc
        hogwild_delay=2,
    )
    return config