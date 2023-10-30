import os
import socket
import argparse

def get_work_index():
    os.system('sleep 10s')
    while True:
        try:
            addr = os.environ.get("MASTER_ADDR", "{}")  # .replace('master','worker')
            if addr == "localhost":
                addr = os.environ.get("HOSTNAME", "{}")
            master_addr = socket.gethostbyname(addr)
            # print("MASTER_ADDR: %s", addr)

            world_size = os.environ.get("WORLD_SIZE", "{}")

            rank = os.environ.get("RANK", "{}")
            # logging.info("RANK: %s", rank)
            break
        except:
            print("get 'TF_CONFIG' failed, sleep for 1 second~")
            os.system('sleep 1s')
            continue
    return int(world_size), int(rank), master_addr

def parse_args():
    parser = argparse.ArgumentParser(
        description='BEVDet train a model with multiple machines')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('root', help='root path of BEVDet repos')
    parser.add_argument('--resume', default=None,  help='path to resume training from')
    parser.add_argument('--ngpus', default=None,  help='num of gpus per-node, by default 8')
    parser.add_argument('--group-size', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    resume_from = None
    config = args.config
    os.system('sleep 10s')
    cfg=dict(
        cluster=dict(
            cluster_name='tc',
            num_gpus=8,
        ),
        config_file=config,
        no_validate=True,
        resume_from = resume_from,
    )

    if args.resume is not None:
        cfg['resume_from'] = args.resume
    if args.ngpus is not None:
        cfg['cluster']['num_gpus']=int(args.ngpus)
    if args.group_size > 0:
        cfg['group_size'] = args.group_size
    num_nodes, node_rank, master_addr = get_work_index()

    print('finish get master addr')

    PWD = args.root
    CMD = './tools/dist_train_mpi.sh %s %d %s %d %d '%(PWD+'/'+cfg['config_file'], cfg['cluster']['num_gpus'], master_addr, num_nodes, node_rank)
    cfg_keys = ['resume_from', 'work_dir', 'no_validate', 'load_from', 'group_size']
    for key in cfg_keys:
        key_val = cfg.get(key, None)
        if key_val is None:
            continue
        if key in ['no_validate']:
            CMD += '--' + key.replace('_', '-') + ' '
        else:
            CMD += '--' + key.replace('_', '-') + ' ' + str(cfg[key]) + ' '
    
    # start train
    print(CMD)
    os.system("cd %s && %s"%(PWD, CMD))
