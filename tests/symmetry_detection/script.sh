MCB_DATASET_DIR=../../assets/MCB python config.py --problem_list=mcb_dgcnn --method_list=pd_hlr --train_step=5000 --restart
MCB_DATASET_DIR=../../assets/MCB python config.py --problem_list=mcb_dgcnn --method_list=gol_single_res_hlr --instance_batch_size=8 --train_step=100000 --restart
MCB_DATASET_DIR=../../assets/MCB python config.py --problem_list=mcb_dgcnn --method_list=pol_res_mot --instance_batch_size=8 --train_step=100000 --restart
