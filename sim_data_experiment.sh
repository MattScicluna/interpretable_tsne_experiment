'''
This code runs the sim data experiments.
It creates the t-SNE embeddings 
Then computes the per cluster aggregations and saves them as .csv tables
'''

#!/bin/bash

# We ran our experiments using Compute Canada. Ignore this if not using SLURM workload manager
#SBATCH --time=6:00:00
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

#84:00:00
echo running job id: $SLURM_JOB_ID

# Paths
source ./set_environment_variables_public.sh

GRAD_STYLE='grad_norm'
SEED=0

# Compute Canada uses modules. See: https://docs.alliancecan.ca/wiki/Utiliser_des_modules/en. Ignore if not using. 
module load python/3.8
# Activate virtual env
source $ENV_LOC/bin/activate
module load python/3.8  # fixes problem with matplotlib

echo 'creating Simulated embeddings and data table ...'

SIM_DSETS=('Sim_2_Clusters_effect=1' 'Sim_4_Clusters_effect1=1_effect2=2' 'Sim_4_Clusters_effect1=2_effect2=5' 'Sim_4_Clusters_effect1=6_effect2=1' 'Sim_4_Clusters_effect1=8_effect2=3' 'Sim_2_Clusters_effect=2' 'Sim_4_Clusters_effect1=1_effect2=3' 'Sim_4_Clusters_effect1=2_effect2=7' 'Sim_4_Clusters_effect1=6_effect2=2' 'Sim_4_Clusters_effect1=8_effect2=5' 'Sim_2_Clusters_effect=3' 'Sim_4_Clusters_effect1=1_effect2=5' 'Sim_4_Clusters_effect1=4_effect2=1' 'Sim_4_Clusters_effect1=6_effect2=3' 'Sim_4_Clusters_effect1=8_effect2=7' 'Sim_2_Clusters_effect=4' 'Sim_4_Clusters_effect1=1_effect2=7' 'Sim_4_Clusters_effect1=4_effect2=2' 'Sim_4_Clusters_effect1=6_effect2=5' 'Sim_2_Clusters_effect=6'             'Sim_4_Clusters_effect1=2_effect2=1' 'Sim_4_Clusters_effect1=4_effect2=3' 'Sim_4_Clusters_effect1=6_effect2=7' 'Sim_2_Clusters_effect=8' 'Sim_4_Clusters_effect1=2_effect2=2' 'Sim_4_Clusters_effect1=4_effect2=5' 'Sim_4_Clusters_effect1=8_effect2=1' 'Sim_4_Clusters_effect1=1_effect2=1' 'Sim_4_Clusters_effect1=2_effect2=3' 'Sim_4_Clusters_effect1=4_effect2=7' 'Sim_4_Clusters_effect1=8_effect2=2')

echo 'making embeddings'
for SIM_DSET in ${SIM_DSETS[*]}
do
bash sim_data_loop_over_seeds.sh $SIM_DSET $TSNE_OUTPUT_DIR_SIM $DATA_DIR_SIM $GRAD_STYLE ;
done

wait

echo 'making data tables'
for SIM_DSET in ${SIM_DSETS[*]}
do
python src/create_sim_data_tables.py --final_csv_path $RESULTS_DIR/$SIM_DSET --tsne_output_dir $TSNE_OUTPUT_DIR_SIM/$SIM_DSET --dset_name $SIM_DSET --seeds 0 1 2 3 4 5 6 7 8 9 --data_dir $DATA_DIR_SIM/$SIM_DSET --grad_style $GRAD_STYLE ;
done

wait

python src/combine_sim_data_tables.py --dsets ${SIM_DSETS[*]} --seeds 1 2 3 4 5 6 7 8 9 --results_dir $RESULTS_DIR --final_csv_path $RESULTS_DIR

echo 'finished making embeddings and data table'
