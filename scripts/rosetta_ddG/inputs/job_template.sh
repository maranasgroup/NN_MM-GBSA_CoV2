#PBS -l nodes=1:ppn=1

#PBS -l walltime=10:00:00

#PBS -l pmem=1gb

#PBS -l mem=1gb

#PBS -A cdm8_b_g_sc_default
#PBS -j oe

set -u

cd $PBS_O_WORKDIR

echo " "

echo " "

echo "JOB Started on $(hostname -s) at $(date)"


