import argparse
import os
import json

# SOME CONSTANTS (for now) 
##TODO##
# Make these as command-line variables later
NUM_TRAJ = 50 # number of trajectories for each mutant
SUB_JOBS = False # if true, write one job for each traj; else, write one job for all NUM_TRAJ together

INTERFACE = True
PARTNERS = 'A_B'
METRICS = ['total_score','dG_binding']

TEMP_DIR = './temp'
INPUT_DIR = './inputs'
WORK_DIR = './work'
ROSETTA_DIR = '/gpfs/group/cdm8/default/protein_external/rosetta_bin_linux_2020.08.61146_bundle/main/source/bin'

FLAGS_FILE = 'flags'
FLAGS_SCORE_FILE = 'flags_score'
XML_FILE = 'mutate_relax_minimize.xml'
JOB_FILE = 'job_template.sh'
RELAX_SHELL = 20
RELAX_REPEATS = 5   #Use 5 by default; 1 for debugging

INPUT_PDB = '6lzg_clean.pdb'

# SPECIFIC TO ACI CLUSTER AT PENN STATE
QUEUES =  {'1':'cdm8_b_g_sc_default','2':'cdm8_i_g_bc_default',
           '3':'cdm8_h_g_gc_default','4':'mms7306_b_g_gc_default','5':'open'}

WRITE_COMMANDS = True
# if True, make .sh files of just the commands (.job files are always written irrespective of this)

class Resfile(object):
    """docstring for Resfile"""
    def __init__(self, content,name):
        super(Resfile, self).__init__()
        self.content = content
        self.name = name
        # self.residue = residue
        self.filepath = TEMP_DIR+'/'+self.name+'.resfile'
        self.file = open(self.filepath,'w')
        self.file.write(self.content)
        self.file.close()    
        self.parse()
        
    def __repr__(self):
        return self.filepath

    def parse(self):
        # The idea is to parse the resfile contents to find out which residues are being designed
        lines = self.content.split('start')[-1].split('\n')
        des_residues = []

        for line in lines:
            if line.strip():
                try:
                    wtres, chain, com, mutres = line.strip().split()
                    des_residues.append(wtres.strip()+chain.strip())
                except ValueError:
                    continue

        self.des_residues = des_residues

class SuperResfile(object):
    """docstring for SuperResfile"""
    def __init__(self, filepath):
        super(SuperResfile, self).__init__()
        self.filepath = filepath
        try:
            self.file = open(self.filepath)
        except IOError:
            self.file = None

    def __next__(self):
        resfile = ''
        while True:
            try:
                line = next(self.file)
            except StopIteration:
                raise StopIteration
            # print(line)
            
            if 'RESFILE' in line:
                _, name = line.strip().split()
            
            elif line.strip()=='':
                continue
            
            elif 'STOP' in line:
                break

            else:
                resfile+=line

        return Resfile(resfile,name)
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return

def create_jobs(SUPER_RESFILE,interface_analysis,initialize=True):
    JOBS = []
    JOBS_ifSUB = []
    # Make SuperResfile and make directories
    sfile = SuperResfile(INPUT_DIR+'/'+SUPER_RESFILE)

    MAIN_DIR = os.getcwd()

    for rf in sfile:
        # Each iteration will be a unique simulation corresponding to a unique resfile
        CUR_DIR = WORK_DIR + '/' + rf.name
        if initialize:
            # Create the directory for current iteration 
            os.system('mkdir '+CUR_DIR)
            
            # Copy input files into the new directory
            os.system('cp {0} {1}/{2}.resfile'.format(str(rf),CUR_DIR,rf.name))
            os.system('cp {0}/{1} {2}/{1}'.format(INPUT_DIR,XML_FILE,CUR_DIR))
            os.system('cp {0}/{1} {2}/{1}'.format(INPUT_DIR,INPUT_PDB,CUR_DIR))
        
        # Read template flags files
        f = open('{0}/{1}'.format(INPUT_DIR,FLAGS_FILE))
        flags = f.read()
        f.close()
        f = open('{0}/{1}'.format(INPUT_DIR,FLAGS_SCORE_FILE))
        flags_score = f.read()
        f.close()

        # Write flags based on templates read from above
        f = open('{0}/{1}'.format(CUR_DIR,FLAGS_FILE),'w')
        f.write(flags.format(INPUT_PDB,NUM_TRAJ,rf.name+'.resfile',','.join(rf.des_residues),RELAX_SHELL,RELAX_REPEATS))
        f.close()
        f = open('{0}/{1}'.format(CUR_DIR,FLAGS_SCORE_FILE),'w')
        f.write(flags_score.format(PARTNERS,rf.name))
        f.close()
        default_job_content = open(INPUT_DIR+'/'+JOB_FILE).read()
        
        if interface_analysis:
            main_job = CUR_DIR+'/'+'{0}_interface.job'.format(rf.name)
        else:
            main_job = CUR_DIR+'/'+'{0}.job'.format(rf.name)
            
        f = open(main_job,'w')
        f.write(default_job_content)
        
        os.chdir(os.path.abspath(CUR_DIR))
        
        if WRITE_COMMANDS:
            if interface_analysis:
                fcom = open('{0}_interface.sh'.format(rf.name),'w')
            else:
                fcom = open('{0}.sh'.format(rf.name),'w')
        
        if SUB_JOBS:
            # Now write all the sub jobs
            for i in range(1,NUM_TRAJ+1):
                if interface_analysis:
                    command = '{0}/InterfaceAnalyzer.static.linuxgccrelease -s *.pdb @flags_score\n'.format(ROSETTA_DIR)
                else:
                    command = '{0}/rosetta_scripts.static.linuxgccrelease @flags -nstruct 1 -out:suffix _struct{1}\n'.format(ROSETTA_DIR,i)
                
                if WRITE_COMMANDS:
                    fcom.write(command)
                    
                sub_job = '{0}_struct{1}.job'.format(rf.name,i)
                ff = open(sub_job,'w')
                ff.write(default_job_content)
                ff.write(command)
                ff.close()
                f.write('qsub {0}\n'.format(sub_job))
                
        else:
            if interface_analysis:
                command = '{0}/InterfaceAnalyzer.static.linuxgccrelease -s *00*.pdb @flags_score\n'.format(ROSETTA_DIR)
            else:
                command = '{0}/rosetta_scripts.static.linuxgccrelease @flags\n'.format(ROSETTA_DIR)
            f.write(command)
            if WRITE_COMMANDS:
                fcom.write(command)
                fcom.close()
                
        f.close()
        JOBS.append(main_job)
        os.chdir(MAIN_DIR)
        
    if SUB_JOBS:
        for each in JOBS:
            for i in range(1,NUM_TRAJ+1):
                JOBS_ifSUB.append(each.replace('.job','struct{0}.job'.format(i)))
                print(i)
        #return JOBS_ifSUB
    
    return JOBS

def submit_jobs(jobs_to_submit, subjobs = False, submit=True, run=False):
    # Two modes of running
    # 1. submit to ACI cluster using PBS scripts
    # submit_jobs (bool)
    # 2. Run rosetta executables directly
    # run_commands (bool)
    #
    # jobs_to_submit (list)
    # List of jobs to submit; This list can be generated by create_jobs()
    print(jobs_to_submit)
    
    def _get_job_args():
        wall_time = int(input("Enter wall time in hrs:"))
        memory = int(input("Enter memory in gb:"))
        print('QUEUE')
        print('1. cdm8_b_g_sc_default\n')
        print('2. cdm8_i_g_bc_default\n')   
        print('3. cdm8_h_g_gc_default\n')   
        print('4. mms7306_b_g_gc_default\n')
        print('5. open\n')
        queue = QUEUES[str(input("Choose queue from above:")).strip()]
    
        return wall_time, memory, queue

    def _change_job_content(old_job_lines,wall_time,memory,queue):
        template = "#PBS -l nodes=1:ppn=1\n#PBS -l walltime={0}:00:00\n#PBS -l pmem={1}gb\n#PBS -l mem={1}gb\n#PBS -A {2}\n#PBS -j oe\n"
        new_job_content = template.format(wall_time,memory,queue)
        
        for line in old_job_lines:
            if not line.startswith('#PBS'):
                #print(line)
                new_job_content+=line
        
        #print(new_job_content)
        return new_job_content
    
    wall_time = None
    memory = None
    queue = None
        
    MAIN_DIR = os.getcwd()
    
    for job in jobs_to_submit:
        items = job.split('/')
        jobfile = items[-1]
        jobname = jobfile[:-4]
        jobdir = '/'.join(items[:-1])
        #print(os.getcwd())
        os.chdir(jobdir)
        
        files = os.listdir('.')
        if submit:
            # There should be a main .job file and a bunch of other job files
            for file in files:
                if file.endswith('.job'):
                    print(file)
                    f = open(file)
                    old_job_lines = f.readlines()
                    f.close()
                    if wall_time==None:
                        wall_time, memory, queue = _get_job_args()
                    else:
                        pass
                    new_job_content = _change_job_content(old_job_lines,wall_time,memory,queue)
                    f = open(file,'w')
                    f.write(new_job_content)
                    f.close()
            
            try:
                submitted = False
                for file in files:
                    if ('struct' in file and file.endswith('.job')) and subjobs:
                        #this means there are sub jobs, so ignore the main job and submit these
                        f = open(file)
                        os.system('qsub '+file)  
                        submitted=True
                
                if not submitted:
                    f = open(jobname+'.job')
                    os.system('qsub '+jobname+'.job')   
                    try:
                        status[jobname] = 'Running'
                    except KeyError:
                        print(jobname,' not found in status dictionary.. Something might be wrong!')
                
            except IOError:
                print('Some job file not found for ',job)
                print('Doing nothing..')
                continue
            
        elif run:
            # There should be a main .sh file
            try:
                f = open(jobname+'.sh')
                for line in f:
                    os.system(line.strip())
                
                try:
                    status[jobname] = 'Running'
                except KeyError:
                    print(jobname,' not found in status dictionary.. Something might be wrong!')
                
            except IOError:
                print('Main sh file not found for ',jobname)
                print('Doing nothing..')
                continue
    
        os.chdir(MAIN_DIR)

def main_interactive(interface_analysis,initialize):
    super_resfile = str(input('Enter the name of super resfile:'))
    assert(super_resfile in os.listdir(INPUT_DIR))
    
    jobs = create_jobs(super_resfile, interface_analysis,initialize)
    
    print(len(jobs),'jobs have been created.')
    n = int(input('How many jobs you want to submit in each iteration:'))
    i = 1
    while True:
        jobs_now = jobs[(i-1)*n:i*n]
        jobs_left = jobs[i*n:]
        
        print('*'*20,' Job submission iteration:',i, '*'*20)
        submit = int(input('Submit jobs (Enter 0 or 1):'))
        
        if submit==1:
            submit = True
        elif submit==0:
            submit = False
        else:
            submit = True
        
        run = int(input('Run jobs (Enter 0 or 1):'))
        if run==1:
            run = True
        elif run==0:
            run = False
        else:
            run = True
        
        submit_jobs(jobs_now, subjobs = SUB_JOBS, submit = submit, run = run)
        
        if len(jobs_left)==0:
            break
        else:
            i+=1
if __name__=='__main__':
	# Argument parser
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    p.add_argument("initialize", type=bool,default=True,
                   help="initialize directories in work?")
    p.add_argument("interface_analysis", type=bool, default=False,
                   help="is this an analysis job?")
    
    args = p.parse_args()

    main_interactive(interface_analysis=args.interface_analysis,initialize=args.initialize)
