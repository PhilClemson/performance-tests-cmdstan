import argparse
import numpy as np
import subprocess
import multiprocessing
import os
import sys

def shexec_make(command, wd = "."):
    print(command)
    returncode = subprocess.call(command, shell=True, cwd=wd)
    if returncode != 0:
        raise FailedCommand(returncode, command)
    return returncode

def shexec(command, wd = "."):
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    #t = Timer(3600, shkill, [process]) # kill if hour passes
    #t.start()
    process.wait()
    #t.cancel()
    output = process.stdout.read()
    print(output)
    output = process.stderr.read()
    print(output)
    sys.stdout.flush() # added so Jenkins log can catch up
    if process.returncode != 0:
        raise FailedCommand(process.returncode, command)
    return process.returncode

def parse_args():
    parser = argparse.ArgumentParser(description="Run tests and record performance.")
    parser.add_argument("--num-samples", dest="num_samples", nargs='+', action="store", default=None, type=int,
                        help="Number of samples to ask Stan programs for if we're sampling.")
    parser.add_argument("--proposal", dest="proposal", action="store", default="rw",
                        help="Proposal used in SMC.")
    return parser.parse_args()


#main start

args = parse_args()

model_dir = "performance-tests-cmdstan/example-models/applications/covid-19/imperial_model_sub/europe"

# extract mean and sd values from "long run"
summary = np.genfromtxt(model_dir+"_summary.csv", skip_header=1, delimiter=",")

print(summary[:,0])

mean_hmc = summary[:,0]
sd = summary[:,2]

num_proc = multiprocessing.cpu_count()

proposal = args.proposal

num_samples = args.num_samples

L=len(num_samples)

comp_eq=num_samples[0]*num_samples[L-1]

# compile model
shexec_make("make -i -j{} {}".format(num_proc, model_dir), wd = ".")

num_proc = num_proc - (num_proc % 2)

for n in range(0,L):
	try:
		shexec("mpirun -np {} {} method=sample algorithm=smcs proposal={} stepsize=0.00952820175 T=1 Tsmc={} num_samples={} {} random seed=1234 output file=output_smc.out".format(num_proc, model_dir, proposal,num_samples[n],comp_eq/num_samples[n], "data file="+model_dir+"_sub.data.R"))
		samps = np.loadtxt("output_smc.out", comments=["#"], delimiter=",", unpack=False)
		mean_smc = samps[num_samples[n]-2,] # temporary fix while seg fault on writing samples is investigated
		error = (mean_smc - mean_hmc) / sd
		sys.stdout.flush() # added so Jenkins log can catch up
		print("mean_smc = {}\n error = {}".format(mean_smc,error))
		sys.stdout.flush() # added so Jenkins log can catch up
		os.remove("output_smc.out")
	except:
		print("ERROR")

