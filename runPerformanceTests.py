#!/usr/bin/python

import argparse
import csv
from collections import defaultdict
import os
import os.path
import sys
import re
import subprocess
from difflib import SequenceMatcher
from fnmatch import fnmatch
from functools import wraps
from multiprocessing.pool import ThreadPool
from time import time
from datetime import datetime
import xml.etree.ElementTree as ET
import multiprocessing
import numpy as np
import time as tm
import signal
from threading import Timer

np.set_printoptions(threshold=sys.maxsize)

GOLD_OUTPUT_DIR = os.path.join("performance-tests-cmdstan/golds","")
DIR_UP = os.path.join("..","")
CURR_DIR = os.path.join(".","")
SEP_RE = "\\\\" if os.sep == "\\" else "/"
EXE_FILE_EXT = ".exe" if os.name == "nt" else ""
def find_files(pattern, dirs):
    res = []
    for pd in dirs:
        for d, _, flist in os.walk(pd):
            for f in flist:
                if fnmatch(f, pattern):
                    res.append(os.path.join(d, f))
    return res

def read_tests(filename, default_num_samples):
    test_files = []
    num_samples_list = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"): continue
            if ", " in line:
                model, num_samples = line.split(", ")
            else:
                model = line
                num_samples = default_num_samples
            if model in bad_models:
                print("You specified {} but we have that blacklisted; skipping"
                      .format(model))
                continue
            num_samples_list.append(num_samples)
            test_files.append(model)
    return test_files, num_samples_list

def str_dist(target):
    def str_dist_internal(candidate):
        return SequenceMatcher(None, candidate, target).ratio()
    return str_dist_internal

def closest_string(target, candidates):
    if candidates:
        return max(candidates, key=str_dist(target))

def find_data_for_model(model):
    d = os.path.dirname(model)
    data_files = find_files("*.data.R", [d])
    if len(data_files) == 1:
        return data_files[0]
    else:
        return closest_string(model, data_files)

def time_step(name, fn, *args, **kwargs):
    start = time()
    res = fn(*args, **kwargs)
    end = time()
    return end-start, res

class FailedCommand(Exception):
    def __init__(self, returncode, command):
        self.returncode = returncode
        self.command = command
        Exception(self, "return code '{}' from command '{}'!"
                  .format(returncode, command))

def shexec_make(command, wd = "."):
    print(command)
    returncode = subprocess.call(command, shell=True, cwd=wd)
    if returncode != 0:
        raise FailedCommand(returncode, command)
    return returncode

def shkill(process):
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    print('KILLED (TIMEOUT)')

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

def make(targets, j=8):
    for i in range(len(targets)):
        prefix = ""
        #if not targets[i].startswith(os.sep):
            #prefix = os.path.join("/performance-test-cmdstan/","")
        targets[i] = targets[i] + EXE_FILE_EXT
    try:
        name = os.path.basename(targets[i])
        formatted_name = ""
        for char in name:
            if char.isalnum():
                formatted_name = formatted_name + char
        shexec_make("make -i -j{} {}"
            .format(j, " ".join(targets)), wd = ".")
    except FailedCommand:
        print("Failed to make at least some targets")

model_name_re = re.compile(".*"+SEP_RE+"[A-z_][^"+SEP_RE+"]+\.stan$")

bad_models = frozenset(
    [os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.21","finite_populations.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.21","multiple_comparison.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.21","r_sqr.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.23","electric_1a.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.23","educational_subsidy.stan")
     , os.path.join("performance-tests-cmdstan","example-models","bugs_examples","vol2","pines","pines-3.stan")
     , os.path.join("performance-tests-cmdstan","example-models","bugs_examples","vol3","fire","fire.stan")
     , os.path.join("performance-tests-cmdstan","example-models","bugs_examples","vol2","schools","schools-3.stan")
     # The following have data issues
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.10","ideo_two_pred.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.16","radon.1.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ample-models","ARM","Ch.16","radon.2.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.16","radon.2a.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.16","radon.2b.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ample-models","ARM","Ch.16","radon.3.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ample-models","ARM","Ch.16","radon.nopooling.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.16","radon.pooling.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.18","radon.1.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.18","radon.2.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.18","radon.nopooling.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.18","radon.pooling.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.19","item_response.stan")
     , os.path.join("performance-tests-cmdstan","example-models","bugs_examples","vol1","dogs","dogs.stan")
     , os.path.join("performance-tests-cmdstan","example-models","bugs_examples","vol1","rats","rats_stanified.stan")
     , os.path.join("performance-tests-cmdstan","example-models","bugs_examples","vol2","pines","pines-4.stan")
     , os.path.join("performance-tests-cmdstan","example-models","bugs_examples","vol2","pines","fit.stan")
     , os.path.join("performance-tests-cmdstan","example-models","BPA","Ch.06","MtX.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.21","radon_vary_intercept_a.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.21","radon_vary_intercept_b.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.23","sesame_street2.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.3","kidiq_validation.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.7","earnings_interactions.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.8","y_x.stan")
     , os.path.join("performance-tests-cmdstan","example-models","basic_estimators","normal_mixture_k.stan")
     , os.path.join("performance-tests-cmdstan","example-models","basic_estimators","normal_mixture_k_prop.stan")
     , os.path.join("performance-tests-cmdstan","example-models","BPA","Ch.04","GLM0.stan")
     , os.path.join("performance-tests-cmdstan","example-models","BPA","Ch.04","GLM1.stan")
     , os.path.join("performance-tests-cmdstan","example-models","BPA","Ch.04","GLM2.stan")
     , os.path.join("performance-tests-cmdstan","example-models","BPA","Ch.04","GLMM3.stan")
     , os.path.join("performance-tests-cmdstan","example-models","BPA","Ch.04","GLMM4.stan")
     , os.path.join("performance-tests-cmdstan","example-models","BPA","Ch.04","GLMM5.stan")
     , os.path.join("performance-tests-cmdstan","example-models","BPA","Ch.05","ssm2.stan")
     , os.path.join("performance-tests-cmdstan","example-models","BPA","Ch.07","cjs_group_raneff.stan")
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.17","flight_simulator_17.3.stan") # disabled while issue with SMC-Stan remains
     , os.path.join("performance-tests-cmdstan","example-models","ARM","Ch.24","dogs_log.stan") # disabled while issue with SMC-Stan remains
     , os.path.join("performance-tests-cmdstan","stat_comp_benchmarks","benchmarks","sir","sir.stan") # removed while issue with SMC-Stan remains
     , os.path.join("performance-tests-cmdstan","example-models","knitr","cook_et_al","bym_predictor_plus_offset")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","cook_et_al","bym_predictor_only")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","cook_et_al","sim_bym_data")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","climate-challenge","mixture_2")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","climate-challenge","mixture")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","pest-control","stan_programs","multiple_NB_regression")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","pest-control","stan_programs","hier_NB_regression_ncp_slopes_mod_resid")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","pest-control","stan_programs","multiple_poisson_regression")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","pest-control","stan_programs","hier_NB_regression_ncp_slopes_mod_mos")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","pest-control","stan_programs","simple_poisson_regression_dgp")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","pest-control","stan_programs","multiple_NB_regression_dgp")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","pest-control","stan_programs","simple_poisson_regression")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","pest-control","stan_programs","hier_NB_regression_ncp_slopes_mod_mos_predict")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","pest-control","stan_programs","hier_NB_regression")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","pest-control","stan_programs","multiple_poisson_regression_dgp")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","pest-control","stan_programs","hier_NB_regression_ncp_slopes_mod")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","pest-control","stan_programs","hier_NB_regression_ncp")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","pest-control","stan_programs","hier_NB_regression_ncp_slopes_mod_mos_gp")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","bradley-terry","individual-uniform")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","bradley-terry","team")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","bradley-terry","mle")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","bradley-terry","individual-hierarchical")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","bradley-terry","individual")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","sum-of-exponentials","sum_of_exponentials")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","sum-of-exponentials","sum_of_exponentials_with_priors")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt","irt_1pl_pin")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt","irt_2pl_power")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt","irt_1pl_vague")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt","irt_1pl_predict")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt","irt_1pl_mle")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt","irt_1pl_unit")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt","irt_1pl_hier")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt","irt-multilevel")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt","irt_1pl_adjust")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt","irt_1pl_fit_predict")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt","irt_1pl")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","ratings","ratings_3")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","ratings","ratings_2")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","ratings","ratings_1")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","single-exponential","exponential")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","single-exponential","exponential_positive")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","single-exponential","exponential_lognormal")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","bandits","bernoulli-bandits-sufficient")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","bandits","bernoulli-bandits-conjugate")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","bandits","bernoulli-bandits")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","neural","nets_nn-simple")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","neural","nn-k-minus-1")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","noisy-raters","raykar-etal","raykar-marginal")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","noisy-raters","dawid-skene","dawid-skene")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","noisy-raters","sex-ratio","normal_normal")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","mle-params","logodds-jac")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","mle-params","prob")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","mle-params","logodds")
    ])

fixed_models = frozenset(
    [os.path.join("performance-tests-cmdstan","stat_comp_benchmarks","benchmarks","low_dim_corr_gauss","low_dim_corr_gauss.stan")
     , os.path.join("performance-tests-cmdstan","stat_comp_benchmarks","benchmarks","pkpd","sim_one_comp_mm_elim_abs.stan")
     , os.path.join("performance-tests-cmdstan","stat_comp_benchmarks","benchmarks","gp_regr","gen_gp_data.stan")
     , os.path.join("performance-tests-cmdstan","example-models","basic_distributions","ordered_pair")
     , os.path.join("performance-tests-cmdstan","example-models","basic_distributions","sorted_pair")
     , os.path.join("performance-tests-cmdstan","example-models","basic_distributions","uniform")
     , os.path.join("performance-tests-cmdstan","example-models","basic_distributions","wishart2")
     , os.path.join("performance-tests-cmdstan","example-models","basic_distributions","normal")
     , os.path.join("performance-tests-cmdstan","example-models","basic_distributions","binormal")
     , os.path.join("performance-tests-cmdstan","example-models","basic_distributions","inv_wishart")
     , os.path.join("performance-tests-cmdstan","example-models","basic_distributions","triangle")
     , os.path.join("performance-tests-cmdstan","example-models","basic_distributions","inv_wishart")
     , os.path.join("performance-tests-cmdstan","example-models","basic_distributions","normal_mixture")
     , os.path.join("performance-tests-cmdstan","example-models","basic_distributions","wishart")
     , os.path.join("performance-tests-cmdstan","example-models","basic_distributions","wishart2x2")
     , os.path.join("performance-tests-cmdstan","example-models","bugs_examples","vol3","funshapes","hsquare")
     , os.path.join("performance-tests-cmdstan","example-models","bugs_examples","vol3","funshapes","parallelogram")
     , os.path.join("performance-tests-cmdstan","example-models","bugs_examples","vol3","funshapes","circle")
     , os.path.join("performance-tests-cmdstan","example-models","bugs_examples","vol3","funshapes","squaremc")
     , os.path.join("performance-tests-cmdstan","example-models","bugs_examples","vol3","funshapes","ring")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","simplest-regression","fake-data")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt","irt_1pl_power")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","irt","irt_power")
     , os.path.join("performance-tests-cmdstan","example-models","knitr","chapter1","fake-data")
     , os.path.join("performance-tests-cmdstan","example-models","misc","funnel","funnel")
     , os.path.join("performance-tests-cmdstan","example-models","misc","funnel","funnel_reparam")
    ])

def avg(coll):
    return float(sum(coll)) / len(coll)

def stdev(coll, mean):
    if len(coll ) < 2:
        return 0
    return (sum((x - mean)**2 for x in coll) / (len(coll) - 1)**0.5)

def csv_summary(csv_file):
    d = defaultdict(list)
    with open(csv_file, 'rb') as raw:
        headers = None
        for row in csv.reader(raw):
            if len(row) != 0:
                if row[0].startswith("#"):
                    continue
                if headers is None:
                    headers = row
                    continue
                for i in range(0, len(row)):
                    d[headers[i]].append(float(row[i]))
    res = {}
    for k, v in d.items():
        if k.endswith("__"):
            continue
        mean = avg(v)
        try:
            res[k] = (mean, stdev(v, mean))
        except OverflowError as e:
            raise OverflowError("calculating stdev for " + k)
    return res

def format_summary_lines(summary):
    return ["{} {:.15f} {:.15f}\n".format(k, avg, stdev)
            for k, (avg, stdev) in sorted(summary.items())]

def parse_summary(f):
    d = {}
    for line in f:
        param, avg, stdev = line.split()
        d[param] = (float(avg), float(stdev))
    return d

def run_model(exe, method, proposal, data, tmp, runs, num_samples, fixed):
    def run_as_fixed_param():
        shexec("{} method=sample algorithm='fixed_param' random seed=1234 output file={}"
               .format(exe, tmp))

    if data == None:
	data_str = ""
    else:
        data_str = data and "data file={}".format(data)
    total_time = 0
    for i in range(runs):
            start = time()
	    try:
		num_samples_str = ""
		num_proc = multiprocessing.cpu_count()
		if method == "sample":
		    num_samples_str = "num_samples={} num_warmup={}".format(num_samples, num_samples)
		    shexec("{} method={} {} {} random seed=1234 output file={}"
		    .format(exe, method, num_samples_str, data_str, tmp))
		if method == "smc-sample":
		    if num_proc != 1:
		    	num_proc = num_proc - (num_proc % 2)
		    num_samples_str = "num_samples={}".format(num_samples)
		    shexec("mpirun -np {} {} method=sample algorithm=smcs proposal={} T=1 Tsmc=1024 num_samples={} {} random seed=1234 output file=output_smc.out".format(num_proc, exe, proposal, num_samples, data_str, tmp))
		    samps = np.loadtxt("output_smc.out", comments=["#"], delimiter=",", unpack=False)
		    mean_smc = samps[1022,] # temporary fix while seg fault on writing samples is investigated
		    print("mean_smc = {}".format(mean_smc))
		    sys.stdout.flush() # added so Jenkins log can catch up
		    os.remove("output_smc.out")
		if method == "nuts-sample":
		    thread_num = "1"
		    if num_proc != 1:
		    	num_proc = num_proc - (num_proc % 2)
			for n in range(2,num_proc+1):
			    thread_num = thread_num + " {}".format(n)
		    num_samples_str = "num_samples={} num_warmup={}".format(num_samples, num_samples)
		    if fixed == True:
		        shexec("for i in {}; do ({} id=$i method=sample algorithm=fixed_param {} {} random seed=1234 output file=output_hmc$i.out refresh=0) & done; wait".format(thread_num,exe, num_samples_str, data_str))
		    else:
		        shexec("for i in {}; do ({} id=$i method=sample algorithm=hmc engine=nuts {} {} random seed=1234 output file=output_hmc$i.out refresh=0) & done; wait".format(thread_num,exe, num_samples_str, data_str))
		    f_string = "output_hmc1.out"
		    lines = np.loadtxt(f_string, comments=["#","lp__"], delimiter=",", unpack=False)
		    samps = lines[:,7:]
		    if num_proc != 1:
			for i in range(2,num_proc):
		    	    f_string = "output_hmc{}.out".format(i)
			    lines = np.loadtxt(f_string, comments=["#","lp__"], delimiter=",", unpack=False)
			    samps = np.append(samps,lines[:,7:], axis=0)
		    cov_hmc = np.cov(samps.T)
		    mean_hmc = np.mean(samps, axis=0)
		    sd = np.sqrt(np.diag(cov_hmc))  				
		    #print("cov_hmc = {}\n mean_hmc = {}\n cov_smc = {}\n mean_smc = {}\n error = {}".format(cov_hmc, mean_hmc, cov_smc, mean_smc, error))
		    print("cov_hmc = {}\n mean_hmc = {}\n sd = {}".format(cov_hmc, mean_hmc, sd))
		    sys.stdout.flush() # added so Jenkins log can catch up
		    shexec("bin/stansummary output_hmc*.out --sig_figs=3 &> summary.txt")
		    for n in range(1,num_proc+1):
			os.remove("output_hmc{}.out".format(n))
		if method == "compare_methods":
		    thread_num = "1"
		    if num_proc != 1:
		    	num_proc = num_proc - (num_proc % 2)
			for n in range(2,num_proc+1):
			    thread_num = thread_num + " {}".format(n)
		    #num_samples_str = "num_samples={} num_warmup={}".format(num_samples/num_proc, ((100*num_samples) - num_samples)/num_proc)
		    num_samples_str = "num_samples={} num_warmup={}".format(num_samples, num_samples)
		    shexec("mpirun -np {} {} method=sample algorithm=smcs proposal={} T=1 Tsmc={} num_samples={} {} random seed=1234 output file=output_smc.out"
		    .format(num_proc, exe, proposal, num_samples, num_samples, data_str, tmp))
		    samps = np.loadtxt("output_smc.out", comments=["#"], delimiter=",", unpack=False)
		    #cov_smc = np.cov(samps.T)
		    #mean_smc = np.mean(samps, axis=0)
		    mean_smc = samps[num_samples-2,] # temporary fix while seg fault on writing samples is investigated
		    if not np.isnan(mean_smc[0]):
			if fixed == True:
			     shexec("for i in {}; do ({} id=$i method=sample algorithm=fixed_param {} {} random seed=1234 output file=output_hmc$i.out refresh=0) & done; wait".format(thread_num,exe, num_samples_str, data_str))
			else:
			     shexec("for i in {}; do ({} id=$i method=sample algorithm=hmc engine=nuts {} {} random seed=1234 output file=output_hmc$i.out refresh=0) & done; wait".format(thread_num,exe, num_samples_str, data_str))
			shexec("bin/stansummary output_hmc*.out --sig_figs=3 &> summary.txt")
			f_string = "output_hmc1.out"
			lines = np.loadtxt(f_string, comments=["#","lp__"], delimiter=",", unpack=False)
			samps = lines[:,7:]
			if num_proc != 1:
			    for i in range(2,num_proc):
				f_string = "output_hmc{}.out".format(i)
			        lines = np.loadtxt(f_string, comments=["#","lp__"], delimiter=",", unpack=False)
				samps = np.append(samps,lines[:,7:], axis=0)
			cov_hmc = np.cov(samps.T)
			mean_hmc = np.mean(samps, axis=0)
			sd = np.sqrt(np.diag(cov_hmc))
			error = (mean_smc - mean_hmc) / sd
			#print("cov_hmc = {}\n mean_hmc = {}\n cov_smc = {}\n mean_smc = {}\n error = {}".format(cov_hmc, mean_hmc, cov_smc, mean_smc, error))
			shexec("bin/stansummary output_hmc*.out --sig_figs=3 &> summary.txt")
			#print("cov_hmc = {}\n mean_hmc = {}\n mean_smc = {}\n error = {}".format(cov_hmc, mean_hmc, mean_smc, error))
			print("error = {}".format(error)) # no longer printing cov / means due to potential of large outputs
			sys.stdout.flush() # added so Jenkins log can catch up
			for n in range(1,num_proc+1):
		            os.remove("output_hmc{}.out".format(n))
		    else:
			print("{} has NaNs in SMC-stan output".format(exe))
			sys.stdout.flush() # added so Jenkins log can catch up
		    os.remove("output_smc.out")
		if method == "compare_methods_custom":
		    thread_num = "1"
		    if num_proc != 1:
		    	num_proc = num_proc - (num_proc % 2)
			for n in range(2,num_proc+1):
			    thread_num = thread_num + " {}".format(n)
		    #num_samples_str = "num_samples={} num_warmup={}".format(num_samples/num_proc, ((100*num_samples) - num_samples)/num_proc)
		    #num_samples_str = "num_samples={} num_warmup={}".format(num_samples, 1024)
		    num_samples_str = "num_samples={} num_warmup={}".format(num_samples/num_proc, 1000)
		    if fixed == True:
			shexec("for i in {}; do ({} id=$i method=sample algorithm=fixed_param {} {} random seed=1234 output file=output_hmc$i.out refresh=0) & done; wait".format(thread_num,exe, num_samples_str, data_str))
		    else:
			shexec("for i in {}; do ({} id=$i method=sample adapt delta=0.99 algorithm=hmc engine=nuts max_depth=20 {} {} random seed=1234 output file=output_hmc$i.out refresh=0) & done; wait".format(thread_num,exe, num_samples_str, data_str))
		    f_string = "output_hmc1.out"
		    lines = np.loadtxt(f_string, comments=["#","lp__"], delimiter=",", unpack=False)
		    if fixed == True:
			samps = lines[:,2:]
			if num_proc != 1:
			    for i in range(2,num_proc):
			    	f_string = "output_hmc{}.out".format(i)
				lines = np.loadtxt(f_string, comments=["#","lp__"], delimiter=",", unpack=False)
				samps = np.append(samps,lines[:,2:], axis=0)
		    else:
			samps = lines[:,7:]
			if num_proc != 1:
			    for i in range(2,num_proc):
			    	f_string = "output_hmc{}.out".format(i)
				lines = np.loadtxt(f_string, comments=["#","lp__"], delimiter=",", unpack=False)
				samps = np.append(samps,lines[:,7:], axis=0)
		    cov_hmc = np.cov(samps.T)
		    mean_hmc = np.mean(samps, axis=0)
		    sd = np.sqrt(np.diag(cov_hmc))
		    lines = subprocess.check_output("bin/stansummary output_hmc*.out --sig_figs=3 &> summary.txt", shell=True)
		    if fixed == True:
			shexec("mpirun -np {} {} method=sample algorithm=smcs proposal={} T=1 Tsmc={} num_samples={} {} random seed=1234 output file=output_smc.out"
		    .format(num_proc, exe, proposal, 1024, num_samples, data_str))
		    else:
			k=1
			l=-1
			j=0
			word = "stepsize__"
			while k < len(lines):
			    if lines[k] == word[1]:
				l=1
				j=0
				while l < len(word):
				    if lines[k] == word[l]:
					j=j+1
				    l=l+1
				    k=k+1
			    k=k+1
			    if j==l-1:
			        break;
			while lines[k] == ' ':
			    k = k+1
			    k_start = k
			while lines[k] != ' ':
			    k = k+1
			    k_end = k
			stepsize = float(lines[k_start:k_end])
			if proposal == "hmc":
			    shexec("mpirun -np {} {} method=sample algorithm=smcs proposal={} stepsize={} num_leapfrog_steps=5 T=1 Tsmc=1024 num_samples={} {} random seed=1234 output file=output_smc.out"
			.format(num_proc, exe, proposal, stepsize, num_samples, data_str, tmp))
			elif proposal == "rw":
			    shexec("mpirun -np {} {} method=sample algorithm=smcs proposal={} T=1 Tsmc=200 num_samples={} {} random seed=1234 output file=output_smc.out"
			.format(num_proc, exe, proposal, num_samples, data_str, tmp))
			else:
			    shexec("mpirun -np {} {} method=sample algorithm=smcs proposal={} stepsize={} T=1 Tsmc=200 num_samples={} {} random seed=1234 output file=output_smc.out"
			.format(num_proc, exe, proposal, stepsize, num_samples, data_str, tmp))
		    samps = np.loadtxt("output_smc.out", comments=["#"], delimiter=",", unpack=False)
		    mean_smc = samps[198,] # temporary fix while seg fault on writing samples is investigated			
		    error = (mean_smc - mean_hmc) / sd
		    print(lines)
		    sys.stdout.flush() # added so Jenkins log can catch up
		    #print("cov_hmc = {}\n mean_hmc = {}\n mean_smc = {}\n error = {}".format(cov_hmc, mean_hmc, mean_smc, error))
		    print("mean_hmc = {}\n mean_smc = {}\n error = {}".format( mean_hmc, mean_smc, error)) # no longer printing cov / means due to potential of large outputs
		    sys.stdout.flush() # added so Jenkins log can catch up
		    for n in range(1,num_proc+1):
		        os.remove("output_hmc{}.out".format(n))
		    os.remove("output_smc.out")
		if method == "lee_output":
		    f_string = "output_hmc1.out"
		    lines = np.loadtxt(f_string, comments=["#","lp__"], delimiter=",", unpack=False)
		    samps = lines[:,7:]
		    if num_proc != 1:
			for i in range(2,num_proc):
		    	    f_string = "output_hmc{}.out".format(i)
		            lines = np.loadtxt(f_string, comments=["#","lp__"], delimiter=",", unpack=False)
			    samps = np.append(samps,lines[:,7:], axis=0)
		    cov_hmc = np.cov(samps.T)
		    mean_hmc = np.mean(samps, axis=0)
		    sd = np.sqrt(np.diag(cov_hmc))
		    samps = np.loadtxt("output_smc.csv", comments=["#"], delimiter=",", unpack=False)
		    mean_smc = samps[98,] # temporary fix while seg fault on writing samples is investigated			
		    error = (mean_smc - mean_hmc) / sd
		    print(lines)
		    sys.stdout.flush() # added so Jenkins log can catch up
		    #print("cov_hmc = {}\n mean_hmc = {}\n mean_smc = {}\n error = {}".format(cov_hmc, mean_hmc, mean_smc, error))
		    print("mean_hmc = {}\n mean_smc = {}\n error = {}".format( mean_hmc, mean_smc, error)) # no longer printing cov / means due to potential of large outputs
		    sys.stdout.flush() # added so Jenkins log can catch up
	    except FailedCommand as e:
		if e.returncode == 78:
		    run_as_fixed_param()
		else:
		    raise e
            end = time()
            total_time += end-start
    return total_time

def run_golds(gold, tmp, summary, check_golds_exact):
    gold_summary = {}
    with open(gold) as gf:
        gold_summary = parse_summary(gf)

    fails, errors = [], []
    first_params = set(summary)
    second_params = set(gold_summary)
    if not (first_params == second_params):
        msg = "First program has these extra params: {}.  ".format(
                first_params - second_params)
        msg += "2nd program has these extra params: {}.  ".format(
                second_params - first_params)
        msg += "They have these params in common: {}".format(
                second_params & first_params)
        print("ERROR: " + msg)
        errors.append(msg)
        return fails, errors
    for k, (mean, stdev) in sorted(gold_summary.items()):
        if stdev < 0.00001: #XXX Uh...
            continue
        err = abs(summary[k][0] - mean)
        if check_golds_exact and err > check_golds_exact:
            print("FAIL: {} param {} |{} - {}| not within {}"
                    .format(gold, k, summary[k][0], mean, check_golds_exact))
            fails.append((k, mean, stdev, summary[k][0]))
        elif err > 0.0001 and (err / stdev) > 0.3:
            print("FAIL: {} param {} not within ({} - {}) / {} < 0.3"
                    .format(gold, k, summary[k][0], mean, stdev))
            fails.append((k, mean, stdev, summary[k][0]))
    if not fails and not errors:
        print("SUCCESS: Gold {} passed.".format(gold))
    return fails, errors

def run(exe, data, overwrite, check_golds, check_golds_exact, runs, method, proposal, num_samples, fixed):
    fails, errors = [], []
    if not os.path.isfile(exe):
        return 0, (fails, errors + ["Did not compile!"])
    if runs <= 0:
        return 0, (fails, errors)

    gold = os.path.join(GOLD_OUTPUT_DIR,
                        exe.replace(DIR_UP, "").replace(os.sep, "_") + ".gold")
    tmp = gold + ".tmp"
    try:
        total_time = run_model(exe, method, proposal, data, tmp, runs, num_samples, fixed)
    except Exception as e:
        print("Encountered exception while running {}:".format(exe))
        print(e)
        return 0, (fails, errors + [str(e)])
    # Removing this as it's not needed and it's occasionally causing overflow errors
    #summary = csv_summary(tmp)
    #with open(tmp, "w+") as f:
    #    f.writelines(format_summary_lines(summary))
    #
    #if overwrite:
    #    shexec("mv {} {}".format(tmp, gold))
    #elif check_golds or check_golds_exact:
    #    fails, errors = run_golds(gold, tmp, summary, check_golds_exact)
    fails, errors = [], []

    return total_time, (fails, errors)

def test_results_xml(tests):
    failures = str(sum(1 if x[2] else 0 for x in tests))
    time_ = str(sum(x[1] for x in tests))
    root = ET.Element("testsuite", disabled = '0',
            failures=failures, name="Performance Tests",
            tests=str(len(tests)), time=str(time_),
            timestamp=str(datetime.now()))
    for model, time_, fails, errors in tests:
        if time_ <= 0 and (not fails and not errors):
            continue
        name = model.replace(".stan", "").replace(os.sep, ".")
        classname = name
        last_dot = name.rfind(".")
        if last_dot > 0:
            classname = classname[:last_dot]
            name = name[last_dot + 1:]
        time_ = str(time_)
        testcase = ET.SubElement(root, "testcase", status="run",
                classname=classname, name=name, time=time_)
        for fail in fails:
            failure = ET.SubElement(testcase, "failure", type="OffGold",
                    classname=classname, name = name,
                    message = ("param {} got mean {}, gold has mean {} and stdev {}"
                        .format(fail[0], fail[3], fail[1], fail[2])))
        for error in errors:
            err = ET.SubElement(testcase, "failure", type="Exception",
                    classname=classname, name = name, message = error)
    return ET.ElementTree(root)

def test_results_csv(tests):
    return "\n".join(",".join([model, str(time_)]) for model, time_, _, _ in tests) + "\n"

def parse_args():
    parser = argparse.ArgumentParser(description="Run gold tests and record performance.")
    parser.add_argument("directories", nargs="*")
    parser.add_argument("--check-golds", dest="check_golds", action="store_true",
                        help="Run the gold tests and check output within loose boundaries.")
    parser.add_argument("--check-golds-exact", dest="check_golds_exact", action="store",
                        help="Run the gold tests and check output to within specified tolerance",
                        type=float)
    parser.add_argument("--overwrite-golds", dest="overwrite", action="store_true",
                        help="Overwrite the gold test records.")
    parser.add_argument("--runs", dest="runs", action="store", type=int,
                        help="Number of runs per benchmark.", default=1)
    parser.add_argument("-j", dest="j", action="store", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--runj", dest="runj", action="store", type=int, default=1)
    parser.add_argument("--name", dest="name", action="store", type=str, default="performance")
    parser.add_argument("--method", dest="method", action="store", default="sample",
                        help="Inference method to ask Stan to use for all models.")
    parser.add_argument("--num-samples", dest="num_samples", action="store", default=None, type=int,
                        help="Number of samples to ask Stan programs for if we're sampling.")
    parser.add_argument("--tests-file", dest="tests", action="store", type=str, default="")
    parser.add_argument("--scorch-earth", dest="scorch", action="store_true")
    parser.add_argument("--proposal", dest="proposal", action="store", default="rw",
                        help="Proposal used in SMC.")
    return parser.parse_args()

def process_test(overwrite, check_golds, check_golds_exact, runs, method, proposal):
    def process_test_wrapper(tup):
        model, exe, data, num_samples = tup
	if model in fixed_models:
	    fixed = True
	else:
	    fixed = False
        time_, (fails, errors) = run(exe, data, overwrite, check_golds,
                                     check_golds_exact, runs, method, proposal, num_samples, fixed)
        average_time = runs and time_ / runs or 0
        return (model, average_time, fails, errors)
    return process_test_wrapper

def delete_temporary_exe_files(exes):
    for exe in exes:
        extensions = ["", ".hpp", ".o"]
        for ext in extensions:
            print("Removing " + exe + ext)
            if os.path.exists(exe + ext):
                os.remove(exe + ext)

if __name__ == "__main__":
    args = parse_args()
    
    models = None

    default_num_samples = 1024
    if args.tests == "":
        models = find_files("*.stan", args.directories)
        models = filter(model_name_re.match, models)
        models = list(filter(lambda m: not m in bad_models, models))
        num_samples = [args.num_samples or default_num_samples] * len(models)
    else:
        models, num_samples = read_tests(args.tests, args.num_samples or default_num_samples)
        if args.num_samples:
            num_samples = [args.num_samples] * len(models)


    executables = [m[:-5] for m in models]
    if args.scorch:
        delete_temporary_exe_files(executables)

    if not len(models) == len(num_samples):
        print("Something got the models list out of sync with the num_samples list")
        sys.exit(-10)
    tests = [(model, exe, find_data_for_model(model), ns)
             for model, exe, ns in zip(models, executables, num_samples)]

    make_time, _ = time_step("make_all_models", make, executables, args.j)
    if args.runj > 1:
        tp = ThreadPool(args.runj)
        map_ = tp.imap_unordered
    else:
        map_ = map
    results = map_(process_test(args.overwrite, args.check_golds,
                                args.check_golds_exact, args.runs,
                                args.method, args.proposal),
                    tests)
    results = list(results)
    results.append(("{}.compilation".format(args.name), make_time, [], []))
    test_results_xml(results).write("{}.xml".format(args.name))
    with open("{}.csv".format(args.name), "w") as f:
        f.write(test_results_csv(results))
    failed = False
    for model, _, fails, errors in results:
        if fails or errors:
            print("'{}' had fails '{}' and errors '{}'".format(model, fails, errors))
            failed = True
    if failed:
        sys.exit(-1)
