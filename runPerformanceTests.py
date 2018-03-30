#!/usr/bin/python

import argparse
import csv
from collections import defaultdict
import os
import re
import subprocess
from difflib import SequenceMatcher
from fnmatch import fnmatch
from functools import wraps
from multiprocessing.pool import ThreadPool
from time import time
import xml.etree.ElementTree as ET

GOLD_OUTPUT_DIR = "golds/"

def find_files(pattern, dirs):
    res = []
    for pd in dirs:
        for d, _, flist in os.walk(pd):
            for f in flist:
                if fnmatch(f, pattern):
                    res.append(os.path.join(d, f))
    return res

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

def shexec(command):
    print(command)
    returncode = subprocess.call(command, shell=True)
    if returncode != 0:
        raise Exception("return code '{}' from command '{}'!"
                        .format(returncode, command))
    return returncode

def make(targets, j=8):
    shexec("cd cmdstan; make -j{} {}"
           .format(j, " ".join("../" + t for t in targets)))

model_name_re = re.compile(".*/[A-z_][^/]+\.stan$")

bad_models = frozenset(
    ["examples/example-models/ARM/Ch.21/finite_populations.stan"
     , "examples/example-models/ARM/Ch.21/multiple_comparison.stan"
     , "examples/example-models/ARM/Ch.21/r_sqr.stan"
     , "examples/example-models/ARM/Ch.23/electric_1a.stan"
     , "examples/example-models/ARM/Ch.23/educational_subsidy.stan"
     , "examples/example-models/bugs_examples/vol2/pines/pines-3.stan"
     , "examples/example-models/bugs_examples/vol3/fire/fire.stan"
     # The following have data issues
              , "examples/example-models/ARM/Ch.10/ideo_two_pred.stan"
     , "examples/example-models/ARM/Ch.16/radon.1.stan"
     , "examples/example-models/ARM/Ch.16/radon.2.stan"
     , "examples/example-models/ARM/Ch.16/radon.2a.stan"
     , "examples/example-models/ARM/Ch.16/radon.2b.stan"
     , "examples/example-models/ARM/Ch.16/radon.3.stan"
     , "examples/example-models/ARM/Ch.16/radon.nopooling.stan"
     , "examples/example-models/ARM/Ch.16/radon.pooling.stan"
     , "examples/example-models/ARM/Ch.18/radon.1.stan"
     , "examples/example-models/ARM/Ch.18/radon.2.stan"
     , "examples/example-models/ARM/Ch.18/radon.nopooling.stan"
     , "examples/example-models/ARM/Ch.18/radon.pooling.stan"
     , "examples/example-models/ARM/Ch.19/item_response.stan"
     , "examples/example-models/bugs_examples/vol1/dogs/dogs.stan"
     , "examples/example-models/bugs_examples/vol1/rats/rats_stanified.stan"
     , "examples/example-models/bugs_examples/vol2/pines/pines-4.stan"
     , "examples/example-models/bugs_examples/vol2/pines/fit.stan"
     , "examples/example-models/BPA/Ch.06/MtX.stan"
     , "examples/example-models/ARM/Ch.21/radon_vary_intercept_a.stan"
     , "examples/example-models/ARM/Ch.21/radon_vary_intercept_b.stan"
     , "examples/example-models/ARM/Ch.23/sesame_street2.stan"
     , "examples/example-models/ARM/Ch.3/kidiq_validation.stan"
     , "examples/example-models/ARM/Ch.7/earnings_interactions.stan"
     , "examples/example-models/ARM/Ch.8/y_x.stan"
     , "examples/example-models/basic_estimators/normal_mixture_k.stan"
     , "examples/example-models/basic_estimators/normal_mixture_k_prop.stan"
     , "examples/example-models/BPA/Ch.04/GLM0.stan"
     , "examples/example-models/BPA/Ch.04/GLM1.stan"
     , "examples/example-models/BPA/Ch.04/GLM2.stan"
     , "examples/example-models/BPA/Ch.04/GLMM3.stan"
     , "examples/example-models/BPA/Ch.04/GLMM4.stan"
     , "examples/example-models/BPA/Ch.04/GLMM5.stan"
     , "examples/example-models/BPA/Ch.05/ssm2.stan"
     , "examples/example-models/BPA/Ch.07/cjs_group_raneff.stan"
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
        res[k] = (mean, stdev(v, mean))
    return res

def format_summary_lines(summary):
    return ["{} {} {}\n".format(k, avg, stdev) for k, (avg, stdev) in summary.items()]

def parse_summary(f):
    d = {}
    for line in f:
        param, avg, stdev = line.split()
        d[param] = (float(avg), float(stdev))
    return d

def run(exe, data, overwrite, check_golds, check_golds_exact, runs, cmdstan_args):
    fails, errors = [], []
    gold = os.path.join(GOLD_OUTPUT_DIR,
                        exe.replace("../", "").replace("/", "_") + ".gold")
    tmp = gold + ".tmp"
    try:
        total_time = 0
        for i in range(runs):
            start = time()
            shexec("{} {} data file={} random seed=1234 output file={}"
                   .format(exe, cmdstan_args, data, tmp))
            end = time()
            total_time += end-start
    except Exception as e:
        print("Encountered exception while running {}:".format(exe))
        print(e)
        return 0, (fails, errors + [str(e)])
    summary = csv_summary(tmp)
    with open(tmp, "w+") as f:
        f.writelines(format_summary_lines(summary))

    if overwrite:
        shexec("mv {} {}".format(tmp, gold))
    elif check_golds or check_golds_exact:
        gold_summary = {}
        with open(gold) as gf:
            gold_summary = parse_summary(gf)

        for k, (mean, stdev) in gold_summary.items():
            if stdev < 0.00001: #XXX Uh...
                continue
            err = abs(summary[k][0] - mean)
            if check_golds_exact and err > check_golds_exact:
                print("FAIL: {} param {} |{} - {}| not within {}"
                      .format(gold, k, summary[k][0], mean, check_golds_exact))
                fails.append((k, mean, stdev, summary[k][0]))
            elif err > 0.0001 and (err / stdev) > 0.5:
                print("FAIL: {} param {} not within ({} - {}) / {} < 0.5"
                      .format(gold, k, summary[k][0], mean, stdev))
                fails.append((k, mean, stdev, summary[k][0]))
    return total_time, (fails, errors)

def test_results_xml(tests):
    failures = str(sum(1 if x[2] else 0 for x in tests))
    time_ = str(sum(x[1] for x in tests))
    root = ET.Element("testsuite", failures=failures, name="Performance Tests",
                      tests=str(len(tests)), time=str(time_))
    for model, time_, fails, errors in tests:
        name = model.replace(".stan", "").replace("/", ".")
        time_ = str(time_)
        testcase = ET.SubElement(root, "testcase", classname=name, time=time_)
        for fail in fails:
            testcase = ET.SubElement(root, "failure", type="param mismatch")
            testcase.text = ("param {} got mean {}, gold has mean {} and stdev {}"
                             .format(fail[0], fail[3], fail[1], fail[2]))
        for error in errors:
            testcase = ET.SubElement(root, "error", type="Exception")
            testcase.text = error
    return ET.ElementTree(root)

def test_results_csv(tests):
    return "\n".join(",".join([model, str(time_)]) for model, time_, _, _ in tests) + "\n"

def parse_args():
    parser = argparse.ArgumentParser(description="Run gold tests and record performance.")
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--check-golds", dest="check_golds", action="store_true",
                        help="Run the gold tests and check output within loose boundaries.")
    parser.add_argument("--check-golds-exact", dest="check_golds_exact", action="store",
                        help="Run the gold tests and check output to within specified tolerance",
                        type=float)
    parser.add_argument("--overwrite-golds", dest="overwrite", action="store_true",
                        help="Overwrite the gold test records.")
    parser.add_argument("--runs", dest="runs", action="store", type=int,
                        help="Number of runs per benchmark.", default=1)
    parser.add_argument("-j", dest="j", action="store", type=int, default=4)
    parser.add_argument("--runj", dest="runj", action="store", type=int, default=1)
    parser.add_argument("--cmdstan-args", dest="cmdstan_args", action="store",
                        default="method=sample",
                        help="Options to cmdstan binary. Must include method=")
    return parser.parse_args()

def process_test(overwrite, check_golds, check_golds_exact, runs, cmdstan_args):
    def process_test_wrapper(tup):
        # TODO: figure out the right place to compute the average or maybe don't compute the average.
        model, exe, data = tup
        time_, (fails, errors) = run(exe, data, overwrite, check_golds,
                                     check_golds_exact, runs, cmdstan_args)
        average_time = time_ / runs
        return (model, average_time, fails, errors)
    return process_test_wrapper

if __name__ == "__main__":
    args = parse_args()

    models = find_files("*.stan", args.directories)
    models = filter(model_name_re.match, models)
    models = list(filter(lambda m: not m in bad_models, models))
    executables = [m[:-5] for m in models]
    make_time, _ = time_step("make_all_models", make, executables, args.j)
    tests = [(model, exe, find_data_for_model(model))
             for model, exe in zip(models, executables)]
    tests = filter(lambda x: x[2], tests)
    if args.runj > 1:
        tp = ThreadPool(args.runj)
        map_ = tp.imap_unordered
    else:
        map_ = map
    results = map_(process_test(args.overwrite, args.check_golds,
                                args.check_golds_exact, args.runs,
                                args.cmdstan_args),
                    tests)
    results = list(results)
    results.append(("compilation", make_time, [], []))
    test_results_xml(results).write("performance.xml")
    with open("performance.csv", "w") as f:
        f.write(test_results_csv(results))
    for model, _, fails, errors in results:
        if fails or errors:
            print("'{}' had fails '{}' and errors '{}'".format(model, fails, errors))
