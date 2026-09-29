"""Run with python -m option_agent_lab --help."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from .data import generate_data
from .evaluation import evaluate, summarize
from .model import load_predictor, train_model
from .report import make_report
from .search import init_session, run_baselines, session_status, step_session, write_json


def audit_model(data, run):
    predictor=load_predictor(run/"model.pt")
    result={}
    for split in ["test","stress"]:
        x=np.load(data/f"{split}.npz")["X"]
        result[split]=summarize(evaluate(x,predictor))
    write_json(run/"audit.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ["demo","generate","train","benchmark","audit","report","session-init","session-status","session-step"]:
        p=sub.add_parser(name)
        p.add_argument("--run",type=Path,default=Path("runs/demo"))
        if name in ["demo","generate","train","audit"]:p.add_argument("--data",type=Path,default=Path("data/synthetic"))
        if name in ["demo","train"]:
            p.add_argument("--epochs",type=int,default=200)
            p.add_argument("--model-seed",type=int,default=7)
        if name in ["demo","benchmark","session-init"]:p.add_argument("--budget",type=int,default=1024)
        if name in ["demo","benchmark"]:p.add_argument("--repeats",type=int,default=5)
        if name.startswith("session-"):p.add_argument("--name",required=True)
        if name=="session-init":p.add_argument("--seed",type=int,default=123)
        if name=="session-step":p.add_argument("--proposal",type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=="demo":
            if args.run.exists():
                raise ValueError("Run directory already exists; choose a new --run to preserve previous results")
            generate_data(args.data)
            train_model(args.data,args.run,epochs=args.epochs,seed=args.model_seed)
            run_baselines(args.run,args.budget,range(123,123+args.repeats))
            audit_model(args.data,args.run)
            versions={p:importlib.metadata.version(p) for p in ["numpy","scipy","torch","matplotlib"]}
            write_json(args.run/"environment.json",{"python":platform.python_version(),"platform":platform.platform(),"packages":versions})
            result=make_report(args.run)
        elif args.command=="generate":result=generate_data(args.data)
        elif args.command=="train":
            if (args.run/"model.pt").exists():raise ValueError("Checkpoint exists; choose another run directory")
            result=train_model(args.data,args.run,epochs=args.epochs,seed=args.model_seed)
        elif args.command=="benchmark":result=run_baselines(args.run,args.budget,range(123,123+args.repeats))
        elif args.command=="audit":result=audit_model(args.data,args.run)
        elif args.command=="report":result=make_report(args.run)
        elif args.command=="session-init":result=init_session(args.run,args.name,args.budget,args.seed)
        elif args.command=="session-status":result=session_status(args.run,args.name)
        elif args.command=="session-step":result=step_session(args.run,args.name,json.loads(args.proposal.read_text()))
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,FileNotFoundError) as exc:
        parser.exit(2,f"Error: {exc}\n")


if __name__=="__main__":main()
