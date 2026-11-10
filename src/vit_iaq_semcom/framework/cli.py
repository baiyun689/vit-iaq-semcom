"""Single command surface for the native experiment framework."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def main(argv=None):
    parser=argparse.ArgumentParser(description="原生图像语义传输实验框架")
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("validate","plan","run","calibrate","evaluate"):
        p=sub.add_parser(name);p.add_argument("--config",required=True)
        if name in ("run","evaluate"):p.add_argument("--resume",action="store_true",help="恢复相同配置的已有运行")
        if name=="evaluate":
            p.add_argument("--checkpoints-from",required=True,help="已完成原始训练实验的输出目录")
            p.add_argument("--allow-channel-shift",action="store_true",help="显式允许冻结评价改变信道类型或接收噪声估计偏差")
    p=sub.add_parser("summarize");p.add_argument("--run-dir",required=True);p.add_argument("--bootstrap-samples",type=int,default=1000)
    p=sub.add_parser("fit-classifier");p.add_argument("--config",required=True);p.add_argument("--output","--out",dest="out",required=True)
    p.add_argument("--full-model",action="store_true");p.add_argument("--resume",action="store_true")
    sub.add_parser("doctor")
    args=parser.parse_args(argv)
    try:
        if args.command=="doctor":
            import torch
            from importlib.metadata import version,PackageNotFoundError
            packages={}
            for name in ("torch","torchvision","timm","numpy","scipy","pyyaml","matplotlib"):
                try:packages[name]=version(name)
                except PackageNotFoundError:packages[name]="missing"
            result={"python":sys.version,"executable":sys.executable,"packages":packages,"cuda_available":torch.cuda.is_available(),"torch_cuda":torch.version.cuda}
            if torch.cuda.is_available():
                result["gpus"]=[{"name":torch.cuda.get_device_name(i),"memory_gb":round(torch.cuda.get_device_properties(i).total_memory/2**30,2)} for i in range(torch.cuda.device_count())]
        elif args.command=="summarize":
            from .metrics import summarize
            result=summarize(args.run_dir,args.bootstrap_samples)
            result={"complete":result["complete"],"observations":result["observations"],"summary_dir":str(Path(args.run_dir).resolve()/"summary")}
        else:
            cfg=load_config(args.config)
            if args.command=="validate":result={"valid":True,"name":cfg["name"],"output_dir":cfg["output_dir"]}
            elif args.command=="plan":
                from .runner import plan
                result=plan(cfg)
                out=Path(cfg["output_dir"]);out.mkdir(parents=True,exist_ok=True)
                target=out/"plan.json"
                serialized=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
                if target.exists() and json.loads(target.read_text(encoding="utf8")) != result:
                    raise ValueError("Existing plan differs; use a new output_dir")
                target.write_text(serialized,encoding="utf8")
                result={"plan":str(target),"jobs":len(result.get("jobs",[])),"training_started":False}
            elif args.command in ("run","evaluate"):
                from .runner import run,evaluate
                result=run(cfg,resume=args.resume) if args.command=="run" else evaluate(cfg,args.checkpoints_from,resume=args.resume,allow_channel_shift=args.allow_channel_shift)
                from .metrics import summarize
                summary=summarize(cfg["output_dir"],cfg["experiment"]["bootstrap_samples"])
                result={"run_dir":cfg["output_dir"],"complete":summary["complete"],"observations":summary["observations"]}
            elif args.command=="fit-classifier":
                from .classifier_training import fit_classifier
                result=fit_classifier(cfg,args.out,full_model=args.full_model,resume=args.resume)
            else:
                from .runner import calibrate
                result=calibrate(cfg)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except (ValueError,FileNotFoundError,RuntimeError) as exc:
        print(f"实验未完成：{exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
