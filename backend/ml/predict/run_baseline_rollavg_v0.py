import argparse
import json

from app.core.db import SessionLocal
from app.api.routes.predictions import run_baseline_rollavg_v0_core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-gw", type=int, default=None)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = run_baseline_rollavg_v0_core(
            db=db,
            target_gw=args.target_gw,
            window=args.window,
        )
        print(json.dumps(result, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()